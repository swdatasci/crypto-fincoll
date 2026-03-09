"""
Enriched Inference API - Layer 2 Output

Provides enriched feature vectors with labels, interpretations, and context
for PIM Layer 2 RL agents.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import numpy as np
import uuid
from threading import Lock
from pathlib import Path
import yaml

from ..features.feature_labeler import FeatureLabeler
from ..features.context_generator import ContextGenerator
from ..features.feature_extractor import FeatureExtractor
from ..providers.alphavantage_client import AlphaVantageClient
from config.dimensions import DIMS  # type: ignore[import]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/inference/enriched", tags=["enriched"])

# Global data provider injected by server.py startup (MultiProviderFetcher)
_data_provider = None

# Initialize shared instances (singleton pattern with thread safety)
_feature_labeler = None
_context_generator = None
_labeler_lock = Lock()
_context_lock = Lock()


def set_provider(data_provider) -> None:
    """Set global data provider (called by server.py on startup)."""
    global _data_provider
    _data_provider = data_provider
    logger.info(
        f"✅ Enriched router: data provider set to {data_provider.__class__.__name__}"
    )


def _require_provider():
    """Return the global provider or raise 503."""
    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")
    return _data_provider


def get_feature_labeler() -> FeatureLabeler:
    """Get or create FeatureLabeler singleton (thread-safe)"""
    global _feature_labeler
    if _feature_labeler is None:
        with _labeler_lock:
            # Double-check pattern to avoid race conditions
            if _feature_labeler is None:
                _feature_labeler = FeatureLabeler()
                logger.info("FeatureLabeler singleton initialized")
    return _feature_labeler


def get_context_generator() -> ContextGenerator:
    """Get or create ContextGenerator singleton (thread-safe)"""
    global _context_generator
    if _context_generator is None:
        with _context_lock:
            # Double-check pattern to avoid race conditions
            if _context_generator is None:
                _context_generator = ContextGenerator()
                logger.info("ContextGenerator singleton initialized")
    return _context_generator


@router.post("/symbol/{symbol}")
async def get_enriched_prediction(
    symbol: str,
    lookback: int = Query(960, description="Number of bars to look back"),
    interval: str = Query("1d", description="Bar interval (30s, 1m, 5m, 30m, 1h, 1d)"),
    provider: str = Query("tradestation", description="Data provider"),
):
    """
    Get enriched prediction for single symbol

    URL: POST /api/v1/inference/enriched/symbol/{symbol}

    Returns:
    - 414D labeled features (organized by category)
    - Velocity predictions from FinVec
    - LLM-friendly context (summary, signals, risks, recommendation)
    - Data quality metadata
    - Confidence scores

    This is the PRIMARY endpoint for PIM Layer 2 RL agents.
    """
    try:
        logger.info(
            f"Enriched prediction request: {symbol} (interval={interval}, lookback={lookback})"
        )

        # Step 1: Extract raw features
        raw_features = await _extract_raw_features(symbol, lookback, interval, provider)

        # Step 2: Get FinVec predictions
        predictions = await _get_finvec_predictions(symbol, raw_features)

        # Step 3: Label features
        labeler = get_feature_labeler()
        labeled_features = labeler.label(raw_features)

        # Step 4: Generate context
        context_gen = get_context_generator()
        context = context_gen.generate(symbol, labeled_features, predictions)

        # Step 5: Calculate data quality
        data_quality = _calculate_data_quality(raw_features, labeled_features)

        # Step 6: Calculate confidence
        confidence = _calculate_confidence(labeled_features, predictions, data_quality)

        # Step 7: Build enriched payload
        enriched_payload = {
            "prediction_id": str(uuid.uuid4()),
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "predictions": predictions,
            "features": labeled_features,
            "context_for_agent": context,
            "raw_features": raw_features.tolist(),
            "data_quality": data_quality,
            "confidence": confidence,
        }

        # Step 8: Validate before returning
        is_valid, error = _validate_enriched_payload(enriched_payload)
        if not is_valid:
            logger.warning(f"Enriched payload validation failed for {symbol}: {error}")
            raise HTTPException(status_code=400, detail=f"Validation failed: {error}")

        logger.info(
            f"Enriched prediction complete for {symbol}: {confidence['overall']:.2f} confidence"
        )
        return enriched_payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enriched prediction failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Enriched prediction failed: {str(e)}"
        )


@router.post("/batch")
async def get_enriched_batch(
    symbols: List[str] = Body(..., description="List of symbols to process"),
    lookback: int = Query(960, description="Number of bars to look back"),
    interval: str = Query("1d", description="Bar interval"),
    provider: str = Query("tradestation", description="Data provider"),
):
    """
    Get enriched predictions for batch of symbols

    URL: POST /api/v1/inference/enriched/batch

    Processes symbols in PARALLEL for optimal performance.

    This endpoint is designed for PIM Layer 2 batch processing.
    """
    import asyncio

    try:
        logger.info(
            f"Enriched batch request: {len(symbols)} symbols (parallel processing)"
        )

        # Process all symbols in parallel
        async def safe_predict(symbol: str):
            """Wrapper to catch exceptions per symbol"""
            try:
                result = await get_enriched_prediction(
                    symbol=symbol,
                    lookback=lookback,
                    interval=interval,
                    provider=provider,
                )
                return {"status": "success", "data": result}
            except Exception as e:
                logger.error(f"Batch item failed for {symbol}: {e}")
                return {"status": "error", "symbol": symbol, "error": str(e)}

        # Execute all predictions in parallel
        batch_results = await asyncio.gather(
            *[safe_predict(symbol) for symbol in symbols],
            return_exceptions=False,  # Already handling exceptions in safe_predict
        )

        # Separate successes and failures
        results = [r["data"] for r in batch_results if r["status"] == "success"]
        errors = [
            {"symbol": r["symbol"], "error": r["error"]}
            for r in batch_results
            if r["status"] == "error"
        ]

        logger.info(f"Batch complete: {len(results)} succeeded, {len(errors)} failed")

        return {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "symbols_requested": len(symbols),
            "symbols_completed": len(results),
            "symbols_failed": len(errors),
            "results": results,
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"Enriched batch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== HELPER FUNCTIONS ==========


async def _extract_raw_features(
    symbol: str, lookback: int, interval: str, provider: str
) -> np.ndarray:
    """
    Extract raw 414D feature vector

    Returns numpy array of shape (414,) or (DIMS.fincoll_total,)
    """
    try:
        from datetime import timedelta

        # Use injected MultiProviderFetcher (provider param is a hint only)
        data_provider = _require_provider()

        # Fetch historical OHLCV data
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback)

        df = data_provider.get_historical_bars(
            symbol=symbol, start_date=start_dt, end_date=end_dt, interval=interval
        )
        if df is None or len(df) < 50:
            raise ValueError(
                f"Insufficient data for {symbol}: got {len(df) if df is not None else 0} bars, need >= 50"
            )

        # Initialize AlphaVantage client for fundamentals
        av_client = AlphaVantageClient()

        # Initialize feature extractor (MPF handles caching internally)
        extractor = FeatureExtractor(
            data_provider=data_provider,
            alpha_vantage_client=av_client,
            enable_senvec=True,
            cache_fundamentals=True,
            cache_news=True,
        )

        # Extract features from OHLCV data
        features = extractor.extract_features(
            df, symbol=symbol, timestamp=datetime.now()
        )

        if features is None or len(features) == 0:
            raise ValueError(f"No features extracted for {symbol}")

        if len(features) != DIMS.fincoll_total:
            raise ValueError(
                f"Feature dimension mismatch: expected {DIMS.fincoll_total}, got {len(features)}"
            )

        return features

    except Exception as e:
        logger.error(f"Raw feature extraction failed for {symbol}: {e}")
        raise


async def _get_finvec_predictions(
    symbol: str, raw_features: np.ndarray
) -> Dict[str, Any]:
    """
    Get velocity predictions from FinVec model

    Uses VelocityEngine to run real model inference on feature vectors.
    Returns velocity predictions for multiple timeframes with confidence scores.
    """
    try:
        from ..inference.velocity_engine import get_velocity_engine

        logger.debug(f"Getting FinVec velocity predictions for {symbol}")

        # Get velocity engine singleton (loads model on first call)
        engine = get_velocity_engine()

        # Get current price from features (close price is in technical indicators)
        # For now, use a placeholder - will be replaced with actual price from OHLCV data
        # The engine needs current price for return calculation
        current_price = 100.0  # Placeholder, will be overridden by context generator

        # Run velocity model inference
        velocity_result = engine.predict(raw_features, symbol, current_price)

        # Format predictions to match enriched API structure
        # Convert from velocity engine format to enriched API format
        # Velocity engine returns both LONG and SHORT for each timeframe, need to aggregate them
        velocities_by_tf = {}
        for vel_pred in velocity_result.get("velocities", []):
            tf = vel_pred["timeframe"]
            if tf not in velocities_by_tf:
                velocities_by_tf[tf] = {
                    "timeframe": tf,
                    "long_velocity": 0.0,
                    "long_bars": 0,
                    "long_confidence": 0.0,
                    "short_velocity": 0.0,
                    "short_bars": 0,
                    "short_confidence": 0.0,
                }

            # Populate long or short based on direction
            if vel_pred["direction"] == "LONG":
                velocities_by_tf[tf]["long_velocity"] = vel_pred["velocity"]
                velocities_by_tf[tf]["long_bars"] = vel_pred["bars"]
                velocities_by_tf[tf]["long_confidence"] = vel_pred["confidence"]
            elif vel_pred["direction"] == "SHORT":
                velocities_by_tf[tf]["short_velocity"] = vel_pred["velocity"]
                velocities_by_tf[tf]["short_bars"] = vel_pred["bars"]
                velocities_by_tf[tf]["short_confidence"] = vel_pred["confidence"]

        # Convert to list
        velocities = list(velocities_by_tf.values())

        # Get best opportunity from velocity engine
        best_opp = velocity_result.get("best_opportunity", {})

        predictions = {
            "velocities": velocities,
            "best_opportunity": {
                "timeframe": best_opp.get("timeframe", "unknown"),
                "direction": best_opp.get("direction", "NONE"),
                "velocity": best_opp.get("velocity", 0.0),
                "bars": best_opp.get("bars", 0),
                "confidence": best_opp.get("confidence", 0.0),
                "expected_return": best_opp.get("expected_return", 0.0),
            },
            "spike_alert": velocity_result.get("spike_alert", {}),
            "metadata": velocity_result.get("metadata", {}),
        }

        logger.info(
            f"Velocity prediction for {symbol}: {predictions['best_opportunity']['direction']} "
            f"@ {predictions['best_opportunity']['velocity']:.4f} "
            f"(confidence: {predictions['best_opportunity']['confidence']:.2f})"
        )

        return predictions

    except Exception as e:
        logger.error(f"FinVec prediction failed for {symbol}: {e}", exc_info=True)
        # Return empty predictions rather than failing
        return {
            "velocities": [],
            "best_opportunity": {},
            "spike_alert": {},
            "metadata": {"error": str(e)},
        }


def _calculate_data_quality(
    raw_features: np.ndarray, labeled_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate data quality metrics per feature group

    Returns:
    - by_group: Dict of {group_name: completeness_ratio}
    - missing_services: List of services that failed
    - stale_data_flags: Indicators of stale data
    """
    # Calculate per-group completeness
    by_group = {}

    for group_name, group_data in labeled_features.items():
        dimensions = group_data.get("dimensions", 0)

        if dimensions == 0:
            by_group[group_name] = 0.0
            continue

        # Count non-zero values in this group's data
        group_features_data = group_data.get("data", {})
        total_values = 0
        non_zero_values = 0

        def count_values(obj):
            nonlocal total_values, non_zero_values
            if isinstance(obj, dict):
                for v in obj.values():
                    count_values(v)
            elif isinstance(obj, (int, float)):
                total_values += 1
                if abs(obj) > 1e-10:
                    non_zero_values += 1

        count_values(group_features_data)

        completeness = (non_zero_values / total_values) if total_values > 0 else 0.0
        by_group[group_name] = completeness

    # Identify missing services (0% completeness)
    missing_services = [
        name for name, completeness in by_group.items() if completeness == 0.0
    ]

    total = len(raw_features)
    non_zero = int(np.count_nonzero(raw_features))
    return {
        "by_group": by_group,
        "feature_completeness": round(non_zero / total, 4) if total > 0 else 0.0,
        "total_features": total,
        "non_zero_features": non_zero,
        "missing_services": missing_services,
        "stale_data_flags": {
            "news": False,  # TODO: Check timestamp
            "fundamentals": False,
            "senvec": False,
        },
    }


def _calculate_confidence(
    labeled_features: Dict[str, Any],
    predictions: Dict[str, Any],
    data_quality: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate confidence scores based on data quality and model predictions

    Returns:
    - overall: Overall confidence (0-1)
    - by_component: Confidence breakdown by feature group
    """
    # Calculate base confidence from data quality
    by_group = data_quality.get("by_group", {})

    # Prefer pre-calculated feature_completeness if available
    if "feature_completeness" in data_quality:
        data_quality_conf = data_quality["feature_completeness"]
    else:
        non_zero_features = data_quality.get("non_zero_features", 0)
        total_features = data_quality.get("total_features", 1)
        data_quality_conf = (
            non_zero_features / total_features if total_features > 0 else 0.0
        )

    # Get prediction confidence from model
    pred_confidence = predictions.get("best_opportunity", {}).get("confidence", 0.5)

    # Calculate component confidences from feature groups
    technical_conf = by_group.get("technical_indicators", 0.5)
    sentiment_conf = by_group.get("senvec", 0.5)
    fundamental_conf = by_group.get("fundamentals", 0.5)

    # Overall confidence is weighted average of components
    # Model prediction has highest weight (70%), data quality 30%
    overall = (data_quality_conf * 0.3) + (pred_confidence * 0.7)

    return {
        "overall": round(overall, 4),
        "by_component": {
            "data_quality": round(data_quality_conf, 4),
            "predictions": round(pred_confidence, 4),
            "technical": round(technical_conf, 4),
            "sentiment": round(sentiment_conf, 4),
            "fundamentals": round(fundamental_conf, 4),
        },
    }


def _validate_enriched_payload(payload: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate enriched payload using thresholds from config/feature_dimensions.yaml

    Uses per-service thresholds and critical service checks defined in config.

    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    required_fields = ["symbol", "predictions", "features", "context_for_agent"]
    for field in required_fields:
        if field not in payload:
            return False, f"Missing required field: {field}"

    features = payload.get("features", {})
    data_quality = payload.get("data_quality", {})
    by_group = data_quality.get("by_group", {})

    # Check overall feature completeness threshold
    feature_completeness = data_quality.get("feature_completeness")
    if feature_completeness is not None and feature_completeness < 0.8:
        return (
            False,
            f"Low feature completeness: {feature_completeness:.2f} (minimum 0.80)",
        )

    # Check overall confidence threshold
    confidence = payload.get("confidence", {})
    overall_confidence = confidence.get("overall")
    if overall_confidence is not None and overall_confidence < 0.5:
        return False, f"Low overall confidence: {overall_confidence:.2f} (minimum 0.50)"

    # Load validation config from YAML
    config_path = (
        Path(__file__).parent.parent.parent / "config" / "feature_dimensions.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    validation_config = config.get("validation", {})
    service_thresholds = validation_config.get("service_thresholds", {})
    critical_services = validation_config.get("critical_services", [])
    fail_on_service_down = validation_config.get("fail_on_service_down", True)

    # Map feature group names to config keys (handle naming differences)
    group_to_config_key = {
        "technical_indicators": "technical",
        "advanced_technical": "advanced_technical",
        "velocity": "velocity",
        "news": "news",
        "fundamentals": "fundamentals",
        "cross_asset": "cross_asset",
        "sector": "sector",
        "options": "options",
        "support_resistance": "support_resistance",
        "vwap": "vwap",
        "senvec": "senvec_social",  # Map to social for now
        "futures": "futures",
        "finnhub": "finnhub",
        "early_signal": "early_signal",
        "market_neutral": "market_neutral",
        "advanced_risk": "advanced_risk",
        "momentum_variations": "momentum_variations",
    }

    # Check critical services (must have SOME non-zero data)
    if fail_on_service_down:
        for service_name in critical_services:
            # Find matching feature group
            matching_group = None
            for group_name, config_key in group_to_config_key.items():
                if config_key == service_name:
                    matching_group = group_name
                    break

            if matching_group and matching_group in by_group:
                completeness = by_group[matching_group]
                if completeness == 0.0:
                    return (
                        False,
                        f"Critical service '{service_name}' has no data (API down?)",
                    )

    # Check per-service thresholds
    for group_name, config_key in group_to_config_key.items():
        if config_key not in service_thresholds:
            continue  # No threshold defined for this group

        threshold = service_thresholds[config_key]

        # Skip if threshold is 1.0 (means disabled/not implemented)
        if threshold >= 1.0:
            continue

        if group_name in by_group:
            completeness = by_group[group_name]

            # Convert to "non-zero ratio" (config uses zero_threshold, we have completeness)
            non_zero_ratio = completeness

            if non_zero_ratio < threshold:
                logger.warning(
                    f"Feature group '{group_name}' below threshold: "
                    f"{non_zero_ratio:.2f} < {threshold:.2f} (acceptable if sparse)"
                )

    return True, ""
