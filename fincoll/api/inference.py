"""
Inference API endpoints for real-time data extraction
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional
import logging
import os
import pandas as pd

from ..providers.alphavantage_client import AlphaVantageClient
from ..features.feature_extractor import FeatureExtractor
from ..storage.influxdb_cache import get_cache
from config.dimensions import DIMS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])

# Global data provider injected by server.py startup (MultiProviderFetcher)
_data_provider = None


def set_provider(data_provider) -> None:
    """Set global data provider (called by server.py on startup)."""
    global _data_provider
    _data_provider = data_provider
    logger.info(
        f"✅ Inference router: data provider set to {data_provider.__class__.__name__}"
    )


def _require_provider():
    """Return the global provider or raise 503."""
    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")
    return _data_provider


def _calculate_lookback_timedelta(lookback_bars: int, interval: str):
    """
    Calculate the timedelta needed for a given number of lookback bars and interval.

    Args:
        lookback_bars: Number of bars to look back
        interval: Time interval (30s, 1m, 5m, 30m, 1h, 1d, etc.)

    Returns:
        timedelta object representing the lookback period
    """
    interval_seconds = {
        "30s": 30,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }

    seconds = interval_seconds.get(interval, 86400)  # Default to 1 day

    # For intraday intervals, add extra time to account for market closures
    if interval in ["30s", "1m", "5m", "15m", "30m", "1h"]:
        # Market is open ~6.5 hours per day (390 minutes)
        # Add 50% buffer for market closures and extended hours
        total_seconds = lookback_bars * seconds * 1.5
    else:
        total_seconds = lookback_bars * seconds

    return timedelta(seconds=total_seconds)


def _fetch_ohlcv(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    influx_cache,
    provider_label: str,
):
    """
    Cache-first OHLCV fetch using the global MultiProviderFetcher.

    Returns DataFrame or raises HTTPException.
    """
    data_provider = _require_provider()

    # Check InfluxDB cache first
    if influx_cache.enabled:
        cached_df = influx_cache.get_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            source="any",
        )
        if cached_df is not None and len(cached_df) >= 50:
            logger.info(f"✅ Cache HIT: {symbol} {interval} - {len(cached_df)} bars")
            return cached_df, True

    # Cache miss – fetch via MultiProviderFetcher
    logger.info(f"Cache MISS: {symbol} {interval} - fetching from provider")
    df = data_provider.get_historical_bars(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )

    if df is not None and not df.empty and influx_cache.enabled:
        influx_cache.store_bars(symbol, df, interval=interval, source=provider_label)
        logger.info(f"Stored {len(df)} bars in InfluxDB cache for {symbol}")

    return df, False


def _validate_ohlcv(df, symbol: str, min_bars: int = 50):
    """Validate OHLCV DataFrame; raise HTTPException on failure."""
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    if len(df) < min_bars:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {symbol}: need at least {min_bars} bars, got {len(df)}",
        )


def _resolve_timestamp(ts, symbol: str) -> datetime:
    """Coerce index value to datetime; raise HTTPException on bad value."""
    if ts is None or (hasattr(ts, "__class__") and pd.isna(ts)):
        raise HTTPException(
            status_code=500, detail=f"Internal error: Null timestamp for {symbol}"
        )
    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: NaT timestamp for {symbol}",
            )
        return ts.to_pydatetime()
    if isinstance(ts, datetime):
        return ts
    raise HTTPException(
        status_code=500,
        detail=f"Internal error: Invalid timestamp type {type(ts)} for {symbol}",
    )


@router.get("/features/{symbol}")
async def get_inference_features(
    symbol: str,
    lookback: int = Query(960, description="Number of bars to look back"),
    provider: str = Query(
        "tradestation",
        description="Data provider hint (tradestation, yfinance) — actual routing handled by MultiProviderFetcher",
    ),
    interval: str = Query("1d", description="Bar interval (30s, 1m, 5m, 30m, 1h, 1d)"),
    format: str = Query(
        "latest", description="Feature format (uses latest version from config)"
    ),
):
    """
    Extract real-time features for inference

    Returns latest feature vector for prediction (dimension from config).
    Data routing is handled automatically by MultiProviderFetcher (TradeStation → yfinance).

    For live trading with 1-minute bars, use:
    provider=tradestation&interval=1m&lookback=960
    """
    try:
        logger.info(
            f"Inference feature request: {symbol} (provider={provider}, interval={interval}, lookback={lookback})"
        )

        data_provider = _require_provider()

        # Initialize AlphaVantage client for fundamentals/news
        av_client = AlphaVantageClient()

        # Initialize feature extractor
        extractor = FeatureExtractor(
            alpha_vantage_client=av_client,
            cache_fundamentals=True,
            cache_news=True,
            enable_senvec=True,
            enable_futures=True,
            data_provider=data_provider,
        )

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - _calculate_lookback_timedelta(lookback, interval)

        logger.info(f"Fetching data from {start_date} to {end_date}")

        influx_cache = get_cache()
        ohlcv_data, _cache_hit = _fetch_ohlcv(
            symbol, interval, start_date, end_date, influx_cache, provider.lower()
        )
        _validate_ohlcv(ohlcv_data, symbol)

        logger.info(f"Fetched {len(ohlcv_data)} bars for {symbol}")

        latest_timestamp = _resolve_timestamp(ohlcv_data.index[-1], symbol)

        try:
            feature_vector = extractor.extract_features(
                ohlcv_data=ohlcv_data,
                symbol=symbol,
                timestamp=latest_timestamp,
            )

            logger.info(
                f"Extracted {len(feature_vector)}D feature vector for {symbol} at {latest_timestamp}"
            )

            return {
                "symbol": symbol,
                "timestamp": latest_timestamp.isoformat(),
                "features": feature_vector.tolist(),
                "metadata": {
                    "feature_dimensions": len(feature_vector),
                    "lookback_bars": len(ohlcv_data),
                    "interval": interval,
                    "provider": data_provider.get_name()
                    if hasattr(data_provider, "get_name")
                    else data_provider.__class__.__name__,
                    "senvec_enabled": extractor.enable_senvec,
                },
            }

        except Exception as e:
            logger.error(f"Feature extraction failed for {symbol}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Feature extraction failed: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference feature extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def get_batch_inference_features(request: dict):
    """
    Batch inference feature extraction for multiple symbols

    Expects JSON with:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "lookback": 960,
        "format": "latest"
    }
    """
    try:
        symbols = request.get("symbols", [])
        logger.info(f"Batch inference feature request for {len(symbols)} symbols")

        results = []
        for symbol in symbols:
            # TODO: Process each symbol
            results.append({"symbol": symbol, "status": "not_implemented"})

        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "symbols_requested": len(symbols),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{symbol}")
async def get_latest_bars(
    symbol: str,
    lookback: int = Query(960, description="Number of bars"),
    timeframe: str = Query("30s", description="Timeframe (30s, 1m, 5m, 30m)"),
):
    """
    Get latest OHLCV bars for a symbol (without feature extraction)

    Useful for debugging and data verification.
    """
    try:
        logger.info(f"Latest bars request: {symbol} ({timeframe}, lookback={lookback})")

        # TODO: Fetch from provider
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback": lookback,
            "timestamp": datetime.now().isoformat(),
            "status": "not_implemented",
            "message": "Use /api/v1/bars/{symbol} for OHLCV data",
        }

    except Exception as e:
        logger.error(f"Latest bars error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL INFERENCE ENDPOINTS
# ============================================================================


@router.post("/predict/batch", include_in_schema=False)
async def predict_batch_alias(request: dict):
    """Alias registered BEFORE /predict/{symbol} so FastAPI routes /predict/batch here."""
    return await batch_predict_velocity(request)


@router.post("/predict/{symbol}")
async def predict_symbol(
    symbol: str,
    lookback: int = Query(960, description="Number of bars to look back"),
    provider: str = Query(
        "tradestation",
        description="Data provider hint — actual routing via MultiProviderFetcher",
    ),
    interval: str = Query("1d", description="Bar interval (30s, 1m, 5m, 30m, 1h, 1d)"),
):
    """
    Velocity Prediction Endpoint

    Extracts features from FinColl and runs velocity model inference.
    Returns velocity predictions for multiple timeframes.

    CRITICAL: Stores SymVectors (feature vectors) to InfluxDB for training data collection.
    """
    try:
        import asyncio
        from ..inference import get_velocity_engine
        from ..storage.influxdb_saver import InfluxDBFeatureSaver
        import numpy as np

        logger.info(
            f"Velocity prediction request: {symbol} (provider={provider}, interval={interval})"
        )

        data_provider = _require_provider()

        # Initialize SymVector saver (fails gracefully if InfluxDB unavailable)
        feature_saver = None
        try:
            feature_saver = InfluxDBFeatureSaver(
                url=os.getenv("INFLUXDB_URL"),
                token=os.getenv("INFLUXDB_TOKEN"),
                org=os.getenv("INFLUXDB_ORG"),
                bucket=os.getenv("INFLUXDB_FEATURE_BUCKET", "feature_vectors"),
            )
            logger.info(f"✅ SymVector storage enabled for {symbol}")
        except Exception as e:
            logger.error(f"❌ SymVector storage FAILED (InfluxDB unavailable): {e}")

        # Get velocity engine (loads model on first call)
        engine = get_velocity_engine()

        # Initialize AlphaVantage client for fundamentals/news
        av_client = AlphaVantageClient()

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - _calculate_lookback_timedelta(lookback, interval)

        # BATCH OPTIMIZATION: Pre-fetch main symbol + SPY + sector ETF
        sector_etf_map = {
            "AAPL": "XLK",
            "MSFT": "XLK",
            "GOOGL": "XLK",
            "AMZN": "XLY",
            "META": "XLC",
            "NVDA": "XLK",
            "TSLA": "XLK",
        }
        sector_etf = sector_etf_map.get(symbol, "XLK")
        symbols_to_prefetch = [symbol, "SPY", sector_etf]

        influx_cache = get_cache()
        for prefetch_symbol in symbols_to_prefetch:
            try:
                _df = data_provider.get_historical_bars(
                    symbol=prefetch_symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                )
                if _df is not None and not _df.empty and influx_cache.enabled:
                    influx_cache.store_bars(
                        prefetch_symbol, _df, interval=interval, source=provider.lower()
                    )
                logger.info(f"  ✅ Pre-fetched {prefetch_symbol}")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to pre-fetch {prefetch_symbol}: {e}")

        # Initialize feature extractor (uses the global provider for SPY/ETF lookups)
        extractor = FeatureExtractor(
            alpha_vantage_client=av_client,
            cache_fundamentals=True,
            cache_news=True,
            enable_senvec=True,
            enable_futures=True,
            data_provider=data_provider,
        )

        # Fetch symbol OHLCV (cache-first)
        ohlcv_data, _cache_hit = _fetch_ohlcv(
            symbol, interval, start_date, end_date, influx_cache, provider.lower()
        )
        _validate_ohlcv(ohlcv_data, symbol)

        latest_timestamp = _resolve_timestamp(ohlcv_data.index[-1], symbol)

        feature_vector = extractor.extract_features(
            ohlcv_data=ohlcv_data, symbol=symbol, timestamp=latest_timestamp
        )
        logger.debug(
            f"Feature vector length for {symbol}: {feature_vector.shape[0]} "
            f"(expected {DIMS.fincoll_total})"
        )

        # CRITICAL: Store SymVector to InfluxDB for training data collection
        if feature_saver is not None:
            try:
                saved = feature_saver.save_feature_vector(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    features=feature_vector,
                    source=provider.lower(),
                    metadata={"interval": interval, "lookback": lookback},
                )
                if saved:
                    logger.info(
                        f"💾 SymVector stored: {symbol} @ {latest_timestamp} ({feature_vector.shape[0]}D)"
                    )
                else:
                    logger.warning(f"⚠️  SymVector storage failed for {symbol}")
            except Exception as e:
                logger.warning(
                    f"⚠️  SymVector storage error for {symbol}: {e} (non-critical)"
                )

        # Get current price from last bar
        close_col = "Close" if "Close" in ohlcv_data.columns else "close"
        current_price = float(ohlcv_data[close_col].iloc[-1])

        # Run velocity model inference
        result = engine.predict(feature_vector, symbol, current_price)

        logger.info(
            f"Velocity prediction for {symbol}: {result.get('best_opportunity', {}).get('direction', 'NONE')}"
        )

        provider_name = (
            data_provider.get_name()
            if hasattr(data_provider, "get_name")
            else data_provider.__class__.__name__
        )
        result.setdefault("metadata", {})["provider"] = provider_name.lower()

        # Add A/B testing metadata (Phase 4C)
        from ..ab_testing_manager import ab_manager

        result["ab_test"] = ab_manager.get_variant_metadata(symbol)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch")
async def batch_predict_velocity(request: dict):
    """
    Batch Velocity Prediction Endpoint (OPTIMIZED)

    Predicts velocities for multiple symbols in parallel.
    This is the RECOMMENDED endpoint for PIM to use.

    Request body:
        {
            "symbols": ["AAPL", "GOOGL", "MSFT", ...],
            "lookback": 960,
            "provider": "tradestation",
            "interval": "1d"
        }
    """
    try:
        import asyncio
        import time
        from ..inference import get_velocity_engine
        from ..storage.influxdb_saver import InfluxDBFeatureSaver

        symbols = request.get("symbols", [])
        lookback = request.get("lookback", 960)
        provider_str = request.get("provider", "tradestation")
        interval = request.get("interval", "1d")

        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")

        logger.info(
            f"🚀 Batch velocity prediction: {len(symbols)} symbols (provider={provider_str}, interval={interval})"
        )

        data_provider = _require_provider()
        start_time = time.time()

        # Initialize SymVector saver (fails gracefully)
        feature_saver = None
        try:
            feature_saver = InfluxDBFeatureSaver(
                url=os.getenv("INFLUXDB_URL"),
                token=os.getenv("INFLUXDB_TOKEN"),
                org=os.getenv("INFLUXDB_ORG"),
                bucket=os.getenv("INFLUXDB_FEATURE_BUCKET", "feature_vectors"),
            )
            logger.info(
                f"✅ SymVector storage enabled for batch ({len(symbols)} symbols)"
            )
        except Exception as e:
            logger.error(f"❌ SymVector storage FAILED (InfluxDB unavailable): {e}")

        engine = get_velocity_engine()
        av_client = AlphaVantageClient()

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - _calculate_lookback_timedelta(lookback, interval)

        influx_cache = get_cache()

        extractor = FeatureExtractor(
            alpha_vantage_client=av_client,
            cache_fundamentals=True,
            cache_news=True,
            enable_senvec=True,
            enable_futures=True,
            data_provider=data_provider,
        )

        async def predict_single_symbol(symbol: str):
            """Predict velocity for a single symbol (async)."""
            try:
                # Fetch OHLCV (cache-first, blocking call in thread pool)
                loop = asyncio.get_event_loop()
                ohlcv_data, _cache_hit = await loop.run_in_executor(
                    None,
                    lambda: _fetch_ohlcv(
                        symbol,
                        interval,
                        start_date,
                        end_date,
                        influx_cache,
                        provider_str.lower(),
                    ),
                )

                if ohlcv_data is None or ohlcv_data.empty:
                    return {"symbol": symbol, "error": "No data found"}
                if len(ohlcv_data) < 50:
                    return {
                        "symbol": symbol,
                        "error": f"Insufficient data: {len(ohlcv_data)} bars",
                    }

                latest_timestamp = _resolve_timestamp(ohlcv_data.index[-1], symbol)
                feature_vector = extractor.extract_features(
                    ohlcv_data=ohlcv_data, symbol=symbol, timestamp=latest_timestamp
                )

                # Store SymVector
                if feature_saver is not None:
                    try:
                        from datetime import datetime as dt

                        feature_saver.save_feature_vector(
                            symbol=symbol,
                            timestamp=dt.now(),
                            features=feature_vector,
                            source=provider_str.lower(),
                            metadata={
                                "interval": interval,
                                "lookback": lookback,
                                "batch": True,
                            },
                        )
                    except Exception as e:
                        logger.debug(f"⚠️  SymVector storage error for {symbol}: {e}")

                close_col = "Close" if "Close" in ohlcv_data.columns else "close"
                current_price = float(ohlcv_data[close_col].iloc[-1])

                result = await loop.run_in_executor(
                    None, lambda: engine.predict(feature_vector, symbol, current_price)
                )

                from ..ab_testing_manager import ab_manager

                result["ab_test"] = ab_manager.get_variant_metadata(symbol)

                logger.info(
                    f"✅ {symbol}: {result.get('best_opportunity', {}).get('direction', 'NONE')}"
                )
                return result

            except HTTPException as he:
                return {"symbol": symbol, "error": he.detail}
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                return {"symbol": symbol, "error": str(e)}

        # Predict all symbols with concurrency limit to avoid OOM
        logger.info(
            f"⚡ Parallel prediction: {len(symbols)} symbols (max 5 concurrent)"
        )
        _infer_sem = asyncio.Semaphore(5)

        async def _predict_with_sem(sym: str):
            async with _infer_sem:
                return await predict_single_symbol(sym)

        predictions = await asyncio.gather(*[_predict_with_sem(s) for s in symbols])

        successful = len([p for p in predictions if "error" not in p])
        failed = len([p for p in predictions if "error" in p])
        elapsed = time.time() - start_time

        logger.info(
            f"✅ Batch complete: {successful} successful, {failed} failed in {elapsed:.1f}s"
        )

        provider_name = (
            data_provider.get_name()
            if hasattr(data_provider, "get_name")
            else data_provider.__class__.__name__
        )

        return {
            "predictions": predictions,
            "summary": {
                "total": len(symbols),
                "successful": successful,
                "failed": failed,
                "elapsed_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
                "provider": provider_name,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch velocity prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_predict(
    symbols: list[str],
    lookback: int = Query(100, description="Number of days of OHLC data"),
    provider: str = Query(
        "tradestation",
        description="Data provider hint (tradestation, yfinance)",
    ),
):
    """
    Batch Prediction Endpoint (LEGACY - Use /predict/batch for velocity predictions)

    Run model inference on multiple symbols using TradeStation data.
    NOTE: This is the OLD endpoint for backward compatibility.
    New code should use /predict/batch which returns velocity predictions.
    """
    try:
        import asyncio
        from ..inference import get_prediction_engine

        logger.info(
            f"Batch prediction request: {len(symbols)} symbols (provider={provider})"
        )

        data_provider = _require_provider()
        engine = get_prediction_engine()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback)
        influx_cache = get_cache()

        async def fetch_symbol_data(symbol: str):
            """Fetch data for a single symbol (runs in thread pool)."""
            try:
                loop = asyncio.get_event_loop()
                ohlcv_data, _cache_hit = await loop.run_in_executor(
                    None,
                    lambda: _fetch_ohlcv(
                        symbol,
                        "1d",
                        start_date,
                        end_date,
                        influx_cache,
                        provider.lower(),
                    ),
                )

                if ohlcv_data is not None and len(ohlcv_data) >= 20:
                    if isinstance(ohlcv_data, pd.DataFrame):
                        if "Open" in ohlcv_data.columns:
                            ohlc_data = ohlcv_data[
                                ["Open", "High", "Low", "Close", "Volume"]
                            ].values
                        else:
                            ohlc_data = ohlcv_data[
                                ["open", "high", "low", "close", "volume"]
                            ].values
                    else:
                        ohlc_data = ohlcv_data
                    return symbol, ohlc_data, None
                else:
                    bars = len(ohlcv_data) if ohlcv_data is not None else 0
                    logger.warning(
                        f"Skipping {symbol}: insufficient data ({bars} bars)"
                    )
                    return symbol, None, f"Insufficient data ({bars} bars)"
            except Exception as e:
                logger.warning(f"Failed to download {symbol}: {e}")
                return symbol, None, str(e)

        logger.info(f"🚀 Parallel fetch: {len(symbols)} symbols")
        fetch_results = await asyncio.gather(*[fetch_symbol_data(s) for s in symbols])

        ohlc_data_list = []
        valid_symbols = []
        for sym, data, error in fetch_results:
            if data is not None:
                ohlc_data_list.append(data)
                valid_symbols.append(sym)

        results = engine.batch_predict(ohlc_data_list, valid_symbols)

        provider_name = (
            data_provider.get_name()
            if hasattr(data_provider, "get_name")
            else data_provider.__class__.__name__
        )
        timestamp = datetime.now().isoformat()
        for result in results:
            if "error" not in result:
                result["timestamp"] = timestamp
                result["model_version"] = "finvec-continuous"
                result["provider"] = provider_name

        successful = len([r for r in results if "error" not in r])
        failed = len([r for r in results if "error" in r])
        logger.info(f"Batch prediction: {successful} successful, {failed} failed")

        return {
            "predictions": results,
            "summary": {
                "total": len(symbols),
                "successful": successful,
                "failed": failed,
                "timestamp": timestamp,
                "provider": provider_name,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
