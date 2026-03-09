"""
Backtesting API endpoints for historical feature extraction and prediction
"""

import os
import uuid
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..features.feature_extractor import FeatureExtractor
from config.dimensions import DIMS  # type: ignore[import]
from ..utils.api_credentials import APICredentials

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/backtesting", tags=["backtesting"])

# Global data provider injected by server.py startup (MultiProviderFetcher)
_data_provider = None


def set_provider(data_provider) -> None:
    """Set global data provider (called by server.py on startup)."""
    global _data_provider
    _data_provider = data_provider
    logger.info(
        f"✅ Backtesting router: data provider set to {data_provider.__class__.__name__}"
    )


def _require_provider():
    """Return the global provider or raise 503."""
    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")
    return _data_provider


# Cache directory for extracted features
CACHE_DIR = Path(
    os.getenv(
        "FINCOLL_CACHE_DIR", str(Path.home() / "caelum" / "ss" / "backtest_cache")
    )
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Job status tracking
extraction_jobs: Dict[str, Dict[str, Any]] = {}

# Velocity model cache (loaded once, reused)
_velocity_model = None
_velocity_model_path = None


# Request/Response Models


class VelocityBacktestRequest(BaseModel):
    """Request for historical velocity prediction (single date)"""

    symbol: str = Field(..., description="Stock symbol")
    date: str = Field(..., description="Historical date for prediction (YYYY-MM-DD)")
    interval: str = Field(default="1d", description="Bar interval (1d default)")
    provider: str = Field(default="yfinance", description="Data provider")


class VelocityBacktestResponse(BaseModel):
    """Response with velocity prediction for historical date"""

    symbol: str
    date: str
    current_price: float
    velocities: List[Dict[str, Any]]
    best_opportunity: Dict[str, Any]
    metadata: Dict[str, Any]


class ExtractFeaturesRequest(BaseModel):
    symbols: List[str] = Field(
        ..., description="List of symbols to extract features for"
    )
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    interval: str = Field(default="1d", description="Bar interval (1d, 1h, 5m, etc.)")
    provider: str = Field(
        default="yfinance", description="Data provider (yfinance, tradestation)"
    )
    output_format: str = Field(default="parquet", description="Output format (parquet)")
    output_path: Optional[str] = Field(
        None, description="Custom output path (optional)"
    )


class ExtractFeaturesResponse(BaseModel):
    job_id: str
    status: str
    symbols_requested: int
    date_range: Dict[str, str]
    estimated_completion_time: str
    estimated_features: int
    output_file: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Dict[str, Any]
    output_file: Optional[str] = None
    file_size_mb: Optional[float] = None
    completion_time: Optional[str] = None
    error: Optional[str] = None


class BacktestFromCacheRequest(BaseModel):
    cache_file: str = Field(..., description="Path to cached feature file (Parquet)")
    symbols: List[str] = Field(..., description="Symbols to backtest")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    model_checkpoint: Optional[str] = Field(
        None, description="FinVec model checkpoint to use"
    )


class BacktestFromCacheResponse(BaseModel):
    backtest_id: str
    symbols: List[str]
    date_range: Dict[str, str]
    predictions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Helper Functions


def _estimate_trading_days(start_date: str, end_date: str) -> int:
    """Estimate number of trading days between two dates (roughly 252 per year)"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days
    # Rough estimate: 5/7 of days are weekdays, 252/365 are trading days
    return int(days * 252 / 365)


def _get_data_provider(provider_name: str):
    """Get the global data provider (provider_name is a hint only; routing is internal)."""
    return _require_provider()


def _load_velocity_model():
    """Load velocity model (cached for reuse)"""
    global _velocity_model, _velocity_model_path

    if _velocity_model is not None:
        return _velocity_model

    # Find latest velocity model checkpoint
    finvec_dir = Path(__file__).parent.parent.parent.parent / "finvec"

    # Check for model in multiple locations
    possible_paths = [
        finvec_dir / "checkpoints" / "velocity" / "best_model.pt",  # Actual location
        finvec_dir / "simple_velocity_20241129.pth",
        finvec_dir / "checkpoints" / "simple_velocity_latest.pth",
        Path.home() / "caelum" / "ss" / "finvec" / "simple_velocity_20241129.pth",
    ]

    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(f"Velocity model not found. Checked: {possible_paths}")

    logger.info(f"Loading velocity model from {model_path}")

    # Import model class
    import sys

    sys.path.insert(0, str(finvec_dir))
    from models.simple_velocity_model import SimpleVelocityModel, ModelConfig

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Create model using checkpoint config
    if "config" in checkpoint:
        # Use config from checkpoint (input_dim, output_dim, etc.)
        ckpt_config = checkpoint["config"]
        config = ModelConfig(
            input_dim=ckpt_config.get("input_dim", DIMS.model_input),
            output_dim=ckpt_config.get("output_dim", 10),
        )
        logger.info(
            f"Using checkpoint config: input_dim={config.input_dim}, output_dim={config.output_dim}"
        )
    else:
        # Fallback to default config
        config = ModelConfig()
        logger.warning("No config in checkpoint, using default ModelConfig")

    model = SimpleVelocityModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _velocity_model = model
    _velocity_model_path = str(model_path)

    logger.info(f"✅ Velocity model loaded from {model_path}")

    return model


async def _extract_features_background(
    job_id: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str,
    provider_name: str,
    output_file: str,
):
    """
    Background task for feature extraction

    Extracts feature vectors for all symbols and saves to Parquet (dimension from config)
    """
    try:
        logger.info(f"[Job {job_id}] Starting feature extraction")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Interval: {interval}")
        logger.info(f"  Provider: {provider_name}")

        # Update job status
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["progress"]["symbols_completed"] = 0
        extraction_jobs[job_id]["progress"]["features_extracted"] = 0

        # Initialize providers
        data_provider = _get_data_provider(provider_name)
        av_client = None  # AlphaVantageClient() if needed

        # Initialize feature extractor
        extractor = FeatureExtractor(
            alpha_vantage_client=av_client,
            cache_fundamentals=True,
            cache_news=True,
            enable_senvec=True,
        )

        # Extract features for all symbols
        all_features = []

        for i, symbol in enumerate(symbols):
            try:
                logger.info(
                    f"[Job {job_id}] Processing {symbol} ({i + 1}/{len(symbols)})"
                )

                # Fetch historical data
                ohlcv_data = data_provider.get_historical_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                )

                if ohlcv_data is None or ohlcv_data.empty:
                    logger.warning(f"  No data for {symbol}, skipping")
                    continue

                if len(ohlcv_data) < 50:
                    logger.warning(
                        f"  Insufficient data for {symbol} ({len(ohlcv_data)} bars), skipping"
                    )
                    continue

                logger.info(f"  Fetched {len(ohlcv_data)} bars for {symbol}")

                # Extract features for each timestamp
                for timestamp in ohlcv_data.index:
                    # Get all data up to this timestamp
                    historical_data = ohlcv_data.loc[:timestamp]

                    if len(historical_data) < 50:
                        continue  # Need at least 50 bars for indicators

                    try:
                        feature_vector = extractor.extract_features(
                            ohlcv_data=historical_data,
                            symbol=symbol,
                            timestamp=timestamp,
                        )

                        # Store feature with metadata
                        all_features.append(
                            {
                                "symbol": symbol,
                                "timestamp": timestamp,
                                "features": feature_vector.tolist(),
                                "open": float(ohlcv_data.loc[timestamp, "open"]),
                                "high": float(ohlcv_data.loc[timestamp, "high"]),
                                "low": float(ohlcv_data.loc[timestamp, "low"]),
                                "close": float(ohlcv_data.loc[timestamp, "close"]),
                                "volume": float(ohlcv_data.loc[timestamp, "volume"]),
                            }
                        )

                    except Exception as e:
                        logger.error(
                            f"  Feature extraction failed for {symbol} at {timestamp}: {e}"
                        )
                        continue

                # Update progress
                extraction_jobs[job_id]["progress"]["symbols_completed"] = i + 1
                extraction_jobs[job_id]["progress"]["features_extracted"] = len(
                    all_features
                )
                extraction_jobs[job_id]["progress"]["percent_complete"] = int(
                    (i + 1) / len(symbols) * 100
                )

                logger.info(f"  Extracted {len(all_features)} total features so far")

            except Exception as e:
                logger.error(f"  Error processing {symbol}: {e}", exc_info=True)
                continue

        # Save to Parquet
        logger.info(f"[Job {job_id}] Saving {len(all_features)} features to Parquet")

        if len(all_features) == 0:
            raise ValueError("No features extracted for any symbols")

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Save as Parquet
        df.to_parquet(output_file, engine="pyarrow", compression="snappy")

        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)

        logger.info(
            f"[Job {job_id}] ✅ Complete! Saved to {output_file} ({file_size_mb:.1f} MB)"
        )

        # Update job status
        extraction_jobs[job_id]["status"] = "completed"
        extraction_jobs[job_id]["progress"]["percent_complete"] = 100
        extraction_jobs[job_id]["output_file"] = output_file
        extraction_jobs[job_id]["file_size_mb"] = file_size_mb
        extraction_jobs[job_id]["completion_time"] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"[Job {job_id}] ❌ Failed: {e}", exc_info=True)
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = str(e)


# API Endpoints


@router.post("/velocity", response_model=VelocityBacktestResponse)
async def get_historical_velocity_prediction(request: VelocityBacktestRequest):
    """
    Get velocity prediction for a specific historical date

    This endpoint:
    1. Fetches historical data UP TO the specified date (no look-ahead)
    2. Extracts features from that data
    3. Runs velocity model inference
    4. Returns multi-timeframe velocity predictions

    Used by backtesting system to get predictions AS IF it were that date.
    """
    try:
        logger.info(f"Historical velocity request: {request.symbol} on {request.date}")

        # Parse target date
        target_date = pd.Timestamp(request.date)

        # Fetch historical data UP TO target date (+ buffer for indicators)
        start_date = (target_date - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = target_date.strftime("%Y-%m-%d")

        # Get data provider
        data_provider = _get_data_provider(request.provider)

        # Fetch OHLCV data
        ohlcv_data = data_provider.get_historical_bars(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
            interval=request.interval,
        )

        if ohlcv_data is None or ohlcv_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data available for {request.symbol} up to {request.date}",
            )

        if len(ohlcv_data) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {request.symbol} ({len(ohlcv_data)} bars, need 50+)",
            )

        logger.info(f"  Fetched {len(ohlcv_data)} bars up to {request.date}")

        # Extract features
        extractor = FeatureExtractor(
            alpha_vantage_client=None,
            cache_fundamentals=True,
            cache_news=True,
            enable_senvec=True,
        )

        feature_vector = extractor.extract_features(
            ohlcv_data=ohlcv_data, symbol=request.symbol, timestamp=target_date
        )

        logger.info(f"  Extracted {len(feature_vector)}D feature vector")

        # Load velocity model
        model = _load_velocity_model()

        # Run inference
        with torch.no_grad():
            features_tensor = torch.tensor(
                feature_vector, dtype=torch.float32
            ).unsqueeze(0)
            predictions = model(features_tensor)

        # Parse model output (depends on model architecture)
        # Assuming output is [batch, num_timeframes * 4] where 4 = [long_vel, long_bars, short_vel, short_bars]

        timeframes = ["1min", "5min", "15min", "1hour", "daily"]
        velocities = []

        pred_array = predictions.squeeze(0).numpy()

        for i, tf in enumerate(timeframes):
            # Extract predictions for this timeframe
            idx = i * 4
            long_vel = float(pred_array[idx])
            long_bars = int(pred_array[idx + 1])
            short_vel = float(pred_array[idx + 2])
            short_bars = int(pred_array[idx + 3])

            # Choose best direction (highest absolute velocity)
            if abs(long_vel) > abs(short_vel):
                direction = "LONG"
                velocity = long_vel
                bars = long_bars
            else:
                direction = "SHORT"
                velocity = short_vel
                bars = short_bars

            velocities.append(
                {
                    "timeframe": tf,
                    "direction": direction,
                    "velocity": velocity,
                    "bars_to_target": bars,
                    "confidence": min(
                        0.95, 0.5 + abs(velocity) * 10
                    ),  # Simple confidence
                    "long_velocity": long_vel,
                    "long_bars": long_bars,
                    "short_velocity": short_vel,
                    "short_bars": short_bars,
                }
            )

        # Find best opportunity (highest absolute velocity with reasonable confidence)
        best = max(velocities, key=lambda v: abs(v["velocity"]) * v["confidence"])

        best_opportunity = {
            "timeframe": best["timeframe"],
            "direction": best["direction"],
            "velocity": best["velocity"],
            "confidence": best["confidence"],
            "bars_to_target": best["bars_to_target"],
            "expected_return_pct": best["velocity"] * 100,
        }

        # Get current price from last bar
        current_price = float(ohlcv_data.iloc[-1]["close"])

        logger.info(
            f"  ✅ Prediction: {best_opportunity['direction']} {best_opportunity['velocity']:.2%} ({best_opportunity['timeframe']})"
        )

        return VelocityBacktestResponse(
            symbol=request.symbol,
            date=request.date,
            current_price=current_price,
            velocities=velocities,
            best_opportunity=best_opportunity,
            metadata={
                "model_path": _velocity_model_path,
                "feature_dim": len(feature_vector),
                "data_bars": len(ohlcv_data),
                "provider": request.provider,
                "timeframes": timeframes,
                "inference_time_ms": 50,  # Rough estimate
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Historical velocity prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-features", response_model=ExtractFeaturesResponse)
async def extract_historical_features(
    request: ExtractFeaturesRequest, background_tasks: BackgroundTasks
):
    """
    Extract historical feature vectors for backtesting (dimension from config)

    This endpoint starts a background job to pre-compute features for all symbols
    across the specified date range. Features are cached to Parquet for fast
    backtesting.

    Returns job_id to poll for status.
    """
    try:
        # Generate job ID
        job_id = (
            f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        # Determine output file
        if request.output_path:
            output_file = request.output_path
        else:
            output_file = str(CACHE_DIR / f"features_{job_id}.parquet")

        # Estimate completion
        trading_days = _estimate_trading_days(request.start_date, request.end_date)
        estimated_features = len(request.symbols) * trading_days

        # Rough estimate: 1 symbol-day takes ~0.5 seconds
        estimated_seconds = len(request.symbols) * trading_days * 0.5
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)

        # Create job record
        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "symbols": request.symbols,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "interval": request.interval,
            "provider": request.provider,
            "output_file": output_file,
            "progress": {
                "symbols_completed": 0,
                "symbols_total": len(request.symbols),
                "features_extracted": 0,
                "features_total": estimated_features,
                "percent_complete": 0,
            },
            "created_at": datetime.now().isoformat(),
        }

        # Start background task
        background_tasks.add_task(
            _extract_features_background,
            job_id=job_id,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            provider_name=request.provider,
            output_file=output_file,
        )

        logger.info(f"✅ Feature extraction job created: {job_id}")

        return ExtractFeaturesResponse(
            job_id=job_id,
            status="queued",
            symbols_requested=len(request.symbols),
            date_range={
                "start": request.start_date,
                "end": request.end_date,
                "trading_days": str(trading_days),
            },
            estimated_completion_time=estimated_completion.isoformat(),
            estimated_features=estimated_features,
            output_file=output_file,
        )

    except Exception as e:
        logger.error(f"Extract features error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract-features/{job_id}", response_model=JobStatusResponse)
async def get_extraction_job_status(job_id: str):
    """
    Get status of feature extraction job

    Poll this endpoint to check progress of background extraction job.
    """
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = extraction_jobs[job_id]

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        output_file=job.get("output_file"),
        file_size_mb=job.get("file_size_mb"),
        completion_time=job.get("completion_time"),
        error=job.get("error"),
    )


@router.post("/predict-from-cache", response_model=BacktestFromCacheResponse)
async def backtest_from_cached_features(request: BacktestFromCacheRequest):
    """
    Run FinVec inference on pre-computed features (fast backtesting)

    This endpoint loads cached feature vectors from Parquet and runs
    FinVec inference to generate predictions. Much faster than extracting
    features on the fly.

    NOTE: Currently returns mock predictions. Will integrate with actual
    FinVec model checkpoint when available.
    """
    try:
        logger.info(f"Backtest from cache request: {request.cache_file}")

        # Verify cache file exists
        cache_path = Path(request.cache_file)
        if not cache_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Cache file not found: {request.cache_file}"
            )

        # Load cached features
        logger.info(f"Loading features from {cache_path}")
        df = pd.read_parquet(cache_path)

        logger.info(f"Loaded {len(df)} cached features")
        logger.info(f"  Unique symbols: {df['symbol'].nunique()}")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Filter by requested symbols and date range
        start_date = pd.Timestamp(request.start_date)
        end_date = pd.Timestamp(request.end_date)

        df_filtered = df[
            (df["symbol"].isin(request.symbols))
            & (df["timestamp"] >= start_date)
            & (df["timestamp"] <= end_date)
        ]

        logger.info(
            f"Filtered to {len(df_filtered)} features for requested symbols/dates"
        )

        if len(df_filtered) == 0:
            raise HTTPException(
                status_code=404,
                detail="No cached features found for requested symbols and date range",
            )

        # TODO: Load FinVec model checkpoint and run inference
        # For now, return mock predictions

        backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        predictions = []

        for _, row in df_filtered.iterrows():
            # Mock prediction (will replace with actual FinVec inference)
            prediction = {
                "symbol": row["symbol"],
                "timestamp": row["timestamp"].isoformat(),
                "current_price": row["close"],
                "prediction": {
                    "direction": "LONG",  # Mock
                    "confidence": 0.65,
                    "entry_signal": 0.70,
                    "expected_profit_pct": 1.5,
                    "expected_holding_days": 3.0,
                    "multi_horizon": {
                        "1d": {"return": 0.008, "confidence": 0.65},
                        "5d": {"return": 0.015, "confidence": 0.60},
                        "20d": {"return": 0.030, "confidence": 0.55},
                    },
                    "risk": {
                        "expected_drawdown": -0.02,
                        "volatility": 0.015,
                        "sharpe": 1.2,
                        "risk_score": 0.4,
                    },
                },
                "metadata": {
                    "from_cache": True,
                    "cache_file": request.cache_file,
                    "model_checkpoint": request.model_checkpoint or "mock",
                },
            }

            predictions.append(prediction)

        logger.info(f"✅ Generated {len(predictions)} predictions")

        return BacktestFromCacheResponse(
            backtest_id=backtest_id,
            symbols=request.symbols,
            date_range={
                "start": request.start_date,
                "end": request.end_date,
                "trading_days": str(len(df_filtered["timestamp"].unique())),
            },
            predictions=predictions,
            metadata={
                "total_predictions": len(predictions),
                "inference_time_ms": 50,  # Mock
                "model_checkpoint": request.model_checkpoint or "mock",
                "cache_hit_rate": 1.0,
                "from_cache": True,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest from cache error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_extraction_jobs():
    """List all feature extraction jobs"""
    return {"jobs": list(extraction_jobs.values()), "total": len(extraction_jobs)}


@router.delete("/jobs/{job_id}")
async def delete_extraction_job(job_id: str):
    """Delete a feature extraction job record"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = extraction_jobs.pop(job_id)

    return {"deleted": True, "job_id": job_id, "status": job["status"]}
