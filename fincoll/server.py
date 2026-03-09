#!/usr/bin/env python3
"""
FinColl REST API Server
Centralized financial data collection service

Architecture (2025-11-29):
- Uses FinCollEngine from finvec for unified training/inference
- Multi-timeframe profit velocity predictions (NOT continuous horizons)
- Feature vectors sized from config.dimensions.DIMS
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from caelum_secrets import load_env_from_secrets as _load_env_from_secrets

    _dotenv_load = None
except Exception:
    from dotenv import load_dotenv as _dotenv_load  # type: ignore

    _load_env_from_secrets = None
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.dimensions import DIMS

# Add finvec to path for engine imports
_finvec_path = os.getenv(
    "FINVEC_PATH", str(Path(__file__).parent.parent.parent.parent / "finvec")
)
if _finvec_path not in sys.path:
    sys.path.insert(0, _finvec_path)

# Import API routers
from .api.backtesting import router as backtesting_router
from .api.bars import router as bars_router
from .api.inference import router as inference_router
from .api.enriched import router as enriched_router
from .api.training import router as training_router
from .api.ab_testing import router as ab_testing_router
from .api.market_data import router as market_data_router
from .api.symvector_batch import router as symvector_batch_router
from .auth.tradestation_auth import TradeStationAuth

# Import SenVec controller
from .services.senvec_controller import SenVecController
from .tools.diagnostics_api import router as diagnostics_router

# Import velocity inference server app from finvec
try:
    from inference.velocity_inference_server import app as velocity_app

    VELOCITY_AVAILABLE = True
except ImportError as e:
    VELOCITY_AVAILABLE = False
    logging.warning(f"Velocity inference server not available: {e}")

# Import engine components
try:
    from engine.fincoll_engine import EngineConfig, EngineMode, FinCollEngine
    from models.simple_velocity_model import ModelConfig, SimpleVelocityModel

    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    logging.warning(f"FinCollEngine not available: {e}")
    # Define placeholder types for type hints
    FinCollEngine = None  # type: ignore

# Load environment variables (caelum-secrets → .env → os.environ fallback chain)
if _load_env_from_secrets is not None:
    _load_env_from_secrets("/prod/fincoll/")
elif _dotenv_load is not None:
    _dotenv_load()

# Setup logging
logging.basicConfig(
    level=os.getenv("FINCOLL_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce httpx logging verbosity (prevent spamming logs with every HTTP request)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fincoll.features.feature_extractor").setLevel(logging.WARNING)

# Initialize FastAPI app
app = FastAPI(
    title="FinColl API",
    description="Financial Data Collection Service for FinVec Training & PIM Inference",
    version="0.1.0",
)

# Include API routers
app.include_router(training_router)
app.include_router(inference_router)
app.include_router(enriched_router)  # Layer 2 enriched output
app.include_router(backtesting_router)
app.include_router(bars_router)
app.include_router(market_data_router)  # Real-time quotes for trade execution
app.include_router(ab_testing_router)  # A/B testing endpoints
app.include_router(symvector_batch_router)  # GPU-accelerated batch predictions
app.include_router(diagnostics_router, prefix="/api/diagnostics", tags=["diagnostics"])

# Mount velocity inference server at /api/v1/velocity
if VELOCITY_AVAILABLE:
    app.mount("/api/v1/velocity", velocity_app)

# Configuration
FINCOLL_HOST = os.getenv("FINCOLL_HOST", "0.0.0.0")
FINCOLL_PORT = int(os.getenv("FINCOLL_PORT", 8100))
CREDENTIALS_DIR = Path(os.getenv("CREDENTIALS_DIR", str(Path.home() / "caelum" / "ss")))

# Global engine instance (initialized on startup)
_engine: Optional[FinCollEngine] = None
_feature_extractor = None
_data_provider = None
_senvec_controller: Optional[SenVecController] = None
_alphavantage_client = None


class TradeStationTokenIngest(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    expires_at: Optional[str] = None


class TradeStationTokenStatus(BaseModel):
    authenticated: bool
    is_expired: bool
    expires_in: int
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize data providers and engine on startup"""
    global \
        _engine, \
        _feature_extractor, \
        _data_provider, \
        _senvec_controller, \
        _alphavantage_client

    logger.info("=" * 60)
    logger.info("FinColl Service Starting...")
    logger.info("=" * 60)
    logger.info(f"Host: {FINCOLL_HOST}")
    logger.info(f"Port: {FINCOLL_PORT}")
    logger.info(f"Credentials Dir: {CREDENTIALS_DIR}")

    # Check credentials exist
    av_creds = CREDENTIALS_DIR / ".alpha_vantage_credentials.json"
    ts_token = CREDENTIALS_DIR / ".tradestation_token.json"

    if av_creds.exists():
        logger.info("✅ Alpha Vantage credentials found")
    else:
        logger.warning(f"⚠️  Alpha Vantage credentials not found at {av_creds}")

    if ts_token.exists():
        logger.info("✅ TradeStation token found")
    else:
        logger.warning(f"⚠️  TradeStation token not found at {ts_token}")

    # Initialize engine components
    if ENGINE_AVAILABLE:
        try:
            from .features.feature_extractor import FeatureExtractor
            from .providers.multi_provider_fetcher import MultiProviderFetcher
            from .providers.tradestation_trading_provider import (
                TradeStationTradingProvider,
            )

            # Build MultiProviderFetcher as the single data-provider gateway.
            # yfinance is added automatically by MultiProviderFetcher.__init__.
            # TradeStationTradingProvider is added when credentials are available.
            fetcher = MultiProviderFetcher()

            try:
                # Read TradeStation tokens from file if available
                token_file = Path.home() / ".tradestation_token.json"
                refresh_token = None
                if token_file.exists():
                    try:
                        import json
                        with open(token_file) as f:
                            token_data = json.load(f)
                        refresh_token = token_data.get("refresh_token")
                        logger.info(f"✅ Loaded TradeStation refresh token from {token_file}")
                    except Exception as token_error:
                        logger.warning(f"Failed to read token file: {token_error}")

                ts_provider = TradeStationTradingProvider(refresh_token=refresh_token)
                fetcher.add_provider("tradestation", ts_provider)
                logger.info(
                    "✅ TradeStation provider registered with MultiProviderFetcher"
                )
            except Exception as ts_error:
                logger.warning(
                    f"TradeStation unavailable ({ts_error}), proceeding with yfinance only"
                )

            _data_provider = fetcher
            logger.info(
                f"✅ MultiProviderFetcher ready — active providers: {list(fetcher.providers.keys())}"
            )

            _feature_extractor = FeatureExtractor(
                enable_senvec=True,
                enable_futures=True,
                data_provider=_data_provider,  # CME/CBOE futures: @ES.D, @NQ.D, @VX, @CL, @GC (25D macro)
            )
            logger.info("✅ FinCollEngine components initialized")
            logger.info(f"   Data provider: {_data_provider.__class__.__name__}")
            logger.info(
                f"   Feature extractor: FeatureExtractor ({DIMS.fincoll_total}D)"
            )

            # Initialize AlphaVantage Premium for intraday bars
            try:
                from .providers.alphavantage_client import AlphaVantageClient

                _alphavantage_client = AlphaVantageClient()
                logger.info(
                    "✅ AlphaVantage Premium client initialized (150 calls/min)"
                )

                # Pass providers to bars router
                from .api import bars

                bars.set_providers(_data_provider, _alphavantage_client)
            except Exception as av_error:
                logger.warning(
                    f"⚠️  AlphaVantage client initialization failed: {av_error}"
                )
                # Still pass TradeStation to bars router
                from .api import bars

                bars.set_providers(_data_provider, None)

            # Wire MultiProviderFetcher into inference router
            from .api import inference as inference_module

            inference_module.set_provider(_data_provider)
            logger.info("✅ Inference router: MultiProviderFetcher wired")

            # Wire MultiProviderFetcher into pim_interface library
            from .api import pim_interface as pim_interface_module

            pim_interface_module.set_provider(_data_provider)
            logger.info("✅ PimInterface: MultiProviderFetcher wired")

            # Wire MultiProviderFetcher into training router
            from .api import training as training_module

            training_module.set_provider(_data_provider)
            logger.info("✅ Training router: MultiProviderFetcher wired")

            # Wire MultiProviderFetcher into backtesting router
            from .api import backtesting as backtesting_module

            backtesting_module.set_provider(_data_provider)
            logger.info("✅ Backtesting router: MultiProviderFetcher wired")

            # Wire MultiProviderFetcher into enriched router
            from .api import enriched as enriched_module

            enriched_module.set_provider(_data_provider)
            logger.info("✅ Enriched router: MultiProviderFetcher wired")

            # Wire MultiProviderFetcher into market_data router
            from .api import market_data as market_data_module

            market_data_module.set_provider(_data_provider)
            logger.info("✅ Market data router: MultiProviderFetcher wired")

            # Wire MultiProviderFetcher into symvector_batch router (GPU-accelerated)
            from .api import symvector_batch as symvector_module

            symvector_module.set_provider(_data_provider)
            logger.info("✅ SymVector batch router: MultiProviderFetcher wired")

        except Exception as e:
            logger.error(f"❌ Failed to initialize engine components: {e}")
    else:
        logger.warning("⚠️  FinCollEngine not available - velocity endpoints disabled")

    # Initialize SenVec Cache Controller
    senvec_enabled = os.getenv("SENVEC_CACHE_CONTROL", "true").lower() == "true"
    if senvec_enabled:
        try:
            senvec_url = os.getenv("SENVEC_URL", "http://localhost:18000")
            _senvec_controller = SenVecController(
                senvec_url=senvec_url,
                check_interval=60,  # Check market status every minute
                cache_interval=300,  # SenVec refresh every 5 minutes
                symbols=None,  # Use SenVec's default symbol universe
            )
            await _senvec_controller.start()
            logger.info("✅ SenVec Cache Controller started")
        except Exception as e:
            logger.error(f"❌ Failed to start SenVec controller: {e}")
    else:
        logger.info("⚠️  SenVec Cache Controller disabled (SENVEC_CACHE_CONTROL=false)")

    logger.info("=" * 60)
    logger.info("FinColl Service Ready")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global _senvec_controller

    logger.info("FinColl Service Shutting Down...")

    # Stop SenVec controller
    if _senvec_controller:
        try:
            await _senvec_controller.stop()
            logger.info("✅ SenVec Cache Controller stopped")
        except Exception as e:
            logger.error(f"Error stopping SenVec controller: {e}")

    logger.info("FinColl Service Stopped")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "FinColl",
        "version": "0.1.0",
        "description": "Financial Data Collection Service",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "training": "/api/v1/training/*",
            "inference": "/api/v1/inference/*",
            "backtesting": "/api/v1/backtesting/*",
            "market_data": {
                "quote": "/api/v1/market-data/quote/{symbol}",
                "history": "/api/v1/market-data/history/{symbol}",
            },
            "fundamentals": {
                "earnings": "/api/v1/fundamentals/earnings/{symbol}",
                "dividends": "/api/v1/fundamentals/dividends/{symbol}",
            },
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""

    # Check data source availability
    sources = {}

    # Alpha Vantage
    av_creds = CREDENTIALS_DIR / ".alpha_vantage_credentials.json"
    sources["alpha_vantage"] = {
        "available": av_creds.exists(),
        "status": "configured" if av_creds.exists() else "not_configured",
    }

    # TradeStation
    ts_token = CREDENTIALS_DIR / ".tradestation_token.json"
    sources["tradestation"] = {
        "available": ts_token.exists(),
        "status": "configured" if ts_token.exists() else "not_configured",
    }

    # SenVec
    senvec_enabled = os.getenv("SENVEC_ENABLED", "true").lower() == "true"
    sources["senvec"] = {
        "available": senvec_enabled,
        "status": "enabled" if senvec_enabled else "disabled",
        "url": os.getenv("SENVEC_API_URL", "http://10.32.3.27:8000"),
    }

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "data_sources": sources,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response

    from .monitoring.metrics import get_content_type, get_metrics

    return Response(content=get_metrics(), media_type=get_content_type())


@app.get("/api/v1/safe-mode/status")
async def safe_mode_status():
    """Get safe mode status"""
    from .monitoring.safe_mode import get_safe_mode_manager

    manager = get_safe_mode_manager()
    status = manager.get_status()
    provider_health = manager.get_provider_health()

    return {
        "state": status.state.value,
        "is_active": manager.is_safe_mode_active(),
        "entered_at": status.entered_at.isoformat() if status.entered_at else None,
        "reason": status.reason,
        "auto_resume_at": (
            status.auto_resume_at.isoformat() if status.auto_resume_at else None
        ),
        "manual_override": status.manual_override,
        "recent_events": [
            {
                "timestamp": datetime.fromtimestamp(e.timestamp).isoformat(),
                "type": e.event_type,
                "provider": e.provider,
                "details": e.details,
            }
            for e in status.recent_events
        ],
        "provider_health": provider_health,
        "config": {
            "rate_limit_threshold": manager.rate_limit_threshold,
            "server_error_threshold": manager.server_error_threshold,
            "window_seconds": manager.window_seconds,
            "auto_resume_minutes": manager.auto_resume_minutes,
        },
    }


@app.post("/api/v1/safe-mode/trigger")
async def trigger_safe_mode(
    reason: str = Query(..., description="Reason for manual trigger"),
):
    """Manually trigger safe mode"""
    from .monitoring.safe_mode import get_safe_mode_manager

    manager = get_safe_mode_manager()
    manager.trigger_manual(reason)

    return {
        "success": True,
        "message": "Safe mode manually triggered",
        "reason": reason,
        "state": manager.state.value,
    }


@app.post("/api/v1/safe-mode/resume")
async def resume_safe_mode():
    """Manually resume from safe mode"""
    from .monitoring.safe_mode import get_safe_mode_manager

    manager = get_safe_mode_manager()
    success = manager.resume_manual()

    if success:
        return {
            "success": True,
            "message": "Safe mode manually resumed",
            "state": manager.state.value,
        }
    else:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Cannot resume - not in safe mode",
                "state": manager.state.value,
            },
        )


@app.get("/api/v1/sources/status")
async def sources_status():
    """Get detailed status of all data sources"""

    status = {"timestamp": datetime.now().isoformat(), "sources": {}}

    # Alpha Vantage
    av_creds = CREDENTIALS_DIR / ".alpha_vantage_credentials.json"
    if av_creds.exists():
        import json

        with open(av_creds, "r") as f:
            av_data = json.load(f)
        status["sources"]["alpha_vantage"] = {
            "configured": True,
            "tier": av_data.get("tier", "unknown"),
            "rate_limit": av_data.get("rate_limit", "unknown"),
        }
    else:
        status["sources"]["alpha_vantage"] = {
            "configured": False,
            "error": "Credentials file not found",
        }

    # TradeStation
    ts_token = CREDENTIALS_DIR / ".tradestation_token.json"
    status["sources"]["tradestation"] = {
        "configured": ts_token.exists(),
        "api_url": os.getenv("TRADESTATION_API_URL", "https://api.tradestation.com/v3"),
    }

    # SenVec
    status["sources"]["senvec"] = {
        "enabled": os.getenv("SENVEC_ENABLED", "true").lower() == "true",
        "url": os.getenv("SENVEC_API_URL", "http://10.32.3.27:8000"),
    }

    return status


@app.post("/api/v1/auth/tradestation/ingest")
async def ingest_tradestation_token(payload: TradeStationTokenIngest):
    """Store TradeStation OAuth tokens in FinColl credentials store."""
    try:
        auth = TradeStationAuth()
        expires_at = None
        if payload.expires_at:
            expires_at = datetime.fromisoformat(payload.expires_at)

        auth.save_tokens(
            access_token=payload.access_token,
            refresh_token=payload.refresh_token,
            expires_at=expires_at,
            expires_in=payload.expires_in,
        )
        return {"success": True, "message": "Token stored"}
    except Exception as e:
        logger.error(f"Failed to ingest TradeStation token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/auth/tradestation/refresh")
async def refresh_tradestation_token():
    """Refresh TradeStation OAuth token in FinColl."""
    try:
        auth = TradeStationAuth()
        new_token = auth.refresh_access_token()
        return {"success": True, "access_token_present": bool(new_token)}
    except Exception as e:
        logger.error(f"Failed to refresh TradeStation token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/auth/tradestation/status")
async def tradestation_token_status() -> TradeStationTokenStatus:
    """Return TradeStation token status (no secrets)."""
    try:
        auth = TradeStationAuth()
        authenticated = auth.is_authenticated()
        expires_in = 0
        is_expired = True

        if auth.expires_at:
            now = datetime.now()
            is_expired = auth.expires_at <= now
            expires_in = max(0, int((auth.expires_at - now).total_seconds()))

        message = "Authenticated" if authenticated else "Not authenticated"
        if authenticated and is_expired:
            message = "Token expired"

        return TradeStationTokenStatus(
            authenticated=authenticated,
            is_expired=is_expired,
            expires_in=expires_in,
            message=message,
        )
    except Exception as e:
        logger.error(f"Failed to get TradeStation token status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/training/features/{symbol}")
async def get_training_features(
    symbol: str,
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    timeframes: Optional[str] = Query(
        "30s,1m,5m,30m", description="Comma-separated timeframes"
    ),
):
    """
    Get historical features for training (DIMS.fincoll_total feature vectors)
    """
    if _feature_extractor is None or _data_provider is None:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")

    try:
        import asyncio
        from datetime import timedelta

        # Default date range: 1 year
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Fetch historical data in a thread to avoid blocking the event loop
        _end_dt = datetime.strptime(end, "%Y-%m-%d")
        _start_dt = datetime.strptime(start, "%Y-%m-%d")
        df = await asyncio.to_thread(
            lambda: _data_provider.get_historical_bars(
                symbol=symbol, interval="1d", start_date=_start_dt, end_date=_end_dt
            )
        )
        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=404, detail=f"Insufficient data for {symbol}"
            )

        # Extract features in a thread to avoid blocking the event loop
        _ts = datetime.now()
        features = await asyncio.to_thread(
            _feature_extractor.extract_features, df, symbol=symbol, timestamp=_ts
        )

        return {
            "symbol": symbol,
            "start": start,
            "end": end,
            "num_samples": len(df),
            "feature_dim": len(features) if features is not None else 0,
            "features": features.tolist() if hasattr(features, "tolist") else features,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting features for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/inference/features/{symbol}")
async def get_inference_features(
    symbol: str,
    lookback: int = Query(100, description="Number of days to look back"),
    format: str = Query(
        "latest", description="Feature format (uses latest version from config)"
    ),
):
    """
    Get real-time features for inference (DIMS.fincoll_total feature vectors)
    """
    if _feature_extractor is None or _data_provider is None:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")

    try:
        import asyncio
        from datetime import timedelta

        _end_dt2 = datetime.now()
        _start_dt2 = _end_dt2 - timedelta(days=lookback)

        # Fetch historical data in a thread to avoid blocking the event loop
        df = await asyncio.to_thread(
            lambda: _data_provider.get_historical_bars(
                symbol=symbol, interval="1d", start_date=_start_dt2, end_date=_end_dt2
            )
        )
        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=404, detail=f"Insufficient data for {symbol}"
            )

        # Extract features in a thread to avoid blocking the event loop
        _ts = datetime.now()
        features = await asyncio.to_thread(
            _feature_extractor.extract_features, df, symbol=symbol, timestamp=_ts
        )

        return {
            "symbol": symbol,
            "feature_dim": len(features) if features is not None else 0,
            "features": features.tolist() if hasattr(features, "tolist") else features,
            "data_bars": len(df),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting features for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/velocity/predict/{symbol}")
async def predict_velocity(symbol: str):
    """
    Predict multi-timeframe profit velocities for a symbol

    Returns best velocity opportunities for each timeframe (1min, 5min, 15min, 1hour, daily)
    """
    if not ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Velocity engine not available")

    if _feature_extractor is None or _data_provider is None:
        raise HTTPException(status_code=503, detail="Engine components not initialized")

    try:
        # Create temporary engine for inference
        model = SimpleVelocityModel(
            ModelConfig(input_dim=DIMS.fincoll_total, output_dim=10)
        )
        model.eval()

        # Try to load checkpoint if exists
        checkpoint_path = (
            Path(__file__).parent.parent.parent
            / "finvec"
            / "checkpoints"
            / "velocity"
            / "best_model.pt"
        )
        if checkpoint_path.exists():
            import torch

            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                elif "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded velocity model from {checkpoint_path}")
            except Exception as e:
                logger.warning(
                    f"Could not load checkpoint (architecture mismatch?): {e}"
                )
                logger.warning("Using untrained model - predictions will be random")

        class SingleSymbolScanner:
            def __init__(self, sym):
                self.symbols, self.current_index = [sym], 0

            def get_next_batch(self, n):
                return self.symbols

            def reset_cycle(self):
                pass

            def get_stats(self):
                return {"total": 1}

        config = EngineConfig(
            mode=EngineMode.INFER,
            batch_size=1,
            device="cpu",
            feature_dim=DIMS.fincoll_total,  # Fallback to CPU due to CUDA version mismatch (trained on cu121, driver is cu130)
        )
        engine = FinCollEngine(
            config=config,
            symbol_scanner=SingleSymbolScanner(symbol),
            model=model,
            feature_extractor=_feature_extractor,
            data_provider=_data_provider,
        )

        result = engine.process_batch()
        symbol_result = result.get("results", {}).get(symbol, {})

        if symbol_result.get("status") == "success":
            return {
                "symbol": symbol,
                "status": "success",
                "predictions": symbol_result.get("predictions", {}),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=500, detail=symbol_result.get("error", "Unknown error")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Velocity prediction error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/velocity/batch")
async def predict_velocity_batch(symbols: List[str]):
    """
    Predict velocities for multiple symbols
    """
    if not ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Velocity engine not available")

    results = {}
    for symbol in symbols[:10]:  # Limit to 10 symbols per batch
        try:
            # Simplified - could optimize with batch inference
            from datetime import timedelta

            _end_dt3 = datetime.now()
            _start_dt3 = _end_dt3 - timedelta(days=100)

            df = _data_provider.get_historical_bars(
                symbol=symbol, interval="1d", start_date=_start_dt3, end_date=_end_dt3
            )
            if df is not None and len(df) >= 50:
                features = _feature_extractor.extract_features(
                    df, symbol=symbol, timestamp=datetime.now()
                )
                results[symbol] = {"status": "success", "features_extracted": True}
            else:
                results[symbol] = {"status": "insufficient_data"}
        except Exception as e:
            results[symbol] = {"status": "error", "error": str(e)}

    return {"results": results, "timestamp": datetime.now().isoformat()}


# ============================================================================
# Market Data API Endpoints
# Added 2025-11-29 to fix PIM boundary violations
# ============================================================================


@app.get("/api/v1/market-data/quote/{symbol}")
async def get_market_quote(symbol: str):
    """
    Get real-time bid/ask quote for a symbol

    Returns:
        {
            'symbol': str,
            'price': float,
            'bid': float,
            'ask': float,
            'spread': float,
            'spread_pct': float,
            'volume': int,
            'timestamp': str,
            'provider': str
        }

    Used by:
        - PIM tradability_evaluator.py (bid/ask spread analysis)
        - PIM event_calendar.py (real-time price checks)
    """
    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")

    try:
        import asyncio

        # Run blocking provider call in thread pool to avoid blocking the event loop
        quote = await asyncio.to_thread(_data_provider.get_quote, symbol)

        if quote is None:
            raise HTTPException(
                status_code=404, detail=f"No quote data available for {symbol}"
            )

        # Calculate spread metrics
        bid = quote.get("bid")
        ask = quote.get("ask")

        # Coerce bid/ask to float if they come back as strings
        try:
            bid = float(bid) if bid is not None else None
        except (TypeError, ValueError):
            bid = None
        try:
            ask = float(ask) if ask is not None else None
        except (TypeError, ValueError):
            ask = None

        spread = None
        spread_pct = None

        if bid is not None and ask is not None and bid > 0:
            spread = ask - bid
            spread_pct = (spread / bid) * 100

        return {
            "symbol": symbol,
            "price": quote.get("price"),
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "volume": quote.get("volume"),
            "timestamp": quote.get("timestamp") or datetime.now().isoformat(),
            "provider": quote.get("provider", _data_provider.get_name()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quote: {str(e)}")


@app.post("/api/v1/market-data/quotes")
async def get_market_quotes_batch(request: Request):
    """
    Get real-time quotes for multiple symbols in a single batch request.

    Request body:
        {"symbols": ["AAPL", "MSFT", ...]}

    Returns:
        {
            "quotes": [
                {
                    "symbol": str,
                    "price": float | null,
                    "bid": float | null,
                    "ask": float | null,
                    "spread": float | null,
                    "spread_pct": float | null,
                    "volume": int | null,
                    "timestamp": str,
                    "provider": str,
                    "error": str | null
                },
                ...
            ],
            "count": int,
            "provider": str
        }

    Used by:
        - PIM pim_scheduler.py (live position price updates)
    """
    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")

    try:
        body = await request.json()
        symbols = body.get("symbols", [])
    except Exception:
        raise HTTPException(
            status_code=400, detail='Invalid JSON body; expected {"symbols": [...]}'
        )

    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    import asyncio

    async def fetch_quote(symbol: str):
        """Fetch a single quote in a thread pool to avoid blocking the event loop."""
        try:
            return symbol, await asyncio.to_thread(_data_provider.get_quote, symbol)
        except Exception as e:
            return symbol, None

    # Fetch all quotes concurrently in thread pool
    raw_results = await asyncio.gather(*[fetch_quote(s) for s in symbols])

    quotes = []
    for symbol, quote in raw_results:
        try:
            if quote is None:
                quotes.append(
                    {
                        "symbol": symbol,
                        "price": None,
                        "error": "No quote data available",
                    }
                )
                continue

            bid = quote.get("bid")
            ask = quote.get("ask")

            # Coerce bid/ask to float if they come back as strings
            try:
                bid = float(bid) if bid is not None else None
            except (TypeError, ValueError):
                bid = None
            try:
                ask = float(ask) if ask is not None else None
            except (TypeError, ValueError):
                ask = None

            spread = None
            spread_pct = None
            if bid is not None and ask is not None and bid > 0:
                spread = ask - bid
                spread_pct = (spread / bid) * 100

            quotes.append(
                {
                    "symbol": symbol,
                    "price": quote.get("price"),
                    "bid": bid,
                    "ask": ask,
                    "spread": spread,
                    "spread_pct": spread_pct,
                    "volume": quote.get("volume"),
                    "timestamp": quote.get("timestamp") or datetime.now().isoformat(),
                    "provider": quote.get("provider", _data_provider.get_name()),
                    "error": None,
                }
            )
        except Exception as e:
            logger.warning(f"Error getting quote for {symbol} in batch: {e}")
            quotes.append({"symbol": symbol, "price": None, "error": str(e)})

    return {
        "quotes": quotes,
        "count": len(quotes),
        "provider": _data_provider.get_name() if _data_provider else "unknown",
    }


@app.get("/api/v1/market-data/history/{symbol}")
async def get_market_history(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Interval: 1m, 5m, 15m, 30m, 1h, 1d"),
):
    """
    Get historical OHLCV bars for a symbol

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
        end_date: End date (YYYY-MM-DD), defaults to today
        interval: Time interval (1m, 5m, 15m, 30m, 1h, 1d)

    Returns:
        {
            'symbol': str,
            'start_date': str,
            'end_date': str,
            'interval': str,
            'bars': [
                {
                    'timestamp': str,
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': int
                },
                ...
            ],
            'count': int,
            'provider': str
        }

    Used by:
        - PIM tradability_evaluator.py (volume profile, VWAP, support/resistance)
        - PIM event_calendar.py (historical price patterns)
    """
    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")

    try:
        import asyncio
        from datetime import timedelta

        # Default date range: 30 days
        _end_dt4 = (
            datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        )
        _start_dt4 = (
            datetime.strptime(start_date, "%Y-%m-%d")
            if start_date
            else _end_dt4 - timedelta(days=30)
        )

        # Fetch historical data in thread pool to avoid blocking the event loop
        df = await asyncio.to_thread(
            lambda: _data_provider.get_historical_bars(
                symbol=symbol,
                interval=interval,
                start_date=_start_dt4,
                end_date=_end_dt4,
            )
        )

        if df is None or len(df) == 0:
            raise HTTPException(
                status_code=404, detail=f"No historical data available for {symbol}"
            )

        # Convert DataFrame to list of dicts
        bars = []
        for idx, row in df.iterrows():
            bars.append(
                {
                    "timestamp": idx.isoformat()
                    if hasattr(idx, "isoformat")
                    else str(idx),
                    "open": float(row["open"]) if pd.notna(row["open"]) else None,
                    "high": float(row["high"]) if pd.notna(row["high"]) else None,
                    "low": float(row["low"]) if pd.notna(row["low"]) else None,
                    "close": float(row["close"]) if pd.notna(row["close"]) else None,
                    "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                }
            )

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "bars": bars,
            "count": len(bars),
            "provider": _data_provider.get_name(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history for {symbol}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get historical data: {str(e)}"
        )


@app.get("/api/v1/fundamentals/earnings/{symbol}")
async def get_earnings_date(symbol: str):
    """
    Get next earnings announcement date for a symbol

    Returns:
        {
            'symbol': str,
            'next_earnings_date': str (YYYY-MM-DD) or null,
            'days_until_earnings': int or null,
            'estimated': bool,
            'provider': str,
            'timestamp': str
        }

    Used by:
        - PIM event_calendar.py (earnings event risk scoring)

    Note:
        - Uses yfinance ticker.info for earnings calendar
        - TradeStation doesn't provide earnings calendar in v3 API
        - Falls back to yfinance even if TradeStation is primary provider
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info

        # yfinance provides earnings date in various formats
        earnings_date = None
        estimated = False

        # Try different yfinance fields
        if "earningsDate" in info and info["earningsDate"]:
            # earningsDate can be a list of timestamps
            earnings_dates = info["earningsDate"]
            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                # Take the first future date
                import pandas as pd

                earnings_timestamp = pd.Timestamp(earnings_dates[0])
                earnings_date = earnings_timestamp.strftime("%Y-%m-%d")
                estimated = True

        # Calculate days until earnings
        days_until = None
        if earnings_date:
            earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
            days_until = (earnings_dt - datetime.now()).days

        return {
            "symbol": symbol,
            "next_earnings_date": earnings_date,
            "days_until_earnings": days_until,
            "estimated": estimated,
            "provider": "yfinance",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting earnings date for {symbol}: {e}")
        # Return null data rather than error - earnings date is optional
        return {
            "symbol": symbol,
            "next_earnings_date": None,
            "days_until_earnings": None,
            "estimated": False,
            "provider": "yfinance",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/api/v1/fundamentals/dividends/{symbol}")
async def get_dividend_info(symbol: str):
    """
    Get next dividend ex-date and amount for a symbol

    Returns:
        {
            'symbol': str,
            'next_ex_date': str (YYYY-MM-DD) or null,
            'days_until_ex_date': int or null,
            'dividend_amount': float or null,
            'dividend_yield': float or null (annual %),
            'payment_date': str or null,
            'provider': str,
            'timestamp': str
        }

    Used by:
        - PIM event_calendar.py (dividend event risk scoring)

    Note:
        - Uses yfinance for dividend calendar
        - TradeStation doesn't provide dividend calendar in v3 API
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get dividend history
        dividends = ticker.dividends

        next_ex_date = None
        dividend_amount = None
        days_until = None

        # Estimate next ex-date based on historical pattern
        if len(dividends) > 0:
            # Get last dividend
            last_div_date = dividends.index[-1]
            last_div_amount = dividends.iloc[-1]

            # Calculate average dividend frequency
            if len(dividends) >= 2:
                # Get dates of last few dividends
                recent_divs = dividends.tail(4)
                if len(recent_divs) >= 2:
                    # Calculate average days between dividends
                    date_diffs = []
                    for i in range(1, len(recent_divs)):
                        diff = (recent_divs.index[i] - recent_divs.index[i - 1]).days
                        date_diffs.append(diff)

                    avg_frequency = (
                        np.mean(date_diffs) if date_diffs else 90
                    )  # Default to quarterly

                    # Estimate next ex-date
                    estimated_next = last_div_date + pd.Timedelta(
                        days=int(avg_frequency)
                    )

                    # Only return if it's in the future
                    if estimated_next > pd.Timestamp.now():
                        next_ex_date = estimated_next.strftime("%Y-%m-%d")
                        dividend_amount = float(last_div_amount)
                        days_until = (estimated_next - pd.Timestamp.now()).days

        # Get annual dividend yield from info
        dividend_yield = info.get("dividendYield")
        if dividend_yield:
            dividend_yield = dividend_yield * 100  # Convert to percentage

        return {
            "symbol": symbol,
            "next_ex_date": next_ex_date,
            "days_until_ex_date": days_until,
            "dividend_amount": dividend_amount,
            "dividend_yield": dividend_yield,
            "payment_date": None,  # yfinance doesn't provide payment date easily
            "provider": "yfinance",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting dividend info for {symbol}: {e}")
        # Return null data rather than error - dividend info is optional
        return {
            "symbol": symbol,
            "next_ex_date": None,
            "days_until_ex_date": None,
            "dividend_amount": None,
            "dividend_yield": None,
            "payment_date": None,
            "provider": "yfinance",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting FinColl server on {FINCOLL_HOST}:{FINCOLL_PORT}")

    uvicorn.run(
        "fincoll.server:app",
        host=FINCOLL_HOST,
        port=FINCOLL_PORT,
        reload=os.getenv("FINCOLL_DEBUG", "false").lower() == "true",
        log_level=os.getenv("FINCOLL_LOG_LEVEL", "info").lower(),
    )
