"""
Diagnostics API for FinColl Vector Analysis

FastAPI endpoints for the PIM Vue3 dashboard to fetch vector diagnostics,
service health, and optimization analysis.

Usage:
    # Mount in existing FastAPI app
    from fincoll.tools.diagnostics_api import router as diagnostics_router
    app.include_router(diagnostics_router, prefix="/api/diagnostics")

    # Or run standalone
    uvicorn fincoll.tools.diagnostics_api:app --port 8003
"""

import asyncio
import logging
import os
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .feature_registry import (
    FINCOLL_CATEGORIES,
    FINCOLL_FEATURE_GROUPS,
    get_feature_summary,
    get_total_dimensions,
)
from .vector_analyzer import VectorAnalyzer, get_mock_analysis, load_npz_data

logger = logging.getLogger(__name__)

router = APIRouter(tags=["diagnostics"])

# ============================================================================
# Analysis cache — persisted to disk so results survive restarts
# ============================================================================

# Cache file lives alongside this module's package root
_ANALYSIS_CACHE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "analysis_cache.json"
)
_analysis_lock = threading.Lock()
_analysis_status: Dict[str, Any] = {
    "status": "idle"
}  # in-memory status for running/idle


# ============================================================================
# Service Configuration
# ============================================================================

SERVICES = {
    "fincoll": {
        "name": "FinColl API",
        "port": 8002,
        "health_endpoint": "/health",
        "description": "FinVec prediction service",
    },
    "senvec_aggregator": {
        "name": "SenVec Aggregator",
        "port": 18000,
        "health_endpoint": "/health",
        "description": "Sentiment feature gateway",
    },
    "senvec_alphavantage": {
        "name": "Alpha Vantage Service",
        "port": 18002,
        "health_endpoint": "/health",
        "description": "Cross-asset signals (18D)",
    },
    "senvec_social": {
        "name": "Social Sentiment Service",
        "port": 18006,
        "health_endpoint": "/health",
        "description": "Twitter/Reddit/StockTwits (23D, MongoDB-backed)",
    },
    "senvec_news": {
        "name": "News Sentiment Service",
        "port": 18005,
        "health_endpoint": "/health",
        "description": "News analysis (20D, MongoDB-backed)",
    },
    "pim_engine": {
        "name": "PIM Engine",
        "port": 5002,
        "health_endpoint": "/api/pim/status",
        "description": "Trading engine",
    },
}


# ============================================================================
# Pydantic Models
# ============================================================================


class ServiceStatus(BaseModel):
    name: str
    port: int
    status: str  # 'healthy', 'unhealthy', 'unknown'
    latency_ms: Optional[float] = None
    description: str
    last_checked: str


class FeatureSegment(BaseModel):
    name: str
    category: str
    dimensions: int
    start_idx: int
    end_idx: int
    description: str
    data_source: str
    update_frequency: str


class CategorySummary(BaseModel):
    name: str
    dimensions: int
    groups: int
    color: str
    description: str


class AnalysisRequest(BaseModel):
    use_mock: bool = True  # Default to mock for demo
    symbol: Optional[str] = None


class DataFreshnessItem(BaseModel):
    source: str
    last_update: str
    age_minutes: int
    status: str  # 'fresh', 'stale', 'error'


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/health")
async def diagnostics_health():
    """Health check for diagnostics API."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/services/status")
async def get_services_status() -> Dict[str, Any]:
    """
    Check health of all related services.

    Returns status of FinColl, SenVec, and PIM services.
    """
    statuses = []

    async def check_service(service_id: str, config: dict) -> ServiceStatus:
        url = f"http://10.32.3.27:{config['port']}{config['health_endpoint']}"
        status = "unknown"
        latency = None

        try:
            start = datetime.now()
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(url) as response:
                    latency = (datetime.now() - start).total_seconds() * 1000
                    if response.status == 200:
                        status = "healthy"
                    else:
                        status = "unhealthy"
        except asyncio.TimeoutError:
            status = "timeout"
        except Exception as e:
            logger.debug(f"Service {service_id} check failed: {e}")
            status = "unreachable"

        return ServiceStatus(
            name=config["name"],
            port=config["port"],
            status=status,
            latency_ms=round(latency, 2) if latency else None,
            description=config["description"],
            last_checked=datetime.now().isoformat(),
        )

    # Check all services concurrently
    tasks = [check_service(sid, cfg) for sid, cfg in SERVICES.items()]
    results = await asyncio.gather(*tasks)

    # Determine overall health
    healthy_count = sum(1 for r in results if r.status == "healthy")
    total_count = len(results)

    if healthy_count == total_count:
        overall = "healthy"
    elif healthy_count > total_count // 2:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "overall_status": overall,
        "healthy_count": healthy_count,
        "total_count": total_count,
        "services": [s.model_dump() for s in results],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/features/summary")
async def get_features_summary() -> Dict[str, Any]:
    """
    Get summary of the feature vector structure (dimension from config).

    Returns total dimensions, categories, and group counts.
    """
    return {
        **get_feature_summary(),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/features/segments")
async def get_feature_segments() -> List[Dict[str, Any]]:
    """
    Get all feature segments with their definitions.

    Returns detailed info for each of the ~20 feature groups.
    """
    segments = []

    for name, info in FINCOLL_FEATURE_GROUPS.items():
        indices = info["indices"]
        segments.append(
            {
                "name": name,
                "category": info["category"],
                "dimensions": info["dimensions"],
                "start_idx": min(indices),
                "end_idx": max(indices),
                "description": info["description"],
                "data_source": info["data_source"],
                "provider": info.get("provider", ""),
                "update_frequency": info["update_frequency"],
            }
        )

    # Sort by start index
    segments.sort(key=lambda x: x["start_idx"])
    return segments


@router.get("/features/categories")
async def get_feature_categories() -> List[Dict[str, Any]]:
    """
    Get feature categories (higher-level groupings).

    Returns 8 categories with their colors and dimensions.
    """
    categories = []

    for name, info in FINCOLL_CATEGORIES.items():
        categories.append(
            {
                "name": name,
                "dimensions": info["dimensions"],
                "groups": len(info["groups"]),
                "color": info["color"],
                "description": info["description"],
                "group_names": info["groups"],
            }
        )

    # Sort by dimensions descending
    categories.sort(key=lambda x: x["dimensions"], reverse=True)
    return categories


@router.get("/data/freshness")
async def get_data_freshness() -> Dict[str, Any]:
    """
    Check data freshness for each data source.

    Returns age and status of each data source.
    """
    # This would normally query actual data timestamps
    # For now, return mock data with realistic patterns
    now = datetime.now()

    sources = [
        {
            "source": "Market Data (OHLCV)",
            "description": "Price and volume data",
            "last_update": now.isoformat(),
            "age_minutes": 1,
            "status": "fresh",
            "segments": [
                "technical",
                "advanced_technical",
                "velocity",
                "support_resistance",
                "vwap",
            ],
        },
        {
            "source": "Alpha Vantage",
            "description": "Cross-asset macro indicators",
            "last_update": now.replace(minute=0).isoformat(),
            "age_minutes": now.minute,
            "status": "fresh" if now.minute < 60 else "stale",
            "segments": ["senvec_alphavantage", "cross_asset"],
        },
        {
            "source": "Social APIs",
            "description": "Twitter, Reddit, StockTwits",
            "last_update": now.isoformat(),
            "age_minutes": 5,
            "status": "fresh",
            "segments": ["senvec_social"],
        },
        {
            "source": "News API",
            "description": "FinLight news + TradeStation news",
            "last_update": now.isoformat(),
            "age_minutes": 3,
            "status": "fresh",
            "segments": ["senvec_news", "news"],
        },
        {
            "source": "Fundamentals",
            "description": "Financial statements (TradeStation + Finnhub)",
            "last_update": now.replace(day=1).isoformat(),
            "age_minutes": (now - now.replace(day=1)).days * 24 * 60,
            "status": "fresh",  # Quarterly data is always "fresh" within quarter
            "segments": ["fundamentals", "finnhub"],
        },
        {
            "source": "Futures Contracts",
            "description": "ES, NQ, CL, GC futures (TradeStation)",
            "last_update": now.isoformat(),
            "age_minutes": 1,
            "status": "fresh",
            "segments": ["futures"],
        },
        {
            "source": "Sector ETFs",
            "description": "Sector relative performance",
            "last_update": now.isoformat(),
            "age_minutes": 1,
            "status": "fresh",
            "segments": ["sector"],
        },
    ]

    return {
        "sources": sources,
        "timestamp": now.isoformat(),
    }


@router.post("/analysis/run")
async def run_analysis(
    request: AnalysisRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run vector importance analysis.

    Args:
        use_mock: If True, return mock results immediately (fast).
                  If False, kick off a real background analysis using stored
                  NPZ feature vectors. Does not affect live trading.

    Returns:
        Analysis results (mock) or accepted/status (real).
    """
    if request.use_mock:
        # Return mock analysis for demo/testing
        result = get_mock_analysis()
        return {
            "status": "complete",
            "is_mock": True,
            "result": {
                "timestamp": result.timestamp,
                "total_dimensions": result.total_dimensions,
                "baseline_score": result.baseline_score,
                "segments": [asdict(s) for s in result.segments],
                "categories": [asdict(c) for c in result.categories],
                "recommendations": result.recommendations,
                "compression_potential": result.compression_potential,
            },
        }

    # Real analysis — check if already running
    with _analysis_lock:
        if _analysis_status.get("status") == "running":
            return {
                "status": "running",
                "is_mock": False,
                "message": "Analysis already in progress. Poll /analysis/latest for results.",
                "started_at": _analysis_status.get("started_at"),
            }
        _analysis_status["status"] = "running"
        _analysis_status["started_at"] = datetime.now().isoformat()
        _analysis_status["error"] = None

    background_tasks.add_task(_run_real_analysis)
    return {
        "status": "running",
        "is_mock": False,
        "message": "Analysis started in background. Poll /analysis/latest for results.",
        "started_at": _analysis_status["started_at"],
    }


def _run_real_analysis() -> None:
    """
    Background worker: load NPZ feature data, run VectorAnalyzer, cache to disk.

    Runs in a background thread (via FastAPI BackgroundTasks) — safe to call
    at any time, will not interrupt live market operations.
    """
    global _analysis_status
    try:
        logger.info("Starting real feature importance analysis...")

        # Load feature data from cached NPZ files
        X, y = load_npz_data()
        if X is None or y is None:
            raise RuntimeError(
                "No feature data found in finvec/data/velocity_cache/. "
                "Run train_velocity.py to generate NPZ files first."
            )

        logger.info(f"Loaded data: X={X.shape}, y={y.shape}")

        # Run analysis (uses RandomForestRegressor with 5-fold CV — CPU only)
        analyzer = VectorAnalyzer(X=X, y=y)
        result = analyzer.full_analysis(verbose=True)

        # Persist result to cache file
        _ANALYSIS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "timestamp": result.timestamp,
            "total_dimensions": result.total_dimensions,
            "baseline_score": result.baseline_score,
            "segments": [asdict(s) for s in result.segments],
            "categories": [asdict(c) for c in result.categories],
            "recommendations": result.recommendations,
            "compression_potential": result.compression_potential,
            "is_mock": False,
            "data_shape": list(X.shape),
        }
        with open(_ANALYSIS_CACHE_PATH, "w") as f:
            import json

            json.dump(cache_data, f, indent=2)

        with _analysis_lock:
            _analysis_status["status"] = "complete"
            _analysis_status["completed_at"] = datetime.now().isoformat()

        logger.info(f"Analysis complete. Cached to {_ANALYSIS_CACHE_PATH}")

    except Exception as e:
        logger.error(f"Real analysis failed: {e}", exc_info=True)
        with _analysis_lock:
            _analysis_status["status"] = "error"
            _analysis_status["error"] = str(e)


@router.get("/analysis/latest")
async def get_latest_analysis() -> Dict[str, Any]:
    """
    Get the most recent analysis results.

    Returns cached real analysis if available, otherwise falls back to mock.
    Also reflects current run status if an analysis is in progress.
    """
    # If analysis is currently running, report that
    with _analysis_lock:
        current_status = _analysis_status.get("status", "idle")
        started_at = _analysis_status.get("started_at")
        error = _analysis_status.get("error")

    if current_status == "running":
        return {
            "status": "running",
            "is_mock": False,
            "message": "Analysis in progress...",
            "started_at": started_at,
        }

    if current_status == "error":
        return {
            "status": "error",
            "is_mock": False,
            "message": error or "Unknown error during analysis",
        }

    # Try to load from disk cache
    if _ANALYSIS_CACHE_PATH.exists():
        try:
            import json

            with open(_ANALYSIS_CACHE_PATH) as f:
                cache_data = json.load(f)
            return {
                "status": "complete",
                "is_mock": cache_data.get("is_mock", False),
                "result": cache_data,
            }
        except Exception as e:
            logger.warning(f"Failed to load analysis cache: {e}")

    # Fall back to mock
    result = get_mock_analysis()
    return {
        "status": "complete",
        "is_mock": True,
        "result": {
            "timestamp": result.timestamp,
            "total_dimensions": result.total_dimensions,
            "baseline_score": result.baseline_score,
            "segments": [asdict(s) for s in result.segments],
            "categories": [asdict(c) for c in result.categories],
            "recommendations": result.recommendations,
            "compression_potential": result.compression_potential,
        },
    }


@router.get("/symbol/{symbol}/features")
async def get_symbol_features(symbol: str) -> Dict[str, Any]:
    """
    Get current feature values for a symbol.

    Returns the feature vector with segment breakdown.
    """
    # In production, this would call FinColl to get actual features
    # For now, return mock structure
    return {
        "symbol": symbol.upper(),
        "timestamp": datetime.now().isoformat(),
        "total_dimensions": get_total_dimensions(),
        "is_mock": True,
        "message": "Real feature retrieval not implemented. Shows structure only.",
        "segments": {
            name: {
                "dimensions": info["dimensions"],
                "values": [0.0] * info["dimensions"],  # Mock zeros
                "mean": 0.0,
                "std": 0.0,
            }
            for name, info in FINCOLL_FEATURE_GROUPS.items()
        },
    }


# ============================================================================
# Standalone App (for testing)
# ============================================================================


def create_app():
    """Create standalone FastAPI app for testing."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="FinColl Vector Diagnostics API",
        description="API for analyzing and monitoring the FinColl feature vector",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/diagnostics")

    return app


# For standalone running: uvicorn fincoll.tools.diagnostics_api:app --port 8003
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
