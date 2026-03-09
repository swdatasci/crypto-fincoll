"""
PIM Interface - Python Library API for Passive Income Maximizer

Pure Python functions for PIM to import directly (NO HTTP).

Flow:
    1. Check InfluxDB cache (fast path)
    2. On cache miss: TradeStation → SenVec → Features → FinVec
    3. Return velocity predictions

Usage:
    from fincoll import get_velocity_predictions

    predictions = get_velocity_predictions('AAPL', current_price=175.50)
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.dimensions import DIMS

from ..features.feature_extractor import FeatureExtractor
from ..inference.velocity_engine import get_velocity_engine

# Import FinColl components
from ..storage.influxdb_cache import get_cache as get_influx_cache

logger = logging.getLogger(__name__)

# Global data provider injected by server.py startup (MultiProviderFetcher).
# Falls back to a lazily-created MultiProviderFetcher if set_provider() was never called
# (e.g. when pim_interface is used as a library outside the server process).
_data_provider = None


def set_provider(data_provider) -> None:
    """Set global data provider (called by server.py on startup)."""
    global _data_provider
    _data_provider = data_provider
    logger.info(
        f"✅ PimInterface: data provider set to {data_provider.__class__.__name__}"
    )


def _get_provider():
    """Return the global provider, lazily creating a MultiProviderFetcher if needed."""
    global _data_provider
    if _data_provider is None:
        logger.warning(
            "PimInterface: no provider set — lazily creating MultiProviderFetcher"
        )
        from ..providers.multi_provider_fetcher import MultiProviderFetcher

        _data_provider = MultiProviderFetcher()
    return _data_provider


@dataclass
class VelocityPredictionConfig:
    """Configuration for velocity predictions"""

    # Data fetching
    use_cache: bool = True  # Check InfluxDB first
    cache_ttl_minutes: int = 5  # Cache freshness tolerance
    lookback_days: int = 90  # Historical data lookback

    # Feature extraction
    include_sentiment: bool = True  # Use SenVec sentiment (dimension from config)
    include_fundamentals: bool = True  # Use fundamental features

    # Model inference
    checkpoint_path: Optional[str] = None  # Path to velocity model
    device: str = "auto"  # 'cpu', 'cuda', or 'auto'

    # Performance
    timeout_seconds: int = 30  # Max time for prediction


def get_velocity_predictions(
    symbol: str,
    current_price: Optional[float] = None,
    config: Optional[VelocityPredictionConfig] = None,
) -> Dict[str, Any]:
    """
    Get velocity predictions for a single symbol.

    This is the PRIMARY function PIM will call for predictions.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        current_price: Current price (if None, fetches latest)
        config: Prediction configuration

    Returns:
        Velocity predictions in format:
        {
            "symbol": str,
            "timestamp": str (ISO format),
            "current_price": float,
            "velocities": [
                {
                    "velocity": float,        # % return per bar
                    "timeframe": str,         # "1min", "5min", etc.
                    "bars": int,              # Bars until target
                    "seconds": int,           # Time in seconds
                    "direction": str,         # "LONG" or "SHORT"
                    "confidence": float,      # 0-1 scale
                    "expected_return": float  # velocity * bars
                },
                ...
            ],
            "best_opportunity": {...},  # Highest |velocity|
            "metadata": {
                "cache_hit": bool,
                "feature_dim": int,
                "model": str,
                "data_sources": [str],
                "latency_ms": float
            }
        }

    Raises:
        ValueError: If symbol is invalid
        RuntimeError: If prediction fails
    """
    start_time = datetime.now()

    if config is None:
        config = VelocityPredictionConfig()

    logger.info(f"Getting velocity predictions for {symbol}")

    try:
        # Step 1: Get current price if not provided
        if current_price is None:
            current_price = _fetch_current_price(symbol, config)
            logger.info(f"Fetched current price for {symbol}: ${current_price:.2f}")

        # Step 2: Get historical bars (cache-first)
        bars, cache_hit = _fetch_historical_bars(symbol, config)
        logger.info(f"Fetched {len(bars)} bars for {symbol} (cache_hit={cache_hit})")

        # Step 3: Extract feature vector (dimension from config)
        features = _extract_features(symbol, bars, config)
        logger.info(f"Extracted {features.shape[0]}D feature vector for {symbol}")

        # Step 4: Run velocity inference
        velocity_engine = get_velocity_engine(
            checkpoint_path=config.checkpoint_path, device=config.device
        )

        predictions = velocity_engine.predict(
            feature_vector=features, symbol=symbol, current_price=current_price
        )

        # Step 5: Add metadata
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        predictions["metadata"].update(
            {
                "cache_hit": cache_hit,
                "feature_dim": features.shape[0],
                "data_sources": _get_data_sources(cache_hit, config),
                "latency_ms": round(latency_ms, 2),
            }
        )

        logger.info(
            f"Velocity predictions for {symbol} completed in {latency_ms:.0f}ms"
        )
        return predictions

    except Exception as e:
        logger.error(f"Failed to get velocity predictions for {symbol}: {e}")
        raise RuntimeError(f"Prediction failed for {symbol}: {e}") from e


# ==================== PRIVATE HELPER FUNCTIONS ====================


def _fetch_current_price(symbol: str, config: VelocityPredictionConfig) -> float:
    """
    Fetch current price for symbol.

    Flow: InfluxDB cache → TradeStation → fallback providers

    Args:
        symbol: Stock symbol
        config: Prediction config

    Returns:
        Current price

    Raises:
        RuntimeError: If price cannot be fetched
    """
    # Try cache first
    if config.use_cache:
        cache = get_influx_cache()
        latest_bars = cache.get_bars(
            symbol=symbol,
            start_date=datetime.now() - timedelta(minutes=config.cache_ttl_minutes),
            end_date=datetime.now(),
            interval="1m",
            source="tradestation",
        )

        if latest_bars is not None and not latest_bars.empty:
            price = float(latest_bars.iloc[-1]["close"])
            logger.info(f"Got current price from cache: ${price:.2f}")
            return price

    # Cache miss - fetch via MultiProviderFetcher (TradeStation → yfinance fallback)
    provider = _get_provider()
    quote = provider.get_quote(symbol)
    price = quote.get("price")
    if price is None:
        raise RuntimeError(f"Could not fetch current price for {symbol}")

    return float(price)


def _fetch_historical_bars(
    symbol: str, config: VelocityPredictionConfig
) -> tuple[pd.DataFrame, bool]:
    """
    Fetch historical bars with cache-first strategy.

    Args:
        symbol: Stock symbol
        config: Prediction config

    Returns:
        (bars_dataframe, cache_hit)

    Raises:
        RuntimeError: If bars cannot be fetched
    """
    cache_hit = False
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.lookback_days)

    # Try cache first
    if config.use_cache:
        cache = get_influx_cache()
        bars = cache.get_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            source="tradestation",
        )

        if bars is not None and not bars.empty:
            # Check if cache is fresh enough
            latest_timestamp = pd.Timestamp(bars.iloc[-1]["timestamp"])
            age_hours = (
                datetime.now() - latest_timestamp.to_pydatetime()  # type: ignore[operator]
            ).total_seconds() / 3600

            if age_hours <= (config.cache_ttl_minutes / 60):
                logger.info(f"Cache hit for {symbol} (age: {age_hours:.1f}h)")
                return bars, True

    # Cache miss or stale - fetch via MultiProviderFetcher (TradeStation → yfinance fallback)
    logger.info(f"Cache miss for {symbol}, fetching from provider")

    provider = _get_provider()
    bars = provider.get_historical_bars(
        symbol=symbol,
        interval="1d",
        start_date=start_date,
        end_date=end_date,
    )

    if bars is None or bars.empty:
        raise RuntimeError(f"Could not fetch historical bars for {symbol}")

    # Store in cache for next time
    if config.use_cache:
        cache = get_influx_cache()
        cache.store_bars(symbol=symbol, bars=bars, interval="1d", source="tradestation")

    return bars, cache_hit


def _extract_features(
    symbol: str, bars: pd.DataFrame, config: VelocityPredictionConfig
) -> np.ndarray:
    """
    Extract feature vector (dimension from config).

    Flow:
        - Technical indicators from bars
        - Sentiment from SenVec if enabled
        - Fundamental features if enabled
        - Combine into configured feature vector

    Args:
        symbol: Stock symbol
        bars: Historical OHLCV bars
        config: Prediction config

    Returns:
        Numpy array sized to DIMS.fincoll_total

    Raises:
        RuntimeError: If feature extraction fails
    """
    extractor = FeatureExtractor()

    try:
        # FeatureExtractor.extract_features signature: (ohlcv_data, symbol, timestamp)
        # Use the latest bar timestamp (coerce index element → datetime)
        timestamp: datetime = datetime.now()
        if not bars.empty:  # type: ignore[truthy-bool]
            _ts_candidate = pd.Timestamp(str(bars.index[-1])).to_pydatetime()
            if _ts_candidate is not pd.NaT:
                timestamp = _ts_candidate  # type: ignore[assignment]

        features = extractor.extract_features(
            ohlcv_data=bars, symbol=symbol, timestamp=timestamp
        )

        # Features should match DIMS.fincoll_total
        return features

    except Exception as e:
        logger.error(f"Feature extraction failed for {symbol}: {e}")
        raise RuntimeError(f"Feature extraction failed: {e}") from e


def _get_data_sources(cache_hit: bool, config: VelocityPredictionConfig) -> List[str]:
    """Get list of data sources used"""
    sources = []

    if cache_hit:
        sources.append("influxdb")
    else:
        sources.append("tradestation")

    if config.include_sentiment:
        sources.append("senvec")

    if config.include_fundamentals:
        sources.append("alphavantage")

    return sources
