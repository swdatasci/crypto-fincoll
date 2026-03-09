"""
Cache Manager for FinColl - InfluxDB Cache Operations

This module provides cache-first data access:
1. Check InfluxDB cache for existing data
2. Return cached data if available and fresh
3. Provide interface to store fetched data

Database: InfluxDB at 10.32.3.27:8086
Bucket: market_data
Organization: caelum
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

from ..storage.influxdb_cache import get_cache as get_influx_cache

logger = logging.getLogger(__name__)


def check_cache(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    source: str = 'tradestation'
) -> Optional[pd.DataFrame]:
    """
    Check InfluxDB cache for existing bars.

    This is the FIRST step in the cache-first orchestration flow.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        timeframe: Time interval ('1m', '15m', '1h', '1d')
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        source: Data source ('tradestation', 'alpaca', 'yfinance')

    Returns:
        DataFrame with OHLCV data if cache hit, None if cache miss

    Example:
        >>> bars = check_cache('AAPL', '1d', '2024-01-01', '2024-12-01')
        >>> if bars is not None:
        >>>     print(f"Cache hit! Got {len(bars)} bars")
        >>> else:
        >>>     print("Cache miss - need to fetch from provider")
    """
    try:
        cache = get_influx_cache()

        if not cache.enabled:
            logger.warning("InfluxDB cache is disabled - skipping cache check")
            return None

        # Convert string dates to datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Query cache
        bars = cache.get_bars(
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
            interval=timeframe,
            source=source
        )

        if bars is not None and not bars.empty:
            logger.info(f"Cache HIT for {symbol} ({timeframe}): {len(bars)} bars")
            return bars
        else:
            logger.info(f"Cache MISS for {symbol} ({timeframe})")
            return None

    except Exception as e:
        logger.error(f"Cache check failed for {symbol}: {e}")
        return None


def save_to_cache(
    symbol: str,
    timeframe: str,
    data: pd.DataFrame,
    source: str = 'tradestation'
) -> bool:
    """
    Save fetched bars to InfluxDB cache for future use.

    This is called AFTER fetching from a provider to populate the cache.

    Args:
        symbol: Stock symbol
        timeframe: Time interval ('1m', '15m', '1h', '1d')
        data: DataFrame with OHLCV data (must have 'timestamp' column/index)
        source: Data source name

    Returns:
        True if stored successfully, False otherwise

    Example:
        >>> bars = provider.get_historical_bars('AAPL', ...)
        >>> success = save_to_cache('AAPL', '1d', bars)
        >>> if success:
        >>>     print("Cached for next time!")
    """
    try:
        cache = get_influx_cache()

        if not cache.enabled:
            logger.warning("InfluxDB cache is disabled - cannot save")
            return False

        if data is None or data.empty:
            logger.warning(f"Cannot save empty data to cache for {symbol}")
            return False

        # Ensure timestamp is in the data (as index or column)
        if 'timestamp' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            logger.error(f"Data must have 'timestamp' column or DatetimeIndex to cache")
            return False

        # Add timestamp column if it's the index
        if isinstance(data.index, pd.DatetimeIndex) and 'timestamp' not in data.columns:
            data = data.copy()
            data['timestamp'] = data.index

        # Store in cache
        success = cache.store_bars(
            symbol=symbol,
            bars=data,
            interval=timeframe,
            source=source
        )

        if success:
            logger.info(f"Cached {len(data)} bars for {symbol} ({timeframe})")
        else:
            logger.warning(f"Failed to cache bars for {symbol}")

        return success

    except Exception as e:
        logger.error(f"Failed to save to cache for {symbol}: {e}")
        return False


def get_cache_stats(
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    source: str = 'tradestation'
) -> Dict[str, Any]:
    """
    Get cache coverage statistics for a list of symbols.

    Useful for reporting cache hit rates and identifying missing data.

    Args:
        symbols: List of stock symbols
        timeframe: Time interval
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source: Data source

    Returns:
        Dictionary with cache statistics:
        {
            'total_symbols': int,
            'cached_symbols': [str],
            'missing_symbols': [str],
            'cache_hit_rate': float,
            'total_bars_cached': int,
        }

    Example:
        >>> stats = get_cache_stats(['AAPL', 'MSFT', 'GOOGL'], '1d', '2024-01-01', '2024-12-01')
        >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        >>> print(f"Need to fetch: {stats['missing_symbols']}")
    """
    cached_symbols = []
    missing_symbols = []
    total_bars = 0

    for symbol in symbols:
        bars = check_cache(symbol, timeframe, start_date, end_date, source)

        if bars is not None and not bars.empty:
            cached_symbols.append(symbol)
            total_bars += len(bars)
        else:
            missing_symbols.append(symbol)

    total = len(symbols)
    hit_rate = len(cached_symbols) / total if total > 0 else 0.0

    stats = {
        'total_symbols': total,
        'cached_symbols': cached_symbols,
        'missing_symbols': missing_symbols,
        'cache_hit_rate': hit_rate,
        'total_bars_cached': total_bars,
    }

    logger.info(f"Cache stats: {len(cached_symbols)}/{total} symbols cached ({hit_rate:.1%})")

    return stats
