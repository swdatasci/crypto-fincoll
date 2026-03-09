"""
Data fetching and caching module for FinColl.

This module provides orchestration for:
- Cache-first data retrieval (InfluxDB)
- Multi-provider data fetching (TradeStation, etc.)
- Transparent caching of fetched data
"""

from .cache_manager import check_cache, save_to_cache, get_cache_stats
from .data_fetcher import fetch_bars

__all__ = [
    'check_cache',
    'save_to_cache',
    'get_cache_stats',
    'fetch_bars',
]
