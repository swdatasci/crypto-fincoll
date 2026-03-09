"""
Data Fetcher for FinColl - Multi-Provider Data Fetching with Cache-First Strategy

This module orchestrates data fetching:
1. CHECK InfluxDB cache first (fast path)
2. On cache miss: Fetch from providers with automatic fallback
3. SAVE fetched data to cache for next time
4. Return data to caller

Supports multi-timeframe fetching for velocity training.

Provider Priority (via MultiProviderFetcher):
- Daily Bars: yfinance > TradeStation > Alpaca > Public
- Intraday: TradeStation > Alpaca > Public > yfinance
- Real-time Quotes: TradeStation > Alpaca > Public > yfinance
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from .cache_manager import check_cache, save_to_cache

logger = logging.getLogger(__name__)

# Singleton instance of MultiProviderFetcher for reuse
_multi_provider_fetcher = None


def get_multi_provider_fetcher():
    """Get or create the singleton MultiProviderFetcher instance."""
    global _multi_provider_fetcher

    if _multi_provider_fetcher is None:
        try:
            from ..providers.alpaca_trading_provider import AlpacaTradingProvider
            from ..providers.multi_provider_fetcher import MultiProviderFetcher
            from ..providers.public_trading_provider import PublicTradingProvider
            from ..providers.tradestation_trading_provider import (
                TradeStationTradingProvider,
            )

            _multi_provider_fetcher = MultiProviderFetcher()

            # Add pim-api-clients providers if credentials are available
            try:
                ts_provider = TradeStationTradingProvider()
                _multi_provider_fetcher.add_provider("tradestation", ts_provider)
                logger.info("Added TradeStation provider")
            except Exception as e:
                logger.debug(f"TradeStation provider not available: {e}")

            try:
                alpaca_provider = AlpacaTradingProvider()
                _multi_provider_fetcher.add_provider("alpaca", alpaca_provider)
                logger.info("Added Alpaca provider")
            except Exception as e:
                logger.debug(f"Alpaca provider not available: {e}")

            try:
                public_provider = PublicTradingProvider()
                _multi_provider_fetcher.add_provider("public", public_provider)
                logger.info("Added Public provider")
            except Exception as e:
                logger.debug(f"Public provider not available: {e}")

            logger.info(
                f"MultiProviderFetcher initialized with providers: {list(_multi_provider_fetcher.providers.keys())}"
            )

        except ImportError as e:
            logger.warning(
                f"MultiProviderFetcher not available, falling back to legacy: {e}"
            )
            _multi_provider_fetcher = None

    return _multi_provider_fetcher


def fetch_bars(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    provider: str = "auto",
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars with cache-first strategy and multi-provider fallback.

    This is the PRIMARY function for data fetching in FinColl orchestration.

    Flow:
        1. Check InfluxDB cache (if use_cache=True)
        2. If cache hit: return cached data
        3. If cache miss: fetch from providers with automatic fallback
        4. Save fetched data to cache (for next time)
        5. Return data

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        timeframe: Time interval ('1m', '15m', '1h', '1d')
        start: Start date (YYYY-MM-DD format)
        end: End date (YYYY-MM-DD format)
        provider: Data provider ('auto', 'tradestation', 'alpaca', 'yfinance', 'public')
                  'auto' (default) uses MultiProviderFetcher with automatic fallback
        use_cache: Whether to check cache first (default: True)

    Returns:
        DataFrame with OHLCV data, or None if fetch fails

    Example:
        >>> # Fetch with cache and auto provider selection (recommended)
        >>> bars = fetch_bars('AAPL', '1d', '2024-01-01', '2024-12-01')
        >>>
        >>> # Force specific provider
        >>> bars = fetch_bars('AAPL', '1d', '2024-01-01', '2024-12-01', provider='yfinance')
        >>>
        >>> # Force fresh fetch (bypass cache)
        >>> bars = fetch_bars('AAPL', '1d', '2024-01-01', '2024-12-01', use_cache=False)
    """
    # Step 1: Check cache first (if enabled)
    cache_source = provider if provider != "auto" else "multi"
    if use_cache:
        cached_bars = check_cache(symbol, timeframe, start, end, source=cache_source)

        if cached_bars is not None and not cached_bars.empty:
            logger.info(f"Using cached data for {symbol} ({timeframe})")
            return cached_bars

    # Step 2: Cache miss - fetch from provider(s)
    logger.info(f"Fetching {symbol} ({timeframe}) with provider={provider}")

    try:
        bars = _fetch_from_provider(symbol, timeframe, start, end, provider)

        if bars is None or bars.empty:
            logger.error(f"Provider returned no data for {symbol}")
            return None

        # Step 3: Save to cache for next time (if cache is enabled)
        if use_cache:
            save_to_cache(symbol, timeframe, bars, source=cache_source)

        logger.info(f"Fetched {len(bars)} bars for {symbol} ({timeframe})")
        return bars

    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None


def fetch_multi_timeframe(
    symbol: str,
    timeframes: List[str],
    start: str,
    end: str,
    provider: str = "auto",
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple timeframes (for velocity training).

    This is used when training velocity models that need multiple
    timeframes (1m, 15m, 1h, 1d) simultaneously.

    Args:
        symbol: Stock symbol
        timeframes: List of timeframes to fetch (e.g., ['1m', '15m', '1h', '1d'])
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        provider: Data provider name ('auto' for multi-provider fallback)
        use_cache: Whether to use cache-first strategy

    Returns:
        Dictionary mapping timeframe -> DataFrame
        {
            '1m': DataFrame(...),
            '15m': DataFrame(...),
            '1h': DataFrame(...),
            '1d': DataFrame(...),
        }

    Example:
        >>> # Fetch all timeframes for velocity training
        >>> data = fetch_multi_timeframe('AAPL', ['1m', '15m', '1h', '1d'], '2024-01-01', '2024-12-01')
        >>> print(f"Got {len(data['1m'])} 1-minute bars")
        >>> print(f"Got {len(data['1d'])} daily bars")
    """
    result = {}

    for tf in timeframes:
        bars = fetch_bars(symbol, tf, start, end, provider, use_cache)

        if bars is not None and not bars.empty:
            result[tf] = bars
        else:
            logger.warning(f"Failed to fetch {symbol} for timeframe {tf}")

    if not result:
        logger.error(f"Failed to fetch any timeframes for {symbol}")

    return result


def fetch_batch(
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    provider: str = "auto",
    use_cache: bool = True,
    skip_failures: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple symbols (batch operation).

    Args:
        symbols: List of stock symbols
        timeframe: Time interval (applies to all symbols)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        provider: Data provider name ('auto' for multi-provider fallback)
        use_cache: Whether to use cache-first strategy
        skip_failures: If True, continue on errors; if False, raise on first error

    Returns:
        Dictionary mapping symbol -> DataFrame
        {
            'AAPL': DataFrame(...),
            'MSFT': DataFrame(...),
            ...
        }

    Example:
        >>> # Fetch multiple symbols with auto provider fallback
        >>> data = fetch_batch(['AAPL', 'MSFT', 'GOOGL'], '1d', '2024-01-01', '2024-12-01')
        >>> print(f"Fetched data for {len(data)} symbols")
    """
    result = {}
    failed_symbols = []

    for symbol in symbols:
        try:
            bars = fetch_bars(symbol, timeframe, start, end, provider, use_cache)

            if bars is not None and not bars.empty:
                result[symbol] = bars
            else:
                failed_symbols.append(symbol)
                if not skip_failures:
                    raise ValueError(f"Failed to fetch data for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            failed_symbols.append(symbol)

            if not skip_failures:
                raise

    logger.info(f"Batch fetch: {len(result)}/{len(symbols)} symbols successful")

    if failed_symbols:
        logger.warning(f"Failed symbols: {failed_symbols}")

    return result


def get_current_price(symbol: str, provider: str = "auto") -> Optional[float]:
    """
    Get current price for a symbol using multi-provider fallback.

    Args:
        symbol: Stock symbol
        provider: Provider to use ('auto' for automatic fallback)

    Returns:
        Current price, or None if all providers fail
    """
    fetcher = get_multi_provider_fetcher()

    if fetcher is None:
        logger.error("MultiProviderFetcher not available")
        return None

    try:
        preferred = provider if provider != "auto" else None
        return fetcher.get_current_price(symbol, preferred_provider=preferred)
    except Exception as e:
        logger.error(f"Failed to get current price for {symbol}: {e}")
        return None


def get_provider_status() -> Dict[str, Dict]:
    """
    Get status of all configured providers.

    Returns:
        Dictionary with provider status information
    """
    fetcher = get_multi_provider_fetcher()

    if fetcher is None:
        return {"error": "MultiProviderFetcher not available"}

    return fetcher.get_provider_status()


# ==================== PRIVATE HELPER FUNCTIONS ====================


def _fetch_from_provider(
    symbol: str, timeframe: str, start: str, end: str, provider: str
) -> Optional[pd.DataFrame]:
    """
    Fetch data from the specified provider or use multi-provider fallback.

    Args:
        symbol: Stock symbol
        timeframe: Time interval
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        provider: Provider name ('auto' for multi-provider fallback)

    Returns:
        DataFrame with OHLCV data, or None on failure
    """
    # Parse dates
    start_dt = datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
    end_dt = datetime.strptime(end, "%Y-%m-%d") if isinstance(end, str) else end

    # Use MultiProviderFetcher for automatic fallback
    if provider == "auto":
        fetcher = get_multi_provider_fetcher()
        if fetcher:
            try:
                return fetcher.get_historical_bars(
                    symbol=symbol,
                    interval=timeframe,
                    start_date=start_dt,
                    end_date=end_dt,
                )
            except Exception as e:
                logger.error(f"MultiProviderFetcher failed for {symbol}: {e}")
                return None
        else:
            # Fall back to legacy if MultiProviderFetcher not available
            provider = "tradestation"

    # Specific provider requested - use MultiProviderFetcher with preference
    fetcher = get_multi_provider_fetcher()
    if fetcher:
        try:
            return fetcher.get_historical_bars(
                symbol=symbol,
                interval=timeframe,
                start_date=start_dt,
                end_date=end_dt,
                preferred_provider=provider,
            )
        except Exception as e:
            logger.error(
                f"Fetch failed for {symbol} with preferred provider {provider}: {e}"
            )
            return None

    # Legacy fallback - direct provider instantiation
    return _fetch_from_legacy_provider(symbol, timeframe, start, end, provider)


def _fetch_from_legacy_provider(
    symbol: str, timeframe: str, start: str, end: str, provider: str
) -> Optional[pd.DataFrame]:
    """
    Fallback provider fetching when MultiProviderFetcher is not available.

    Used for graceful degradation in environments without fincoll.providers.
    """
    if provider == "tradestation":
        return _fetch_from_tradestation(symbol, timeframe, start, end)
    elif provider == "yfinance":
        return _fetch_from_yfinance(symbol, timeframe, start, end)
    else:
        logger.error(f"Unsupported legacy provider: {provider}")
        return None


def _fetch_from_tradestation(
    symbol: str, timeframe: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    """
    Fetch data via MultiProviderFetcher (TradeStation → yfinance fallback).
    """
    try:
        from ..providers.multi_provider_fetcher import MultiProviderFetcher
        from datetime import datetime

        provider = MultiProviderFetcher()
        start_dt = (
            datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
        )
        end_dt = datetime.strptime(end, "%Y-%m-%d") if isinstance(end, str) else end

        bars = provider.get_historical_bars(
            symbol=symbol, start_date=start_dt, end_date=end_dt, interval=timeframe
        )

        return bars

    except Exception as e:
        logger.error(f"Provider fetch failed for {symbol}: {e}")
        return None


def _fetch_from_yfinance(
    symbol: str, timeframe: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    """
    Fetch data from yfinance (fallback data source).
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)

        # Map timeframe to yfinance interval
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo",
        }
        yf_interval = interval_map.get(timeframe, "1d")

        df = ticker.history(start=start, end=end, interval=yf_interval)

        if df.empty:
            return None

        # Normalize column names
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    except Exception as e:
        logger.error(f"yfinance fetch failed for {symbol}: {e}")
        return None
