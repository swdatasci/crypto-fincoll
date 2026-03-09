"""
MultiProviderFetcher - Orchestrates multiple data providers with fallback.

Implements:
- Round-robin provider selection
- Health-based routing (skip unhealthy providers)
- Automatic fallback on failure
- Provider priority based on data type
- Safe mode circuit breaker integration
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Type

import pandas as pd

from .base_trading_provider import BaseTradingProvider

logger = logging.getLogger(__name__)


def _get_safe_mode_manager():
    """Lazy import safe mode manager to avoid circular imports."""
    try:
        from ..monitoring.safe_mode import get_safe_mode_manager

        return get_safe_mode_manager()
    except ImportError:
        return None


class DataType(Enum):
    """Types of data requests with different provider preferences."""

    DAILY_BARS = "daily_bars"  # Daily historical data
    INTRADAY_BARS = "intraday_bars"  # Intraday (< 1d interval)
    REALTIME_QUOTE = "realtime_quote"  # Current price
    HISTORICAL_LONG = "historical_long"  # 1+ year historical


class MultiProviderFetcher:
    """
    Orchestrates multiple trading data providers with intelligent fallback.

    Priority order by data type (from FINCOLL_PIM_API_CLIENTS_INTEGRATION_ANALYSIS.md):
    - Daily Bars: yfinance > TradeStation > Alpaca > Public
    - Intraday: TradeStation > Alpaca > Public > yfinance
    - Real-time Quotes: TradeStation > Alpaca > Public > yfinance
    - Historical (1+ year): yfinance > TradeStation > Alpaca
    """

    # Default provider priority by data type
    # yfinance removed from DAILY_BARS and HISTORICAL_LONG (15-min delayed, unreliable)
    DEFAULT_PRIORITIES: Dict[DataType, List[str]] = {
        DataType.DAILY_BARS: ["tradestation", "alpaca", "public"],
        DataType.INTRADAY_BARS: ["tradestation", "alpaca", "public"],
        DataType.REALTIME_QUOTE: ["tradestation", "alpaca", "public"],
        DataType.HISTORICAL_LONG: ["tradestation", "alpaca"],
    }

    def __init__(
        self,
        providers: Optional[Dict[str, BaseTradingProvider]] = None,
        priorities: Optional[Dict[DataType, List[str]]] = None,
    ):
        """
        Initialize MultiProviderFetcher.

        Args:
            providers: Dict mapping provider name to provider instance.
                       If None, will attempt to create default providers.
            priorities: Custom priority order by data type.
        """
        self.providers: Dict[str, BaseTradingProvider] = providers or {}
        self.priorities = priorities or self.DEFAULT_PRIORITIES
        self._round_robin_index: Dict[str, int] = {}

        # yfinance is NOT auto-initialized (15-min delayed, unreliable for price data)
        # It remains available via add_provider() for fundamentals/dividends if needed

    def _init_yfinance(self) -> None:
        """Initialize yfinance provider (always available)."""
        try:
            # Create a wrapper that matches BaseTradingProvider interface
            self.providers["yfinance"] = YFinanceProviderWrapper()
            logger.info("Initialized yfinance provider")
        except Exception as e:
            logger.warning(f"Failed to initialize yfinance: {e}")

    def add_provider(self, name: str, provider: BaseTradingProvider) -> None:
        """Add a provider to the fetcher."""
        self.providers[name] = provider
        logger.info(f"Added provider: {name}")

    def remove_provider(self, name: str) -> None:
        """Remove a provider from the fetcher."""
        if name in self.providers:
            del self.providers[name]
            logger.info(f"Removed provider: {name}")

    def get_historical_bars(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bar_count: Optional[int] = None,
        preferred_provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical bars with automatic fallback.

        Args:
            symbol: Stock symbol
            interval: Bar interval ("1m", "5m", "15m", "1h", "1d")
            start_date: Start date
            end_date: End date
            bar_count: Number of bars
            preferred_provider: Specific provider to try first

        Returns:
            DataFrame with OHLCV data
        """
        # Check safe mode
        safe_mode = _get_safe_mode_manager()
        if safe_mode and safe_mode.is_safe_mode_active():
            status = safe_mode.get_status()
            raise RuntimeError(
                f"Safe mode is active: {status.reason}. "
                f"Trading operations are paused. Resume via /api/v1/safe-mode/resume"
            )

        # Determine data type based on interval
        data_type = self._classify_data_type(interval, start_date, end_date)

        # Get provider order
        provider_order = self._get_provider_order(data_type, preferred_provider)

        last_error = None
        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            # Skip unhealthy providers
            if hasattr(provider, "is_healthy") and not provider.is_healthy():
                logger.debug(f"Skipping unhealthy provider: {provider_name}")
                continue

            try:
                logger.debug(f"Trying provider {provider_name} for {symbol} {interval}")
                df = provider.get_historical_bars(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    bar_count=bar_count,
                )

                if df is not None and not df.empty:
                    logger.info(f"Got {len(df)} bars from {provider_name} for {symbol}")
                    return df

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                last_error = e

                # Record error in safe mode manager
                safe_mode = _get_safe_mode_manager()
                if safe_mode:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str:
                        safe_mode.record_rate_limit_error(provider_name, str(e))
                    elif any(
                        code in error_str for code in ["500", "502", "503", "504"]
                    ):
                        # Extract status code from error message
                        for code in [500, 502, 503, 504]:
                            if str(code) in error_str:
                                safe_mode.record_server_error(
                                    provider_name, code, str(e)
                                )
                                break
                    elif "timeout" in error_str:
                        safe_mode.record_timeout(provider_name, str(e))

                continue

        # All providers failed
        if last_error:
            raise RuntimeError(
                f"All providers failed for {symbol}. Last error: {last_error}"
            )
        return pd.DataFrame()

    def get_current_price(
        self,
        symbol: str,
        preferred_provider: Optional[str] = None,
    ) -> float:
        """
        Get current price with automatic fallback.

        Args:
            symbol: Stock symbol
            preferred_provider: Specific provider to try first

        Returns:
            Current price
        """
        # Check safe mode
        safe_mode = _get_safe_mode_manager()
        if safe_mode and safe_mode.is_safe_mode_active():
            status = safe_mode.get_status()
            raise RuntimeError(
                f"Safe mode is active: {status.reason}. "
                f"Trading operations are paused. Resume via /api/v1/safe-mode/resume"
            )

        provider_order = self._get_provider_order(
            DataType.REALTIME_QUOTE, preferred_provider
        )

        last_error = None
        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            if hasattr(provider, "is_healthy") and not provider.is_healthy():
                continue

            try:
                price = provider.get_current_price(symbol)
                if price and price > 0:
                    logger.debug(f"Got price {price} from {provider_name} for {symbol}")
                    return price
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed for quote {symbol}: {e}"
                )
                last_error = e

                # Record error in safe mode manager
                safe_mode = _get_safe_mode_manager()
                if safe_mode:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str:
                        safe_mode.record_rate_limit_error(provider_name, str(e))
                    elif any(
                        code in error_str for code in ["500", "502", "503", "504"]
                    ):
                        for code in [500, 502, 503, 504]:
                            if str(code) in error_str:
                                safe_mode.record_server_error(
                                    provider_name, code, str(e)
                                )
                                break
                    elif "timeout" in error_str:
                        safe_mode.record_timeout(provider_name, str(e))

                continue

        if last_error:
            raise RuntimeError(
                f"All providers failed for quote {symbol}. Last error: {last_error}"
            )
        return 0.0

    def get_name(self) -> str:
        """Return the name of the active/primary provider for logging."""
        # Return the first healthy provider name, or "multi"
        for name, provider in self.providers.items():
            if not hasattr(provider, "is_healthy") or provider.is_healthy():
                return name
        return "multi"

    def get_quote(self, symbol: str) -> Dict:
        """
        Get a quote dict for *symbol* with automatic provider fallback.

        Delegates to the first healthy provider that has a ``get_quote``
        method returning a non-None price.  Falls back to
        ``get_current_price`` if no provider returns a full quote.

        Returns a dict matching the standard quote schema::

            {
                "symbol": str,
                "price":  float | None,
                "bid":    float | None,
                "ask":    float | None,
                "volume": float | None,
                "provider": str,
            }
        """
        provider_order = self._get_provider_order(DataType.REALTIME_QUOTE)

        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            if hasattr(provider, "is_healthy") and not provider.is_healthy():
                continue

            try:
                if hasattr(provider, "get_quote"):
                    quote = provider.get_quote(symbol)
                    if quote and quote.get("price") is not None:
                        return quote
                else:
                    price = provider.get_current_price(symbol)
                    if price and price > 0:
                        return {
                            "symbol": symbol,
                            "price": price,
                            "bid": None,
                            "ask": None,
                            "volume": None,
                            "provider": provider_name,
                        }
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} get_quote failed for {symbol}: {e}"
                )
                continue

        return {
            "symbol": symbol,
            "price": None,
            "bid": None,
            "ask": None,
            "volume": None,
            "provider": "multi",
        }

    def get_provider_status(self) -> Dict[str, Dict]:
        """Get status of all providers."""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                "available": True,
                "healthy": provider.is_healthy()
                if hasattr(provider, "is_healthy")
                else True,
                "type": type(provider).__name__,
            }
        return status

    def _classify_data_type(
        self,
        interval: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> DataType:
        """Classify request to determine best provider order."""
        # Check if intraday
        intraday_intervals = {"1m", "5m", "15m", "30m", "1h"}
        if interval in intraday_intervals:
            return DataType.INTRADAY_BARS

        # Check if long historical (> 1 year)
        if start_date and end_date:
            days = (end_date - start_date).days
            if days > 365:
                return DataType.HISTORICAL_LONG

        return DataType.DAILY_BARS

    def _get_provider_order(
        self,
        data_type: DataType,
        preferred_provider: Optional[str] = None,
    ) -> List[str]:
        """Get provider order based on data type and preferences."""
        order = list(self.priorities.get(data_type, ["yfinance"]))

        # Move preferred provider to front if specified
        if preferred_provider and preferred_provider in order:
            order.remove(preferred_provider)
            order.insert(0, preferred_provider)

        # Filter to only available providers
        return [p for p in order if p in self.providers]


class YFinanceProviderWrapper(BaseTradingProvider):
    """Wrapper to make YFinanceProvider compatible with BaseTradingProvider interface."""

    def __init__(self):
        super().__init__(name="yfinance")
        self._healthy = True

    def _get_historical_bars(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bar_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch bars using yfinance."""
        try:
            # YFinanceProvider may have different interface - adapt as needed
            import yfinance as yf

            ticker = yf.Ticker(symbol)

            # Map interval
            yf_interval = self._map_interval(interval)

            # Determine period or date range
            if start_date and end_date:
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                )
            elif bar_count:
                # Use period for bar count
                period = self._bars_to_period(bar_count, interval)
                df = ticker.history(period=period, interval=yf_interval)
            else:
                df = ticker.history(period="100d", interval=yf_interval)

            if df.empty:
                return pd.DataFrame()

            # Normalize column names
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Rename 'date' or 'datetime' to 'timestamp'
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            elif "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})

            self._healthy = True
            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"yfinance failed for {symbol}: {e}")
            self._healthy = False
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get current price using yfinance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("regularMarketPrice") or info.get("currentPrice") or 0.0
            self._healthy = True
            return price
        except Exception as e:
            logger.error(f"yfinance quote failed for {symbol}: {e}")
            self._healthy = False
            raise

    def is_healthy(self) -> bool:
        return self._healthy

    def _map_interval(self, interval: str) -> str:
        """Map generic interval to yfinance interval."""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo",
        }
        return mapping.get(interval, "1d")

    def _bars_to_period(self, bar_count: int, interval: str) -> str:
        """Convert bar count to yfinance period string."""
        if interval in {"1m", "5m", "15m", "30m"}:
            days = max(1, bar_count // 78)  # ~78 5-min bars per day
            return f"{days}d"
        elif interval == "1h":
            days = max(1, bar_count // 7)  # ~7 hours per day
            return f"{days}d"
        else:
            return f"{bar_count}d"
