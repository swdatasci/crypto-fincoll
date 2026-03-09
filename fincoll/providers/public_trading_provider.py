"""Public.com provider wrapper using pim-api-clients SDK."""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from .base_trading_provider import BaseTradingProvider

logger = logging.getLogger(__name__)


class PublicTradingProvider(BaseTradingProvider):
    """
    Public.com data provider using pim-public-sdk.

    Leverages the SDK's built-in:
    - API authentication
    - Rate limiting
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
    ):
        super().__init__(name="public")

        # Lazy import to avoid import errors if SDK not installed
        from public_api_sdk import PublicApiClient

        self.client = PublicApiClient(
            api_key=api_key or os.getenv("PUBLIC_API_KEY", ""),
            account_id=account_id or os.getenv("PUBLIC_ACCOUNT_ID"),
        )
        self._healthy = True

    def _get_historical_bars(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bar_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical bars from Public.com.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            interval: Bar interval ("1d", "1w", "1M")
            start_date: Start date for historical data
            end_date: End date for historical data
            bar_count: Number of bars to fetch

        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
        """
        try:
            # Map interval to Public.com period
            period = self._map_interval(interval)

            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                from datetime import timedelta

                # Compute a time window that covers bar_count bars for the
                # given interval.  Using raw bar_count as days is wrong for
                # intraday intervals (e.g. 100 bars of 1-minute data ≠ 100 days).
                _bars = bar_count or 100
                _interval_seconds = {
                    "1m": 60,
                    "5m": 300,
                    "15m": 900,
                    "30m": 1800,
                    "1h": 3600,
                    "1d": 86400,
                    "1w": 604800,
                    "1M": 2592000,
                }.get(interval, 86400)
                # Add 50 % buffer to account for market closures / weekends
                _window_seconds = _bars * _interval_seconds * 1.5
                start_date = end_date - timedelta(seconds=_window_seconds)

            history = self.client.get_history(
                symbol=symbol,
                period=period,
                start_date=start_date.strftime("%Y-%m-%d")
                if isinstance(start_date, datetime)
                else start_date,
                end_date=end_date.strftime("%Y-%m-%d")
                if isinstance(end_date, datetime)
                else end_date,
            )

            if not history:
                return pd.DataFrame()

            # Convert to DataFrame - adjust based on actual response structure
            df = pd.DataFrame(
                [
                    {
                        "timestamp": bar.date
                        if hasattr(bar, "date")
                        else bar.get("date"),
                        "open": bar.open if hasattr(bar, "open") else bar.get("open"),
                        "high": bar.high if hasattr(bar, "high") else bar.get("high"),
                        "low": bar.low if hasattr(bar, "low") else bar.get("low"),
                        "close": bar.close
                        if hasattr(bar, "close")
                        else bar.get("close"),
                        "volume": bar.volume
                        if hasattr(bar, "volume")
                        else bar.get("volume", 0),
                    }
                    for bar in history
                ]
            )

            self._healthy = True
            return df

        except Exception as e:
            logger.error(f"Public get_history failed for {symbol}: {e}")
            self._healthy = False
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get current price from Public.com quote."""
        try:
            quotes = self.client.get_quotes([symbol])
            if quotes and len(quotes) > 0:
                quote = quotes[0]
                self._healthy = True
                # Use last price or mid of bid/ask
                if hasattr(quote, "last_price"):
                    return quote.last_price
                elif hasattr(quote, "bid") and hasattr(quote, "ask"):
                    return (quote.bid + quote.ask) / 2
                return 0.0
            return 0.0
        except Exception as e:
            logger.error(f"Public get_quotes failed for {symbol}: {e}")
            self._healthy = False
            raise

    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self._healthy

    def _map_interval(self, interval: str) -> str:
        """Map generic interval to Public.com period."""
        mapping = {
            "1d": "day",
            "1w": "week",
            "1M": "month",
            "1y": "year",
        }
        return mapping.get(interval, "day")
