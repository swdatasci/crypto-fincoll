"""Alpaca provider wrapper using pim-api-clients SDK."""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from .base_trading_provider import BaseTradingProvider

logger = logging.getLogger(__name__)


class AlpacaTradingProvider(BaseTradingProvider):
    """
    Alpaca data provider using pim-alpaca-sdk.

    Leverages the SDK's built-in:
    - API key authentication
    - Rate limiting
    - Market hours validation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        super().__init__(name="alpaca")

        # Lazy import to avoid import errors if SDK not installed
        from alpaca_sdk import AlpacaClient

        self.client = AlpacaClient(
            api_key=api_key or os.getenv("ALPACA_API_KEY", ""),
            api_secret=api_secret or os.getenv("ALPACA_API_SECRET", ""),
            paper=paper,
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
        Fetch historical bars from Alpaca.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            interval: Bar interval ("1m", "5m", "15m", "1h", "1d")
            start_date: Start date for historical data
            end_date: End date for historical data
            bar_count: Number of bars to fetch

        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
        """
        try:
            # Map interval to Alpaca timeframe
            timeframe = self._map_interval(interval)

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

            bars = self.client.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date.isoformat()
                if isinstance(start_date, datetime)
                else start_date,
                end=end_date.isoformat()
                if isinstance(end_date, datetime)
                else end_date,
                limit=bar_count or 1000,
            )

            if not bars:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                    for bar in bars
                ]
            )

            self._healthy = True
            return df

        except Exception as e:
            logger.error(f"Alpaca get_bars failed for {symbol}: {e}")
            self._healthy = False
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get current price from Alpaca quote."""
        try:
            quote = self.client.get_latest_quote(symbol)
            self._healthy = True
            # Use ask price as current price, fallback to bid
            return (
                quote.ask_price
                if quote and quote.ask_price
                else (quote.bid_price if quote else 0.0)
            )
        except Exception as e:
            logger.error(f"Alpaca get_latest_quote failed for {symbol}: {e}")
            self._healthy = False
            raise

    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self._healthy

    def _map_interval(self, interval: str) -> str:
        """Map generic interval to Alpaca timeframe."""
        mapping = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "1d": "1Day",
            "1w": "1Week",
            "1M": "1Month",
        }
        return mapping.get(interval, "1Day")
