"""TradeStation provider wrapper using pim-api-clients SDK."""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from .base_trading_provider import BaseTradingProvider

logger = logging.getLogger(__name__)


class TradeStationTradingProvider(BaseTradingProvider):
    """
    TradeStation data provider using pim-tradestation-sdk.

    Leverages the SDK's built-in:
    - OAuth token management with auto-refresh
    - Rate limiting with Redis backend
    - Market hours validation
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ):
        super().__init__(name="tradestation")

        # Lazy import to avoid import errors if SDK not installed
        from tradestation_sdk import TradeStationClient

        # Let SDK determine base_url from API_ENVIRONMENT env variable
        # API_ENVIRONMENT: mock (local), paper (sim), live (real market data)
        # For market data quotes, use 'live' even when paper trading
        # DO NOT pass base_url - let SDK handle URL selection
        self.client = TradeStationClient(
            client_id=client_id or os.getenv("TRADESTATION_CLIENT_ID", ""),
            client_secret=client_secret or os.getenv("TRADESTATION_CLIENT_SECRET", ""),
            refresh_token=refresh_token or os.getenv("TRADESTATION_REFRESH_TOKEN"),
            max_requests_per_minute=30,
            rate_limit_period=60,
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
        Fetch historical bars from TradeStation.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            interval: Bar interval ("1m", "5m", "15m", "1h", "1d")
            start_date: Start date for historical data
            end_date: End date for historical data
            bar_count: Number of bars to fetch (alternative to date range)

        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
        """
        try:
            # Map interval to TradeStation interval format
            ts_interval = self._map_interval(interval)

            bars = self.client.get_bars(
                symbol=symbol,
                interval=ts_interval,
                bars_back=bar_count or 100,
            )

            if not bars:
                return pd.DataFrame()

            # Convert to DataFrame
            # Note: SDK uses PascalCase (TimeStamp, TotalVolume)
            # CRITICAL: TradeStation returns ALL values as strings, including timestamps
            df = pd.DataFrame(
                [
                    {
                        "timestamp": bar.TimeStamp,
                        "open": bar.Open,
                        "high": bar.High,
                        "low": bar.Low,
                        "close": bar.Close,
                        "volume": bar.TotalVolume,
                    }
                    for bar in bars
                ]
            )

            # Convert timestamp column to datetime for filtering
            # Make timezone-naive to match start_date/end_date (which are tz-naive)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                # Remove timezone info (tz_localize(None) doesn't work on already-aware timestamps)
                df["timestamp"] = df["timestamp"].dt.tz_convert(None)

            # Convert OHLCV columns from strings to numeric (TradeStation returns all as strings)
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Filter by date range if provided
            if start_date and "timestamp" in df.columns:
                df = df[df["timestamp"] >= start_date]
            if end_date and "timestamp" in df.columns:
                df = df[df["timestamp"] <= end_date]

            # Set timestamp as index (expected by training.py and feature extractor)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            self._healthy = True
            return df

        except Exception as e:
            logger.error(f"TradeStation get_bars failed for {symbol}: {e}")
            self._healthy = False
            raise

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price from TradeStation quote.

        Returns 0.0 for custom/broker-specific symbols (e.g., PUBLIC.com CUSIPs)
        that TradeStation doesn't recognize.
        """
        try:
            quote = self.client.get_quote(symbol)
            self._healthy = True
            # SDK returns None for invalid symbols instead of raising
            if quote is None:
                logger.info(
                    f"Symbol {symbol} not recognized by TradeStation (likely custom/broker-specific)"
                )
                return 0.0
            return float(quote.Last) if quote else 0.0
        except Exception as e:
            logger.error(f"TradeStation get_quote failed for {symbol}: {e}")
            self._healthy = False
            raise

    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self._healthy

    def _map_interval(self, interval: str) -> str:
        """Map generic interval to TradeStation interval format."""
        mapping = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "60Min",
            "1d": "Daily",
            "1w": "Weekly",
            "1M": "Monthly",
        }
        return mapping.get(interval, "Daily")
