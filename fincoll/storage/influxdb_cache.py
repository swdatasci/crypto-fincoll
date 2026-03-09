"""
InfluxDB Cache for Market Data

Stores TradeStation/Alpaca/yfinance bars in InfluxDB for:
- Avoiding repeated API calls (rate limit savings)
- Fast historical data retrieval
- Automatic retention policy (90 days minute bars, 10 years daily)

Usage:
    from fincoll.storage.influxdb_cache import InfluxDBCache

    cache = InfluxDBCache()

    # Try to get from cache first
    bars = cache.get_bars('AAPL', start_date, end_date, interval='1m')

    if bars is None:
        # Cache miss - fetch from API
        bars = provider.get_historical_bars(...)
        cache.store_bars('AAPL', bars, interval='1m')

    return bars
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
# influxdb_client is imported lazily inside InfluxDBCache.__init__ so that
# importing fincoll.features.feature_extractor (e.g. from finvec) works
# without requiring influxdb_client in the caller's venv.


class InfluxDBCache:
    """Cache market data in InfluxDB with automatic retention"""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        """
        Initialize InfluxDB cache

        Args:
            url: InfluxDB URL (default: from env INFLUXDB_URL)
            token: Auth token (default: from env INFLUXDB_TOKEN)
            org: Organization name (default: from env INFLUXDB_ORG)
            bucket: Bucket name for market data (default: from env INFLUXDB_BUCKET)
        """
        import os

        # Read from environment with fallbacks (consistent with influxdb_saver.py)
        self.url = url or os.getenv("INFLUXDB_URL", "http://10.32.3.27:8086")
        self.org = org or os.getenv("INFLUXDB_ORG", "caelum")
        self.bucket = bucket or os.getenv("INFLUXDB_BUCKET", "market_data")
        token = token or os.getenv("INFLUXDB_TOKEN", "")

        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS

            self.client = InfluxDBClient(url=self.url, token=token, org=self.org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.enabled = True
            print(f"✅ InfluxDB cache enabled: {self.url}")
        except ImportError:
            print("⚠️  influxdb_client not installed — InfluxDB cache disabled")
            self.enabled = False
        except Exception as e:
            print(f"⚠️  InfluxDB cache disabled: {e}")
            self.enabled = False

    @staticmethod
    def _sanitize_flux_string(value: str) -> str:
        """
        Sanitize string values for Flux queries to prevent injection attacks.

        Args:
            value: String to sanitize

        Returns:
            Sanitized string safe for Flux queries
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value)}")

        # Remove or escape dangerous characters
        # Flux string literals use double quotes, so escape them
        sanitized = value.replace('"', '\\"').replace("\\", "\\\\")

        # Only allow alphanumeric, dash, underscore, and dot for symbols/intervals
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        )
        if not all(c in allowed_chars or c in '\\"' for c in sanitized):
            raise ValueError(f"Invalid characters in Flux query value: {value}")

        return sanitized

    def store_bars(
        self,
        symbol: str,
        bars: pd.DataFrame,
        interval: str = "1d",
        source: str = "tradestation",
    ) -> bool:
        """
        Store OHLCV bars in InfluxDB

        Args:
            symbol: Stock symbol
            bars: DataFrame with columns: timestamp, open, high, low, close, volume
            interval: Time interval (1m, 5m, 15m, 1h, 1d)
            source: Data source (tradestation, alpaca, yfinance)

        Returns:
            True if stored successfully
        """
        if not self.enabled or bars is None or bars.empty:
            return False

        try:
            from influxdb_client import Point, WritePrecision

            points = []

            for idx, row in bars.iterrows():
                # Get timestamp - either from column or from index
                if "timestamp" in bars.columns:
                    timestamp = pd.Timestamp(row["timestamp"])
                else:
                    # Timestamp is the index
                    timestamp = pd.Timestamp(idx)

                # Create InfluxDB point
                point = (
                    Point("ohlcv")
                    .tag("symbol", symbol)
                    .tag("interval", interval)
                    .tag("source", source)
                    .field("open", float(row["open"]))
                    .field("high", float(row["high"]))
                    .field("low", float(row["low"]))
                    .field("close", float(row["close"]))
                    .field("volume", int(row["volume"]))
                    .time(timestamp, WritePrecision.NS)
                )
                points.append(point)

            # Write in batches
            self.write_api.write(bucket=self.bucket, record=points)
            print(f"✅ Stored {len(points)} bars for {symbol} ({interval}) in InfluxDB")
            return True

        except Exception as e:
            print(f"⚠️  Failed to store bars in InfluxDB: {e}")
            return False

    def get_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        source: str = "tradestation",
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve OHLCV bars from InfluxDB cache

        Args:
            symbol: Stock symbol
            start_date: Start datetime
            end_date: End datetime
            interval: Time interval
            source: Preferred data source

        Returns:
            DataFrame with OHLCV data, or None if not in cache
        """
        if not self.enabled:
            return None

        try:
            # Sanitize inputs to prevent injection attacks
            safe_symbol = self._sanitize_flux_string(symbol)
            safe_interval = self._sanitize_flux_string(interval)

            # Build Flux query with sanitized inputs
            # Skip source filter if source is None or "any" to accept data from any provider
            source_filter = ""
            if source and source != "any":
                safe_source = self._sanitize_flux_string(source)
                source_filter = f'|> filter(fn: (r) => r["source"] == "{safe_source}")'

            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
              |> filter(fn: (r) => r["_measurement"] == "ohlcv")
              |> filter(fn: (r) => r["symbol"] == "{safe_symbol}")
              |> filter(fn: (r) => r["interval"] == "{safe_interval}")
              {source_filter}
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            print(
                f"🔍 InfluxDB query: {symbol} {interval} from {start_date.isoformat()} to {end_date.isoformat()}"
            )
            result = self.query_api.query(query, org=self.org)
            print(f"🔍 Query returned {len(result) if result else 0} tables")

            if not result or len(result) == 0:
                return None

            # Convert to DataFrame
            records = []
            for table in result:
                for record in table.records:
                    records.append(
                        {
                            "timestamp": record.get_time(),
                            "open": record.values.get("open"),
                            "high": record.values.get("high"),
                            "low": record.values.get("low"),
                            "close": record.values.get("close"),
                            "volume": record.values.get("volume"),
                        }
                    )

            if not records:
                return None

            df = pd.DataFrame(records)

            # BUGFIX: Set timestamp as index (was causing integer index = 1969 date bug)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                # Ensure index is datetime, not integers
                if not isinstance(df.index, pd.DatetimeIndex):
                    print(
                        f"⚠️  WARNING: Timestamp index is not DatetimeIndex for {symbol}"
                    )

            print(
                f"✅ Retrieved {len(df)} bars for {symbol} ({interval}) from InfluxDB cache"
            )
            return df

        except Exception as e:
            print(f"⚠️  InfluxDB cache miss: {e}")
            return None

    def get_latest_timestamp(
        self, symbol: str, interval: str = "1d", source: str = "tradestation"
    ) -> Optional[datetime]:
        """
        Get the latest timestamp for a symbol in cache

        Useful for incremental updates: only fetch new bars since last cached timestamp

        Args:
            symbol: Stock symbol
            interval: Time interval
            source: Data source

        Returns:
            Latest timestamp in cache, or None if not cached
        """
        if not self.enabled:
            return None

        try:
            # Sanitize inputs to prevent injection attacks
            safe_symbol = self._sanitize_flux_string(symbol)
            safe_interval = self._sanitize_flux_string(interval)
            safe_source = self._sanitize_flux_string(source)

            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -90d)
              |> filter(fn: (r) => r["_measurement"] == "ohlcv")
              |> filter(fn: (r) => r["symbol"] == "{safe_symbol}")
              |> filter(fn: (r) => r["interval"] == "{safe_interval}")
              |> filter(fn: (r) => r["source"] == "{safe_source}")
              |> last()
            '''

            result = self.query_api.query(query, org=self.org)

            if result and len(result) > 0 and len(result[0].records) > 0:
                return result[0].records[0].get_time()

            return None

        except Exception as e:
            print(f"⚠️  Failed to get latest timestamp: {e}")
            return None

    def close(self):
        """Close InfluxDB connection"""
        if self.enabled and self.client:
            self.client.close()


# Global cache instance (initialized on first import)
_global_cache = None


def get_cache() -> InfluxDBCache:
    """Get global InfluxDB cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = InfluxDBCache()
    return _global_cache
