#!/usr/bin/env python3
"""
InfluxDB Feature Vector Loader

Loads historical feature vectors from InfluxDB for backtesting.
Ensures backtest uses EXACT same data model saw during training.

Usage:
    from fincoll.storage.influxdb_loader import InfluxDBFeatureLoader

    loader = InfluxDBFeatureLoader()
    features = loader.get_feature_vector('AAPL', datetime(2024, 1, 15))
    if features is not None:
        # Use stored features (preferred)
        prediction = model.predict(features)
    else:
        # Fallback: reconstruct (not ideal)
        logger.warning("No stored features, reconstructing...")
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging
import os

from config.dimensions import DIMS

logger = logging.getLogger(__name__)


class InfluxDBFeatureLoader:
    """
    Loads historical feature vectors from InfluxDB.

    Supports config-aware queries to handle dimension changes over time.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None,
        config_version: Optional[str] = None,
    ):
        """
        Initialize InfluxDB feature loader.

        Args:
            url: InfluxDB URL (default: from env INFLUX_URL)
            token: Auth token (default: from env INFLUX_TOKEN)
            org: Organization (default: from env INFLUX_ORG)
            bucket: Bucket name (default: from env INFLUX_BUCKET)
            config_version: Specific config version to query (default: current)
        """
        # Load from environment if not provided
        self.url = url or os.getenv('INFLUX_URL', 'http://10.32.3.27:8086')
        self.token = token or os.getenv('INFLUX_TOKEN', 'caelum-influx-token-2026-change-me')
        self.org = org or os.getenv('INFLUX_ORG', 'caelum')
        self.bucket = bucket or os.getenv('INFLUX_BUCKET', 'feature_vectors')

        # Config version to query (default: current)
        if config_version is None:
            from fincoll.storage.config_version import get_config_version
            self.config_version = get_config_version()
        else:
            self.config_version = config_version

        self.feature_dim = DIMS.fincoll_total

        # Initialize InfluxDB client
        self._init_client()

        # Track stats
        self.stats = {
            'queries': 0,
            'hits': 0,
            'misses': 0,
            'errors': 0,
        }

        logger.info(
            f"InfluxDB Feature Loader initialized: "
            f"config_version={self.config_version}, "
            f"feature_dim={self.feature_dim}D, "
            f"bucket={self.bucket}"
        )

    def _init_client(self):
        """Initialize InfluxDB client and query API."""
        try:
            from influxdb_client import InfluxDBClient

            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=30000,  # 30s timeout
            )

            self.query_api = self.client.query_api()

            # Test connection
            self.client.ping()
            logger.info(f"✅ Connected to InfluxDB at {self.url}")

        except ImportError:
            logger.error("influxdb-client not installed. Install with: pip install influxdb-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    def get_feature_vector(
        self,
        symbol: str,
        date: datetime,
        window_hours: int = 24,
    ) -> Optional[np.ndarray]:
        """
        Retrieve feature vector for symbol at specific date.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            date: Target date/time
            window_hours: Search window around target date (default: ±24 hours)

        Returns:
            np.ndarray of shape (DIMS.fincoll_total,) if found, None otherwise
        """
        self.stats['queries'] += 1

        try:
            # Query InfluxDB for feature vector
            start = date - timedelta(hours=window_hours)
            stop = date + timedelta(hours=window_hours)

            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start.isoformat()}Z, stop: {stop.isoformat()}Z)
              |> filter(fn: (r) => r._measurement == "feature_vectors")
              |> filter(fn: (r) => r.symbol == "{symbol}")
              |> filter(fn: (r) => r.config_version == "{self.config_version}")
              |> filter(fn: (r) => r.feature_dim == "{self.feature_dim}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> limit(n: 1)
            '''

            result = self.query_api.query(query)

            if not result or len(result) == 0:
                self.stats['misses'] += 1
                logger.debug(f"No feature vector found for {symbol} on {date}")
                return None

            # Extract features from result
            features = self._parse_influx_result(result)

            if features is not None:
                self.stats['hits'] += 1
                logger.debug(
                    f"✅ Loaded feature vector: {symbol} @ {date} "
                    f"({self.feature_dim}D, v{self.config_version})"
                )

            return features

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to load feature vector for {symbol} on {date}: {e}")
            return None

    def get_feature_vectors_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Dict[datetime, np.ndarray]:
        """
        Retrieve all feature vectors for symbol in date range.
        Used for batch backtest loading.

        Args:
            symbol: Stock ticker
            start: Start date
            end: End date

        Returns:
            Dict mapping datetime -> feature vector (np.ndarray)
        """
        self.stats['queries'] += 1

        try:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
              |> filter(fn: (r) => r._measurement == "feature_vectors")
              |> filter(fn: (r) => r.symbol == "{symbol}")
              |> filter(fn: (r) => r.config_version == "{self.config_version}")
              |> filter(fn: (r) => r.feature_dim == "{self.feature_dim}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            result = self.query_api.query(query)

            vectors = {}
            for table in result:
                for record in table.records:
                    timestamp = record.get_time()
                    features = self._parse_record(record)
                    if features is not None:
                        vectors[timestamp] = features
                        self.stats['hits'] += 1

            if len(vectors) == 0:
                self.stats['misses'] += 1

            logger.info(
                f"✅ Loaded {len(vectors)} feature vectors for {symbol} "
                f"({start.date()} to {end.date()})"
            )

            return vectors

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to load feature vectors for {symbol}: {e}")
            return {}

    def _parse_influx_result(self, result) -> Optional[np.ndarray]:
        """
        Parse InfluxDB query result into numpy array.

        Args:
            result: InfluxDB query result

        Returns:
            np.ndarray of shape (DIMS.fincoll_total,) or None
        """
        if not result or len(result) == 0:
            return None

        # Get first table and first record
        table = result[0]
        if len(table.records) == 0:
            return None

        record = table.records[0]
        return self._parse_record(record)

    def _parse_record(self, record) -> Optional[np.ndarray]:
        """
        Parse single InfluxDB record into numpy array.

        Args:
            record: InfluxDB record

        Returns:
            np.ndarray of shape (DIMS.fincoll_total,) or None
        """
        try:
            # Extract features f0, f1, ..., f{N-1}
            features = np.zeros(self.feature_dim, dtype=np.float32)

            for i in range(self.feature_dim):
                field_name = f"f{i}"
                if field_name in record.values:
                    features[i] = record.values[field_name]
                else:
                    logger.warning(f"Missing field {field_name} in record")
                    return None

            return features

        except Exception as e:
            logger.error(f"Failed to parse record: {e}")
            return None

    def get_stats(self) -> Dict[str, any]:
        """Get loader statistics."""
        hit_rate = (
            self.stats['hits'] / self.stats['queries']
            if self.stats['queries'] > 0
            else 0.0
        )

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'config_version': self.config_version,
            'feature_dim': self.feature_dim,
            'bucket': self.bucket,
        }

    def close(self):
        """Close InfluxDB client."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("InfluxDB client closed")


if __name__ == "__main__":
    # Test feature vector loading
    print("InfluxDB Feature Loader Test")
    print("=" * 60)

    # Create loader
    loader = InfluxDBFeatureLoader()

    # Try to load test vector (from saver test)
    features = loader.get_feature_vector(
        symbol='TEST',
        date=datetime.now(),
        window_hours=1,
    )

    if features is not None:
        print(f"✅ Test vector loaded successfully ({features.shape[0]}D)")
        print(f"   Config version: {loader.config_version}")
        print(f"   First 10 values: {features[:10]}")
        print(f"   Stats: {loader.get_stats()}")
    else:
        print("❌ Test vector not found (run influxdb_saver.py test first)")

    loader.close()
