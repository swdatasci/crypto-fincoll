#!/usr/bin/env python3
"""
InfluxDB Feature Vector Saver

Saves complete feature vectors to InfluxDB at generation time.
Uses DIMS for dynamic dimension handling - no hardcoded sizes.

CRITICAL: This is the fix for the training/test data mismatch problem.
By storing features at generation time, we preserve:
- SenVec sentiment from that exact moment
- Options flow data
- All temporal features that cannot be reconstructed

Usage:
    from fincoll.storage.influxdb_saver import InfluxDBFeatureSaver

    saver = InfluxDBFeatureSaver()
    saver.save_feature_vector(
        symbol='AAPL',
        timestamp=datetime.now(),
        features=feature_array,  # np.ndarray of shape (DIMS.fincoll_total,)
        source='tradestation'
    )
"""

import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import os

from config.dimensions import DIMS
from fincoll.storage.config_version import get_config_version, get_config_snapshot

logger = logging.getLogger(__name__)


class InfluxDBFeatureSaver:
    """
    Saves feature vectors to InfluxDB with config-aware versioning.

    Features are stored as generic f0, f1, ..., f{N-1} fields where N = DIMS.fincoll_total.
    Config version tags enable backward compatibility when dimensions change.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None,
        batch_size: int = 100,
        flush_interval: int = 1000,  # ms
    ):
        """
        Initialize InfluxDB feature saver.

        Args:
            url: InfluxDB URL (default: from env INFLUX_URL)
            token: Auth token (default: from env INFLUX_TOKEN)
            org: Organization (default: from env INFLUX_ORG)
            bucket: Bucket name (default: from env INFLUX_BUCKET)
            batch_size: Number of points to batch before writing
            flush_interval: Max time (ms) to wait before flushing batch
        """
        # Load from environment if not provided
        # CRITICAL: Use INFLUXDB_* env vars (with DB suffix) to match .env file
        self.url = url or os.getenv('INFLUXDB_URL') or 'http://10.32.3.27:8086'
        self.token = token or os.getenv('INFLUXDB_TOKEN')
        self.org = org or os.getenv('INFLUXDB_ORG') or 'caelum'
        self.bucket = bucket or os.getenv('INFLUXDB_FEATURE_BUCKET') or 'feature_vectors'

        # Validate required token
        if not self.token:
            raise ValueError("INFLUXDB_TOKEN environment variable is required but not set")

        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Get current config version
        self.config_version = get_config_version()
        self.feature_dim = DIMS.fincoll_total

        # Initialize InfluxDB client
        self._init_client()

        # Track stats
        self.stats = {
            'vectors_saved': 0,
            'batches_written': 0,
            'errors': 0,
            'last_write_time': None,
        }

        logger.info(
            f"InfluxDB Feature Saver initialized: "
            f"config_version={self.config_version}, "
            f"feature_dim={self.feature_dim}D, "
            f"bucket={self.bucket}"
        )

    def _init_client(self):
        """Initialize InfluxDB client and write API."""
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS

            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=30000,  # 30s timeout
            )

            # Use synchronous write for reliability (can change to async for performance)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

            # Test connection
            self.client.ping()
            logger.info(f"✅ Connected to InfluxDB at {self.url}")

            # Store config snapshot on first init
            self._store_config_snapshot()

        except ImportError:
            logger.error("influxdb-client not installed. Install with: pip install influxdb-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    def _store_config_snapshot(self):
        """
        Store config snapshot for auditability.

        This allows future retrieval of exact feature meanings for old data.
        """
        try:
            from influxdb_client import Point

            snapshot = get_config_snapshot()

            # Create point with config metadata
            point = Point("config_snapshots") \
                .tag("config_version", snapshot['version']) \
                .field("fincoll_total", snapshot['dimensions']['fincoll_total']) \
                .field("senvec_total", snapshot['dimensions']['senvec_total']) \
                .field("model_input", snapshot['dimensions']['model_input']) \
                .field("model_output", snapshot['dimensions']['model_output']) \
                .field("hash", snapshot['hash']) \
                .time(datetime.now())

            # Write config snapshot (non-critical, don't fail if it errors)
            self.write_api.write(bucket=self.bucket, record=point)
            logger.info(f"✅ Stored config snapshot: {snapshot['version']}")

        except Exception as e:
            logger.warning(f"Failed to store config snapshot (non-critical): {e}")

    def save_feature_vector(
        self,
        symbol: str,
        timestamp: datetime,
        features: np.ndarray,
        source: str = 'tradestation',
        metadata: Optional[Dict[str, Any]] = None,
        velocities: Optional[Dict[str, float]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save complete SymVec triad (input + output + raw data) to InfluxDB.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            timestamp: When features were generated
            features: Feature array of shape (DIMS.fincoll_total,)
            source: Data source (tradestation, yfinance, etc.)
            metadata: Optional metadata to store as fields
            velocities: Optional velocity predictions (velocity_1m, velocity_15m, etc.)
            raw_data: Optional raw market data (price, volume, etc.)

        Returns:
            True if saved successfully, False otherwise

        Raises:
            ValueError: If feature dimensions don't match DIMS.fincoll_total
        """
        # Validate dimensions
        if features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {features.shape[0]}. Config version: {self.config_version}"
            )

        try:
            from influxdb_client import Point

            # Create point with tags
            point = Point("feature_vectors") \
                .tag("symbol", symbol) \
                .tag("source", source) \
                .tag("config_version", self.config_version) \
                .tag("feature_dim", str(self.feature_dim)) \
                .time(timestamp)

            # Add all features as generic fields f0, f1, ..., f{N-1}
            for i in range(self.feature_dim):
                point.field(f"f{i}", float(features[i]))

            # Add velocity predictions (output vector) if provided
            if velocities:
                point.field("velocity_1m", float(velocities.get("1min", 0.0)))
                point.field("velocity_15m", float(velocities.get("15min", 0.0)))
                point.field("velocity_1h", float(velocities.get("1hour", 0.0)))
                point.field("velocity_1d", float(velocities.get("daily", 0.0)))
                point.field("confidence", float(velocities.get("confidence", 0.0)))

            # Add raw market data if provided
            if raw_data:
                for key, value in raw_data.items():
                    if isinstance(value, (int, float)):
                        point.field(f"raw_{key}", float(value))
                    elif isinstance(value, str):
                        point.field(f"raw_{key}", value)

            # Add optional metadata fields
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str, bool)):
                        point.field(f"meta_{key}", value)

            # Write to InfluxDB
            self.write_api.write(bucket=self.bucket, record=point)

            # Update stats
            self.stats['vectors_saved'] += 1
            self.stats['last_write_time'] = datetime.now()

            # Log what was saved
            components = [f"{self.feature_dim}D features"]
            if velocities:
                components.append("velocities")
            if raw_data:
                components.append(f"{len(raw_data)} raw fields")

            logger.debug(
                f"✅ Saved SymVec triad: {symbol} @ {timestamp} "
                f"({', '.join(components)}, v{self.config_version})"
            )

            return True

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to save feature vector for {symbol}: {e}")
            return False

    def save_batch(
        self,
        vectors: list[Dict[str, Any]]
    ) -> int:
        """
        Save multiple feature vectors in batch.

        Args:
            vectors: List of dicts with keys: symbol, timestamp, features, source

        Returns:
            Number of vectors successfully saved
        """
        from influxdb_client import Point

        points = []

        for vec in vectors:
            symbol = vec['symbol']
            timestamp = vec['timestamp']
            features = vec['features']
            source = vec.get('source', 'tradestation')

            # Validate dimensions
            if features.shape[0] != self.feature_dim:
                logger.warning(
                    f"Skipping {symbol}: dimension mismatch "
                    f"(expected {self.feature_dim}, got {features.shape[0]})"
                )
                continue

            # Create point
            point = Point("feature_vectors") \
                .tag("symbol", symbol) \
                .tag("source", source) \
                .tag("config_version", self.config_version) \
                .tag("feature_dim", str(self.feature_dim)) \
                .time(timestamp)

            # Add features
            for i in range(self.feature_dim):
                point.field(f"f{i}", float(features[i]))

            points.append(point)

        # Write batch
        try:
            self.write_api.write(bucket=self.bucket, record=points)
            saved_count = len(points)
            self.stats['vectors_saved'] += saved_count
            self.stats['batches_written'] += 1
            self.stats['last_write_time'] = datetime.now()

            logger.info(f"✅ Saved batch: {saved_count} vectors")
            return saved_count

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to save batch: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get saver statistics."""
        return {
            **self.stats,
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
    # Test feature vector saving
    print("InfluxDB Feature Saver Test")
    print("=" * 60)

    # Create saver
    saver = InfluxDBFeatureSaver()

    # Generate test feature vector
    test_features = np.random.randn(DIMS.fincoll_total)

    # Save test vector
    success = saver.save_feature_vector(
        symbol='TEST',
        timestamp=datetime.now(),
        features=test_features,
        source='test',
        metadata={'test_run': True, 'test_id': 'phase2_test'}
    )

    if success:
        print(f"✅ Test vector saved successfully ({DIMS.fincoll_total}D)")
        print(f"   Config version: {saver.config_version}")
        print(f"   Stats: {saver.get_stats()}")
    else:
        print("❌ Test vector save failed")

    saver.close()
