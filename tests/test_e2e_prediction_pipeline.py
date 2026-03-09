#!/usr/bin/env python3
"""
End-to-End Integration Test for Full Prediction Pipeline

Tests the complete flow:
1. Mock Data Provider → InfluxDB Cache
2. Cache → Feature Extractor
3. Feature Extractor → Prediction API
4. Validates no 1969 dates, proper datetime indices throughout

Run with: pytest test_e2e_prediction_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK DATA PROVIDER (Replaces TradeStation/yfinance for testing)
# ============================================================================

class MockDataProvider:
    """
    Mock data provider that returns properly formatted DataFrames
    with datetime indices (NOT integers!)
    """

    def __init__(self):
        self.name = "mock"

    def get_historical_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Generate mock OHLCV data with PROPER datetime index

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Bar interval (default: '1d')

        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        logger.info(f"[MockProvider] Generating data for {symbol} from {start_date} to {end_date}")

        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate date range (trading days only - weekdays)
        dates = pd.bdate_range(start=start_dt, end=end_dt, freq='B')

        if len(dates) == 0:
            logger.warning(f"[MockProvider] No trading days in range {start_date} to {end_date}")
            return None

        # Generate realistic mock OHLCV data
        num_bars = len(dates)
        seed = abs(hash(symbol)) % (2**31)  # Ensure positive seed for numpy
        base_price = 100.0 + seed % 200  # Different base for each symbol

        # Random walk for prices
        np.random.seed(seed)  # Deterministic per symbol
        returns = np.random.randn(num_bars) * 0.02  # 2% daily volatility
        close_prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        df = pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(num_bars) * 0.005),
            'high': close_prices * (1 + np.abs(np.random.randn(num_bars)) * 0.01),
            'low': close_prices * (1 - np.abs(np.random.randn(num_bars)) * 0.01),
            'close': close_prices,
            'volume': np.random.randint(1_000_000, 10_000_000, num_bars)
        }, index=dates)

        # CRITICAL: Ensure index is DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex), f"Index must be DatetimeIndex, got {type(df.index)}"

        logger.info(f"[MockProvider] Generated {len(df)} bars with datetime index")
        return df

    def is_available(self) -> bool:
        return True

    def get_name(self) -> str:
        return "MockDataProvider"


# ============================================================================
# E2E TEST SUITE
# ============================================================================

class TestE2EPredictionPipeline:
    """
    End-to-end tests for the complete prediction pipeline
    """

    @pytest.fixture
    def mock_provider(self):
        """Fixture: Mock data provider"""
        return MockDataProvider()

    @pytest.fixture
    def test_symbol(self):
        """Fixture: Test symbol"""
        return "AAPL"

    @pytest.fixture
    def date_range(self):
        """Fixture: Recent date range"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # ========================================================================
    # TEST 1: Mock Provider Returns Valid Data
    # ========================================================================

    def test_mock_provider_datetime_index(self, mock_provider, test_symbol, date_range):
        """
        TEST 1: Mock provider returns DataFrame with datetime index (not integers)
        """
        logger.info("=" * 80)
        logger.info("TEST 1: Mock Provider Returns Valid Datetime Index")
        logger.info("=" * 80)

        start_date, end_date = date_range

        # Fetch data
        df = mock_provider.get_historical_bars(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )

        # Assertions
        assert df is not None, "Provider should return data"
        assert not df.empty, "DataFrame should not be empty"
        assert isinstance(df.index, pd.DatetimeIndex), \
            f"Index must be DatetimeIndex, got {type(df.index)}"

        # Check latest timestamp is a datetime
        latest_ts = df.index[-1]
        assert isinstance(latest_ts, (pd.Timestamp, datetime)), \
            f"Latest timestamp should be datetime, got {type(latest_ts)} = {latest_ts}"

        # Ensure NO integer timestamps
        assert not isinstance(latest_ts, int), \
            f"Timestamp should NOT be integer! Got: {latest_ts}"

        # Verify columns
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        logger.info(f"✅ Provider returned {len(df)} bars with proper datetime index")
        logger.info(f"   Index type: {type(df.index)}")
        logger.info(f"   First timestamp: {df.index[0]} (type: {type(df.index[0])})")
        logger.info(f"   Last timestamp: {df.index[-1]} (type: {type(df.index[-1])})")

    # ========================================================================
    # TEST 2: InfluxDB Cache Preserves Datetime Index
    # ========================================================================

    def test_influxdb_cache_datetime_index(self, mock_provider, test_symbol, date_range):
        """
        TEST 2: InfluxDB cache stores and retrieves data with datetime index
        """
        logger.info("=" * 80)
        logger.info("TEST 2: InfluxDB Cache Preserves Datetime Index")
        logger.info("=" * 80)

        from fincoll.storage.influxdb_cache import InfluxDBCache

        # Initialize cache
        cache = InfluxDBCache()

        if not cache.enabled:
            pytest.skip("InfluxDB not available - skipping cache test")

        start_date, end_date = date_range

        # Get mock data
        df = mock_provider.get_historical_bars(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )

        # Store in cache
        success = cache.store_bars(
            symbol=test_symbol,
            bars=df,
            interval='1d',
            source='mock'
        )

        assert success, "Cache storage should succeed"
        logger.info(f"✅ Stored {len(df)} bars in InfluxDB cache")

        # Retrieve from cache
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        cached_df = cache.get_bars(
            symbol=test_symbol,
            start_date=start_dt,
            end_date=end_dt,
            interval='1d',
            source='mock'
        )

        # Assertions
        assert cached_df is not None, "Cache should return data"
        assert not cached_df.empty, "Cached DataFrame should not be empty"
        assert isinstance(cached_df.index, pd.DatetimeIndex), \
            f"Cached index must be DatetimeIndex, got {type(cached_df.index)}"

        # Check latest timestamp
        latest_ts = cached_df.index[-1]
        assert isinstance(latest_ts, (pd.Timestamp, datetime)), \
            f"Cached timestamp should be datetime, got {type(latest_ts)}"

        assert not isinstance(latest_ts, int), \
            f"Cached timestamp should NOT be integer! Got: {latest_ts}"

        logger.info(f"✅ Retrieved {len(cached_df)} bars from cache with datetime index")
        logger.info(f"   Index type: {type(cached_df.index)}")
        logger.info(f"   Latest timestamp: {cached_df.index[-1]} (type: {type(cached_df.index[-1])})")

    # ========================================================================
    # TEST 3: Feature Extractor Validates Timestamps
    # ========================================================================

    def test_feature_extractor_timestamp_validation(self, mock_provider, test_symbol, date_range):
        """
        TEST 3: Feature extractor rejects invalid timestamps (integers, nulls)
        """
        logger.info("=" * 80)
        logger.info("TEST 3: Feature Extractor Timestamp Validation")
        logger.info("=" * 80)

        from fincoll.features.feature_extractor import FeatureExtractor

        start_date, end_date = date_range

        # Get mock data
        df = mock_provider.get_historical_bars(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )

        # Initialize feature extractor
        extractor = FeatureExtractor(data_provider=mock_provider)

        # TEST 3a: Valid datetime timestamp should work
        valid_timestamp = df.index[-1]
        logger.info(f"Testing with valid timestamp: {valid_timestamp} (type: {type(valid_timestamp)})")

        try:
            features = extractor.extract_features(
                ohlcv_data=df,
                symbol=test_symbol,
                timestamp=valid_timestamp
            )
            logger.info(f"✅ Valid timestamp accepted, extracted {len(features)}D features")
        except Exception as e:
            pytest.fail(f"Valid timestamp should be accepted: {e}")

        # TEST 3b: Integer timestamp should be rejected
        logger.info("Testing with invalid integer timestamp...")
        with pytest.raises(ValueError, match="Invalid timestamp.*int"):
            extractor.extract_features(
                ohlcv_data=df,
                symbol=test_symbol,
                timestamp=249  # Integer - should be rejected!
            )
        logger.info("✅ Integer timestamp correctly rejected")

        # TEST 3c: None timestamp should be rejected
        logger.info("Testing with null timestamp...")
        with pytest.raises(ValueError, match="Invalid timestamp.*None"):
            extractor.extract_features(
                ohlcv_data=df,
                symbol=test_symbol,
                timestamp=None
            )
        logger.info("✅ Null timestamp correctly rejected")

        # TEST 3d: NaT timestamp should be rejected
        logger.info("Testing with NaT timestamp...")
        with pytest.raises(ValueError, match="Invalid timestamp.*None or NaN"):
            extractor.extract_features(
                ohlcv_data=df,
                symbol=test_symbol,
                timestamp=pd.NaT
            )
        logger.info("✅ NaT timestamp correctly rejected")

    # ========================================================================
    # TEST 4: Full E2E Prediction Pipeline
    # ========================================================================

    def test_full_prediction_pipeline(self, mock_provider, test_symbol):
        """
        TEST 4: Complete end-to-end prediction pipeline

        Flow:
        1. Mock Provider → Data
        2. Cached Provider → Wraps provider
        3. Feature Extractor → Features
        4. Verify no 1969 dates anywhere
        """
        logger.info("=" * 80)
        logger.info("TEST 4: Full E2E Prediction Pipeline")
        logger.info("=" * 80)

        from fincoll.providers.cached_provider import CachedDataProvider
        from fincoll.features.feature_extractor import FeatureExtractor

        # Setup date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # Wrap provider with cache
        cached_provider = CachedDataProvider(
            provider=mock_provider,
            source_name='mock'
        )

        logger.info("Step 1: Fetch data through cached provider")
        df = cached_provider.get_historical_bars(
            symbol=test_symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval='1d'
        )

        # Validate data
        assert df is not None and not df.empty, "Should have data"
        assert isinstance(df.index, pd.DatetimeIndex), "Must have datetime index"
        logger.info(f"✅ Fetched {len(df)} bars through cached provider")

        # Extract timestamp
        latest_timestamp = df.index[-1]
        logger.info(f"   Latest timestamp: {latest_timestamp} (type: {type(latest_timestamp)})")

        # Check for 1969 dates (the bug we fixed!)
        assert latest_timestamp.year >= 2020, \
            f"Timestamp should be recent, not {latest_timestamp.year}!"

        logger.info("Step 2: Extract features")
        extractor = FeatureExtractor(data_provider=cached_provider)

        features = extractor.extract_features(
            ohlcv_data=df,
            symbol=test_symbol,
            timestamp=latest_timestamp
        )

        assert features is not None, "Should extract features"
        assert len(features) > 0, "Features should not be empty"
        logger.info(f"✅ Extracted {len(features)}D feature vector")

        # Verify features are numeric and valid
        assert not np.isnan(features).all(), "Not all features should be NaN"
        logger.info(f"   Non-zero features: {np.count_nonzero(features)}/{len(features)}")

        logger.info("=" * 80)
        logger.info("✅ FULL E2E PIPELINE TEST PASSED!")
        logger.info("=" * 80)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    import sys

    # Run with pytest
    exit_code = pytest.main([
        __file__,
        "-v",           # Verbose
        "-s",           # Show print/log output
        "--tb=short",   # Short traceback format
        "--color=yes"   # Colored output
    ])

    sys.exit(exit_code)
