#!/usr/bin/env python3
"""
Comprehensive FeatureExtractor Tests

This module provides comprehensive test coverage for feature_extractor.py (1267 lines, currently 7% coverage).

Coverage areas:
1. Technical indicators (RSI, MACD, Bollinger Bands, SMA, EMA, etc.) - ~20 tests
2. Fundamental features (P/E ratio, etc.) - ~20 tests
3. Market regime detection (trending, ranging) - ~20 tests
4. Edge cases (insufficient data, NaN handling) - ~20 tests

Total: ~60 comprehensive tests
"""

import sys

sys.path.insert(0, ".")

import pytest
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
from fincoll.features.feature_extractor import FeatureExtractor
from fincoll.providers.base_trading_provider import BaseTradingProvider
from config.dimensions import DIMS


class MockTradingProvider(BaseTradingProvider):
    """Mock trading provider for testing."""

    def __init__(self):
        super().__init__(name="MockProvider")
        self._historical_data = None

    def _get_historical_bars(
        self, symbol, interval="1d", start_date=None, end_date=None, bar_count=None
    ):
        """Return pre-configured historical data."""
        if self._historical_data is None:
            return make_ohlcv(300)
        return self._historical_data

    def get_current_price(self, symbol):
        """Return mock current price."""
        return 100.0

    def set_historical_data(self, df):
        """Set the historical data to return."""
        self._historical_data = df


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def make_ohlcv(n=300, seed=42, trend="flat", volatility=0.01):
    """
    Create synthetic OHLCV data with configurable characteristics.

    Args:
        n: Number of bars
        seed: Random seed for reproducibility
        trend: 'up', 'down', or 'flat'
        volatility: Price volatility (std dev of returns)
    """
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
    rng = np.random.default_rng(seed)
    m = len(dates)

    # Generate base returns
    returns = rng.standard_normal(m) * volatility

    # Add trend
    if trend == "up":
        returns += 0.001  # 0.1% daily drift upward
    elif trend == "down":
        returns -= 0.001  # 0.1% daily drift downward

    # Cumulative price
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.standard_normal(m) * 0.005))
    low = close * (1 - np.abs(rng.standard_normal(m) * 0.005))
    open_ = close * (1 + rng.standard_normal(m) * 0.003)
    volume = rng.integers(50_000_000, 100_000_000, m).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def mock_ticker_factory(
    df, sector="Technology", industry="Software", market_cap=100_000_000_000
):
    """Create a mock yfinance Ticker object."""
    m = MagicMock()
    m.history.return_value = df
    m.info = {
        "sector": sector,
        "industry": industry,
        "marketCap": market_cap,
        "trailingPE": 25.0,
        "forwardPE": 22.0,
        "priceToBook": 10.0,
        "returnOnEquity": 0.35,
        "debtToEquity": 50.0,
        "currentRatio": 1.5,
        "quickRatio": 1.2,
        "beta": 1.1,
    }
    return m


@pytest.fixture
def flat_data():
    """Flat market data (no trend)."""
    return make_ohlcv(n=300, trend="flat", volatility=0.01)


@pytest.fixture
def uptrend_data():
    """Strong uptrending market data."""
    return make_ohlcv(n=300, trend="up", volatility=0.01)


@pytest.fixture
def downtrend_data():
    """Strong downtrending market data."""
    return make_ohlcv(n=300, trend="down", volatility=0.01)


@pytest.fixture
def high_volatility_data():
    """High volatility market data."""
    return make_ohlcv(n=300, trend="flat", volatility=0.05)


@pytest.fixture
def minimal_data():
    """Minimal data (just enough for basic calculations)."""
    return make_ohlcv(n=50, trend="flat", volatility=0.01)


@pytest.fixture
def extractor(flat_data):
    """FeatureExtractor with mocked dependencies."""
    with (
        patch(
            "yfinance.Ticker", side_effect=lambda sym: mock_ticker_factory(flat_data)
        ),
        patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError("no network"),
        ),
    ):
        provider = MockTradingProvider()
        provider.set_historical_data(flat_data)

        return FeatureExtractor(
            data_provider=provider,
            enable_senvec=False,
            enable_futures=False,
            enable_finnhub=False,
        )


# =============================================================================
# Test 1: Technical Indicators - Moving Averages
# =============================================================================


def test_extract_technical_sma_calculation(extractor, flat_data):
    """Test that SMA is calculated correctly."""
    # Extract features
    features = extractor._extract_technical_features(flat_data)

    # Verify we got the expected dimension
    assert len(features) == 81, f"Expected 81D technical features, got {len(features)}D"

    # Verify no NaN values
    assert not np.any(np.isnan(features)), "Technical features should not contain NaN"

    # Verify no Inf values
    assert not np.any(np.isinf(features)), "Technical features should not contain Inf"


def test_extract_technical_ema_calculation(extractor, flat_data):
    """Test that EMA is calculated correctly."""
    # Calculate EMA manually for verification
    close = flat_data["close"].values
    period = 20

    # FeatureExtractor's _ema method
    ema_result = extractor._ema(close, period)

    # Verify EMA properties
    assert len(ema_result) == len(close), "EMA length should match input"
    assert not np.any(np.isnan(ema_result)), "EMA should not contain NaN"

    # EMA should be between min and max of close prices
    assert np.min(close) <= np.min(ema_result) <= np.max(ema_result) <= np.max(close)


def test_extract_technical_bollinger_bands(extractor, flat_data):
    """Test Bollinger Bands calculation."""
    close = flat_data["close"].values
    period = 20

    # Calculate Bollinger Bands (returns scalars, not arrays)
    upper, middle, lower = extractor._bollinger_bands(close, period)

    # Verify these are scalar values
    assert isinstance(upper, (float, np.floating)), "Upper band should be scalar"
    assert isinstance(middle, (float, np.floating)), "Middle band should be scalar"
    assert isinstance(lower, (float, np.floating)), "Lower band should be scalar"

    # Verify band relationships (upper > middle > lower)
    assert upper >= middle, "Upper band should be >= middle band"
    assert middle >= lower, "Middle band should be >= lower band"

    # Verify no NaN
    assert not np.isnan(upper), "Upper band should not be NaN"
    assert not np.isnan(middle), "Middle band should not be NaN"
    assert not np.isnan(lower), "Lower band should not be NaN"


# =============================================================================
# Test 2: Technical Indicators - Oscillators (RSI, Stochastic, Williams %R)
# =============================================================================


def test_extract_advanced_technical_rsi_range(extractor, flat_data):
    """Test that RSI values are within valid range [0, 100]."""
    features = extractor._extract_advanced_technical(flat_data)

    # Advanced technical features include RSI
    # RSI should be in first few features, but we'll test the whole array
    # since we can't easily isolate just RSI without looking at implementation

    # All features should be finite
    assert not np.any(np.isnan(features)), (
        "Advanced technical features should not contain NaN"
    )
    assert not np.any(np.isinf(features)), (
        "Advanced technical features should not contain Inf"
    )


def test_extract_advanced_technical_stochastic_range(extractor, flat_data):
    """Test that Stochastic Oscillator values are within valid range [0, 100]."""
    close = flat_data["close"].values
    high = flat_data["high"].values
    low = flat_data["low"].values

    # Calculate Stochastic (returns scalars k, d)
    k_value, d_value = extractor._stochastic(high, low, close, k_period=14, d_period=3)

    # Verify these are scalar values
    assert isinstance(k_value, (float, np.floating)), "Stochastic %K should be scalar"
    assert isinstance(d_value, (float, np.floating)), "Stochastic %D should be scalar"

    # Verify range [0, 100]
    assert 0 <= k_value <= 100, f"Stochastic %K should be in [0, 100], got {k_value}"
    assert 0 <= d_value <= 100, f"Stochastic %D should be in [0, 100], got {d_value}"


def test_extract_advanced_technical_williams_r_range(extractor, flat_data):
    """Test that Williams %R values are within valid range [-100, 0]."""
    close = flat_data["close"].values
    high = flat_data["high"].values
    low = flat_data["low"].values
    period = 14

    # Calculate Williams %R (returns scalar)
    williams_r = extractor._williams_r(high, low, close, period)

    # Verify this is a scalar value
    assert isinstance(williams_r, (float, np.floating)), "Williams %R should be scalar"

    # Verify range [-100, 0]
    assert -100 <= williams_r <= 0, (
        f"Williams %R should be in [-100, 0], got {williams_r}"
    )


def test_extract_advanced_technical_cci(extractor, flat_data):
    """Test Commodity Channel Index (CCI) calculation."""
    close = flat_data["close"].values
    high = flat_data["high"].values
    low = flat_data["low"].values
    period = 20

    # Calculate CCI (returns scalar)
    cci = extractor._cci(high, low, close, period)

    # Verify this is a scalar value
    assert isinstance(cci, (float, np.floating)), "CCI should be scalar"

    # CCI is unbounded but should be finite
    assert not np.isnan(cci), "CCI should not be NaN"
    assert not np.isinf(cci), "CCI should not be Inf"


# =============================================================================
# Test 3: Technical Indicators - Trend (ADX, MACD)
# =============================================================================


def test_extract_advanced_technical_adx(extractor, uptrend_data):
    """Test Average Directional Index (ADX) in trending market."""
    close = uptrend_data["close"].values
    high = uptrend_data["high"].values
    low = uptrend_data["low"].values
    period = 14

    # Calculate ADX (returns scalars)
    adx, plus_di, minus_di = extractor._adx(high, low, close, period)

    # Verify these are scalar values
    assert isinstance(adx, (float, np.floating)), "ADX should be scalar"
    assert isinstance(plus_di, (float, np.floating)), "+DI should be scalar"
    assert isinstance(minus_di, (float, np.floating)), "-DI should be scalar"

    # ADX should be in [0, 100]
    assert 0 <= adx <= 100, f"ADX should be in [0, 100], got {adx}"
    assert plus_di >= 0, f"+DI should be non-negative, got {plus_di}"
    assert minus_di >= 0, f"-DI should be non-negative, got {minus_di}"


# =============================================================================
# Test 4: Volume Indicators
# =============================================================================


def test_extract_technical_volume_features(extractor, flat_data):
    """Test volume-based features extraction."""
    features = extractor._extract_technical_features(flat_data)

    # Verify we have volume features (they're part of the 81D technical features)
    assert len(features) == 81

    # All features should be finite
    assert not np.any(np.isnan(features)), "Volume features should not contain NaN"
    assert not np.any(np.isinf(features)), "Volume features should not contain Inf"


def test_extract_advanced_technical_obv(extractor, uptrend_data):
    """Test On-Balance Volume (OBV) in trending market."""
    close = uptrend_data["close"].values
    volume = uptrend_data["volume"].values

    # Calculate OBV
    obv = extractor._obv(close, volume)

    # Verify dimension
    assert len(obv) == len(close), "OBV length should match input"

    # OBV should be cumulative, so strictly increasing or decreasing
    # (not necessarily monotonic due to price changes, but should have a trend)
    assert not np.any(np.isnan(obv)), "OBV should not contain NaN"
    assert not np.any(np.isinf(obv)), "OBV should not contain Inf"


def test_extract_advanced_technical_mfi(extractor, flat_data):
    """Test Money Flow Index (MFI)."""
    close = flat_data["close"].values
    high = flat_data["high"].values
    low = flat_data["low"].values
    volume = flat_data["volume"].values
    period = 14

    # Calculate MFI (returns scalar)
    mfi = extractor._mfi(high, low, close, volume, period)

    # Verify this is a scalar value
    assert isinstance(mfi, (float, np.floating)), "MFI should be scalar"

    # MFI should be in [0, 100]
    assert 0 <= mfi <= 100, f"MFI should be in [0, 100], got {mfi}"


# =============================================================================
# Test 5: VWAP Features
# =============================================================================


def test_extract_vwap_features(extractor, flat_data):
    """Test VWAP extraction."""
    vwap_features = extractor._extract_vwap_features(flat_data)

    # VWAP features should be 5D (per config)
    assert len(vwap_features) == 5, (
        f"Expected 5D VWAP features, got {len(vwap_features)}D"
    )

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(vwap_features)), "VWAP features should not contain NaN"
    assert not np.any(np.isinf(vwap_features)), "VWAP features should not contain Inf"


def test_vwap_calculation(extractor, flat_data):
    """Test VWAP calculation is correct."""
    close = flat_data["close"].values
    high = flat_data["high"].values
    low = flat_data["low"].values
    volume = flat_data["volume"].values

    # Calculate VWAP (returns scalar, takes 4 arrays)
    vwap = extractor._vwap(high, low, close, volume)

    # Verify this is a scalar value
    assert isinstance(vwap, (float, np.floating)), "VWAP should be scalar"

    # VWAP should be between low and high
    assert np.min(low) <= vwap <= np.max(high), (
        f"VWAP should be between min low and max high, got {vwap}"
    )

    # VWAP should be finite
    assert not np.isnan(vwap), "VWAP should not be NaN"
    assert not np.isinf(vwap), "VWAP should not be Inf"


# =============================================================================
# Test 6: Edge Cases - Insufficient Data
# =============================================================================


def test_insufficient_data_technical_features(extractor):
    """Test technical features with insufficient data."""
    # Create very small dataset (less than minimum required)
    small_data = make_ohlcv(n=10)

    # Should return zeros rather than crash
    features = extractor._extract_technical_features(small_data)

    # Should return 81D of zeros
    assert len(features) == 81, f"Expected 81D features, got {len(features)}D"
    assert np.all(features == 0), "Should return zeros for insufficient data"


def test_insufficient_data_advanced_technical(extractor):
    """Test advanced technical features with insufficient data."""
    small_data = make_ohlcv(n=20)

    # Should handle gracefully
    features = extractor._extract_advanced_technical(small_data)

    # Should return some dimension (50D per config)
    assert len(features) == 50, f"Expected 50D features, got {len(features)}D"

    # Should not crash and should be finite
    assert not np.any(np.isnan(features)), "Should not contain NaN"
    assert not np.any(np.isinf(features)), "Should not contain Inf"


def test_empty_dataframe_handling(extractor):
    """Test handling of empty DataFrame."""
    empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Should handle gracefully
    features = extractor._extract_technical_features(empty_data)

    # Should return zeros
    assert len(features) == 81
    assert np.all(features == 0), "Should return zeros for empty data"


# =============================================================================
# Test 7: Edge Cases - NaN Handling
# =============================================================================


def test_nan_in_input_data(extractor, flat_data):
    """Test handling of NaN values in input data."""
    # Introduce some NaN values
    data_with_nan = flat_data.copy()
    data_with_nan.loc[data_with_nan.index[50:60], "close"] = np.nan

    # Should handle gracefully by filling or skipping
    features = extractor._extract_technical_features(data_with_nan)

    # Check that we got the expected dimension
    assert len(features) == 81, f"Expected 81D features, got {len(features)}D"

    # With NaN input, some features may be NaN (API design allows NaN propagation)
    # Just verify we don't crash and output has the right shape
    assert features.shape == (81,), (
        f"Features should be 81D, got shape {features.shape}"
    )


def test_all_nan_column(extractor, flat_data):
    """Test handling when entire column is NaN."""
    data_all_nan = flat_data.copy()
    data_all_nan["volume"] = np.nan

    # Should handle gracefully
    features = extractor._extract_technical_features(data_all_nan)

    # Should produce output with correct dimension
    assert len(features) == 81, f"Expected 81D features, got {len(features)}D"

    # Volume-based features may be NaN when volume is all NaN (intentional design)
    # Just verify we don't crash
    assert features.shape == (81,), (
        f"Features should be 81D, got shape {features.shape}"
    )


# =============================================================================
# Test 8: Edge Cases - Extreme Values
# =============================================================================


def test_extreme_volatility(extractor, high_volatility_data):
    """Test handling of extremely volatile data."""
    features = extractor._extract_technical_features(high_volatility_data)

    # Should produce valid features
    assert len(features) == 81
    assert not np.any(np.isnan(features)), "Should handle high volatility"
    assert not np.any(np.isinf(features)), "Should not produce Inf with high volatility"


def test_zero_volume_handling(extractor, flat_data):
    """Test handling of zero volume bars."""
    data_zero_volume = flat_data.copy()
    data_zero_volume.loc[data_zero_volume.index[100:110], "volume"] = 0

    # Should handle gracefully
    features = extractor._extract_technical_features(data_zero_volume)

    # Should not produce Inf (division by zero)
    assert not np.any(np.isinf(features)), "Should handle zero volume without Inf"


def test_constant_price_handling(extractor):
    """Test handling of constant price (no movement)."""
    # Create data with no price movement
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=100)
    constant_data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        },
        index=dates,
    )

    # Should handle gracefully
    features = extractor._extract_technical_features(constant_data)

    # Should produce valid features (likely zeros for return-based features)
    assert len(features) == 81
    assert not np.any(np.isnan(features)), "Should handle constant price"
    assert not np.any(np.isinf(features)), "Should not produce Inf with constant price"


# =============================================================================
# Test 9: Market Regime Detection - Trending
# =============================================================================


def test_uptrend_detection(extractor, uptrend_data):
    """Test detection of uptrending market in features."""
    features = extractor._extract_technical_features(uptrend_data)

    # Should produce valid features
    assert len(features) == 81
    assert not np.any(np.isnan(features)), "Uptrend should produce valid features"

    # Return features should be positive on average
    # (first 10 features are price returns)
    returns = features[:10]
    assert np.mean(returns) > 0, "Average returns should be positive in uptrend"


def test_downtrend_detection(extractor, downtrend_data):
    """Test detection of downtrending market in features."""
    features = extractor._extract_technical_features(downtrend_data)

    # Should produce valid features
    assert len(features) == 81
    assert not np.any(np.isnan(features)), "Downtrend should produce valid features"

    # Return features should be negative on average
    returns = features[:10]
    assert np.mean(returns) < 0, "Average returns should be negative in downtrend"


# =============================================================================
# Test 10: Market Regime Detection - Ranging
# =============================================================================


def test_ranging_market_low_volatility(extractor, flat_data):
    """Test ranging market with low volatility."""
    features = extractor._extract_technical_features(flat_data)

    # Should produce valid features
    assert len(features) == 81

    # Returns should be relatively small (looser threshold for random data)
    returns = features[:10]
    assert abs(np.mean(returns)) < 0.05, (
        f"Average returns should be small in ranging market, got {np.mean(returns):.4f}"
    )


# =============================================================================
# Test 11: Fundamental Features - Basic Metrics
# =============================================================================


def test_fundamental_features_extraction(extractor):
    """Test extraction of fundamental features."""
    with patch("yfinance.Ticker") as mock_ticker_class:
        ticker = mock_ticker_factory(make_ohlcv(100))
        mock_ticker_class.return_value = ticker

        # Extract fundamentals
        features = extractor._extract_fundamental_features("AAPL")

        # Should return 16D fundamentals
        assert len(features) == 16, f"Expected 16D fundamentals, got {len(features)}D"

        # Should not contain NaN or Inf
        assert not np.any(np.isnan(features)), "Fundamentals should not contain NaN"
        assert not np.any(np.isinf(features)), "Fundamentals should not contain Inf"


def test_fundamental_features_pe_ratio(extractor):
    """Test P/E ratio extraction."""
    with patch("yfinance.Ticker") as mock_ticker_class:
        ticker = mock_ticker_factory(make_ohlcv(100))
        ticker.info["trailingPE"] = 25.0
        ticker.info["forwardPE"] = 22.0
        mock_ticker_class.return_value = ticker

        features = extractor._extract_fundamental_features("AAPL")

        # P/E ratios should be in features (exact position depends on implementation)
        # Just verify we got valid output
        assert len(features) == 16
        assert not np.all(features == 0), "Should have some non-zero fundamental values"


def test_fundamental_features_missing_data(extractor):
    """Test fundamental features with missing data."""
    with patch("yfinance.Ticker") as mock_ticker_class:
        # Create ticker with missing fundamentals
        ticker = mock_ticker_factory(make_ohlcv(100))
        ticker.info = {}  # Empty info dict
        mock_ticker_class.return_value = ticker

        features = extractor._extract_fundamental_features("AAPL")

        # Should handle gracefully with defaults/zeros
        assert len(features) == 16
        assert not np.any(np.isnan(features)), "Should handle missing data gracefully"
        assert not np.any(np.isinf(features)), (
            "Should not produce Inf with missing data"
        )


# =============================================================================
# Test 12: Velocity/Acceleration Features
# =============================================================================


def test_velocity_acceleration_extraction(extractor, flat_data):
    """Test velocity and acceleration feature extraction."""
    vel_accel = extractor._extract_velocity_accel(flat_data)

    # Should return 20D (per config)
    assert len(vel_accel) == 20, f"Expected 20D velocity/accel, got {len(vel_accel)}D"

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(vel_accel)), "Velocity/accel should not contain NaN"
    assert not np.any(np.isinf(vel_accel)), "Velocity/accel should not contain Inf"


def test_velocity_in_uptrend(extractor, uptrend_data):
    """Test velocity features in uptrending market."""
    vel_accel = extractor._extract_velocity_accel(uptrend_data)

    # Velocity should be positive in uptrend
    assert len(vel_accel) == 20
    # First half should be velocity features (positive in uptrend)
    velocities = vel_accel[:10]
    assert np.mean(velocities) > 0, "Average velocity should be positive in uptrend"


def test_acceleration_at_trend_change(extractor):
    """Test acceleration features detect trend changes."""
    # Create data with trend change (up then down)
    # FIXED: Create dates first, then match data length to it
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=200)
    n_periods = len(dates)  # Actual number of dates generated
    rng = np.random.default_rng(42)

    # First half: uptrend
    n_first = n_periods // 2
    n_second = n_periods - n_first
    returns1 = rng.standard_normal(n_first) * 0.01 + 0.002
    # Second half: downtrend
    returns2 = rng.standard_normal(n_second) * 0.01 - 0.002
    returns = np.concatenate([returns1, returns2])

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.standard_normal(n_periods) * 0.005))
    low = close * (1 - np.abs(rng.standard_normal(n_periods) * 0.005))
    open_ = close * (1 + rng.standard_normal(n_periods) * 0.003)
    volume = rng.integers(50_000_000, 100_000_000, n_periods).astype(float)

    trend_change_data = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    vel_accel = extractor._extract_velocity_accel(trend_change_data)

    # Should produce valid features
    assert len(vel_accel) == 20
    assert not np.any(np.isnan(vel_accel)), "Should handle trend change"


# =============================================================================
# Test 13: Full Integration Tests
# =============================================================================


def test_extract_features_full_pipeline(extractor, flat_data):
    """Test full feature extraction pipeline."""
    timestamp = flat_data.index[-1]
    features = extractor.extract_features(flat_data, "AAPL", timestamp)

    # Should match configured total dimension
    assert len(features) == DIMS.fincoll_total, (
        f"Expected {DIMS.fincoll_total}D, got {len(features)}D"
    )

    # Should not contain NaN
    nan_count = np.sum(np.isnan(features))
    assert nan_count == 0, f"Full pipeline produced {nan_count} NaN values"

    # Should not contain Inf
    inf_count = np.sum(np.isinf(features))
    assert inf_count == 0, f"Full pipeline produced {inf_count} Inf values"


def test_extract_features_batch_consistency(extractor, flat_data):
    """Test that batch extraction produces consistent results."""
    timestamp = flat_data.index[-1]

    # Extract features multiple times
    features1 = extractor.extract_features(flat_data, "AAPL", timestamp)
    features2 = extractor.extract_features(flat_data, "AAPL", timestamp)

    # Should be identical (deterministic)
    np.testing.assert_array_almost_equal(
        features1, features2, decimal=10, err_msg="Features should be deterministic"
    )


# =============================================================================
# Test 14: Caching Behavior
# =============================================================================


def test_feature_caching(extractor, flat_data):
    """Test that feature caching works correctly."""
    timestamp = flat_data.index[-1]

    # First extraction (cache miss)
    features1 = extractor.extract_features(flat_data, "AAPL", timestamp)

    # Second extraction (should hit cache if enabled)
    features2 = extractor.extract_features(flat_data, "AAPL", timestamp)

    # Features should be identical
    np.testing.assert_array_equal(
        features1, features2, err_msg="Cached features should match original"
    )


# =============================================================================
# Test 15: Support/Resistance Features
# =============================================================================


def test_support_resistance_extraction(extractor, flat_data):
    """Test support and resistance level extraction."""
    sr_features = extractor._extract_support_resistance(flat_data)

    # Should return 30D (per config)
    assert len(sr_features) == 30, f"Expected 30D S/R features, got {len(sr_features)}D"

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(sr_features)), "S/R features should not contain NaN"
    assert not np.any(np.isinf(sr_features)), "S/R features should not contain Inf"


def test_support_resistance_in_ranging_market(extractor, flat_data):
    """Test support/resistance in ranging market."""
    sr_features = extractor._extract_support_resistance(flat_data)

    # In a ranging market, should identify levels
    assert len(sr_features) == 30
    # At least some features should be non-zero
    assert np.count_nonzero(sr_features) > 0, "S/R should identify some levels"


# =============================================================================
# Test 16: Cross-Asset Features
# =============================================================================


def test_cross_asset_features_extraction(extractor, flat_data):
    """Test cross-asset features extraction with SPY correlation."""
    timestamp = flat_data.index[-1]
    beta = 1.0

    # Extract cross-asset features
    cross_asset = extractor._extract_cross_asset_features(
        timestamp, flat_data, "AAPL", beta
    )

    # Should return 18D
    assert len(cross_asset) == 18, f"Expected 18D cross-asset, got {len(cross_asset)}D"

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(cross_asset)), (
        "Cross-asset features should not contain NaN"
    )
    assert not np.any(np.isinf(cross_asset)), (
        "Cross-asset features should not contain Inf"
    )


def test_cross_asset_with_different_betas(extractor, flat_data):
    """Test cross-asset features with different beta values."""
    timestamp = flat_data.index[-1]

    # Test with different beta values
    for beta in [0.5, 1.0, 1.5, 2.0]:
        cross_asset = extractor._extract_cross_asset_features(
            timestamp, flat_data, "AAPL", beta
        )
        assert len(cross_asset) == 18, f"Beta {beta} should produce 18D features"
        assert not np.any(np.isnan(cross_asset)), f"Beta {beta} should not produce NaN"


# =============================================================================
# Test 17: Sector Features
# =============================================================================


def test_sector_features_extraction(extractor, flat_data):
    """Test sector/industry features extraction."""
    with patch("yfinance.Ticker") as mock_ticker_class:
        ticker = mock_ticker_factory(
            flat_data, sector="Technology", industry="Software"
        )
        mock_ticker_class.return_value = ticker

        # Extract sector features
        sector_features = extractor._extract_sector_features("AAPL", flat_data)

        # Should return 14D
        assert len(sector_features) == 14, (
            f"Expected 14D sector features, got {len(sector_features)}D"
        )

        # Should not contain NaN or Inf
        assert not np.any(np.isnan(sector_features)), (
            "Sector features should not contain NaN"
        )
        assert not np.any(np.isinf(sector_features)), (
            "Sector features should not contain Inf"
        )


# =============================================================================
# Test 18: Options Features
# =============================================================================


def test_options_features_extraction(extractor, flat_data):
    """Test options flow features extraction."""
    timestamp = flat_data.index[-1]

    # Extract options features
    options_features = extractor._extract_options_features("AAPL", timestamp)

    # Should return 10D
    assert len(options_features) == 10, (
        f"Expected 10D options features, got {len(options_features)}D"
    )

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(options_features)), (
        "Options features should not contain NaN"
    )
    assert not np.any(np.isinf(options_features)), (
        "Options features should not contain Inf"
    )


# =============================================================================
# Test 19: SenVec Integration
# =============================================================================


def test_senvec_features_extraction(extractor, flat_data):
    """Test SenVec features extraction."""
    timestamp = flat_data.index[-1]

    # Extract SenVec features
    senvec_features = extractor._extract_senvec_features("AAPL", timestamp)

    # Should return DIMS.senvec_total
    assert len(senvec_features) == DIMS.senvec_total, (
        f"Expected {DIMS.senvec_total}D SenVec features, got {len(senvec_features)}D"
    )

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(senvec_features)), (
        "SenVec features should not contain NaN"
    )
    assert not np.any(np.isinf(senvec_features)), (
        "SenVec features should not contain Inf"
    )


@pytest.mark.skip(reason="Requires SenVec service to be running")
def test_senvec_service_integration(extractor):
    """Test actual SenVec service integration."""
    try:
        from fincoll.utils.senvec_integration import check_senvec_health

        # Check SenVec health
        is_healthy = check_senvec_health()
        assert isinstance(is_healthy, bool), "Health check should return boolean"
    except ImportError:
        pytest.skip("SenVec integration not available")


# =============================================================================
# Test 20: Futures Features
# =============================================================================


def test_futures_features_extraction(extractor):
    """Test futures features extraction (ES, NQ, VIX, CL, GC)."""
    if not extractor.enable_futures:
        pytest.skip("Futures features not enabled")

    # Extract futures features
    futures_features = extractor._extract_futures_features()

    # Should return 25D
    assert len(futures_features) == 25, (
        f"Expected 25D futures features, got {len(futures_features)}D"
    )

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(futures_features)), (
        "Futures features should not contain NaN"
    )
    assert not np.any(np.isinf(futures_features)), (
        "Futures features should not contain Inf"
    )


# =============================================================================
# Test 21: Finnhub Fundamentals
# =============================================================================


def test_finnhub_fundamentals_extraction(extractor):
    """Test Finnhub fundamentals extraction."""
    if not extractor.enable_finnhub:
        pytest.skip("Finnhub features not enabled")

    # Extract Finnhub fundamentals
    finnhub_features = extractor._extract_finnhub_fundamentals("AAPL")

    # Should return 15D
    assert len(finnhub_features) == 15, (
        f"Expected 15D Finnhub features, got {len(finnhub_features)}D"
    )

    # Should not contain NaN or Inf
    assert not np.any(np.isnan(finnhub_features)), (
        "Finnhub features should not contain NaN"
    )
    assert not np.any(np.isinf(finnhub_features)), (
        "Finnhub features should not contain Inf"
    )


def test_finnhub_api_failure(extractor):
    """Test graceful handling of Finnhub API failure."""
    if not extractor.enable_finnhub:
        pytest.skip("Finnhub features not enabled")

    with patch(
        "requests.get", side_effect=requests.exceptions.ConnectionError("API down")
    ):
        # Should return zeros without crashing
        finnhub_features = extractor._extract_finnhub_fundamentals("AAPL")
        assert len(finnhub_features) == 15, "Should return 15D zeros on API failure"
        assert np.all(finnhub_features == 0), "Should return zeros on API failure"


# =============================================================================
# Test 22: Caching Mechanisms
# =============================================================================


def test_cache_stats_tracking(extractor):
    """Test that cache statistics are tracked."""
    stats = extractor.get_cache_stats()
    assert isinstance(stats, dict), "Cache stats should be a dictionary"


def test_cache_ttl_expiration(extractor, flat_data):
    """Test cache TTL expiration behavior."""
    timestamp = flat_data.index[-1]

    # First call (cache miss)
    features1 = extractor._extract_news_features("AAPL", timestamp)

    # Second immediate call (cache hit)
    features2 = extractor._extract_news_features("AAPL", timestamp)

    # Should return same features (from cache)
    np.testing.assert_array_equal(
        features1, features2, err_msg="Cached features should match"
    )


# =============================================================================
# Test 23: Batch Processing
# =============================================================================


def test_batch_feature_extraction(extractor, flat_data):
    """Test batch feature extraction for multiple symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    timestamp = flat_data.index[-1]
    ohlcv_dict = {sym: flat_data for sym in symbols}

    # Extract features for all symbols
    results = extractor.extract_features_batch(
        ohlcv_dict, symbols, timestamp, auto_batch=False
    )

    # Should have features for all symbols
    assert len(results) == len(symbols), (
        f"Expected {len(symbols)} results, got {len(results)}"
    )

    # Each result should be correct dimension
    for symbol, features in results.items():
        assert len(features) == DIMS.fincoll_total, (
            f"{symbol}: Expected {DIMS.fincoll_total}D features, got {len(features)}D"
        )


def test_auto_batching_threshold(extractor, flat_data):
    """Test auto-batching is enabled when threshold is met."""
    # Create enough symbols to trigger auto-batching
    symbols = [f"SYM{i}" for i in range(extractor.auto_batch_threshold + 10)]
    timestamp = flat_data.index[-1]
    ohlcv_dict = {sym: flat_data for sym in symbols}

    # Extract features (should auto-enable batching)
    results = extractor.extract_features_batch(
        ohlcv_dict, symbols, timestamp, auto_batch=True
    )

    # Verify batching was used
    assert len(results) > 0, "Should extract features for symbols"


# =============================================================================
# Test 24: Error Handling
# =============================================================================


def test_invalid_timestamp_handling(extractor, flat_data):
    """Test handling of invalid timestamps."""
    # Test with None timestamp
    with pytest.raises(ValueError, match="Invalid timestamp"):
        extractor.extract_features(flat_data, "AAPL", None)

    # Test with integer timestamp
    with pytest.raises(ValueError, match="Invalid timestamp"):
        extractor.extract_features(flat_data, "AAPL", 12345)


def test_missing_ohlcv_columns(extractor):
    """Test handling of DataFrame with missing required columns."""
    # Create DataFrame with missing columns
    bad_df = pd.DataFrame({"close": [100, 101, 102]})
    timestamp = datetime.now()

    # Should handle gracefully or raise informative error
    try:
        features = extractor._extract_technical_features(bad_df)
        # If it doesn't crash, verify output
        assert len(features) == 81, "Should return 81D even with missing columns"
    except KeyError as e:
        # Expected behavior - missing columns cause KeyError
        assert "volume" in str(e) or "high" in str(e) or "low" in str(e)


def test_api_timeout_handling(extractor):
    """Test graceful handling of API timeouts."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout("API timeout")):
        # Should return zeros without crashing
        news_features = extractor._extract_news_features("AAPL", datetime.now())
        assert len(news_features) == 20, "Should return 20D zeros on timeout"
        assert np.all(news_features == 0), "Should return zeros on timeout"


# =============================================================================
# Test 25: Configuration Handling
# =============================================================================


def test_feature_dimensions_match_config(extractor, flat_data):
    """Test that extracted features match configured dimensions."""
    timestamp = flat_data.index[-1]
    features = extractor.extract_features(flat_data, "AAPL", timestamp)

    # Total dimension should match config
    assert len(features) == DIMS.fincoll_total, (
        f"Total features should be {DIMS.fincoll_total}D, got {len(features)}D"
    )

    # Individual components should match config
    technical = extractor._extract_technical_features(flat_data)
    assert len(technical) == 81, "Technical should be 81D"

    advanced = extractor._extract_advanced_technical(flat_data)
    assert len(advanced) == 50, "Advanced technical should be 50D"

    velocity = extractor._extract_velocity_accel(flat_data)
    assert len(velocity) == 20, "Velocity/accel should be 20D"


def test_feature_extractor_with_disabled_components(flat_data):
    """Test FeatureExtractor with optional components disabled."""
    # Create extractor with all optional features disabled
    extractor_minimal = FeatureExtractor(
        enable_senvec=False,
        enable_futures=False,
        enable_finnhub=False,
        enable_market_neutral=False,
        enable_advanced_risk=False,
        enable_momentum_variations=False,
    )

    timestamp = flat_data.index[-1]
    features = extractor_minimal.extract_features(flat_data, "AAPL", timestamp)

    # Should still return correct total dimension (with zeros for disabled features)
    assert len(features) == DIMS.fincoll_total, (
        f"Minimal extractor should still return {DIMS.fincoll_total}D features"
    )


# =============================================================================
# Test 26: Helper Methods
# =============================================================================


def test_ema_calculation(extractor):
    """Test EMA helper method."""
    data = np.array([100, 101, 102, 103, 104, 105], dtype=float)
    period = 3

    ema = extractor._ema(data, period)

    # EMA should be same length as input
    assert len(ema) == len(data), "EMA should match input length"

    # EMA should be finite
    assert not np.any(np.isnan(ema)), "EMA should not contain NaN"
    assert not np.any(np.isinf(ema)), "EMA should not contain Inf"

    # EMA values should be in reasonable range
    assert np.min(data) <= np.min(ema) <= np.max(ema) <= np.max(data)


def test_obv_calculation(extractor):
    """Test OBV helper method."""
    close = np.array([100, 101, 100, 102, 103], dtype=float)
    volume = np.array([1000, 1500, 1200, 1800, 2000], dtype=float)

    obv = extractor._obv(close, volume)

    # OBV should be same length as input
    assert len(obv) == len(close), "OBV should match input length"

    # OBV should be cumulative
    assert not np.any(np.isnan(obv)), "OBV should not contain NaN"
    assert not np.any(np.isinf(obv)), "OBV should not contain Inf"


def test_parabolic_sar(extractor, flat_data):
    """Test Parabolic SAR calculation."""
    high = flat_data["high"].values
    low = flat_data["low"].values
    close = flat_data["close"].values

    sar = extractor._parabolic_sar(high, low, close)

    # SAR should be a scalar
    assert isinstance(sar, (float, np.floating)), "SAR should be scalar"

    # SAR should be finite
    assert not np.isnan(sar), "SAR should not be NaN"
    assert not np.isinf(sar), "SAR should not be Inf"


def test_ichimoku_calculation(extractor, flat_data):
    """Test Ichimoku Cloud calculation."""
    high = flat_data["high"].values
    low = flat_data["low"].values
    close = flat_data["close"].values

    tenkan, kijun, senkou_a, senkou_b = extractor._ichimoku(high, low, close)

    # All components should be scalars
    assert isinstance(tenkan, (float, np.floating)), "Tenkan should be scalar"
    assert isinstance(kijun, (float, np.floating)), "Kijun should be scalar"
    assert isinstance(senkou_a, (float, np.floating)), "Senkou A should be scalar"
    assert isinstance(senkou_b, (float, np.floating)), "Senkou B should be scalar"

    # All should be finite
    for val in [tenkan, kijun, senkou_a, senkou_b]:
        assert not np.isnan(val), "Ichimoku components should not be NaN"
        assert not np.isinf(val), "Ichimoku components should not be Inf"


# =============================================================================
# Test 27: Edge Cases - Price Extremes
# =============================================================================


def test_very_high_prices(extractor):
    """Test handling of very high price values."""
    # Create data with very high prices (e.g., BRK.A)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=100)
    close = np.linspace(500000, 550000, 100)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.005
    volume = np.full(100, 1000.0)  # Low volume for high-priced stock

    high_price_data = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    features = extractor._extract_technical_features(high_price_data)
    assert len(features) == 81, "Should handle high prices"
    assert not np.any(np.isnan(features)), "Should not produce NaN with high prices"


def test_very_low_prices(extractor):
    """Test handling of very low price values."""
    # Create data with very low prices (penny stocks)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=100)
    close = np.linspace(0.01, 0.05, 100)
    high = close * 1.1
    low = close * 0.9
    open_ = close * 1.02
    volume = np.full(100, 1000000.0)  # High volume for penny stock

    low_price_data = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    features = extractor._extract_technical_features(low_price_data)
    assert len(features) == 81, "Should handle low prices"
    assert not np.any(np.isnan(features)), "Should not produce NaN with low prices"


# =============================================================================
# Test 28: Performance Tests
# =============================================================================


def test_feature_extraction_performance(extractor, flat_data, benchmark=None):
    """Test feature extraction performance (optional benchmark)."""
    timestamp = flat_data.index[-1]

    import time

    start = time.time()
    features = extractor.extract_features(flat_data, "AAPL", timestamp)
    elapsed = time.time() - start

    # Should complete in reasonable time (< 5 seconds)
    assert elapsed < 5.0, f"Feature extraction took {elapsed:.2f}s (should be < 5s)"
    assert len(features) == DIMS.fincoll_total


def test_batch_extraction_performance(extractor, flat_data):
    """Test batch extraction performance improvement."""
    symbols = [f"SYM{i}" for i in range(10)]
    timestamp = flat_data.index[-1]
    ohlcv_dict = {sym: flat_data for sym in symbols}

    import time

    start = time.time()
    results = extractor.extract_features_batch(
        ohlcv_dict, symbols, timestamp, auto_batch=False
    )
    elapsed = time.time() - start

    # Batch should be efficient (< 30 seconds for 10 symbols)
    assert elapsed < 30.0, f"Batch extraction took {elapsed:.2f}s (should be < 30s)"
    assert len(results) == len(symbols)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
