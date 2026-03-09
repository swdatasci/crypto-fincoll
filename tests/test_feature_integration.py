#!/usr/bin/env python3
"""
Test Feature Integration - Config-Driven Feature Vector

Verifies feature extraction end-to-end using config dimensions.
"""
import sys
sys.path.insert(0, '.')

import pytest
import numpy as np
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from fincoll.features.feature_extractor import FeatureExtractor
from fincoll.providers.yfinance_provider import YFinanceProvider
from config.dimensions import DIMS


def _make_ohlcv(n=300, seed=42):
    dates = pd.bdate_range(end=pd.Timestamp('2026-02-13'), periods=n)
    rng = np.random.default_rng(seed)
    m = len(dates)
    close = 185.0 * np.exp(np.cumsum(rng.standard_normal(m) * 0.01))
    high = close * (1 + np.abs(rng.standard_normal(m) * 0.005))
    low  = close * (1 - np.abs(rng.standard_normal(m) * 0.005))
    open_ = close * (1 + rng.standard_normal(m) * 0.003)
    volume = rng.integers(50_000_000, 100_000_000, m).astype(float)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=dates)


def _mock_ticker_factory(df):
    m = MagicMock()
    m.history.return_value = df
    m.info = {'sector': 'Technology', 'industry': 'Consumer Electronics', 'marketCap': 3_000_000_000_000}
    return m


@pytest.fixture(scope="module")
def aapl_feature_vector():
    """Build synthetic OHLCV and extract full feature vector once for all tests."""
    synthetic = _make_ohlcv(300)

    with patch('yfinance.Ticker', side_effect=lambda sym: _mock_ticker_factory(synthetic)), \
         patch('requests.get', side_effect=requests.exceptions.ConnectionError("no network")):

        provider = YFinanceProvider()
        provider.get_historical_bars = MagicMock(return_value=synthetic)

        extractor = FeatureExtractor(
            data_provider=provider,
            enable_senvec=False,
            enable_futures=False,
            enable_finnhub=False,
        )

        features = extractor.extract_features(synthetic, "AAPL", synthetic.index[-1])

    return features


def test_feature_vector_correct_dimension(aapl_feature_vector):
    """Extracted feature vector must match the configured total dimension."""
    assert len(aapl_feature_vector) == DIMS.fincoll_total, \
        f"Expected {DIMS.fincoll_total}D features, got {len(aapl_feature_vector)}D"


def test_feature_vector_no_nan(aapl_feature_vector):
    """Feature vector must contain no NaN values."""
    nan_count = int(np.isnan(aapl_feature_vector).sum())
    assert nan_count == 0, f"Feature vector contains {nan_count} NaN values"


def test_feature_vector_no_inf(aapl_feature_vector):
    """Feature vector must contain no infinite values."""
    inf_count = int(np.isinf(aapl_feature_vector).sum())
    assert inf_count == 0, f"Feature vector contains {inf_count} Inf values"


def test_feature_vector_has_nonzero_values(aapl_feature_vector):
    """Feature vector must have meaningful non-zero content (>40% non-zero).

    Threshold is 40% (not 50%) because optional features (senvec, finnhub) are
    disabled in mocked tests, reducing the non-zero count vs production.
    """
    non_zero = int(np.count_nonzero(aapl_feature_vector))
    total = len(aapl_feature_vector)
    pct = non_zero / total
    assert pct > 0.4, \
        f"Only {non_zero}/{total} ({pct:.1%}) features are non-zero — likely extraction failure"


def test_feature_vector_is_numpy_array(aapl_feature_vector):
    """Feature vector must be a floating-point numpy array."""
    assert isinstance(aapl_feature_vector, np.ndarray), \
        f"Expected np.ndarray, got {type(aapl_feature_vector)}"
    assert np.issubdtype(aapl_feature_vector.dtype, np.floating), \
        f"Expected floating-point dtype, got {aapl_feature_vector.dtype}"
