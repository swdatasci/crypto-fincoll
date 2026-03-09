#!/usr/bin/env python3
"""
Tests for sector features implementation
"""
import sys
sys.path.insert(0, '.')

import pytest
import numpy as np
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from fincoll.providers.yfinance_provider import YFinanceProvider
from fincoll.features.feature_extractor import FeatureExtractor


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
def aapl_features():
    """Build synthetic OHLCV and extract features once for all tests in module."""
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


def test_sector_features_extracted(aapl_features):
    """Sector feature slice has correct length (14D: 11 one-hot + 3 rel perf)."""
    sector_features = aapl_features[205:219]
    assert len(sector_features) == 14, \
        f"Expected 14 sector features, got {len(sector_features)}"


def test_sector_onehot_is_valid(aapl_features):
    """Sector one-hot encoding sums to exactly 1.0 (exactly one sector active)."""
    sector_onehot = aapl_features[205:216]
    total = float(np.sum(sector_onehot))
    assert total == pytest.approx(1.0), \
        f"Sector one-hot should sum to 1.0, got {total:.4f}"


def test_sector_onehot_single_bit(aapl_features):
    """Only one sector flag should be set; all others must be 0."""
    sector_onehot = aapl_features[205:216]
    active = int(np.count_nonzero(sector_onehot))
    assert active == 1, \
        f"Expected exactly 1 active sector bit, got {active}: {sector_onehot}"


def test_sector_relative_performance_present(aapl_features):
    """Relative performance features (f216-f218) have correct shape and no NaN/Inf."""
    rel_perf = aapl_features[216:219]
    assert len(rel_perf) == 3, f"Expected 3 rel-perf features, got {len(rel_perf)}"
    assert not np.any(np.isnan(rel_perf)), f"NaN in relative performance features: {rel_perf}"
    assert not np.any(np.isinf(rel_perf)), f"Inf in relative performance features: {rel_perf}"


def test_sector_features_no_nan_or_inf(aapl_features):
    """Sector features must not contain NaN or Inf values."""
    sector_features = aapl_features[205:219]
    assert not np.any(np.isnan(sector_features)), \
        f"NaN in sector features: {sector_features}"
    assert not np.any(np.isinf(sector_features)), \
        f"Inf in sector features: {sector_features}"
