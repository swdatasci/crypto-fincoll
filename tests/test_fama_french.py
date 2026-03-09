#!/usr/bin/env python3
"""Tests for Fama-French factor fetching and exposure calculation."""
import sys
sys.path.insert(0, '.')

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from datetime import timedelta
from fincoll.features.fama_french import get_fama_french_factors


def _make_ff_factors(n=300):
    dates = pd.bdate_range(end=pd.Timestamp('2026-02-13'), periods=n)
    rng = np.random.default_rng(42)
    n = len(dates)
    return pd.DataFrame({
        'Mkt-RF': rng.standard_normal(n) * 0.01,
        'SMB':    rng.standard_normal(n) * 0.005,
        'HML':    rng.standard_normal(n) * 0.005,
        'RMW':    rng.standard_normal(n) * 0.003,
        'CMA':    rng.standard_normal(n) * 0.003,
        'RF':     np.ones(n) * 0.0001,
        'Mom':    rng.standard_normal(n) * 0.008,
    }, index=dates)


@pytest.fixture(scope="module")
def ff_factors():
    """Create FamaFrench instance with fetch_factors mocked to return synthetic data."""
    ff = get_fama_french_factors()
    synthetic = _make_ff_factors()
    with patch.object(ff, 'fetch_factors', return_value=synthetic):
        factors = ff.fetch_factors()
    return ff, factors


def test_fama_french_returns_dataframe(ff_factors):
    """fetch_factors() must return a non-empty DataFrame."""
    _, factors = ff_factors
    assert isinstance(factors, pd.DataFrame), \
        f"Expected DataFrame, got {type(factors)}"
    assert not factors.empty, "Fama-French factors DataFrame is empty"


def test_fama_french_has_required_columns(ff_factors):
    """Factor DataFrame must contain the standard FF3/FF5 factor columns."""
    _, factors = ff_factors
    required = {'Mkt-RF', 'SMB', 'HML'}
    missing = required - set(factors.columns)
    assert not missing, \
        f"Fama-French factors missing required columns: {missing}"


def test_fama_french_has_recent_data(ff_factors):
    """Factor data must include recent history (within last 6 months)."""
    _, factors = ff_factors
    most_recent = factors.index.max()
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    assert most_recent >= six_months_ago, \
        f"Most recent factor data is {most_recent.date()}, expected within last 6 months"


def test_fama_french_no_nan_in_core_factors(ff_factors):
    """Core FF factors (Mkt-RF, SMB, HML) must not contain NaN in the last 252 days."""
    _, factors = ff_factors
    recent = factors.tail(252)[['Mkt-RF', 'SMB', 'HML']]
    nan_count = recent.isna().sum().sum()
    assert nan_count == 0, \
        f"Found {nan_count} NaN values in recent core Fama-French factors"


def test_fama_french_factor_exposures_calculated(ff_factors):
    """calculate_factor_exposures() returns a dict with beta values for each factor."""
    ff, factors = ff_factors
    np.random.seed(42)
    dates = factors.index[-252:]
    returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

    exposures = ff.calculate_factor_exposures(returns, lookback_days=252)

    assert isinstance(exposures, dict), \
        f"Expected dict of exposures, got {type(exposures)}"
    assert len(exposures) > 0, "Factor exposures dict is empty"

    for factor, beta in exposures.items():
        assert isinstance(beta, float), \
            f"Beta for {factor!r} should be float, got {type(beta)}"
        assert not np.isnan(beta), f"Beta for {factor!r} is NaN"
        assert not np.isinf(beta), f"Beta for {factor!r} is Inf"
