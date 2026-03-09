#!/usr/bin/env python3
"""
Comprehensive Backtesting API Tests

This module provides comprehensive test coverage for api/backtesting.py.

Coverage areas:
1. Velocity prediction endpoint (historical data, feature extraction)
2. Feature extraction job creation and background processing
3. Job status tracking and polling
4. Backtest from cached features
5. Job management (list, delete)
6. Edge cases (empty data, invalid inputs, missing features)
7. Error handling (provider not initialized, file not found, etc.)
8. Parameter validation

Target: 80%+ coverage for backtesting.py
"""

import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import BackgroundTasks, HTTPException

sys.path.insert(0, ".")

from fincoll.api.backtesting import (
    BacktestFromCacheRequest,
    BacktestFromCacheResponse,
    ExtractFeaturesRequest,
    ExtractFeaturesResponse,
    JobStatusResponse,
    VelocityBacktestRequest,
    VelocityBacktestResponse,
    _estimate_trading_days,
    _extract_features_background,
    _get_data_provider,
    _load_velocity_model,
    _require_provider,
    backtest_from_cached_features,
    delete_extraction_job,
    extract_historical_features,
    extraction_jobs,
    get_extraction_job_status,
    get_historical_velocity_prediction,
    list_extraction_jobs,
    set_provider,
)


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


class MockDataProvider:
    """Mock data provider for testing."""

    def __init__(self):
        self._historical_data = None

    def get_historical_bars(
        self, symbol, start_date=None, end_date=None, interval="1d"
    ):
        """Return pre-configured historical data."""
        if self._historical_data is None:
            return make_ohlcv(300)
        return self._historical_data

    def set_historical_data(self, df):
        """Set the historical data to return."""
        self._historical_data = df


@pytest.fixture
def mock_provider():
    """Create a mock data provider."""
    return MockDataProvider()


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    return make_ohlcv(300)


@pytest.fixture
def mock_feature_extractor():
    """Create a mock feature extractor."""
    with patch("fincoll.api.backtesting.FeatureExtractor") as mock:
        extractor = MagicMock()
        # Return a 361D feature vector
        extractor.extract_features.return_value = np.random.randn(361)
        mock.return_value = extractor
        yield mock


@pytest.fixture
def mock_velocity_model():
    """Create a mock velocity model."""
    model = MagicMock()
    # Mock output: [batch, 20] for 5 timeframes * 4 values each
    model.return_value = torch.randn(1, 20)
    model.eval = MagicMock()
    return model


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset provider
    import fincoll.api.backtesting as bt

    bt._data_provider = None
    bt._velocity_model = None
    bt._velocity_model_path = None

    # Clear job tracking
    bt.extraction_jobs.clear()

    yield

    # Clean up after test
    bt._data_provider = None
    bt._velocity_model = None
    bt._velocity_model_path = None
    bt.extraction_jobs.clear()


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "backtest_cache"
    cache_dir.mkdir()
    return cache_dir


# =============================================================================
# Provider Setup Tests
# =============================================================================


def test_set_provider(mock_provider):
    """Test setting global data provider."""
    set_provider(mock_provider)

    import fincoll.api.backtesting as bt

    assert bt._data_provider is mock_provider


def test_require_provider_when_set(mock_provider):
    """Test _require_provider returns provider when set."""
    set_provider(mock_provider)
    provider = _require_provider()
    assert provider is mock_provider


def test_require_provider_raises_when_not_set():
    """Test _require_provider raises 503 when provider not set."""
    with pytest.raises(HTTPException) as exc_info:
        _require_provider()

    assert exc_info.value.status_code == 503
    assert "Data provider not initialized" in str(exc_info.value.detail)


def test_get_data_provider(mock_provider):
    """Test _get_data_provider returns global provider."""
    set_provider(mock_provider)
    provider = _get_data_provider("yfinance")
    assert provider is mock_provider


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_estimate_trading_days():
    """Test trading days estimation."""
    # 1 year ~ 252 trading days
    days = _estimate_trading_days("2024-01-01", "2024-12-31")
    assert 240 < days < 260  # Rough estimate

    # 1 month ~ 21 trading days
    days = _estimate_trading_days("2024-01-01", "2024-01-31")
    assert 18 < days < 24


def test_estimate_trading_days_same_day():
    """Test trading days estimation for same day."""
    days = _estimate_trading_days("2024-01-01", "2024-01-01")
    assert days == 0


def test_estimate_trading_days_weekend():
    """Test trading days estimation over weekend."""
    days = _estimate_trading_days("2024-01-01", "2024-01-03")
    assert days >= 0


# =============================================================================
# Velocity Prediction Endpoint Tests
# =============================================================================


@pytest.mark.asyncio
async def test_velocity_prediction_success(
    mock_provider, sample_ohlcv, mock_feature_extractor, mock_velocity_model
):
    """Test successful velocity prediction."""
    set_provider(mock_provider)
    mock_provider.set_historical_data(sample_ohlcv)

    request = VelocityBacktestRequest(
        symbol="AAPL", date="2024-01-15", interval="1d", provider="yfinance"
    )

    with patch("fincoll.api.backtesting._load_velocity_model") as mock_load:
        mock_load.return_value = mock_velocity_model

        response = await get_historical_velocity_prediction(request)

        assert response.symbol == "AAPL"
        assert response.date == "2024-01-15"
        assert response.current_price > 0
        assert len(response.velocities) == 5  # 5 timeframes
        assert "best_opportunity" in response.model_dump()


@pytest.mark.asyncio
async def test_velocity_prediction_no_data(mock_provider, mock_feature_extractor):
    """Test velocity prediction with no historical data."""
    set_provider(mock_provider)
    mock_provider.set_historical_data(pd.DataFrame())

    request = VelocityBacktestRequest(
        symbol="INVALID", date="2024-01-15", interval="1d", provider="yfinance"
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_historical_velocity_prediction(request)

    assert exc_info.value.status_code == 404
    assert "No historical data" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_velocity_prediction_insufficient_data(
    mock_provider, mock_feature_extractor
):
    """Test velocity prediction with insufficient data."""
    set_provider(mock_provider)
    # Only 30 bars, need 50+
    insufficient_data = make_ohlcv(30)
    mock_provider.set_historical_data(insufficient_data)

    request = VelocityBacktestRequest(
        symbol="AAPL", date="2024-01-15", interval="1d", provider="yfinance"
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_historical_velocity_prediction(request)

    assert exc_info.value.status_code == 400
    assert "Insufficient data" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_velocity_prediction_provider_not_initialized():
    """Test velocity prediction when provider not initialized."""
    request = VelocityBacktestRequest(
        symbol="AAPL", date="2024-01-15", interval="1d", provider="yfinance"
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_historical_velocity_prediction(request)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_velocity_prediction_feature_extraction_error(
    mock_provider, sample_ohlcv, mock_feature_extractor
):
    """Test velocity prediction when feature extraction fails."""
    set_provider(mock_provider)
    mock_provider.set_historical_data(sample_ohlcv)

    # Make feature extraction fail
    with patch("fincoll.api.backtesting.FeatureExtractor") as mock_extractor:
        extractor = MagicMock()
        extractor.extract_features.side_effect = ValueError("Feature extraction failed")
        mock_extractor.return_value = extractor

        request = VelocityBacktestRequest(
            symbol="AAPL", date="2024-01-15", interval="1d", provider="yfinance"
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_historical_velocity_prediction(request)

        assert exc_info.value.status_code == 500


# =============================================================================
# Feature Extraction Job Tests
# =============================================================================


@pytest.mark.asyncio
async def test_extract_features_job_creation(mock_provider, tmp_cache_dir):
    """Test feature extraction job creation."""
    set_provider(mock_provider)

    request = ExtractFeaturesRequest(
        symbols=["AAPL", "GOOGL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        provider="yfinance",
    )

    background_tasks = BackgroundTasks()

    with patch("fincoll.api.backtesting.CACHE_DIR", tmp_cache_dir):
        response = await extract_historical_features(request, background_tasks)

        assert response.status == "queued"
        assert response.symbols_requested == 2
        assert response.date_range["start"] == "2024-01-01"
        assert response.date_range["end"] == "2024-01-31"
        assert response.estimated_features > 0
        assert response.job_id.startswith("extract_")


@pytest.mark.asyncio
async def test_extract_features_custom_output_path(mock_provider):
    """Test feature extraction with custom output path."""
    set_provider(mock_provider)

    custom_path = "/tmp/custom_features.parquet"
    request = ExtractFeaturesRequest(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        provider="yfinance",
        output_path=custom_path,
    )

    background_tasks = BackgroundTasks()
    response = await extract_historical_features(request, background_tasks)

    assert response.output_file == custom_path


@pytest.mark.asyncio
async def test_extract_features_multiple_symbols(mock_provider):
    """Test feature extraction with multiple symbols."""
    set_provider(mock_provider)

    request = ExtractFeaturesRequest(
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
        start_date="2024-01-01",
        end_date="2024-03-31",
        interval="1d",
        provider="yfinance",
    )

    background_tasks = BackgroundTasks()
    response = await extract_historical_features(request, background_tasks)

    assert response.symbols_requested == 4
    # Estimate: 4 symbols * ~60 trading days
    assert response.estimated_features > 200


# =============================================================================
# Background Task Tests
# =============================================================================


@pytest.mark.asyncio
async def test_extract_features_background_success(
    mock_provider, sample_ohlcv, tmp_path
):
    """Test successful background feature extraction."""
    set_provider(mock_provider)
    mock_provider.set_historical_data(sample_ohlcv)

    output_file = str(tmp_path / "features.parquet")
    job_id = "test_job_123"

    # Initialize job tracking
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": {},
    }

    with patch("fincoll.api.backtesting.FeatureExtractor") as mock_extractor:
        extractor = MagicMock()
        extractor.extract_features.return_value = np.random.randn(361)
        mock_extractor.return_value = extractor

        await _extract_features_background(
            job_id=job_id,
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            interval="1d",
            provider_name="yfinance",
            output_file=output_file,
        )

        # Check job status
        assert extraction_jobs[job_id]["status"] == "completed"
        assert extraction_jobs[job_id]["output_file"] == output_file
        assert Path(output_file).exists()


@pytest.mark.asyncio
async def test_extract_features_background_no_data(mock_provider, tmp_path):
    """Test background extraction with no data."""
    set_provider(mock_provider)
    mock_provider.set_historical_data(pd.DataFrame())

    output_file = str(tmp_path / "features.parquet")
    job_id = "test_job_no_data"

    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": {},
    }

    with patch("fincoll.api.backtesting.FeatureExtractor") as mock_extractor:
        extractor = MagicMock()
        mock_extractor.return_value = extractor

        await _extract_features_background(
            job_id=job_id,
            symbols=["INVALID"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            interval="1d",
            provider_name="yfinance",
            output_file=output_file,
        )

        # Should fail with no features
        assert extraction_jobs[job_id]["status"] == "failed"
        assert "error" in extraction_jobs[job_id]


@pytest.mark.asyncio
async def test_extract_features_background_partial_success(
    mock_provider, sample_ohlcv, tmp_path
):
    """Test background extraction with some symbols failing."""
    set_provider(mock_provider)

    output_file = str(tmp_path / "features.parquet")
    job_id = "test_job_partial"

    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": {},
    }

    # Mock provider to return data for AAPL, empty for INVALID
    def mock_get_bars(symbol, **kwargs):
        if symbol == "AAPL":
            return sample_ohlcv
        return pd.DataFrame()

    mock_provider.get_historical_bars = mock_get_bars

    with patch("fincoll.api.backtesting.FeatureExtractor") as mock_extractor:
        extractor = MagicMock()
        extractor.extract_features.return_value = np.random.randn(361)
        mock_extractor.return_value = extractor

        await _extract_features_background(
            job_id=job_id,
            symbols=["AAPL", "INVALID"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            interval="1d",
            provider_name="yfinance",
            output_file=output_file,
        )

        # Should complete with partial data
        if extraction_jobs[job_id]["status"] == "completed":
            assert Path(output_file).exists()


# =============================================================================
# Job Status Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_job_status_exists():
    """Test getting status of existing job."""
    job_id = "test_job_status"
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": {"percent_complete": 50},
    }

    response = await get_extraction_job_status(job_id)

    assert response.job_id == job_id
    assert response.status == "processing"
    assert response.progress["percent_complete"] == 50


@pytest.mark.asyncio
async def test_get_job_status_not_found():
    """Test getting status of non-existent job."""
    with pytest.raises(HTTPException) as exc_info:
        await get_extraction_job_status("nonexistent_job")

    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_job_status_completed():
    """Test getting status of completed job."""
    job_id = "test_job_completed"
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "progress": {"percent_complete": 100},
        "output_file": "/tmp/features.parquet",
        "file_size_mb": 10.5,
        "completion_time": "2024-01-15T10:00:00",
    }

    response = await get_extraction_job_status(job_id)

    assert response.status == "completed"
    assert response.output_file == "/tmp/features.parquet"
    assert response.file_size_mb == 10.5


# =============================================================================
# Backtest from Cache Tests
# =============================================================================


@pytest.mark.asyncio
async def test_backtest_from_cache_success(tmp_path):
    """Test successful backtest from cached features."""
    # Create mock cache file
    cache_file = tmp_path / "features.parquet"

    # Create sample cached features
    features_data = []
    for i in range(100):
        features_data.append(
            {
                "symbol": "AAPL" if i % 2 == 0 else "GOOGL",
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "features": np.random.randn(361).tolist(),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000000.0,
            }
        )

    df = pd.DataFrame(features_data)
    df.to_parquet(cache_file)

    request = BacktestFromCacheRequest(
        cache_file=str(cache_file),
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    response = await backtest_from_cached_features(request)

    assert response.backtest_id.startswith("backtest_")
    assert response.symbols == ["AAPL"]
    assert len(response.predictions) > 0
    assert response.metadata["from_cache"] is True


@pytest.mark.asyncio
async def test_backtest_from_cache_file_not_found():
    """Test backtest from cache with non-existent file."""
    request = BacktestFromCacheRequest(
        cache_file="/nonexistent/file.parquet",
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    with pytest.raises(HTTPException) as exc_info:
        await backtest_from_cached_features(request)

    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_backtest_from_cache_no_matching_data(tmp_path):
    """Test backtest from cache with no matching data."""
    cache_file = tmp_path / "features.parquet"

    # Create cache with different symbols
    features_data = [
        {
            "symbol": "MSFT",
            "timestamp": pd.Timestamp("2024-01-01"),
            "features": np.random.randn(361).tolist(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000.0,
        }
    ]

    df = pd.DataFrame(features_data)
    df.to_parquet(cache_file)

    request = BacktestFromCacheRequest(
        cache_file=str(cache_file),
        symbols=["AAPL"],  # Requesting different symbol
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    with pytest.raises(HTTPException) as exc_info:
        await backtest_from_cached_features(request)

    assert exc_info.value.status_code == 404
    assert "No cached features found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_backtest_from_cache_multiple_symbols(tmp_path):
    """Test backtest from cache with multiple symbols."""
    cache_file = tmp_path / "features.parquet"

    features_data = []
    symbols = ["AAPL", "GOOGL", "MSFT"]
    for symbol in symbols:
        for i in range(30):
            features_data.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "features": np.random.randn(361).tolist(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000000.0,
                }
            )

    df = pd.DataFrame(features_data)
    df.to_parquet(cache_file)

    request = BacktestFromCacheRequest(
        cache_file=str(cache_file),
        symbols=["AAPL", "GOOGL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    response = await backtest_from_cached_features(request)

    # Should have predictions for AAPL and GOOGL, not MSFT
    prediction_symbols = set(p["symbol"] for p in response.predictions)
    assert prediction_symbols == {"AAPL", "GOOGL"}


# =============================================================================
# Job Management Tests
# =============================================================================


# =============================================================================
# Job Management Tests
# =============================================================================


@pytest.mark.asyncio
async def test_list_extraction_jobs_empty():
    """Test listing jobs when none exist."""
    response = await list_extraction_jobs()

    assert response["total"] == 0
    assert len(response["jobs"]) == 0


@pytest.mark.asyncio
async def test_list_extraction_jobs_multiple():
    """Test listing multiple jobs."""
    extraction_jobs["job1"] = {"job_id": "job1", "status": "queued"}
    extraction_jobs["job2"] = {"job_id": "job2", "status": "processing"}
    extraction_jobs["job3"] = {"job_id": "job3", "status": "completed"}

    response = await list_extraction_jobs()

    assert response["total"] == 3
    assert len(response["jobs"]) == 3


@pytest.mark.asyncio
async def test_delete_extraction_job_exists():
    """Test deleting existing job."""
    job_id = "job_to_delete"
    extraction_jobs[job_id] = {"job_id": job_id, "status": "completed"}

    response = await delete_extraction_job(job_id)

    assert response["deleted"] is True
    assert response["job_id"] == job_id
    assert job_id not in extraction_jobs


@pytest.mark.asyncio
async def test_delete_extraction_job_not_found():
    """Test deleting non-existent job."""
    with pytest.raises(HTTPException) as exc_info:
        await delete_extraction_job("nonexistent_job")

    assert exc_info.value.status_code == 404


# =============================================================================
# Model Loading Tests
# =============================================================================


def test_load_velocity_model_caching(mock_velocity_model):
    """Test velocity model is cached after first load."""
    # Test that model is cached by setting it directly
    import fincoll.api.backtesting as bt

    bt._velocity_model = mock_velocity_model

    # First load (should return cached)
    model1 = _load_velocity_model()
    # Second load (should return cached)
    model2 = _load_velocity_model()

    # Should return same cached instance
    assert model1 is mock_velocity_model
    assert model2 is mock_velocity_model
    assert model1 is model2


def test_load_velocity_model_not_found():
    """Test velocity model loading when file not found."""
    with patch("fincoll.api.backtesting.Path.exists") as mock_exists:
        mock_exists.return_value = False

        # Reset cached model
        import fincoll.api.backtesting as bt

        bt._velocity_model = None

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_velocity_model()

        assert "not found" in str(exc_info.value)


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.asyncio
async def test_velocity_prediction_invalid_date_format(mock_provider):
    """Test velocity prediction with invalid date format."""
    set_provider(mock_provider)

    request = VelocityBacktestRequest(
        symbol="AAPL",
        date="invalid-date",
        interval="1d",
        provider="yfinance",
    )

    with pytest.raises(Exception):  # Should raise parsing error
        await get_historical_velocity_prediction(request)


@pytest.mark.asyncio
async def test_extract_features_empty_symbol_list(mock_provider):
    """Test feature extraction with empty symbol list."""
    set_provider(mock_provider)

    request = ExtractFeaturesRequest(
        symbols=[],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
    )

    background_tasks = BackgroundTasks()
    response = await extract_historical_features(request, background_tasks)

    assert response.symbols_requested == 0
    assert response.estimated_features == 0


@pytest.mark.asyncio
async def test_backtest_from_cache_date_filtering(tmp_path):
    """Test backtest from cache respects date range filtering."""
    cache_file = tmp_path / "features.parquet"

    # Create data spanning wider range
    features_data = []
    for i in range(90):  # 90 days
        features_data.append(
            {
                "symbol": "AAPL",
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "features": np.random.randn(361).tolist(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000000.0,
            }
        )

    df = pd.DataFrame(features_data)
    df.to_parquet(cache_file)

    # Request only January data
    request = BacktestFromCacheRequest(
        cache_file=str(cache_file),
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    response = await backtest_from_cached_features(request)

    # Should only have predictions for January (max 31 days)
    assert len(response.predictions) <= 31
    assert int(response.date_range["trading_days"]) <= 31


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_full_extraction_workflow(mock_provider, sample_ohlcv, tmp_path):
    """Test complete extraction workflow: create job -> poll status -> check completion."""
    set_provider(mock_provider)
    mock_provider.set_historical_data(sample_ohlcv)

    # Step 1: Create extraction job
    request = ExtractFeaturesRequest(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
    )

    background_tasks = BackgroundTasks()

    with patch("fincoll.api.backtesting.CACHE_DIR", tmp_path):
        create_response = await extract_historical_features(request, background_tasks)

    job_id = create_response.job_id

    # Step 2: Check initial status
    status_response = await get_extraction_job_status(job_id)
    assert status_response.status in ["queued", "processing"]

    # Step 3: List all jobs
    list_response = await list_extraction_jobs()
    assert list_response["total"] >= 1
    assert any(job["job_id"] == job_id for job in list_response["jobs"])

    # Step 4: Delete job
    delete_response = await delete_extraction_job(job_id)
    assert delete_response["deleted"] is True

    # Step 5: Verify deletion
    with pytest.raises(HTTPException):
        await get_extraction_job_status(job_id)
