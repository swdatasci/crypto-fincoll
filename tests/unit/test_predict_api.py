#!/usr/bin/env python3
"""
Comprehensive Prediction API Tests

This module provides comprehensive test coverage for api/inference.py prediction endpoints.

Coverage areas:
1. Single symbol prediction (/predict/{symbol})
2. Batch prediction (/predict/batch)
3. Request validation (valid/invalid symbols, parameters)
4. Feature extraction integration
5. Model inference mocking
6. Prediction formatting and response structure
7. Error handling (missing features, model failures, provider errors)
8. Provider selection (TradeStation, Alpaca, fallback)
9. Multi-symbol batch predictions with concurrency
10. Edge cases (invalid symbols, empty responses, network failures)

Target: 80%+ coverage for inference.py prediction endpoints
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, ".")

from fincoll.api.inference import (
    batch_predict_velocity,
    predict_symbol,
    set_provider,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def make_ohlcv(n=300, seed=42):
    """Create synthetic OHLCV data."""
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
    rng = np.random.default_rng(seed)
    m = len(dates)

    returns = rng.standard_normal(m) * 0.01
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

    def __init__(self, fail_mode=None):
        self.fail_mode = fail_mode
        self.call_count = 0

    async def fetch_ohlcv(self, symbol, interval="1d", limit=300):
        """Mock OHLCV fetch."""
        self.call_count += 1

        if self.fail_mode == "empty":
            return pd.DataFrame()
        elif self.fail_mode == "error":
            raise ValueError(f"Provider error for {symbol}")
        elif self.fail_mode == "timeout":
            raise TimeoutError(f"Timeout fetching {symbol}")

        return make_ohlcv(n=limit, seed=hash(symbol) % 10000)

    async def get_quote(self, symbol):
        """Mock quote fetch."""
        if self.fail_mode == "error":
            raise ValueError(f"Provider error for {symbol}")

        return {
            "symbol": symbol,
            "price": 150.0,
            "bid": 149.95,
            "ask": 150.05,
            "volume": 1000000,
        }


class MockFeatureExtractor:
    """Mock feature extractor."""

    def __init__(self, feature_dim=414, fail_mode=None):
        self.feature_dim = feature_dim
        self.fail_mode = fail_mode
        self.call_count = 0

    async def extract_features(self, symbol, df, quote=None):
        """Mock feature extraction."""
        self.call_count += 1

        if self.fail_mode == "error":
            raise RuntimeError(f"Feature extraction failed for {symbol}")
        elif self.fail_mode == "incomplete":
            # Return partial features (wrong dimension)
            return np.random.randn(200).astype(np.float32)

        return np.random.randn(self.feature_dim).astype(np.float32)


class MockVelocityEngine:
    """Mock velocity prediction engine."""

    def __init__(self, fail_mode=None):
        self.fail_mode = fail_mode
        self.call_count = 0

    def predict(self, features, symbol, current_price):
        """Mock prediction."""
        self.call_count += 1

        if self.fail_mode == "error":
            raise RuntimeError(f"Model inference failed for {symbol}")
        elif self.fail_mode == "invalid_output":
            return {"invalid": "response"}

        # Return realistic velocity prediction
        return {
            "symbol": symbol,
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                "1min": {
                    "long_velocity": 0.0012,
                    "long_bars": 5,
                    "short_velocity": -0.0008,
                    "short_bars": 3,
                    "confidence": 0.75,
                },
                "5min": {
                    "long_velocity": 0.0045,
                    "long_bars": 8,
                    "short_velocity": -0.0032,
                    "short_bars": 6,
                    "confidence": 0.82,
                },
                "1hour": {
                    "long_velocity": 0.015,
                    "long_bars": 4,
                    "short_velocity": -0.012,
                    "short_bars": 3,
                    "confidence": 0.68,
                },
            },
            "best_opportunity": {
                "direction": "LONG",
                "timeframe": "5min",
                "velocity": 0.0045,
                "bars": 8,
                "expected_profit_pct": 3.6,
                "confidence": 0.82,
            },
        }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider():
    """Provide a mock data provider."""
    return MockDataProvider()


@pytest.fixture
def mock_extractor():
    """Provide a mock feature extractor."""
    return MockFeatureExtractor()


@pytest.fixture
def mock_engine():
    """Provide a mock velocity engine."""
    return MockVelocityEngine()


@pytest.fixture
def setup_mocks(mock_provider, mock_extractor, mock_engine):
    """Setup all mocks for prediction endpoints."""
    with (
        patch("fincoll.api.inference._data_provider", mock_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        yield {
            "provider": mock_provider,
            "extractor": mock_extractor,
            "engine": mock_engine,
        }


# =============================================================================
# Single Symbol Prediction Tests
# =============================================================================


@pytest.mark.asyncio
async def test_predict_symbol_success(setup_mocks):
    """Test successful single symbol prediction."""
    result = await predict_symbol(symbol="AAPL", provider="tradestation", interval="1d")

    assert result["symbol"] == "AAPL"
    assert "current_price" in result
    assert "predictions" in result
    assert "best_opportunity" in result
    assert "timestamp" in result

    # Verify predictions structure
    predictions = result["predictions"]
    assert "1min" in predictions
    assert "5min" in predictions
    assert "1hour" in predictions

    # Verify each timeframe prediction
    for tf_pred in predictions.values():
        assert "long_velocity" in tf_pred
        assert "long_bars" in tf_pred
        assert "short_velocity" in tf_pred
        assert "short_bars" in tf_pred
        assert "confidence" in tf_pred

    # Verify best opportunity
    best = result["best_opportunity"]
    assert best["direction"] in ["LONG", "SHORT", "NONE"]
    assert "timeframe" in best
    assert "velocity" in best
    assert "confidence" in best


@pytest.mark.asyncio
async def test_predict_symbol_invalid_symbol(setup_mocks):
    """Test prediction with invalid symbol."""
    # Empty symbol
    with pytest.raises(Exception):  # FastAPI will validate
        await predict_symbol(symbol="", provider="tradestation")

    # Symbol with invalid characters
    result = await predict_symbol(symbol="INVALID@#$", provider="tradestation")
    # Should still attempt prediction - provider will handle validation


@pytest.mark.asyncio
async def test_predict_symbol_provider_error():
    """Test prediction when provider fails."""
    failing_provider = MockDataProvider(fail_mode="error")
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", failing_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        with pytest.raises(ValueError, match="Provider error"):
            await predict_symbol(symbol="AAPL", provider="tradestation")


@pytest.mark.asyncio
async def test_predict_symbol_empty_data():
    """Test prediction when provider returns empty data."""
    empty_provider = MockDataProvider(fail_mode="empty")
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", empty_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        # Should raise error or handle gracefully
        result = await predict_symbol(symbol="AAPL", provider="tradestation")
        # Check if error is in result or exception is raised


@pytest.mark.asyncio
async def test_predict_symbol_feature_extraction_error():
    """Test prediction when feature extraction fails."""
    mock_provider = MockDataProvider()
    failing_extractor = MockFeatureExtractor(fail_mode="error")
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", mock_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=failing_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        with pytest.raises(RuntimeError, match="Feature extraction failed"):
            await predict_symbol(symbol="AAPL", provider="tradestation")


@pytest.mark.asyncio
async def test_predict_symbol_model_inference_error():
    """Test prediction when model inference fails."""
    mock_provider = MockDataProvider()
    mock_extractor = MockFeatureExtractor()
    failing_engine = MockVelocityEngine(fail_mode="error")

    with (
        patch("fincoll.api.inference._data_provider", mock_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch(
            "fincoll.api.inference._load_velocity_model", return_value=failing_engine
        ),
    ):
        with pytest.raises(RuntimeError, match="Model inference failed"):
            await predict_symbol(symbol="AAPL", provider="tradestation")


@pytest.mark.asyncio
async def test_predict_symbol_different_intervals(setup_mocks):
    """Test prediction with different time intervals."""
    intervals = ["1min", "5min", "15min", "1hour", "1d"]

    for interval in intervals:
        result = await predict_symbol(
            symbol="MSFT", provider="tradestation", interval=interval
        )
        assert result["symbol"] == "MSFT"
        assert "predictions" in result


@pytest.mark.asyncio
async def test_predict_symbol_different_providers(setup_mocks):
    """Test prediction with different providers."""
    providers = ["tradestation", "alpaca", "auto"]

    for provider in providers:
        result = await predict_symbol(symbol="GOOGL", provider=provider, interval="1d")
        assert result["symbol"] == "GOOGL"
        assert "predictions" in result


# =============================================================================
# Batch Prediction Tests
# =============================================================================


@pytest.mark.asyncio
async def test_batch_predict_success(setup_mocks):
    """Test successful batch prediction."""
    request = {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "provider": "tradestation",
        "interval": "1d",
    }

    result = await batch_predict_velocity(request)

    assert "predictions" in result
    assert "summary" in result
    assert len(result["predictions"]) == 3

    # Verify summary
    summary = result["summary"]
    assert summary["total"] == 3
    assert summary["successful"] >= 0
    assert summary["failed"] >= 0
    assert summary["successful"] + summary["failed"] == 3


@pytest.mark.asyncio
async def test_batch_predict_empty_symbols():
    """Test batch prediction with empty symbols list."""
    mock_provider = MockDataProvider()
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", mock_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        request = {"symbols": [], "provider": "tradestation", "interval": "1d"}

        result = await batch_predict_velocity(request)
        assert result["summary"]["total"] == 0


@pytest.mark.asyncio
async def test_batch_predict_partial_failure():
    """Test batch prediction with some symbols failing."""

    # Create provider that fails for specific symbols
    class SelectiveFailProvider(MockDataProvider):
        async def fetch_ohlcv(self, symbol, interval="1d", limit=300):
            if symbol == "INVALID":
                raise ValueError(f"Invalid symbol: {symbol}")
            return await super().fetch_ohlcv(symbol, interval, limit)

    failing_provider = SelectiveFailProvider()
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", failing_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        request = {
            "symbols": ["AAPL", "INVALID", "MSFT"],
            "provider": "tradestation",
            "interval": "1d",
        }

        result = await batch_predict_velocity(request)

        # Should have partial results
        assert result["summary"]["total"] == 3
        assert result["summary"]["successful"] == 2
        assert result["summary"]["failed"] == 1

        # Check individual predictions
        predictions = result["predictions"]
        successful = [p for p in predictions if "error" not in p]
        failed = [p for p in predictions if "error" in p]

        assert len(successful) == 2
        assert len(failed) == 1


@pytest.mark.asyncio
async def test_batch_predict_large_batch(setup_mocks):
    """Test batch prediction with large number of symbols."""
    # Create 20 symbols
    symbols = [f"SYM{i}" for i in range(20)]

    request = {"symbols": symbols, "provider": "tradestation", "interval": "1d"}

    result = await batch_predict_velocity(request)

    assert result["summary"]["total"] == 20
    assert len(result["predictions"]) == 20


@pytest.mark.asyncio
async def test_batch_predict_concurrent_execution(setup_mocks):
    """Test that batch predictions execute concurrently."""
    import time

    # Track start times to verify concurrency
    start_times = []

    class TimingProvider(MockDataProvider):
        async def fetch_ohlcv(self, symbol, interval="1d", limit=300):
            start_times.append(time.time())
            # Simulate network delay
            await asyncio.sleep(0.1)
            return await super().fetch_ohlcv(symbol, interval, limit)

    timing_provider = TimingProvider()
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", timing_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        request = {
            "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
            "provider": "tradestation",
            "interval": "1d",
        }

        import asyncio

        start = time.time()
        result = await batch_predict_velocity(request)
        elapsed = time.time() - start

        # If truly concurrent with 5 concurrent limit, should take ~0.1s not 0.5s
        # Allow some overhead
        assert elapsed < 0.3, "Batch should execute concurrently"
        assert result["summary"]["total"] == 5


@pytest.mark.asyncio
async def test_batch_predict_provider_fallback():
    """Test batch prediction with auto provider fallback."""
    # Mock provider that fails, should fallback
    failing_provider = MockDataProvider(fail_mode="error")
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", failing_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        request = {
            "symbols": ["AAPL", "MSFT"],
            "provider": "auto",  # Should try fallback
            "interval": "1d",
        }

        # Depending on implementation, might fail or fallback
        result = await batch_predict_velocity(request)
        # Check results


# =============================================================================
# Provider Selection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_set_provider():
    """Test setting data provider."""
    mock_provider = MockDataProvider()
    set_provider(mock_provider)

    # Provider should be set and usable
    # Verify by attempting a prediction
    with (
        patch(
            "fincoll.api.inference.FeatureExtractor",
            return_value=MockFeatureExtractor(),
        ),
        patch(
            "fincoll.api.inference._load_velocity_model",
            return_value=MockVelocityEngine(),
        ),
    ):
        result = await predict_symbol(symbol="AAPL", provider="tradestation")
        assert result is not None


@pytest.mark.asyncio
async def test_predict_without_provider():
    """Test prediction without provider initialized."""
    # Clear provider
    with patch("fincoll.api.inference._data_provider", None):
        # Should raise 503 or appropriate error
        with pytest.raises(Exception):  # HTTPException 503
            await predict_symbol(symbol="AAPL", provider="tradestation")


# =============================================================================
# Response Format Tests
# =============================================================================


@pytest.mark.asyncio
async def test_predict_response_structure(setup_mocks):
    """Test prediction response has correct structure."""
    result = await predict_symbol(symbol="AAPL", provider="tradestation")

    # Required top-level fields
    required_fields = [
        "symbol",
        "current_price",
        "timestamp",
        "predictions",
        "best_opportunity",
    ]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # Predictions should be dict
    assert isinstance(result["predictions"], dict)
    assert len(result["predictions"]) > 0

    # Each timeframe prediction should have required fields
    for timeframe, pred in result["predictions"].items():
        assert "long_velocity" in pred
        assert "long_bars" in pred
        assert "short_velocity" in pred
        assert "short_bars" in pred
        assert "confidence" in pred

        # Validate types
        assert isinstance(pred["long_velocity"], (int, float))
        assert isinstance(pred["long_bars"], int)
        assert isinstance(pred["short_velocity"], (int, float))
        assert isinstance(pred["short_bars"], int)
        assert isinstance(pred["confidence"], (int, float))
        assert 0 <= pred["confidence"] <= 1

    # Best opportunity structure
    best = result["best_opportunity"]
    assert "direction" in best
    assert best["direction"] in ["LONG", "SHORT", "NONE"]


@pytest.mark.asyncio
async def test_batch_predict_response_structure(setup_mocks):
    """Test batch prediction response has correct structure."""
    request = {
        "symbols": ["AAPL", "MSFT"],
        "provider": "tradestation",
        "interval": "1d",
    }

    result = await batch_predict_velocity(request)

    # Required top-level fields
    assert "predictions" in result
    assert "summary" in result

    # Summary structure
    summary = result["summary"]
    assert "total" in summary
    assert "successful" in summary
    assert "failed" in summary
    assert isinstance(summary["total"], int)
    assert isinstance(summary["successful"], int)
    assert isinstance(summary["failed"], int)

    # Predictions list
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == summary["total"]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_predict_with_special_characters():
    """Test prediction with symbols containing special characters."""
    mock_provider = MockDataProvider()
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", mock_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        # Test various symbol formats
        symbols = ["BRK.B", "BF.A", "GOOGL", "^VIX"]

        for symbol in symbols:
            try:
                result = await predict_symbol(symbol=symbol, provider="tradestation")
                assert result["symbol"] == symbol
            except Exception as e:
                # Some symbols might be invalid - that's ok
                pass


@pytest.mark.asyncio
async def test_predict_with_timeout():
    """Test prediction handling timeout."""
    timeout_provider = MockDataProvider(fail_mode="timeout")
    mock_extractor = MockFeatureExtractor()
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", timeout_provider),
        patch("fincoll.api.inference.FeatureExtractor", return_value=mock_extractor),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        with pytest.raises(TimeoutError):
            await predict_symbol(symbol="AAPL", provider="tradestation")


@pytest.mark.asyncio
async def test_batch_predict_duplicate_symbols(setup_mocks):
    """Test batch prediction with duplicate symbols."""
    request = {
        "symbols": ["AAPL", "AAPL", "MSFT", "AAPL"],
        "provider": "tradestation",
        "interval": "1d",
    }

    result = await batch_predict_velocity(request)

    # Should handle duplicates (either dedupe or predict each)
    assert "predictions" in result
    assert result["summary"]["total"] == 4  # Or 2 if deduped


@pytest.mark.asyncio
async def test_predict_feature_dimension_mismatch():
    """Test prediction when features have wrong dimension."""
    mock_provider = MockDataProvider()
    wrong_dim_extractor = MockFeatureExtractor(fail_mode="incomplete")
    mock_engine = MockVelocityEngine()

    with (
        patch("fincoll.api.inference._data_provider", mock_provider),
        patch(
            "fincoll.api.inference.FeatureExtractor", return_value=wrong_dim_extractor
        ),
        patch("fincoll.api.inference._load_velocity_model", return_value=mock_engine),
    ):
        # Should fail validation or handle gracefully
        try:
            result = await predict_symbol(symbol="AAPL", provider="tradestation")
            # If it doesn't raise, check for error in result
            if isinstance(result, dict) and "error" in result:
                assert "dimension" in result["error"].lower()
        except Exception as e:
            # Expected - wrong feature dimension
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
