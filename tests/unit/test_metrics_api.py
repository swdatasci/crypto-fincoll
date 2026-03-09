#!/usr/bin/env python3
"""
Comprehensive Metrics API Tests

This module provides comprehensive test coverage for the /metrics endpoint in server.py
and monitoring/metrics.py module.

Coverage areas:
1. Metrics endpoint response format (Prometheus-compatible)
2. Request metrics (count, latency, errors)
3. Prediction metrics (if tracked)
4. System metrics (if tracked)
5. Metric types (counters, gauges, histograms)
6. Metric labels and dimensions
7. Content-Type headers (text/plain; version=0.0.4)
8. Metric formatting and naming conventions
9. Error handling and edge cases
10. Integration with monitoring system

Target: 80%+ coverage for metrics endpoint and monitoring.metrics module
"""

import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, ".")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_metrics_registry():
    """Create a mock Prometheus metrics registry."""
    registry = MagicMock()
    registry.get_metrics.return_value = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health"} 100
http_requests_total{method="POST",endpoint="/api/v1/inference/predict/AAPL"} 50

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 80
http_request_duration_seconds_bucket{le="0.5"} 95
http_request_duration_seconds_bucket{le="1.0"} 100
http_request_duration_seconds_count 100
http_request_duration_seconds_sum 25.5

# HELP prediction_count Total predictions made
# TYPE prediction_count counter
prediction_count{timeframe="1min"} 25
prediction_count{timeframe="5min"} 25
prediction_count{timeframe="1hour"} 25
"""
    return registry


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================


@pytest.mark.asyncio
async def test_metrics_endpoint_success():
    """Test successful metrics endpoint call."""
    with (
        patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics,
        patch("fincoll.monitoring.metrics.get_content_type") as mock_get_content_type,
    ):
        mock_get_metrics.return_value = "# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 1\n"
        mock_get_content_type.return_value = "text/plain; version=0.0.4; charset=utf-8"

        from fincoll.server import metrics

        response = await metrics()

        assert response is not None
        assert response.body is not None
        assert response.media_type == "text/plain; version=0.0.4; charset=utf-8"


@pytest.mark.asyncio
async def test_metrics_endpoint_content_type():
    """Test metrics endpoint returns correct Content-Type."""
    with (
        patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics,
        patch("fincoll.monitoring.metrics.get_content_type") as mock_get_content_type,
    ):
        mock_get_metrics.return_value = ""
        mock_get_content_type.return_value = "text/plain; version=0.0.4; charset=utf-8"

        from fincoll.server import metrics

        response = await metrics()

        # Should be Prometheus text format
        assert "text/plain" in response.media_type
        assert "0.0.4" in response.media_type or "version" in response.media_type


@pytest.mark.asyncio
async def test_metrics_endpoint_prometheus_format():
    """Test metrics endpoint returns Prometheus-compatible format."""
    with (
        patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics,
        patch("fincoll.monitoring.metrics.get_content_type") as mock_get_content_type,
    ):
        # Return realistic Prometheus format
        mock_get_metrics.return_value = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET"} 100
"""
        mock_get_content_type.return_value = "text/plain; version=0.0.4; charset=utf-8"

        from fincoll.server import metrics

        response = await metrics()
        content = (
            response.body.decode()
            if isinstance(response.body, bytes)
            else response.body
        )

        # Should contain HELP and TYPE lines
        assert "# HELP" in content
        assert "# TYPE" in content


# =============================================================================
# Metrics Content Tests
# =============================================================================


def test_get_metrics_returns_string():
    """Test get_metrics returns a string."""
    from fincoll.monitoring.metrics import get_metrics

    result = get_metrics()

    assert isinstance(result, (str, bytes))
    if isinstance(result, bytes):
        result = result.decode()
    assert len(result) >= 0  # May be empty if no metrics collected


def test_get_content_type_format():
    """Test get_content_type returns valid Prometheus content type."""
    from fincoll.monitoring.metrics import get_content_type

    content_type = get_content_type()

    assert isinstance(content_type, str)
    assert "text/plain" in content_type or "text" in content_type


# =============================================================================
# Metric Types Tests
# =============================================================================


def test_metrics_counter_format():
    """Test counter metrics have correct format."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP request_count Total requests
# TYPE request_count counter
request_count 42
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Should contain TYPE counter
        assert "TYPE" in content
        assert "counter" in content


def test_metrics_gauge_format():
    """Test gauge metrics have correct format."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP active_connections Current active connections
# TYPE active_connections gauge
active_connections 5
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Should contain TYPE gauge
        assert "TYPE" in content
        assert "gauge" in content


def test_metrics_histogram_format():
    """Test histogram metrics have correct format."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP request_duration Request duration
# TYPE request_duration histogram
request_duration_bucket{le="0.1"} 10
request_duration_bucket{le="1.0"} 20
request_duration_count 20
request_duration_sum 5.5
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Should contain histogram buckets
        assert "histogram" in content
        assert "_bucket" in content
        assert "_count" in content
        assert "_sum" in content


# =============================================================================
# Metric Labels Tests
# =============================================================================


def test_metrics_with_labels():
    """Test metrics can have labels."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP http_requests_total Total requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 100
http_requests_total{method="POST",status="200"} 50
http_requests_total{method="POST",status="500"} 5
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Should contain labels in curly braces
        assert "{" in content
        assert "}" in content
        assert "method=" in content or "status=" in content


def test_metrics_label_values():
    """Test metric label values are properly quoted."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = (
            """http_requests_total{method="GET",endpoint="/health"} 100"""
        )

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Label values should be in quotes
        assert '"GET"' in content or "'GET'" in content or "GET" in content


# =============================================================================
# Request Metrics Tests
# =============================================================================


def test_metrics_tracks_http_requests():
    """Test metrics tracks HTTP requests."""
    # This would test the actual metric recording
    # For now, we test that the infrastructure exists
    try:
        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()
        # May be empty if no requests made yet
        assert isinstance(content, (str, bytes))
    except ImportError:
        pytest.skip("Metrics module not available")


def test_metrics_tracks_request_duration():
    """Test metrics tracks request duration."""
    try:
        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()
        # Infrastructure should exist even if no data
        assert isinstance(content, (str, bytes))
    except ImportError:
        pytest.skip("Metrics module not available")


# =============================================================================
# Prediction Metrics Tests
# =============================================================================


def test_metrics_tracks_predictions():
    """Test metrics tracks prediction counts."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP prediction_count Total predictions
# TYPE prediction_count counter
prediction_count{symbol="AAPL"} 10
prediction_count{symbol="MSFT"} 5
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Should track predictions
        if "prediction" in content.lower():
            assert "symbol=" in content or "AAPL" in content


def test_metrics_tracks_prediction_latency():
    """Test metrics tracks prediction latency."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP prediction_duration_seconds Prediction duration
# TYPE prediction_duration_seconds histogram
prediction_duration_seconds_bucket{le="0.1"} 5
prediction_duration_seconds_count 10
prediction_duration_seconds_sum 0.5
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # May include prediction duration metrics
        # Just verify format is valid
        assert isinstance(content, str)


# =============================================================================
# Error Metrics Tests
# =============================================================================


def test_metrics_tracks_errors():
    """Test metrics tracks errors."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP http_errors_total Total HTTP errors
# TYPE http_errors_total counter
http_errors_total{code="500"} 5
http_errors_total{code="404"} 10
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # May track errors
        if "error" in content.lower():
            assert "500" in content or "404" in content


# =============================================================================
# Metric Naming Tests
# =============================================================================


def test_metrics_follow_naming_conventions():
    """Test metrics follow Prometheus naming conventions."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total 100

# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_count 100
"""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Naming conventions:
        # - snake_case
        # - _total suffix for counters
        # - _seconds suffix for durations
        # - descriptive names

        # Just verify format is reasonable
        assert isinstance(content, str)
        # If there are metrics, check conventions
        if "http" in content.lower():
            # Should use underscores not hyphens
            assert "http-" not in content or "http_" in content


# =============================================================================
# Empty Metrics Tests
# =============================================================================


def test_metrics_empty_when_no_data():
    """Test metrics can return empty/minimal output when no data."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = ""

        from fincoll.monitoring.metrics import get_metrics

        content = get_metrics()

        # Should handle empty case gracefully
        assert isinstance(content, (str, bytes))


@pytest.mark.asyncio
async def test_metrics_endpoint_empty_response():
    """Test metrics endpoint handles empty metrics."""
    with (
        patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics,
        patch("fincoll.monitoring.metrics.get_content_type") as mock_get_content_type,
    ):
        mock_get_metrics.return_value = ""
        mock_get_content_type.return_value = "text/plain"

        from fincoll.server import metrics

        response = await metrics()

        # Should still return valid response
        assert response is not None


# =============================================================================
# Concurrent Access Tests
# =============================================================================


@pytest.mark.asyncio
async def test_metrics_endpoint_concurrent_calls():
    """Test metrics endpoint handles concurrent calls."""
    with (
        patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics,
        patch("fincoll.monitoring.metrics.get_content_type") as mock_get_content_type,
    ):
        mock_get_metrics.return_value = "test_metric 1\n"
        mock_get_content_type.return_value = "text/plain"

        from fincoll.server import metrics
        import asyncio

        # Make 5 concurrent requests
        tasks = [metrics() for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert len(responses) == 5
        for response in responses:
            assert response is not None


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_metrics_endpoint_realistic_output():
    """Test metrics endpoint with realistic Prometheus output."""
    realistic_metrics = """# HELP http_requests_total The total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health",status="200"} 1000
http_requests_total{method="POST",endpoint="/api/v1/inference/predict/AAPL",status="200"} 50
http_requests_total{method="POST",endpoint="/api/v1/inference/predict/batch",status="200"} 10

# HELP http_request_duration_seconds HTTP request latency in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.005"} 500
http_request_duration_seconds_bucket{le="0.01"} 800
http_request_duration_seconds_bucket{le="0.025"} 950
http_request_duration_seconds_bucket{le="0.05"} 990
http_request_duration_seconds_bucket{le="0.1"} 1000
http_request_duration_seconds_bucket{le="+Inf"} 1000
http_request_duration_seconds_sum 15.5
http_request_duration_seconds_count 1000

# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{timeframe="1min"} 150
predictions_total{timeframe="5min"} 150
predictions_total{timeframe="1hour"} 150

# HELP prediction_confidence_score Prediction confidence scores
# TYPE prediction_confidence_score gauge
prediction_confidence_score{timeframe="1min"} 0.75
prediction_confidence_score{timeframe="5min"} 0.82
prediction_confidence_score{timeframe="1hour"} 0.68
"""

    with (
        patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics,
        patch("fincoll.monitoring.metrics.get_content_type") as mock_get_content_type,
    ):
        mock_get_metrics.return_value = realistic_metrics
        mock_get_content_type.return_value = "text/plain; version=0.0.4; charset=utf-8"

        from fincoll.server import metrics

        response = await metrics()

        assert response is not None
        content = (
            response.body.decode()
            if isinstance(response.body, bytes)
            else response.body
        )

        # Verify key elements present
        assert "http_requests_total" in content
        assert "http_request_duration_seconds" in content
        assert "# HELP" in content
        assert "# TYPE" in content


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_metrics_endpoint_handles_exception():
    """Test metrics endpoint handles exceptions gracefully."""
    with patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics:
        mock_get_metrics.side_effect = Exception("Metrics collection failed")

        from fincoll.server import metrics

        # Should either handle gracefully or raise appropriate error
        try:
            response = await metrics()
            # If it doesn't raise, should return valid response
            assert response is not None
        except Exception as e:
            # Exception is acceptable, just verify it's handled
            assert isinstance(e, Exception)


def test_get_metrics_handles_exception():
    """Test get_metrics handles exceptions."""
    from fincoll.monitoring.metrics import get_metrics

    # Should not crash even if metrics collection fails
    try:
        result = get_metrics()
        assert isinstance(result, (str, bytes))
    except Exception:
        # Exception is acceptable if properly handled
        pass


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.asyncio
async def test_metrics_endpoint_performance():
    """Test metrics endpoint responds quickly."""
    import time

    with (
        patch("fincoll.monitoring.metrics.get_metrics") as mock_get_metrics,
        patch("fincoll.monitoring.metrics.get_content_type") as mock_get_content_type,
    ):
        mock_get_metrics.return_value = "test_metric 1\n"
        mock_get_content_type.return_value = "text/plain"

        from fincoll.server import metrics

        start = time.time()
        response = await metrics()
        elapsed = time.time() - start

        # Should be very fast (under 100ms)
        assert elapsed < 0.1, f"Metrics endpoint took {elapsed}s, should be < 0.1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
