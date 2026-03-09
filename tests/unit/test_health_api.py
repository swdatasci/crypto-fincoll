#!/usr/bin/env python3
"""
Comprehensive Health API Tests

This module provides comprehensive test coverage for the /health endpoint in server.py.

Coverage areas:
1. Basic health check response structure
2. Component health checks (database, model, services)
3. Data source availability (TradeStation, AlphaVantage, SenVec)
4. Configuration status
5. Degraded state handling
6. Service dependency checks
7. Version information
8. Timestamp accuracy
9. Error handling when services are unavailable
10. Health check during various system states

Target: 80%+ coverage for health endpoint
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, ".")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_credentials_dir(tmp_path):
    """Create a temporary credentials directory."""
    creds_dir = tmp_path / "credentials"
    creds_dir.mkdir()
    return creds_dir


@pytest.fixture
def mock_tradestation_token(mock_credentials_dir):
    """Create mock TradeStation token file."""
    token_file = mock_credentials_dir / ".tradestation_token.json"
    token_file.write_text('{"access_token": "test_token", "expires_at": 9999999999}')
    return token_file


@pytest.fixture
def mock_alphavantage_creds(mock_credentials_dir):
    """Create mock AlphaVantage credentials file."""
    creds_file = mock_credentials_dir / ".alpha_vantage_credentials.json"
    creds_file.write_text('{"api_key": "test_key"}')
    return creds_file


# =============================================================================
# Basic Health Check Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_success(
    mock_credentials_dir, mock_tradestation_token, mock_alphavantage_creds
):
    """Test successful health check with all services available."""
    with (
        patch("fincoll.server.CREDENTIALS_DIR", mock_credentials_dir),
        patch.dict(
            "os.environ",
            {"SENVEC_ENABLED": "true", "SENVEC_API_URL": "http://10.32.3.27:8000"},
        ),
    ):
        from fincoll.server import health_check

        result = await health_check()

        # Verify response structure
        assert "status" in result
        assert result["status"] == "healthy"

        assert "timestamp" in result
        assert "version" in result
        assert "data_sources" in result

        # Verify timestamp is recent and valid ISO format
        timestamp_str = result["timestamp"]
        timestamp = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        assert (now - timestamp).total_seconds() < 5, "Timestamp should be recent"


@pytest.mark.asyncio
async def test_health_check_response_structure():
    """Test health check response has required fields."""
    from fincoll.server import health_check

    result = await health_check()

    required_fields = ["status", "timestamp", "version", "data_sources"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # Verify types
    assert isinstance(result["status"], str)
    assert isinstance(result["timestamp"], str)
    assert isinstance(result["version"], str)
    assert isinstance(result["data_sources"], dict)


@pytest.mark.asyncio
async def test_health_check_version_format():
    """Test health check version format."""
    from fincoll.server import health_check

    result = await health_check()

    version = result["version"]
    # Should be semantic version format (e.g., "0.1.0")
    assert isinstance(version, str)
    assert len(version) > 0
    # Basic version format check
    parts = version.split(".")
    assert len(parts) >= 2, "Version should have at least major.minor"


# =============================================================================
# Data Source Availability Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_all_sources_available(
    mock_credentials_dir, mock_tradestation_token, mock_alphavantage_creds
):
    """Test health check when all data sources are available."""
    with (
        patch("fincoll.server.CREDENTIALS_DIR", mock_credentials_dir),
        patch.dict("os.environ", {"SENVEC_ENABLED": "true"}),
    ):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]

        # TradeStation should be available
        assert "tradestation" in sources
        ts = sources["tradestation"]
        assert ts["available"] is True
        assert ts["status"] == "configured"

        # AlphaVantage should be available
        assert "alpha_vantage" in sources
        av = sources["alpha_vantage"]
        assert av["available"] is True
        assert av["status"] == "configured"

        # SenVec should be enabled
        assert "senvec" in sources
        sv = sources["senvec"]
        assert sv["available"] is True
        assert sv["status"] == "enabled"


@pytest.mark.asyncio
async def test_health_check_missing_tradestation(
    mock_credentials_dir, mock_alphavantage_creds
):
    """Test health check when TradeStation credentials are missing."""
    # Don't create TradeStation token
    with patch("fincoll.server.CREDENTIALS_DIR", mock_credentials_dir):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]
        ts = sources["tradestation"]

        assert ts["available"] is False
        assert ts["status"] == "not_configured"


@pytest.mark.asyncio
async def test_health_check_missing_alphavantage(
    mock_credentials_dir, mock_tradestation_token
):
    """Test health check when AlphaVantage credentials are missing."""
    # Don't create AlphaVantage credentials
    with patch("fincoll.server.CREDENTIALS_DIR", mock_credentials_dir):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]
        av = sources["alpha_vantage"]

        assert av["available"] is False
        assert av["status"] == "not_configured"


@pytest.mark.asyncio
async def test_health_check_senvec_disabled():
    """Test health check when SenVec is disabled."""
    with patch.dict("os.environ", {"SENVEC_ENABLED": "false"}):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]
        sv = sources["senvec"]

        assert sv["available"] is False
        assert sv["status"] == "disabled"


@pytest.mark.asyncio
async def test_health_check_senvec_enabled():
    """Test health check when SenVec is enabled."""
    test_url = "http://test-senvec:18000"

    with patch.dict(
        "os.environ", {"SENVEC_ENABLED": "true", "SENVEC_API_URL": test_url}
    ):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]
        sv = sources["senvec"]

        assert sv["available"] is True
        assert sv["status"] == "enabled"
        assert sv["url"] == test_url


@pytest.mark.asyncio
async def test_health_check_senvec_default_url():
    """Test health check uses default SenVec URL."""
    with patch.dict("os.environ", {"SENVEC_ENABLED": "true"}, clear=True):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]
        sv = sources["senvec"]

        assert "url" in sv
        # Should have default URL
        assert sv["url"] == "http://10.32.3.27:8000"


# =============================================================================
# Degraded State Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_no_data_sources(mock_credentials_dir):
    """Test health check when no data sources are configured."""
    # Empty credentials dir, SenVec disabled
    with (
        patch("fincoll.server.CREDENTIALS_DIR", mock_credentials_dir),
        patch.dict("os.environ", {"SENVEC_ENABLED": "false"}),
    ):
        from fincoll.server import health_check

        result = await health_check()

        # Should still return healthy (degraded state)
        assert result["status"] == "healthy"

        sources = result["data_sources"]

        # All sources should be unavailable
        for source in sources.values():
            assert source["available"] is False


@pytest.mark.asyncio
async def test_health_check_partial_availability(
    mock_credentials_dir, mock_tradestation_token
):
    """Test health check with partial data source availability."""
    # Only TradeStation available
    with (
        patch("fincoll.server.CREDENTIALS_DIR", mock_credentials_dir),
        patch.dict("os.environ", {"SENVEC_ENABLED": "false"}),
    ):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]

        # TradeStation available
        assert sources["tradestation"]["available"] is True

        # Others not available
        assert sources["alpha_vantage"]["available"] is False
        assert sources["senvec"]["available"] is False


# =============================================================================
# Timestamp and Version Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_timestamp_format():
    """Test health check timestamp is in ISO format."""
    from fincoll.server import health_check

    result = await health_check()

    timestamp_str = result["timestamp"]

    # Should be valid ISO format
    try:
        dt = datetime.fromisoformat(timestamp_str)
        assert isinstance(dt, datetime)
    except ValueError:
        pytest.fail(f"Timestamp not in ISO format: {timestamp_str}")


@pytest.mark.asyncio
async def test_health_check_timestamp_accuracy():
    """Test health check timestamp is accurate."""
    from fincoll.server import health_check

    before = datetime.now()
    result = await health_check()
    after = datetime.now()

    timestamp = datetime.fromisoformat(result["timestamp"])

    # Timestamp should be between before and after
    assert before <= timestamp <= after


@pytest.mark.asyncio
async def test_health_check_multiple_calls_different_timestamps():
    """Test health check returns different timestamps on multiple calls."""
    from fincoll.server import health_check
    import asyncio

    result1 = await health_check()
    await asyncio.sleep(0.01)  # Small delay
    result2 = await health_check()

    timestamp1 = datetime.fromisoformat(result1["timestamp"])
    timestamp2 = datetime.fromisoformat(result2["timestamp"])

    # Timestamps should be different
    assert timestamp1 != timestamp2
    assert timestamp2 > timestamp1


# =============================================================================
# Data Source Structure Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_data_sources_structure():
    """Test data sources have required structure."""
    from fincoll.server import health_check

    result = await health_check()

    sources = result["data_sources"]

    required_sources = ["tradestation", "alpha_vantage", "senvec"]
    for source_name in required_sources:
        assert source_name in sources, f"Missing data source: {source_name}"

        source = sources[source_name]
        assert "available" in source
        assert "status" in source
        assert isinstance(source["available"], bool)
        assert isinstance(source["status"], str)


@pytest.mark.asyncio
async def test_health_check_senvec_includes_url():
    """Test SenVec data source includes URL when enabled."""
    with patch.dict("os.environ", {"SENVEC_ENABLED": "true"}):
        from fincoll.server import health_check

        result = await health_check()

        sources = result["data_sources"]
        sv = sources["senvec"]

        assert "url" in sv
        assert isinstance(sv["url"], str)
        assert len(sv["url"]) > 0


# =============================================================================
# Status Code Tests (would require TestClient)
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_returns_200():
    """Test health check should return 200 OK (tested via endpoint directly)."""
    # This would typically be tested with TestClient
    # Here we just verify the function doesn't raise exceptions
    from fincoll.server import health_check

    result = await health_check()
    assert result is not None
    assert result["status"] == "healthy"


# =============================================================================
# Concurrent Access Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_concurrent_calls():
    """Test health check handles concurrent calls correctly."""
    from fincoll.server import health_check
    import asyncio

    # Make 10 concurrent health check calls
    tasks = [health_check() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # All should succeed
    assert len(results) == 10

    # All should have status "healthy"
    for result in results:
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "version" in result


# =============================================================================
# Environment Variable Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_senvec_enabled_variations():
    """Test SenVec enabled flag with various values."""
    test_cases = [
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("false", False),
        ("FALSE", False),
        ("False", False),
        ("", False),  # Empty defaults to enabled, but empty string is falsy
        ("1", False),  # Not "true"
        ("0", False),
    ]

    for env_value, expected_available in test_cases:
        with patch.dict("os.environ", {"SENVEC_ENABLED": env_value}):
            from fincoll.server import health_check

            result = await health_check()
            sources = result["data_sources"]
            sv = sources["senvec"]

            assert sv["available"] == expected_available, (
                f"SENVEC_ENABLED={env_value} should result in available={expected_available}"
            )


# =============================================================================
# File System Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_invalid_credentials_dir():
    """Test health check when credentials directory doesn't exist."""
    non_existent_path = Path("/nonexistent/credentials")

    with patch("fincoll.server.CREDENTIALS_DIR", non_existent_path):
        from fincoll.server import health_check

        # Should not crash, just report unavailable
        result = await health_check()

        assert result["status"] == "healthy"
        sources = result["data_sources"]

        # All file-based sources should be unavailable
        assert sources["tradestation"]["available"] is False
        assert sources["alpha_vantage"]["available"] is False


# =============================================================================
# Integration-like Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_realistic_production_scenario(
    mock_credentials_dir, mock_tradestation_token, mock_alphavantage_creds
):
    """Test health check in a realistic production scenario."""
    with (
        patch("fincoll.server.CREDENTIALS_DIR", mock_credentials_dir),
        patch.dict(
            "os.environ",
            {"SENVEC_ENABLED": "true", "SENVEC_API_URL": "http://10.32.3.27:18000"},
        ),
    ):
        from fincoll.server import health_check

        result = await health_check()

        # Should be healthy
        assert result["status"] == "healthy"

        # Should have version
        assert result["version"] is not None

        # All sources should be available
        sources = result["data_sources"]
        assert sources["tradestation"]["available"] is True
        assert sources["alpha_vantage"]["available"] is True
        assert sources["senvec"]["available"] is True

        # SenVec should have correct URL
        assert sources["senvec"]["url"] == "http://10.32.3.27:18000"


@pytest.mark.asyncio
async def test_health_check_minimal_configuration():
    """Test health check with minimal configuration."""
    from fincoll.server import health_check

    # No special mocking - just use defaults
    result = await health_check()

    # Should still return a valid response
    assert isinstance(result, dict)
    assert "status" in result
    assert "timestamp" in result
    assert "version" in result
    assert "data_sources" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
