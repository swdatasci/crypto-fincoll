"""Pytest configuration for E2E tests using supersystem mock servers."""


import pytest
import requests
import time
import logging

logger = logging.getLogger(__name__)

# Mock server configuration
MOCK_SERVERS = {
    'alpaca': {
        'port': 7879,
        'base_url': 'http://localhost:7879',
        'health_endpoint': '/health'
    },
    'public': {
        'port': 7880,
        'base_url': 'http://localhost:7880',
        'health_endpoint': '/health'
    },
    'tradestation': {
        'port': 7878,
        'base_url': 'http://localhost:7878',
        'health_endpoint': '/health'
    }
}


def check_mock_server(name: str, config: dict, retries: int = 3) -> bool:
    """Check if a mock server is running and healthy."""
    url = f"{config['base_url']}{config['health_endpoint']}"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ {name} mock server is healthy on port {config['port']}")
                return True
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                logger.warning(f"⚠️  {name} mock not responding (attempt {attempt + 1}/{retries}), retrying...")
                time.sleep(1)
            else:
                logger.error(f"❌ {name} mock server not available: {e}")
                return False

    return False


@pytest.fixture(scope="session")
def mock_servers():
    """
    Verify mock servers are running.

    NOTE: This fixture does NOT start the servers - they should be started
    manually with: /home/rford/caelum/caelum-supersystem/start-mock-apis.sh

    This fixture only verifies they're running and healthy.
    """
    logger.info("=" * 80)
    logger.info("Checking supersystem mock servers...")
    logger.info("=" * 80)

    available_servers = {}

    for name, config in MOCK_SERVERS.items():
        if check_mock_server(name, config):
            available_servers[name] = config

    if not available_servers:
        pytest.fail(
            "No mock servers available. Start them with:\n"
            "  bash /home/rford/caelum/caelum-supersystem/start-mock-apis.sh"
        )

    logger.info(f"Available mock servers: {list(available_servers.keys())}")

    yield available_servers

    logger.info("Mock server check complete")


@pytest.fixture
def alpaca_mock_url(mock_servers):
    """Get Alpaca mock server URL."""
    if 'alpaca' not in mock_servers:
        pytest.skip("Alpaca mock server not available")
    return mock_servers['alpaca']['base_url']


@pytest.fixture
def public_mock_url(mock_servers):
    """Get Public.com mock server URL."""
    if 'public' not in mock_servers:
        pytest.skip("Public mock server not available")
    return mock_servers['public']['base_url']


@pytest.fixture
def tradestation_mock_url(mock_servers):
    """Get TradeStation mock server URL."""
    if 'tradestation' not in mock_servers:
        pytest.skip("TradeStation mock server not available")
    return mock_servers['tradestation']['base_url']
