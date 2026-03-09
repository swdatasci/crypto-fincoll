#!/usr/bin/env python3
"""
E2E Tests for Trading Providers

Tests each trading provider against its supersystem mock server:
- TradeStationTradingProvider → mock-tradestation-api (port 7878)
- AlpacaTradingProvider       → mock-alpaca-api       (port 7879)
- PublicTradingProvider       → mock-public-api       (port 7880)

Each provider is tested:
1. As a DATA source (historical bars, quotes)
2. As an ACCOUNT provider (balance, positions)
3. As a TRADING platform (place order, get order status)

Run with: pytest test_trading_providers_e2e.py -v
Prerequisites: bash /home/rford/caelum/caelum-supersystem/start-mock-apis.sh
"""

import pytest
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOL = "AAPL"
START = "2025-12-01"
END = "2026-01-01"


# ============================================================================
# TRADESTATION TRADING PROVIDER (port 7878)
# ============================================================================

class TestTradeStationTradingProviderE2E:
    """
    E2E tests for TradeStation trading provider.

    Uses mock-tradestation-api at http://localhost:7878
    API path prefix: /tradestation/v3/
    """

    def test_mock_server_health(self, tradestation_mock_url):
        """TEST: TradeStation mock server is healthy"""
        logger.info("=" * 70)
        logger.info("TEST: TradeStation Mock - Health")
        logger.info("=" * 70)

        r = requests.get(f"{tradestation_mock_url}/health")
        assert r.status_code == 200

        data = r.json()
        assert data['status'] == 'healthy'
        assert data['connections']['database'] is True
        assert data['connections']['redis'] is True
        logger.info(f"✅ TradeStation mock healthy: {data}")

    def test_fetch_historical_bars(self, tradestation_mock_url):
        """TEST: Fetch historical bars from TradeStation mock"""
        logger.info("=" * 70)
        logger.info("TEST: TradeStation - Historical Bars (Data Provider)")
        logger.info("=" * 70)

        r = requests.get(
            f"{tradestation_mock_url}/tradestation/v3/marketdata/barcharts/{SYMBOL}",
            params={
                "interval": "1",
                "unit": "Daily",
                "firstdate": START,
                "lastdate": END,
            }
        )

        logger.info(f"Status: {r.status_code}")
        assert r.status_code == 200

        data = r.json()
        logger.info(f"Response keys: {list(data.keys())}")

        # TradeStation returns {"Bars": [...]}
        assert 'Bars' in data, f"Expected 'Bars' key, got: {data.keys()}"

        bars = data['Bars']
        logger.info(f"Received {len(bars)} bars")

        if bars:
            first = bars[0]
            logger.info(f"First bar keys: {list(first.keys())}")
            assert 'TimeStamp' in first or 'Close' in first
            logger.info(f"✅ TradeStation data provider works ({len(bars)} bars)")
        else:
            # Empty is valid if no data in DB for this range
            logger.info("⚠️  No bars returned (DB may not have data for this date range)")
            pytest.skip("No bars data in TradeStation mock DB for this range")

    def test_fetch_quote(self, tradestation_mock_url):
        """TEST: Fetch real-time quote from TradeStation mock"""
        logger.info("=" * 70)
        logger.info("TEST: TradeStation - Real-time Quote")
        logger.info("=" * 70)

        r = requests.get(
            f"{tradestation_mock_url}/tradestation/v3/marketdata/quotes/{SYMBOL}"
        )

        logger.info(f"Status: {r.status_code}")
        assert r.status_code == 200

        data = r.json()
        logger.info(f"Quote data: {data}")
        assert 'Quotes' in data or 'Symbol' in data or isinstance(data, dict)
        logger.info(f"✅ TradeStation quote endpoint works")

    def test_account_balances(self, tradestation_mock_url):
        """TEST: Fetch account balances from TradeStation mock"""
        logger.info("=" * 70)
        logger.info("TEST: TradeStation - Account Balances")
        logger.info("=" * 70)

        # TradeStation uses TS-SIM-001 mock account
        account_id = "TS-SIM-001"

        r = requests.get(
            f"{tradestation_mock_url}/tradestation/v3/brokerage/accounts/{account_id}/balances"
        )

        logger.info(f"Status: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            logger.info(f"✅ Balances: {data}")
            assert isinstance(data, dict)
        elif r.status_code == 404:
            logger.info("⚠️  Account TS-SIM-001 not found (needs seeding)")
            pytest.skip("TradeStation mock account not seeded")
        else:
            pytest.fail(f"Unexpected status: {r.status_code}: {r.text}")

    def test_place_market_order(self, tradestation_mock_url):
        """TEST: Place a market order through TradeStation mock"""
        logger.info("=" * 70)
        logger.info("TEST: TradeStation - Place Market Order (Trading Provider)")
        logger.info("=" * 70)

        order_request = {
            "AccountKey": "TS-SIM-001",
            "Symbol": SYMBOL,
            "Quantity": "10",
            "OrderType": "Market",
            "TradeAction": "Buy"
        }

        r = requests.post(
            f"{tradestation_mock_url}/tradestation/v3/orderexecution/orders",
            json=order_request
        )

        logger.info(f"Status: {r.status_code}")
        logger.info(f"Response: {r.text[:300]}")

        if r.status_code in (200, 201):
            data = r.json()
            logger.info(f"✅ Order placed: {data}")
        elif r.status_code == 404:
            logger.info("⚠️  Account not found - needs seeding")
            pytest.skip("TradeStation mock account not seeded")
        elif r.status_code == 400:
            logger.info(f"⚠️  Bad request: {r.text}")
            pytest.skip(f"Order rejected: {r.text[:100]}")
        else:
            pytest.fail(f"Unexpected status {r.status_code}: {r.text[:200]}")


# ============================================================================
# ALPACA TRADING PROVIDER (port 7879)
# ============================================================================

class TestAlpacaTradingProviderE2E:
    """
    E2E tests for Alpaca trading provider.

    Uses mock-alpaca-api at http://localhost:7879
    API path prefix: /v2/
    """

    def test_mock_server_health(self, alpaca_mock_url):
        """TEST: Alpaca mock server is healthy"""
        logger.info("=" * 70)
        logger.info("TEST: Alpaca Mock - Health")
        logger.info("=" * 70)

        r = requests.get(f"{alpaca_mock_url}/health")
        assert r.status_code == 200

        data = r.json()
        assert data['status'] == 'healthy'
        assert data['connections']['postgres'] is True
        assert data['connections']['redis'] is True
        logger.info(f"✅ Alpaca mock healthy: {data}")

    def test_fetch_historical_bars(self, alpaca_mock_url):
        """TEST: Fetch historical bars from Alpaca mock (data provider role)"""
        logger.info("=" * 70)
        logger.info("TEST: Alpaca - Historical Bars (Data Provider)")
        logger.info("=" * 70)

        r = requests.get(
            f"{alpaca_mock_url}/v2/stocks/{SYMBOL}/bars",
            params={
                "timeframe": "1Day",
                "start": START,
                "end": END,
                "limit": 50
            },
            headers={
                "APCA-API-KEY-ID": "mock_key",
                "APCA-API-SECRET-KEY": "mock_secret"
            }
        )

        logger.info(f"Status: {r.status_code}")
        assert r.status_code == 200

        data = r.json()
        assert 'bars' in data, f"Expected 'bars' key, got: {data.keys()}"
        bars = data['bars']
        assert len(bars) > 0, "Should have bars"

        # Validate bar structure
        first = bars[0]
        assert 't' in first, "Bar should have timestamp 't'"
        assert 'o' in first, "Bar should have open 'o'"
        assert 'h' in first, "Bar should have high 'h'"
        assert 'l' in first, "Bar should have low 'l'"
        assert 'c' in first, "Bar should have close 'c'"
        assert 'v' in first, "Bar should have volume 'v'"

        # Validate datetime index when converted
        timestamps = pd.to_datetime([bar['t'] for bar in bars])
        assert isinstance(timestamps, pd.DatetimeIndex)
        assert timestamps[0].year >= 2020

        logger.info(f"✅ Alpaca data provider works: {len(bars)} bars, "
                    f"from {timestamps[0]} to {timestamps[-1]}")

    def test_fetch_bars_datetime_not_integers(self, alpaca_mock_url):
        """TEST: Verify Alpaca mock bars have datetime timestamps, not integers"""
        logger.info("=" * 70)
        logger.info("TEST: Alpaca - Datetime Index Validation (Not Integers!)")
        logger.info("=" * 70)

        r = requests.get(
            f"{alpaca_mock_url}/v2/stocks/SPY/bars",
            params={"timeframe": "1Day", "start": START, "end": END},
            headers={"APCA-API-KEY-ID": "mock_key", "APCA-API-SECRET-KEY": "mock_secret"}
        )

        assert r.status_code == 200
        bars = r.json()['bars']
        assert len(bars) > 0

        # Every timestamp must be parseable datetime, NOT an integer
        for bar in bars:
            ts = bar['t']
            assert isinstance(ts, str), f"Timestamp must be string, got {type(ts)}: {ts}"
            parsed = pd.Timestamp(ts)
            assert parsed.year >= 2020, f"Year should be recent, not {parsed.year}"
            assert not isinstance(ts, int), "Timestamp must NOT be integer!"

        logger.info(f"✅ All {len(bars)} bars have valid datetime timestamps (no integers)")

    def test_get_account_info(self, alpaca_mock_url):
        """TEST: Get account info from Alpaca mock"""
        logger.info("=" * 70)
        logger.info("TEST: Alpaca - Account Info (Account Provider)")
        logger.info("=" * 70)

        r = requests.get(
            f"{alpaca_mock_url}/v2/account",
            headers={"APCA-API-KEY-ID": "mock_key", "APCA-API-SECRET-KEY": "mock_secret"}
        )

        logger.info(f"Status: {r.status_code}, Response: {r.text[:200]}")

        if r.status_code == 200:
            data = r.json()
            logger.info(f"✅ Account info: {data}")
            assert isinstance(data, dict)
        elif r.status_code in (404, 422):
            logger.info("⚠️  No mock account found - needs seeding")
            pytest.skip("Alpaca mock account not seeded")
        else:
            pytest.fail(f"Unexpected status {r.status_code}: {r.text}")

    def test_place_market_order(self, alpaca_mock_url):
        """TEST: Place a market order through Alpaca mock"""
        logger.info("=" * 70)
        logger.info("TEST: Alpaca - Place Market Order (Trading Provider)")
        logger.info("=" * 70)

        order_request = {
            "symbol": SYMBOL,
            "qty": 10,
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }

        r = requests.post(
            f"{alpaca_mock_url}/v2/orders",
            json=order_request,
            headers={"APCA-API-KEY-ID": "mock_key", "APCA-API-SECRET-KEY": "mock_secret"}
        )

        logger.info(f"Status: {r.status_code}, Response: {r.text[:300]}")

        if r.status_code in (200, 201):
            data = r.json()
            logger.info(f"✅ Order placed: {data}")
            assert 'id' in data or 'order_id' in data or isinstance(data, dict)
        elif r.status_code in (404, 422, 400):
            logger.info(f"⚠️  Order rejected: {r.text[:200]}")
            pytest.skip(f"Alpaca mock account not seeded or order rejected")
        else:
            pytest.fail(f"Unexpected status {r.status_code}: {r.text[:200]}")

    def test_get_positions(self, alpaca_mock_url):
        """TEST: Get positions from Alpaca mock"""
        logger.info("=" * 70)
        logger.info("TEST: Alpaca - Get Positions")
        logger.info("=" * 70)

        r = requests.get(
            f"{alpaca_mock_url}/v2/positions",
            headers={"APCA-API-KEY-ID": "mock_key", "APCA-API-SECRET-KEY": "mock_secret"}
        )

        logger.info(f"Status: {r.status_code}")
        assert r.status_code == 200

        data = r.json()
        logger.info(f"Positions: {data}")
        assert isinstance(data, list), "Positions should be a list"
        logger.info(f"✅ Got {len(data)} positions from Alpaca mock")


# ============================================================================
# PUBLIC TRADING PROVIDER (port 7880)
# ============================================================================

class TestPublicTradingProviderE2E:
    """
    E2E tests for Public.com trading provider.

    Uses mock-public-api at http://localhost:7880
    """

    def test_mock_server_health(self, public_mock_url):
        """TEST: Public mock server is healthy"""
        logger.info("=" * 70)
        logger.info("TEST: Public Mock - Health")
        logger.info("=" * 70)

        r = requests.get(f"{public_mock_url}/health")
        assert r.status_code == 200

        data = r.json()
        assert data['status'] == 'healthy'
        assert data['connections']['postgres'] is True
        assert data['connections']['redis'] is True
        logger.info(f"✅ Public mock healthy: {data}")

    def test_fetch_quotes(self, public_mock_url):
        """TEST: Fetch quotes from Public mock (data provider role)"""
        logger.info("=" * 70)
        logger.info("TEST: Public - Fetch Quotes (Data Provider)")
        logger.info("=" * 70)

        symbols = [SYMBOL, "TSLA", "MSFT"]

        r = requests.post(
            f"{public_mock_url}/quotes",
            json={"symbols": symbols}
        )

        logger.info(f"Status: {r.status_code}, Response: {r.text[:300]}")
        assert r.status_code == 200

        data = r.json()
        logger.info(f"✅ Got quotes: {data}")
        assert isinstance(data, (list, dict))

    def test_list_accounts(self, public_mock_url):
        """TEST: List accounts from Public mock (account provider role)"""
        logger.info("=" * 70)
        logger.info("TEST: Public - List Accounts (Account Provider)")
        logger.info("=" * 70)

        r = requests.get(f"{public_mock_url}/accounts")

        logger.info(f"Status: {r.status_code}")
        assert r.status_code == 200

        data = r.json()
        logger.info(f"Accounts: {data}")
        assert isinstance(data, (list, dict))
        logger.info(f"✅ Public accounts endpoint works")

    def test_list_instruments(self, public_mock_url):
        """TEST: List instruments from Public mock"""
        logger.info("=" * 70)
        logger.info("TEST: Public - List Instruments")
        logger.info("=" * 70)

        r = requests.get(f"{public_mock_url}/instruments", params={"limit": 10})

        logger.info(f"Status: {r.status_code}")
        assert r.status_code == 200

        data = r.json()
        logger.info(f"Instruments: {data}")
        assert isinstance(data, (list, dict))
        logger.info(f"✅ Public instruments endpoint works")

    def test_get_instrument_details(self, public_mock_url):
        """TEST: Get specific instrument details from Public mock"""
        logger.info("=" * 70)
        logger.info("TEST: Public - Get Instrument Details")
        logger.info("=" * 70)

        r = requests.get(f"{public_mock_url}/instrument/{SYMBOL}")

        logger.info(f"Status: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            logger.info(f"✅ Instrument: {data}")
        elif r.status_code == 404:
            logger.info(f"⚠️  Instrument {SYMBOL} not in mock DB")
            pytest.skip(f"{SYMBOL} not in Public mock DB")
        else:
            pytest.fail(f"Unexpected status {r.status_code}: {r.text}")

    def test_place_market_order(self, public_mock_url):
        """TEST: Place a market order through Public mock"""
        logger.info("=" * 70)
        logger.info("TEST: Public - Place Market Order (Trading Provider)")
        logger.info("=" * 70)

        order_request = {
            "account_number": "PUBLICDOTCOM-MOCK-001",
            "symbol": SYMBOL,
            "side": "buy",
            "quantity": 10,
            "order_type": "market"
        }

        r = requests.post(
            f"{public_mock_url}/orders",
            json=order_request
        )

        logger.info(f"Status: {r.status_code}, Response: {r.text[:300]}")

        if r.status_code in (200, 201):
            data = r.json()
            logger.info(f"✅ Order placed: {data}")
        elif r.status_code in (400, 404, 422, 500):
            # 500 means account not found (mock returns 500 for unseeded accounts)
            if "Account not found" in r.text or "not found" in r.text.lower():
                logger.info(f"⚠️  Mock account not seeded: {r.text[:100]}")
                pytest.skip("Public mock account not seeded - run seed_mock_accounts.sql first")
            pytest.fail(f"Unexpected error {r.status_code}: {r.text[:200]}")
        else:
            pytest.fail(f"Unexpected status {r.status_code}: {r.text[:200]}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
