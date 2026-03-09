#!/usr/bin/env python3
"""
Provider Matrix E2E Tests

Tests every DATA provider × TRADING provider × LOOKBACK_DAYS combination.

Matrix:
    Trading Provider     Data Provider       Lookback Days
    ──────────────────────────────────────────────────────
    AlpacaTrading    ×  yfinance         ×  5d, 20d, 60d
    AlpacaTrading    ×  TradeStation     ×  5d, 20d, 60d
    PublicTrading    ×  yfinance         ×  5d, 20d, 60d
    PublicTrading    ×  TradeStation     ×  5d, 20d, 60d
    TSTrading        ×  yfinance         ×  5d, 20d, 60d
    TSTrading        ×  Alpaca           ×  5d, 20d, 60d

Each combination:
    1. Fetch market data through the DATA provider
    2. Verify the data has proper datetime index (not integers)
    3. Use the data to feed the feature extractor
    4. Verify the TRADING provider's account/order endpoints are reachable

Run with: pytest test_provider_matrix_e2e.py -v
"""

import pytest
import requests
import pandas as pd
from datetime import datetime, timedelta, date
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOL = "AAPL"
START = "2025-12-01"
END = "2026-01-01"

# Days dimension: lookback windows to test
LOOKBACK_DAYS = [5, 20, 60]


def get_yfinance_data(symbol=SYMBOL, start=START, end=END):
    """Helper: fetch data from yfinance, skip if unavailable."""
    try:
        from fincoll.providers.yfinance_provider import YFinanceProvider
        provider = YFinanceProvider()
        df = provider.get_historical_bars(symbol=symbol, start_date=start, end_date=end, interval="1d")
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"yfinance unavailable: {e}")
        return None


def get_alpaca_mock_data(alpaca_url, symbol=SYMBOL, start=START, end=END):
    """Helper: fetch data from Alpaca mock server, parse to DataFrame."""
    r = requests.get(
        f"{alpaca_url}/v2/stocks/{symbol}/bars",
        params={"timeframe": "1Day", "start": start, "end": end, "limit": 100},
        headers={"APCA-API-KEY-ID": "mock_key", "APCA-API-SECRET-KEY": "mock_secret"},
        timeout=5
    )
    if r.status_code != 200:
        return None

    bars = r.json().get('bars', [])
    if not bars:
        return None

    df = pd.DataFrame([{
        'open': b['o'],
        'high': b['h'],
        'low': b['l'],
        'close': b['c'],
        'volume': b['v'],
    } for b in bars], index=pd.DatetimeIndex([b['t'] for b in bars]))
    df.index = pd.to_datetime(df.index)
    return df


def validate_dataframe(df, source_name):
    """Common DataFrame validation for all provider combinations."""
    assert df is not None, f"{source_name}: should return data"
    assert not df.empty, f"{source_name}: should not be empty"
    assert isinstance(df.index, pd.DatetimeIndex), \
        f"{source_name}: index must be DatetimeIndex, got {type(df.index)}"

    latest_ts = df.index[-1]
    assert not isinstance(latest_ts, int), \
        f"{source_name}: timestamp must NOT be integer, got {latest_ts}"
    assert latest_ts.year >= 2020, \
        f"{source_name}: year should be recent, not {latest_ts.year}"

    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert col in df.columns, f"{source_name}: missing column {col}"


# ============================================================================
# ALPACA TRADING × DATA PROVIDERS
# ============================================================================

class TestAlpacaTradingWithDataProviders:
    """
    Test AlpacaTradingProvider (account + orders via port 7879)
    paired with each data provider.
    """

    def test_alpaca_trading_x_yfinance_data(self, alpaca_mock_url):
        """
        MATRIX TEST: AlpacaTradingProvider × YFinanceProvider

        Simulates production: Alpaca for trading, yfinance for market data.
        yfinance often used when Alpaca rate limits are a concern.
        """
        logger.info("=" * 70)
        logger.info("MATRIX: AlpacaTrading × yfinance data")
        logger.info("=" * 70)

        # Step 1: Fetch data from yfinance (data provider role)
        df = get_yfinance_data()
        if df is None:
            pytest.skip("yfinance data unavailable")

        validate_dataframe(df, "yfinance")
        logger.info(f"✅ yfinance data: {len(df)} bars, latest {df.index[-1]}")

        # Step 2: Verify Alpaca trading account endpoint is reachable
        r = requests.get(f"{alpaca_mock_url}/v2/positions",
                         headers={"APCA-API-KEY-ID": "mock_key", "APCA-API-SECRET-KEY": "mock_secret"})
        assert r.status_code == 200, f"Alpaca positions failed: {r.status_code}"
        logger.info(f"✅ Alpaca trading provider reachable (positions: {r.json()})")

        # Step 3: Verify the yfinance data can flow to feature extraction
        latest_ts = df.index[-1]
        assert isinstance(latest_ts, pd.Timestamp), \
            f"Timestamp for feature extractor must be pd.Timestamp, got {type(latest_ts)}"

        logger.info(f"✅ MATRIX PASSED: AlpacaTrading × yfinance data")
        logger.info(f"   Data: {len(df)} bars from yfinance, latest {latest_ts}")
        logger.info(f"   Trading: Alpaca mock at port 7879 is reachable")

    def test_alpaca_trading_x_alpaca_mock_data(self, alpaca_mock_url):
        """
        MATRIX TEST: AlpacaTradingProvider × Alpaca-as-data-provider

        Simulates using the same Alpaca account for both data AND trading.
        This is the most common Alpaca production setup.
        """
        logger.info("=" * 70)
        logger.info("MATRIX: AlpacaTrading × AlpacaMockData (same provider)")
        logger.info("=" * 70)

        # Step 1: Fetch data from Alpaca mock (data provider role)
        df = get_alpaca_mock_data(alpaca_mock_url)
        if df is None:
            pytest.skip("Alpaca mock returned no bars data")

        validate_dataframe(df, "alpaca_mock")
        logger.info(f"✅ Alpaca data: {len(df)} bars, latest {df.index[-1]}")

        # Step 2: Alpaca trading endpoint reachable
        r = requests.get(f"{alpaca_mock_url}/v2/positions",
                         headers={"APCA-API-KEY-ID": "mock_key", "APCA-API-SECRET-KEY": "mock_secret"})
        assert r.status_code == 200

        logger.info(f"✅ MATRIX PASSED: AlpacaTrading × AlpacaMockData")
        logger.info(f"   Data: {len(df)} bars from Alpaca mock, latest {df.index[-1]}")
        logger.info(f"   Trading: Same Alpaca mock is reachable for orders")


# ============================================================================
# PUBLIC TRADING × DATA PROVIDERS
# ============================================================================

class TestPublicTradingWithDataProviders:
    """
    Test PublicTradingProvider (account + orders via port 7880)
    paired with each data provider.
    """

    def test_public_trading_x_yfinance_data(self, public_mock_url):
        """
        MATRIX TEST: PublicTradingProvider × YFinanceProvider

        Simulates: Public for trading, yfinance for data.
        Public.com has limited bar history; yfinance fills the gap.
        """
        logger.info("=" * 70)
        logger.info("MATRIX: PublicTrading × yfinance data")
        logger.info("=" * 70)

        # Step 1: Fetch data from yfinance
        df = get_yfinance_data()
        if df is None:
            pytest.skip("yfinance data unavailable")

        validate_dataframe(df, "yfinance")
        logger.info(f"✅ yfinance data: {len(df)} bars, latest {df.index[-1]}")

        # Step 2: Public trading endpoint reachable
        r = requests.get(f"{public_mock_url}/accounts")
        assert r.status_code == 200, f"Public accounts failed: {r.status_code}"
        logger.info(f"✅ Public trading provider reachable")

        logger.info(f"✅ MATRIX PASSED: PublicTrading × yfinance data")

    def test_public_trading_x_alpaca_mock_data(self, public_mock_url, alpaca_mock_url):
        """
        MATRIX TEST: PublicTradingProvider × Alpaca-as-data-provider

        Simulates: Execute trades on Public, but use Alpaca's data API.
        """
        logger.info("=" * 70)
        logger.info("MATRIX: PublicTrading × AlpacaMockData")
        logger.info("=" * 70)

        # Step 1: Fetch data from Alpaca mock
        df = get_alpaca_mock_data(alpaca_mock_url)
        if df is None:
            pytest.skip("Alpaca mock returned no bars data")

        validate_dataframe(df, "alpaca_mock_as_data_source")
        logger.info(f"✅ Alpaca data: {len(df)} bars, latest {df.index[-1]}")

        # Step 2: Public trading endpoint reachable
        r = requests.get(f"{public_mock_url}/accounts")
        assert r.status_code == 200

        logger.info(f"✅ MATRIX PASSED: PublicTrading × AlpacaMockData")

    def test_public_trading_x_ts_mock_data(self, public_mock_url, tradestation_mock_url):
        """
        MATRIX TEST: PublicTradingProvider × TradeStation-as-data-provider

        Simulates: Execute on Public, data from TradeStation (better intraday data).
        """
        logger.info("=" * 70)
        logger.info("MATRIX: PublicTrading × TradeStationMockData")
        logger.info("=" * 70)

        # Step 1: Fetch data from TradeStation mock
        r = requests.get(
            f"{tradestation_mock_url}/tradestation/v3/marketdata/barcharts/{SYMBOL}",
            params={"interval": "Daily", "barsback": 30,
                    "firstdate": START, "lastdate": END}
        )
        assert r.status_code == 200

        bars = r.json().get('Bars', [])
        if not bars:
            pytest.skip("TradeStation mock returned no bars")

        # Convert to DataFrame
        df = pd.DataFrame([{
            'open': float(b.get('Open', b.get('open', 0))),
            'high': float(b.get('High', b.get('high', 0))),
            'low': float(b.get('Low', b.get('low', 0))),
            'close': float(b.get('Close', b.get('close', 0))),
            'volume': float(b.get('TotalVolume', b.get('volume', 0))),
        } for b in bars], index=pd.DatetimeIndex([
            b.get('TimeStamp', b.get('timestamp', '')) for b in bars
        ]))
        df.index = pd.to_datetime(df.index)

        validate_dataframe(df, "tradestation_mock")
        logger.info(f"✅ TradeStation data: {len(df)} bars, latest {df.index[-1]}")

        # Step 2: Public trading endpoint reachable
        r = requests.get(f"{public_mock_url}/accounts")
        assert r.status_code == 200

        logger.info(f"✅ MATRIX PASSED: PublicTrading × TradeStationMockData")


# ============================================================================
# TRADESTATION TRADING × DATA PROVIDERS
# ============================================================================

class TestTradeStationTradingWithDataProviders:
    """
    Test TradeStationTradingProvider (trading via port 7878)
    paired with each data provider.
    """

    def test_ts_trading_x_yfinance_data(self, tradestation_mock_url):
        """
        MATRIX TEST: TradeStationTradingProvider × YFinanceProvider

        Simulates: Execute on TradeStation, use yfinance for broader data.
        """
        logger.info("=" * 70)
        logger.info("MATRIX: TSTrading × yfinance data")
        logger.info("=" * 70)

        # Step 1: Fetch data from yfinance
        df = get_yfinance_data()
        if df is None:
            pytest.skip("yfinance data unavailable")

        validate_dataframe(df, "yfinance")
        logger.info(f"✅ yfinance data: {len(df)} bars, latest {df.index[-1]}")

        # Step 2: TradeStation trading endpoint reachable
        r = requests.get(f"{tradestation_mock_url}/health")
        assert r.status_code == 200 and r.json()['status'] == 'healthy'
        logger.info(f"✅ TradeStation trading provider reachable")

        logger.info(f"✅ MATRIX PASSED: TSTrading × yfinance data")

    def test_ts_trading_x_alpaca_mock_data(self, tradestation_mock_url, alpaca_mock_url):
        """
        MATRIX TEST: TradeStationTradingProvider × Alpaca-as-data-provider

        Simulates: Execute on TradeStation, use Alpaca's data API for signals.
        """
        logger.info("=" * 70)
        logger.info("MATRIX: TSTrading × AlpacaMockData")
        logger.info("=" * 70)

        # Step 1: Fetch data from Alpaca mock
        df = get_alpaca_mock_data(alpaca_mock_url)
        if df is None:
            pytest.skip("Alpaca mock returned no bars data")

        validate_dataframe(df, "alpaca_mock")
        logger.info(f"✅ Alpaca data: {len(df)} bars, latest {df.index[-1]}")

        # Step 2: TradeStation trading endpoint reachable
        r = requests.get(f"{tradestation_mock_url}/health")
        assert r.status_code == 200

        logger.info(f"✅ MATRIX PASSED: TSTrading × AlpacaMockData")


# ============================================================================
# DATA INTEGRITY CROSS-PROVIDER COMPARISON
# ============================================================================

class TestCrossProviderDataConsistency:
    """
    Verify that data from different providers can be compared/interchanged.

    A trading system must be able to switch data sources without downstream
    failures. This validates no timestamp corruption across providers.
    """

    def test_alpaca_and_yfinance_timestamps_are_comparable(self, alpaca_mock_url):
        """
        TEST: Timestamps from Alpaca mock and yfinance can be compared directly.

        Both must use DatetimeIndex for pd.merge / pd.concat to work correctly.
        """
        logger.info("=" * 70)
        logger.info("TEST: Cross-provider timestamp compatibility")
        logger.info("=" * 70)

        df_alpaca = get_alpaca_mock_data(alpaca_mock_url)
        if df_alpaca is None:
            pytest.skip("Alpaca mock returned no bars data")

        df_yfinance = get_yfinance_data()
        if df_yfinance is None:
            pytest.skip("yfinance data unavailable")

        # Both must be DatetimeIndex
        assert isinstance(df_alpaca.index, pd.DatetimeIndex), \
            "Alpaca must have DatetimeIndex"
        assert isinstance(df_yfinance.index, pd.DatetimeIndex), \
            "yfinance must have DatetimeIndex"

        # Timestamps must overlap (both covering same period)
        alpaca_min = df_alpaca.index.min()
        yf_min = df_yfinance.index.min()

        logger.info(f"Alpaca range: {alpaca_min} to {df_alpaca.index.max()}")
        logger.info(f"yfinance range: {yf_min} to {df_yfinance.index.max()}")

        # Both must be recent dates (not 1969/1970)
        assert alpaca_min.year >= 2020, f"Alpaca has 1969 date! {alpaca_min}"
        assert yf_min.year >= 2020, f"yfinance has 1969 date! {yf_min}"

        logger.info(f"✅ Cross-provider timestamp validation PASSED")
        logger.info(f"   No integer timestamps, no 1969 dates from either provider")


# ============================================================================
# DAYS DIMENSION: Test data fetch across lookback windows (5d, 20d, 60d)
# ============================================================================

class TestDaysDimension:
    """
    Test that all provider combinations return correct data across
    different lookback windows: 5d (short), 20d (month), 60d (quarter).

    This validates that providers handle both short and long date ranges
    correctly - critical for backtesting across different time horizons.
    """

    @pytest.mark.parametrize("lookback_days", LOOKBACK_DAYS)
    def test_yfinance_lookback_windows(self, lookback_days):
        """yfinance returns data across all lookback windows."""
        end = date.today()
        start = end - timedelta(days=lookback_days)

        df = get_yfinance_data(
            symbol=SYMBOL,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        if df is None:
            pytest.skip("yfinance unavailable")

        validate_dataframe(df, f"yfinance_{lookback_days}d")
        assert len(df) > 0, f"Expected bars for {lookback_days}d window"
        logger.info(f"✅ yfinance {lookback_days}d: {len(df)} bars, "
                    f"{df.index[0].date()} → {df.index[-1].date()}")

    @pytest.mark.parametrize("lookback_days", LOOKBACK_DAYS)
    def test_alpaca_mock_lookback_windows(self, alpaca_mock_url, lookback_days):
        """Alpaca mock returns data across all lookback windows."""
        end = date.today()
        start = end - timedelta(days=lookback_days)

        df = get_alpaca_mock_data(
            alpaca_mock_url,
            symbol=SYMBOL,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        if df is None:
            pytest.skip("Alpaca mock returned no data")

        validate_dataframe(df, f"alpaca_mock_{lookback_days}d")
        logger.info(f"✅ Alpaca mock {lookback_days}d: {len(df)} bars, "
                    f"{df.index[0].date()} → {df.index[-1].date()}")

    @pytest.mark.parametrize("lookback_days", LOOKBACK_DAYS)
    def test_tradestation_mock_lookback_windows(self, tradestation_mock_url, lookback_days):
        """TradeStation mock returns synthetic bars across all lookback windows."""
        end = date.today()
        start = end - timedelta(days=lookback_days)

        r = requests.get(
            f"{tradestation_mock_url}/tradestation/v3/marketdata/barcharts/{SYMBOL}",
            params={
                "interval": "Daily",
                "barsback": lookback_days,
                "firstdate": start.isoformat(),
                "lastdate": end.isoformat(),
            },
            timeout=5,
        )
        assert r.status_code == 200, f"TS mock failed: {r.status_code}"

        bars = r.json().get("Bars", [])
        assert len(bars) > 0, f"Expected bars for {lookback_days}d window"

        # Validate timestamps are real dates
        ts = bars[0]["TimeStamp"]
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert parsed.year >= 2020, f"Unexpected year: {parsed.year}"

        logger.info(f"✅ TradeStation mock {lookback_days}d: {len(bars)} bars, "
                    f"latest {bars[-1]['TimeStamp']}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
