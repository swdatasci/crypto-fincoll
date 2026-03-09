#!/usr/bin/env python3
"""
Test FinColl Orchestration Layer

Tests the cache-first data fetching and velocity target computation.
"""

import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_cache_manager():
    """Test cache manager functions"""
    print("\n" + "="*60)
    print("TEST 1: Cache Manager")
    print("="*60)

    from fincoll.data.cache_manager import check_cache, save_to_cache, get_cache_stats

    # Test cache check (should be miss for test data)
    result = check_cache('AAPL', '1d', '2024-01-01', '2024-12-01')

    if result is None:
        print("✓ Cache check works (returned None for likely miss)")
    else:
        print(f"✓ Cache hit! Got {len(result)} bars")

    # Test cache stats
    stats = get_cache_stats(['AAPL', 'MSFT', 'GOOGL'], '1d', '2024-01-01', '2024-12-01')
    print(f"✓ Cache stats: {stats['cache_hit_rate']:.1%} hit rate")
    print(f"  - Cached: {stats['cached_symbols']}")
    print(f"  - Missing: {stats['missing_symbols']}")

    return True


def test_data_fetcher():
    """Test data fetcher with cache-first logic"""
    print("\n" + "="*60)
    print("TEST 2: Data Fetcher")
    print("="*60)

    from fincoll.data.data_fetcher import fetch_bars

    # Use a small date range for testing
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    print(f"Fetching AAPL bars from {start} to {end}")

    try:
        bars = fetch_bars('AAPL', '1d', start, end, use_cache=True)

        if bars is not None and not bars.empty:
            print(f"✓ Fetched {len(bars)} bars for AAPL")
            print(f"  Columns: {bars.columns.tolist()}")
            print(f"  Date range: {bars.index[0]} to {bars.index[-1]}")
            return True
        else:
            print("✗ Failed to fetch bars (returned None or empty)")
            return False

    except Exception as e:
        print(f"✗ Fetch failed with error: {e}")
        return False


def test_multi_timeframe_fetch():
    """Test multi-timeframe data fetching"""
    print("\n" + "="*60)
    print("TEST 3: Multi-Timeframe Fetch")
    print("="*60)

    from fincoll.data.data_fetcher import fetch_multi_timeframe

    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    print(f"Fetching AAPL multi-timeframe data from {start} to {end}")

    try:
        # Note: Only '1d' is likely to work with TradeStation for this short range
        # For full multi-timeframe, we'd need intraday data
        data = fetch_multi_timeframe('AAPL', ['1d'], start, end, use_cache=True)

        if data:
            for tf, df in data.items():
                print(f"✓ {tf}: {len(df)} bars")
            return True
        else:
            print("✗ No data returned")
            return False

    except Exception as e:
        print(f"✗ Multi-timeframe fetch failed: {e}")
        return False


def test_velocity_targets():
    """Test velocity target computation"""
    print("\n" + "="*60)
    print("TEST 4: Velocity Target Computation")
    print("="*60)

    from fincoll.training.target_computer import compute_velocity_targets, validate_targets

    # Create synthetic data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')

    # Simulate price movement with trend + noise
    base_price = 100
    trend = np.linspace(0, 10, 100)  # Uptrend
    noise = np.random.randn(100) * 2
    prices = base_price + trend + noise

    df = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100)) * 0.5,
        'low': prices - np.abs(np.random.randn(100)) * 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Test with single timeframe
    bars_dict = {'1h': df}

    try:
        targets = compute_velocity_targets(bars_dict, timeframes=['1h'])

        if not targets.empty:
            print(f"✓ Computed targets: {targets.shape}")
            print(f"  Columns: {targets.columns.tolist()}")

            # Validate targets
            validation = validate_targets(targets)

            if validation['valid']:
                print(f"✓ Validation passed")
            else:
                print(f"⚠ Validation issues: {validation['issues']}")

            # Show sample
            print("\nSample targets (first 5 rows):")
            print(targets.head())

            return True
        else:
            print("✗ Empty targets DataFrame")
            return False

    except Exception as e:
        print(f"✗ Velocity target computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full orchestration pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Full Orchestration Pipeline")
    print("="*60)

    from fincoll.data.data_fetcher import fetch_bars
    from fincoll.training.target_computer import compute_velocity_targets

    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    print(f"Testing full pipeline: Fetch → Compute Targets")
    print(f"Symbol: AAPL, Timeframe: 1d, Range: {start} to {end}")

    try:
        # Step 1: Fetch data (cache-first)
        bars = fetch_bars('AAPL', '1d', start, end)

        if bars is None or bars.empty:
            print("✗ Failed to fetch data")
            return False

        print(f"✓ Step 1: Fetched {len(bars)} bars")

        # Step 2: Compute velocity targets
        bars_dict = {'1d': bars}
        targets = compute_velocity_targets(bars_dict, timeframes=['1d'])

        if targets.empty:
            print("✗ Failed to compute targets")
            return False

        print(f"✓ Step 2: Computed targets {targets.shape}")

        # Step 3: Verify alignment
        if len(targets) == len(bars):
            print(f"✓ Step 3: Targets aligned with bars")
        else:
            print(f"⚠ Warning: Targets ({len(targets)}) != Bars ({len(bars)})")

        print("\n✓ Full pipeline test PASSED")
        return True

    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FinColl Orchestration Layer Tests")
    print("="*60)

    results = {}

    # Run tests
    results['cache_manager'] = test_cache_manager()
    results['data_fetcher'] = test_data_fetcher()
    results['multi_timeframe'] = test_multi_timeframe_fetch()
    results['velocity_targets'] = test_velocity_targets()
    results['full_pipeline'] = test_full_pipeline()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
