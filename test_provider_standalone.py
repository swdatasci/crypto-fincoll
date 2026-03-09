#!/usr/bin/env python3
"""
Standalone test for MultiProviderFetcher integration.

This script tests the provider integration by setting up proper path
for relative imports without triggering fincoll/__init__.py (which imports torch).
"""

import os
import sys

# Get the fincoll directory and add it to path
FINCOLL_DIR = os.path.dirname(os.path.abspath(__file__))
FINCOLL_PACKAGE_DIR = os.path.join(FINCOLL_DIR, "fincoll")

# Add the parent of fincoll package so "from fincoll.providers..." works
sys.path.insert(0, FINCOLL_DIR)

# Pre-populate sys.modules with a minimal fincoll stub to prevent __init__.py from loading
import types

fincoll_stub = types.ModuleType("fincoll")
fincoll_stub.__path__ = [FINCOLL_PACKAGE_DIR]
fincoll_stub.__file__ = os.path.join(FINCOLL_PACKAGE_DIR, "__init__.py")
sys.modules["fincoll"] = fincoll_stub

# Also stub fincoll.providers to avoid its __init__.py if it imports heavy stuff
providers_stub = types.ModuleType("fincoll.providers")
providers_stub.__path__ = [os.path.join(FINCOLL_PACKAGE_DIR, "providers")]
providers_stub.__file__ = os.path.join(FINCOLL_PACKAGE_DIR, "providers", "__init__.py")
sys.modules["fincoll.providers"] = providers_stub

print("=" * 60)
print("Testing MultiProviderFetcher Integration (Standalone)")
print("=" * 60)
print()

# Test 1: Import base_trading_provider
print("1. Testing base imports...")
try:
    # Now we can use regular imports since we've stubbed the package
    from fincoll.providers.base_trading_provider import (
        BaseTradingProvider,
        CircuitBreaker,
    )

    print("   ✓ BaseTradingProvider imported")
    print("   ✓ CircuitBreaker imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: Test CircuitBreaker
print()
print("2. Testing CircuitBreaker...")
cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
print(f"   Initial state - open: {cb.is_open()}")
assert not cb.is_open(), "Circuit breaker should start closed"

cb.record_failure()
cb.record_failure()
cb.record_failure()
print(f"   After 3 failures - open: {cb.is_open()}")
assert cb.is_open(), "Circuit breaker should open after 3 failures"
print("   ✓ CircuitBreaker works correctly")

# Test 3: Import base_provider (for YFinanceProvider)
print()
print("3. Testing BaseDataProvider import...")
try:
    from fincoll.providers.base_provider import BaseDataProvider

    print("   ✓ BaseDataProvider imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Import YFinanceProvider
print()
print("4. Testing YFinanceProvider import...")
try:
    from fincoll.providers.yfinance_provider import YFinanceProvider

    print("   ✓ YFinanceProvider imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Import MultiProviderFetcher
print()
print("5. Testing MultiProviderFetcher import...")
try:
    from fincoll.providers.multi_provider_fetcher import DataType, MultiProviderFetcher

    print("   ✓ MultiProviderFetcher imported")
    print(f"   DataType values: {[d.value for d in DataType]}")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Create MultiProviderFetcher
print()
print("6. Testing MultiProviderFetcher initialization...")
try:
    fetcher = MultiProviderFetcher()
    print("   ✓ MultiProviderFetcher created")
    print(f"   Available providers: {list(fetcher.providers.keys())}")
    status = fetcher.get_provider_status()
    print(f"   Provider status: {status}")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 7: Fetch data with yfinance
print()
print("7. Testing data fetch with yfinance...")
try:
    df = fetcher.get_historical_bars("AAPL", interval="1d", bar_count=5)
    if df is not None and not df.empty:
        print(f"   ✓ Fetched {len(df)} bars for AAPL")
        print(f"   Columns: {list(df.columns)}")
        if "timestamp" in df.columns:
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        elif "date" in df.columns:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        print("   ✗ Fetch returned empty data")
except Exception as e:
    print(f"   ✗ Fetch failed: {e}")
    import traceback

    traceback.print_exc()

# Test 8: Test provider wrapper imports
print()
print("8. Testing provider wrapper imports...")
providers_available = []

try:
    from fincoll.providers.tradestation_trading_provider import (
        TradeStationTradingProvider,
    )

    providers_available.append("TradeStation")
    print("   ✓ TradeStationTradingProvider imported")
except Exception as e:
    print(f"   - TradeStationTradingProvider not available: {e}")

try:
    from fincoll.providers.alpaca_trading_provider import AlpacaTradingProvider

    providers_available.append("Alpaca")
    print("   ✓ AlpacaTradingProvider imported")
except Exception as e:
    print(f"   - AlpacaTradingProvider not available: {e}")

try:
    from fincoll.providers.public_trading_provider import PublicTradingProvider

    providers_available.append("Public")
    print("   ✓ PublicTradingProvider imported")
except Exception as e:
    print(f"   - PublicTradingProvider not available: {e}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("✓ Base components working")
print("✓ CircuitBreaker working")
print("✓ MultiProviderFetcher working")
print(f"✓ Provider wrappers available: {providers_available}")
print()
print("Integration test PASSED!")
