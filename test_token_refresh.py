#!/usr/bin/env python3
"""
Test TradeStation token refresh with Redis rate limiter.
This verifies we can refresh tokens without spamming the API.
"""

import sys
from pathlib import Path

# Add fincoll to path
sys.path.insert(0, str(Path(__file__).parent))

from fincoll.auth.tradestation_auth import TradeStationAuth
from fincoll.utils.rate_limiter import get_shared_limiter
import time

def test_token_refresh():
    print("\n" + "="*70)
    print("Testing TradeStation Token Refresh with Rate Limiter")
    print("="*70)

    # Initialize rate limiter with verbose mode
    print("\n1. Initializing Redis-based rate limiter...")
    limiter = get_shared_limiter(service_name='token_refresh_test', verbose=True)

    # Check current rate limit stats
    stats = limiter.get_stats('accounts')
    print(f"\n2. Current rate limit stats:")
    print(f"   Mode: {stats['mode']}")
    print(f"   Current count: {stats['current_count']}/{stats['max_requests']}")
    print(f"   Window: {stats['window_seconds']}s")
    print(f"   Available: {stats['available']}")

    # Initialize auth
    print("\n3. Initializing TradeStation auth...")
    auth = TradeStationAuth()

    # Check if authenticated
    print("\n4. Checking authentication status...")
    is_authed = auth.is_authenticated()
    print(f"   Currently authenticated: {is_authed}")

    if is_authed:
        print(f"   Access token: {auth.access_token[:20]}...")
        print(f"   Expires at: {auth.expires_at}")

        # Try to get access token (will refresh if needed)
        print("\n5. Getting access token (will refresh if expired)...")
        start_time = time.time()

        try:
            token = auth.get_access_token()
            elapsed = time.time() - start_time

            print(f"   ✅ Token obtained successfully!")
            print(f"   Token: {token[:20]}...")
            print(f"   Time taken: {elapsed:.2f}s")
            print(f"   New expires at: {auth.expires_at}")

            # Check rate limit stats after refresh
            stats_after = limiter.get_stats('accounts')
            print(f"\n6. Rate limit stats after refresh:")
            print(f"   Current count: {stats_after['current_count']}/{stats_after['max_requests']}")
            print(f"   Utilization: {stats_after['utilization_pct']:.1f}%")

            print("\n" + "="*70)
            print("✅ Token refresh test PASSED!")
            print("="*70)
            return True

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Token refresh failed!")
            print(f"   Error: {e}")
            print(f"   Time taken: {elapsed:.2f}s")

            print("\n" + "="*70)
            print("❌ Token refresh test FAILED!")
            print("="*70)
            return False
    else:
        print("\n⚠️  Not authenticated. Need to run login flow first.")
        print("   Run: python -m fincoll.auth.tradestation_auth")
        return False

if __name__ == '__main__':
    success = test_token_refresh()
    sys.exit(0 if success else 1)
