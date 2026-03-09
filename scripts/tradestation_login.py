#!/usr/bin/env python3
"""
TradeStation OAuth Login Tool
Uses existing fincoll auth classes to get a new token
"""
import sys
from pathlib import Path

# Add fincoll to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.auth.tradestation_auth import TradeStationAuth

def main():
    print("\n" + "="*70)
    print("TradeStation OAuth Login")
    print("="*70)
    print()

    try:
        # Create auth instance
        auth = TradeStationAuth()

        # Start OAuth flow (auto-callback mode)
        print("🔗 Starting OAuth login...")
        print()
        print("   1. A browser window will open (or copy the URL below)")
        print("   2. Log in with your TradeStation credentials")
        print("   3. Authorize the app")
        print("   4. You'll be redirected back automatically")
        print()

        # Get login URL
        login_url = auth.get_login_url()
        print("📋 OAuth URL:")
        print(login_url)
        print()

        # Perform login with auto-callback server
        print("🌐 Opening browser and starting callback server...")
        print()
        code = auth.login(auto_callback=True)

        if code:
            print("✅ Login successful!")
            print(f"   Token saved to: {auth.token_file}")
            print()
            print("="*70)
            print("✅ You're now authenticated with TradeStation!")
            print("="*70)
            print()
            print("Next steps:")
            print("   1. Restart fincoll-server: pm2 restart fincoll-server")
            print("   2. Check logs: pm2 logs fincoll-server")
            print()
            return 0
        else:
            print("❌ Login failed - no authorization code received")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
