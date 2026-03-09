#!/usr/bin/env python3
"""
API Credentials Manager

Securely loads API credentials from JSON files in /home/rford/caelum/ss/
"""

import json
from pathlib import Path
from typing import Dict


class APICredentials:
    """Manager for API credentials"""

    CREDENTIALS_DIR = Path.home() / "caelum" / "ss"

    @classmethod
    def get_alpha_vantage_key(cls) -> str:
        """Get Alpha Vantage API key"""
        cred_file = cls.CREDENTIALS_DIR / ".alpha_vantage_credentials.json"

        if not cred_file.exists():
            raise FileNotFoundError(
                f"Alpha Vantage credentials not found at {cred_file}. "
                "Please create the file with your API key."
            )

        with open(cred_file, 'r') as f:
            creds = json.load(f)

        api_key = creds.get('api_key')
        if not api_key or api_key == "PASTE_API_KEY_HERE":
            raise ValueError("Alpha Vantage API key not configured")

        return api_key

    @classmethod
    def get_sentimentradar_credentials(cls) -> Dict[str, str]:
        """Get SentimentRadar credentials"""
        cred_file = cls.CREDENTIALS_DIR / ".sentimentradar_credentials.json"

        if not cred_file.exists():
            raise FileNotFoundError(
                f"SentimentRadar credentials not found at {cred_file}. "
                "Please create the file with your credentials."
            )

        with open(cred_file, 'r') as f:
            creds = json.load(f)

        email = creds.get('email')
        password = creds.get('password')
        api_key = creds.get('api_key')

        if not email or email == "PASTE_EMAIL_HERE":
            raise ValueError("SentimentRadar email not configured")

        if not password or password == "PASTE_PASSWORD_HERE":
            raise ValueError("SentimentRadar password not configured")

        return {
            'email': email,
            'password': password,
            'api_key': api_key  # May be None initially
        }

    @classmethod
    def get_tradestation_token_path(cls) -> Path:
        """Get path to TradeStation token file"""
        return cls.CREDENTIALS_DIR / ".tradestation_token.json"

    @classmethod
    def test_alpha_vantage(cls) -> bool:
        """Test Alpha Vantage API connection"""
        try:
            import requests
            api_key = cls.get_alpha_vantage_key()

            # Test with a simple quote request
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "Global Quote" in data or "Note" in data:  # "Note" = rate limit message
                    print("✅ Alpha Vantage API: Connected")
                    if "Note" in data:
                        print(f"⚠️  Rate limit message: {data['Note']}")
                    return True

            print(f"❌ Alpha Vantage API: Failed (status {response.status_code})")
            return False

        except Exception as e:
            print(f"❌ Alpha Vantage API: Error - {e}")
            return False

    @classmethod
    def test_all_credentials(cls) -> Dict[str, bool]:
        """Test all configured API credentials"""
        results = {}

        # Test Alpha Vantage
        try:
            cls.get_alpha_vantage_key()
            results['alpha_vantage'] = cls.test_alpha_vantage()
        except Exception as e:
            print(f"❌ Alpha Vantage: {e}")
            results['alpha_vantage'] = False

        # Test SentimentRadar
        try:
            creds = cls.get_sentimentradar_credentials()
            if creds['password'] != "PASTE_PASSWORD_HERE":
                print("✅ SentimentRadar: Credentials configured")
                results['sentimentradar'] = True
            else:
                print("⚠️  SentimentRadar: Password not yet configured")
                results['sentimentradar'] = False
        except Exception as e:
            print(f"❌ SentimentRadar: {e}")
            results['sentimentradar'] = False

        # Test TradeStation
        try:
            token_path = cls.get_tradestation_token_path()
            if token_path.exists():
                print("✅ TradeStation: Token file exists")
                results['tradestation'] = True
            else:
                print("❌ TradeStation: Token file not found")
                results['tradestation'] = False
        except Exception as e:
            print(f"❌ TradeStation: {e}")
            results['tradestation'] = False

        return results


if __name__ == '__main__':
    print("=" * 60)
    print("API CREDENTIALS TEST")
    print("=" * 60)
    print()

    results = APICredentials.test_all_credentials()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_working = all(results.values())

    if all_working:
        print("✅ All APIs configured and working!")
    else:
        print("⚠️  Some APIs need configuration:")
        for api, status in results.items():
            if not status:
                print(f"   - {api}")
