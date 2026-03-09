"""
TradeStation OAuth 2.0 Authentication

Handles authentication with TradeStation API using OAuth 2.0 flow.
Stores and refreshes access tokens as needed.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import webbrowser
from urllib.parse import urlencode


# Load environment variables
try:
    from caelum_secrets import load_env_from_secrets

    load_env_from_secrets("/prod/fincoll/")
except Exception:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # env vars must be set manually


class TradeStationAuth:
    """TradeStation OAuth 2.0 authentication manager"""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        """
        Initialize TradeStation authentication

        Args:
            client_id: TradeStation app client ID (or from env)
            client_secret: TradeStation app client secret (or from env)
            redirect_uri: OAuth redirect URI (or from env)
        """
        self.client_id = client_id or os.getenv("TRADESTATION_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("TRADESTATION_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv(
            "TRADESTATION_REDIRECT_URI", "http://10.32.3.27:8080/callback"
        )

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "TradeStation credentials not found. "
                "Set TRADESTATION_CLIENT_ID and TRADESTATION_CLIENT_SECRET environment variables"
            )

        # TradeStation API endpoints
        self.auth_url = "https://signin.tradestation.com/authorize"
        self.token_url = "https://signin.tradestation.com/oauth/token"
        self.api_url = "https://api.tradestation.com/v3"

        # Token storage (NFS-shared location for multi-machine access)
        credentials_dir = os.getenv("CREDENTIALS_DIR")
        if credentials_dir:
            self.token_file = Path(credentials_dir) / ".tradestation_token.json"
        else:
            self.token_file = Path.home() / "caelum" / "ss" / ".tradestation_token.json"
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None

        # Auth failure tracking (prevent infinite retry loop)
        self.auth_failure_count = 0
        self.auth_failure_backoff_until = None
        self.max_auth_failures = 3
        self.auth_backoff_seconds = 300  # 5 minutes

        # Load existing token if available
        self._load_token()

    def _load_token(self):
        """Load token from file if it exists"""
        if self.token_file.exists():
            try:
                with open(self.token_file, "r") as f:
                    data = json.load(f)
                    self.access_token = data.get("access_token")
                    self.refresh_token = data.get("refresh_token")
                    expires_at_str = data.get("expires_at")
                    if expires_at_str:
                        self.expires_at = datetime.fromisoformat(expires_at_str)
            except Exception as e:
                print(f"Error loading token: {e}")

    def _save_token(self):
        """Save token to file"""
        data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
        with open(self.token_file, "w") as f:
            json.dump(data, f)

    def save_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str],
        expires_at: Optional[datetime] = None,
        expires_in: Optional[int] = None,
    ) -> None:
        """
        Persist tokens to disk (used by external OAuth handlers).

        Args:
            access_token: OAuth access token
            refresh_token: OAuth refresh token (optional)
            expires_at: Absolute expiry timestamp
            expires_in: Expiry seconds from now (used if expires_at not provided)
        """
        self.access_token = access_token
        if refresh_token is not None:
            self.refresh_token = refresh_token

        if expires_at:
            self.expires_at = expires_at
        elif expires_in:
            self.expires_at = datetime.now() + timedelta(seconds=expires_in)

        self._save_token()

    def get_login_url(self) -> str:
        """
        Get OAuth login URL for user authentication

        Returns:
            URL to open in browser for user login
        """
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "MarketData ReadAccount Trade Crypto",  # Request necessary scopes
        }
        return f"{self.auth_url}?{urlencode(params)}"

    def login(self, auto_callback: bool = True) -> str:
        """
        Start OAuth login flow

        Args:
            auto_callback: If True, starts local server to catch callback automatically

        Returns:
            Login URL
        """
        url = self.get_login_url()

        if auto_callback:
            # Import here to avoid circular dependency
            from .oauth_server import start_oauth_server

            # Extract port from redirect_uri
            from urllib.parse import urlparse

            parsed = urlparse(self.redirect_uri)
            port = parsed.port or 8080

            # Start callback server
            server = start_oauth_server(port)

            print(f"\n{'=' * 70}")
            print(f"TradeStation OAuth Login")
            print(f"{'=' * 70}")
            print(f"Opening browser for authentication...")
            print(f"Login URL: {url}")
            print(f"\nWaiting for you to authorize the application...")
            print(f"(This page will automatically detect when you're done)")
            print(f"{'=' * 70}\n")

            # Open browser
            webbrowser.open(url)

            try:
                # Wait for authorization code
                auth_code = server.wait_for_code(timeout=300)  # 5 minutes

                # Exchange code for token
                self.exchange_code(auth_code)

                print("\n✅ Authentication successful!")
                print(f"Access token saved to: {self.token_file}")

            finally:
                # Stop server
                server.stop()

            return url
        else:
            # Manual mode - user copies code themselves
            print(f"Opening browser for TradeStation login...")
            print(f"Login URL: {url}")
            webbrowser.open(url)
            return url

    def exchange_code(self, authorization_code: str):
        """
        Exchange authorization code for access token

        Args:
            authorization_code: Code received from OAuth callback
        """
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": authorization_code,
            "redirect_uri": self.redirect_uri,
        }

        response = requests.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token")

        # Calculate expiration time
        expires_in = token_data.get("expires_in", 1200)  # Default 20 minutes
        self.expires_at = datetime.now() + timedelta(seconds=expires_in)

        self._save_token()

    def _refresh_access_token(self):
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            error_msg = "No refresh token available. Need to login again."
            self._send_auth_failure_notification(error_msg)
            raise ValueError(error_msg)

        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
        }

        try:
            response = requests.post(self.token_url, data=data)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data["access_token"]
            if "refresh_token" in token_data:
                self.refresh_token = token_data["refresh_token"]

            expires_in = token_data.get("expires_in", 1200)
            self.expires_at = datetime.now() + timedelta(seconds=expires_in)

            self._save_token()
        except Exception as e:
            error_msg = f"Token refresh failed: {str(e)}"
            self._send_auth_failure_notification(error_msg)
            raise

    def refresh_access_token(self) -> str:
        """Refresh token and return the new access token."""
        self._refresh_access_token()
        if not self.access_token:
            raise ValueError("Token refresh failed - no access token set")
        return self.access_token

    def _send_auth_failure_notification(self, error_message: str):
        """Send notification when auth fails"""
        try:
            from fincoll.utils.notifications import send_auth_failure_notification

            send_auth_failure_notification(
                service="TradeStation",
                error_message=error_message,
                details={
                    "token_file": str(self.token_file),
                    "has_refresh_token": bool(self.refresh_token),
                    "last_expires_at": self.expires_at.isoformat()
                    if self.expires_at
                    else None,
                },
            )
        except Exception as notify_error:
            # Don't let notification failure break auth flow
            print(f"Warning: Could not send auth failure notification: {notify_error}")

    def get_access_token(self) -> str:
        """
        Get valid access token (reloads from file if needed)

        CRITICAL CHANGE (2026-03-06): Auto-refresh DISABLED to prevent TradeStation API spam.

        Token refresh is now handled by external daemon:
        - Location: finvec/utils/tradestation_token_manager.py
        - PM2 process: tradestation-token-manager
        - Refreshes every 15 minutes automatically

        This method now:
        1. Checks if token is expired
        2. Reloads from file if needed (daemon keeps it fresh)
        3. Returns token or raises error if daemon not running

        Returns:
            Valid access token

        Raises:
            ValueError: If token expired and daemon not refreshing
        """
        if not self.access_token:
            raise ValueError(
                "Not authenticated. Call login() and exchange_code() first."
            )

        # Check if token is expired or about to expire (within 5 minutes)
        if self.expires_at and datetime.now() >= self.expires_at - timedelta(minutes=5):
            print("⚠️  Access token expired - reloading from file (daemon should have refreshed it)...")

            # Reload token from file instead of refreshing
            self._load_token()

            # Check if still expired after reload
            if self.expires_at and datetime.now() >= self.expires_at - timedelta(minutes=5):
                error_msg = (
                    "Token still expired after reload. External daemon may be stopped. "
                    "Start daemon: pm2 start tradestation-token-manager"
                )
                print(f"🚨 {error_msg}")
                raise ValueError(error_msg)

            print("✅ Token reloaded successfully")

        return self.access_token

    def is_authenticated(self) -> bool:
        """
        Check if currently authenticated with valid token

        CRITICAL CHANGE (2026-03-06): Auto-refresh DISABLED in this method.
        Token refresh is handled by external daemon (finvec/utils/tradestation_token_manager.py).
        """
        # Check if we're in auth failure backoff period
        if self.auth_failure_backoff_until:
            if datetime.now() < self.auth_failure_backoff_until:
                # Still in backoff period - don't retry
                return False
            else:
                # Backoff period expired - reset counter
                self.auth_failure_count = 0
                self.auth_failure_backoff_until = None

        if not self.access_token:
            return False

        # Check if token is not expired
        if self.expires_at and datetime.now() < self.expires_at:
            # Token is valid - reset failure counter
            self.auth_failure_count = 0
            return True

        # Token expired - try reloading from file (daemon should have refreshed it)
        if self.refresh_token:
            try:
                # Reload from file instead of refreshing
                self._load_token()

                # Check if still expired after reload
                if self.expires_at and datetime.now() < self.expires_at:
                    # Success - reset failure counter
                    self.auth_failure_count = 0
                    return True
                else:
                    # Still expired after reload - daemon may be stopped
                    print(
                        "⚠️  Token expired after reload. External daemon may be stopped. "
                        "Start: pm2 start tradestation-token-manager"
                    )
                    return False

            except Exception as e:
                # Failure - increment counter
                self.auth_failure_count += 1
                print(
                    f"⚠️  Auth token reload failed ({self.auth_failure_count}/{self.max_auth_failures}): {e}"
                )

                # Check if we've hit max failures
                if self.auth_failure_count >= self.max_auth_failures:
                    self.auth_failure_backoff_until = datetime.now() + timedelta(
                        seconds=self.auth_backoff_seconds
                    )
                    print(
                        f"🚫 Auth failure limit reached. Backing off until {self.auth_failure_backoff_until.isoformat()}"
                    )

                return False

        return False

    def logout(self):
        """Clear authentication tokens"""
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None

        if self.token_file.exists():
            self.token_file.unlink()
