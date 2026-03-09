"""
Simple OAuth callback server for TradeStation authentication

Starts a temporary HTTP server to catch the OAuth authorization code,
then automatically exchanges it for an access token.
"""

import http.server
import urllib.parse
from typing import Optional
import threading
import time


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback"""

    authorization_code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        """Handle OAuth callback GET request"""
        # Parse the callback URL
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if 'code' in params:
            # Success - got authorization code
            OAuthCallbackHandler.authorization_code = params['code'][0]

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = """
            <html>
            <head><title>TradeStation Authentication</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1 style="color: green;">✅ Authentication Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                <p>FinVec is now authenticated with TradeStation.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())

        elif 'error' in params:
            # Error from TradeStation
            OAuthCallbackHandler.error = params.get('error_description', ['Unknown error'])[0]

            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = f"""
            <html>
            <head><title>TradeStation Authentication Failed</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1 style="color: red;">❌ Authentication Failed</h1>
                <p>{OAuthCallbackHandler.error}</p>
                <p>Please try again.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            # Invalid callback
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Invalid OAuth callback")

    def log_message(self, format, *args):
        """Suppress server log messages"""
        pass


class OAuthCallbackServer:
    """Temporary HTTP server for OAuth callbacks"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Start the callback server in a background thread"""
        # Reset class variables
        OAuthCallbackHandler.authorization_code = None
        OAuthCallbackHandler.error = None

        # Create server
        self.server = http.server.HTTPServer(
            ('10.32.3.27', self.port),
            OAuthCallbackHandler
        )

        # Run in background thread
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

        print(f"OAuth callback server started on http://10.32.3.27:{self.port}")

    def wait_for_code(self, timeout: int = 300) -> Optional[str]:
        """
        Wait for authorization code from OAuth callback

        Args:
            timeout: Maximum seconds to wait (default: 300 = 5 minutes)

        Returns:
            Authorization code or None if timeout/error
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if OAuthCallbackHandler.authorization_code:
                return OAuthCallbackHandler.authorization_code

            if OAuthCallbackHandler.error:
                raise ValueError(f"OAuth error: {OAuthCallbackHandler.error}")

            time.sleep(0.5)

        raise TimeoutError(f"OAuth callback timeout after {timeout} seconds")

    def stop(self):
        """Stop the callback server"""
        if self.server:
            self.server.shutdown()
            print("OAuth callback server stopped")


def start_oauth_server(port: int = 8080) -> OAuthCallbackServer:
    """
    Start OAuth callback server

    Args:
        port: Port to listen on (default: 8080)

    Returns:
        OAuthCallbackServer instance
    """
    server = OAuthCallbackServer(port)
    server.start()
    return server
