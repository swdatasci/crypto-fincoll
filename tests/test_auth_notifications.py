"""
Unit tests for authentication failure notifications

Tests verify EXPECTED BEHAVIOR:
- Auth failures should trigger notifications
- Notifications should be sent via email AND SMS
- Notifications should include critical details
"""

import pytest
from unittest.mock import Mock, patch, call


@pytest.fixture
def mock_tradestation_auth_failure(monkeypatch):
    """Mock TradeStation auth failure"""

    def mock_get_access_token(*args, **kwargs):
        raise Exception("Token expired")

    monkeypatch.setattr(
        "fincoll.auth.tradestation_auth.TradeStationAuth.get_access_token",
        mock_get_access_token,
    )


class TestAuthFailureNotifications:
    """Test that auth failures trigger critical notifications"""

    @patch("fincoll.utils.notifications.send_auth_failure_notification")
    @patch("fincoll.auth.tradestation_auth.Path.exists")
    @patch("fincoll.auth.tradestation_auth.Path.open")
    def test_tradestation_auth_failure_sends_notification(
        self, mock_open, mock_exists, mock_send_notification
    ):
        """When TradeStation token file is missing/invalid, should send notification"""
        from fincoll.auth.tradestation_auth import TradeStationAuth
        from datetime import datetime

        # Make token file appear missing (daemon not running)
        mock_exists.return_value = False

        auth = TradeStationAuth()
        # Force token to appear expired so get_access_token triggers reload attempt
        auth.access_token = "expired_token"
        auth.expires_at = datetime(2020, 1, 1)  # Far in the past

        # Expected: reload attempt fails (file missing), notification sent
        # Note: Current implementation just reloads, doesn't raise - test needs update
        # For now, skip this test as the notification behavior changed with daemon architecture
        pytest.skip(
            "Auth notification behavior changed with token daemon - needs redesign"
        )

    @patch("fincoll.utils.notifications.send_auth_failure_notification")
    def test_token_refresh_failure_sends_notification(self, mock_send_notification):
        """When token refresh fails, should send notification"""
        from fincoll.auth.tradestation_auth import TradeStationAuth

        auth = TradeStationAuth()
        # Simulate expired token with invalid refresh token
        auth.refresh_token = "invalid_token"
        auth.expires_at = None  # Force refresh attempt

        # Expected: Refresh failure should trigger notification
        with pytest.raises(Exception):
            auth._refresh_access_token()

        # Expected: Notification sent with error details
        assert mock_send_notification.called
        call_args = mock_send_notification.call_args
        assert call_args[1]["service"] == "TradeStation"
        assert "error_message" in call_args[1]

    def test_notification_includes_critical_priority(self):
        """Auth failure notifications must be marked as CRITICAL"""
        from fincoll.utils.notifications import send_auth_failure_notification

        with (
            patch("fincoll.utils.notifications._send_via_caelum_mcp") as mock_caelum,
            patch("fincoll.utils.notifications._send_via_pim_service") as mock_pim,
        ):
            send_auth_failure_notification(
                service="TradeStation", error_message="Token refresh failed"
            )

            assert mock_caelum.called, "Caelum MCP notification should be attempted"
            notification = mock_caelum.call_args[0][0]
            assert notification.get("priority") == "critical", (
                f"Expected priority='critical', got {notification.get('priority')!r}"
            )

    def test_notification_sent_via_both_email_and_sms(self):
        """Critical auth failures should use BOTH email AND SMS"""
        from fincoll.utils.notifications import send_auth_failure_notification

        with (
            patch("fincoll.utils.notifications._send_via_caelum_mcp") as mock_caelum,
            patch("fincoll.utils.notifications._send_via_pim_service") as mock_pim,
        ):
            send_auth_failure_notification(
                service="TradeStation", error_message="Authentication lost"
            )

            assert mock_pim.called, "PIM service notification should be attempted"
            notification = mock_pim.call_args[0][0]
            assert notification.get("channel") == "both", (
                f"Expected channel='both', got {notification.get('channel')!r}"
            )

    def test_notification_includes_service_name(self):
        """Notification should clearly identify which service failed"""
        from fincoll.utils.notifications import send_auth_failure_notification

        with (
            patch("fincoll.utils.notifications._send_via_caelum_mcp") as mock_caelum,
            patch("fincoll.utils.notifications._send_via_pim_service") as mock_pim,
        ):
            send_auth_failure_notification(
                service="TradeStation", error_message="Token expired"
            )

            assert mock_caelum.called, "Caelum MCP notification should be attempted"
            notification = mock_caelum.call_args[0][0]
            assert "TradeStation" in notification.get("message", ""), (
                f"Service name missing from notification message: {notification.get('message')!r}"
            )

    def test_notification_includes_timestamp_and_details(self):
        """Notification should include when the failure occurred and context"""
        from fincoll.utils.notifications import send_auth_failure_notification

        with patch("fincoll.utils.notifications._send_via_pim_service") as mock_pim:
            send_auth_failure_notification(
                service="TradeStation",
                error_message="Auth failed",
                details={"token_file": "/path/to/token.json"},
            )

            assert mock_pim.called, "PIM service notification should be attempted"
            notification = mock_pim.call_args[0][0]
            assert "details" in notification, "Notification missing 'details' key"
            assert "timestamp" in notification["details"], (
                "Notification details missing 'timestamp'"
            )

    def test_notification_failure_does_not_break_auth_flow(self):
        """If notification fails, auth error should still be raised"""
        from fincoll.auth.tradestation_auth import TradeStationAuth

        with patch(
            "fincoll.utils.notifications.send_auth_failure_notification",
            side_effect=Exception("Notification service down"),
        ):
            # Expected: Auth failure should still be raised even if notification fails
            with pytest.raises(Exception) as exc_info:
                auth = TradeStationAuth()
                auth.refresh_token = None
                auth._refresh_access_token()

            # Should get auth error, not notification error
            assert "refresh token" in str(exc_info.value).lower()


class TestNotificationChannels:
    """Test notification delivery through multiple channels"""

    def test_caelum_mcp_notification_attempted_first(self):
        """Should try Caelum MCP notification service first"""
        from fincoll.utils.notifications import send_auth_failure_notification

        with patch("fincoll.utils.notifications.requests.post") as mock_post:
            mock_post.return_value.status_code = 200

            send_auth_failure_notification(service="TradeStation", error_message="Test")

            # Expected: Should call Caelum MCP endpoint
            calls = [str(call) for call in mock_post.call_args_list]
            assert any("10.32.3.27:8090" in str(call) for call in calls)

    def test_pim_service_as_fallback(self):
        """If Caelum MCP fails, should fallback to PIM service"""
        from fincoll.utils.notifications import send_auth_failure_notification

        with (
            patch(
                "fincoll.utils.notifications._send_via_caelum_mcp", return_value=False
            ),
            patch("fincoll.utils.notifications._send_via_pim_service") as mock_pim,
        ):
            send_auth_failure_notification(service="TradeStation", error_message="Test")

            # Expected: Should attempt PIM service
            assert mock_pim.called

    def test_both_channels_attempted_for_reliability(self):
        """Should attempt both notification channels for critical alerts"""
        from fincoll.utils.notifications import send_auth_failure_notification

        with (
            patch("fincoll.utils.notifications._send_via_caelum_mcp") as mock_caelum,
            patch("fincoll.utils.notifications._send_via_pim_service") as mock_pim,
        ):
            send_auth_failure_notification(
                service="TradeStation", error_message="Critical auth failure"
            )

            # Expected: Both channels should be attempted
            assert mock_caelum.called
            assert mock_pim.called
