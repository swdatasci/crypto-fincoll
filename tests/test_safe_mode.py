"""
Unit tests for Safe Mode Circuit Breaker

Tests cover:
- State transitions
- Threshold detection
- Auto-resume functionality
- Manual trigger/resume
- Error recording
- Provider health tracking
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from fincoll.monitoring.safe_mode import (
    SafeModeEvent,
    SafeModeManager,
    SafeModeState,
    SafeModeStatus,
)


class TestSafeModeStates:
    """Test safe mode state enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert SafeModeState.NORMAL.value == "normal"
        assert SafeModeState.WARNING.value == "warning"
        assert SafeModeState.SAFE_MODE.value == "safe_mode"
        assert SafeModeState.MANUAL_PAUSE.value == "manual_pause"


class TestSafeModeManager:
    """Test SafeModeManager core functionality."""

    def test_initial_state(self):
        """Test manager starts in NORMAL state."""
        manager = SafeModeManager()
        assert manager.state == SafeModeState.NORMAL
        assert not manager.is_safe_mode_active()
        assert manager.entered_at is None
        assert manager.reason is None

    def test_rate_limit_threshold(self):
        """Test safe mode triggers on rate limit threshold."""
        manager = SafeModeManager(rate_limit_threshold=3, window_seconds=60)

        # Record 2 rate limit errors - should not trigger
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")
        assert not manager.is_safe_mode_active()

        # 3rd error should trigger safe mode
        manager.record_rate_limit_error("tradestation", "Error 3")
        assert manager.is_safe_mode_active()
        assert manager.state == SafeModeState.SAFE_MODE
        assert "rate limit" in manager.reason.lower()

    def test_server_error_threshold(self):
        """Test safe mode triggers on server error threshold."""
        manager = SafeModeManager(server_error_threshold=5, window_seconds=60)

        # Record 4 server errors - should not trigger
        for i in range(4):
            manager.record_server_error("alpaca", 503, f"Error {i}")
        assert not manager.is_safe_mode_active()

        # 5th error should trigger
        manager.record_server_error("alpaca", 503, "Error 5")
        assert manager.is_safe_mode_active()
        assert manager.state == SafeModeState.SAFE_MODE
        assert "server error" in manager.reason.lower()

    def test_warning_state(self):
        """Test WARNING state triggers before safe mode."""
        manager = SafeModeManager(rate_limit_threshold=3, window_seconds=60)

        # Record 2 errors - should enter WARNING (threshold - 1)
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        assert manager.state == SafeModeState.WARNING
        assert not manager.is_safe_mode_active()

    def test_window_expiry(self):
        """Test old events expire and don't count toward threshold."""
        manager = SafeModeManager(
            rate_limit_threshold=3,
            window_seconds=1,  # 1 second window
        )

        # Record 2 errors
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        # Wait for window to expire
        time.sleep(1.5)

        # Record 1 more - should not trigger (old events expired)
        manager.record_rate_limit_error("tradestation", "Error 3")
        assert not manager.is_safe_mode_active()

    def test_manual_trigger(self):
        """Test manual safe mode trigger."""
        manager = SafeModeManager()

        manager.trigger_manual("Testing manual trigger")

        assert manager.is_safe_mode_active()
        assert manager.state == SafeModeState.MANUAL_PAUSE
        assert manager.manual_override is True
        assert manager.reason == "Testing manual trigger"
        assert manager.auto_resume_at is None  # Manual triggers don't auto-resume

    def test_manual_resume(self):
        """Test manual resume from safe mode."""
        manager = SafeModeManager()

        # Trigger safe mode
        manager.trigger_manual("Test")
        assert manager.is_safe_mode_active()

        # Resume manually
        success = manager.resume_manual()
        assert success is True
        assert not manager.is_safe_mode_active()
        assert manager.state == SafeModeState.NORMAL

    def test_resume_when_not_in_safe_mode(self):
        """Test resume returns False when not in safe mode."""
        manager = SafeModeManager()

        success = manager.resume_manual()
        assert success is False

    def test_auto_resume(self):
        """Test automatic resume after timeout."""
        manager = SafeModeManager(
            rate_limit_threshold=2,
            auto_resume_minutes=0.01,  # 0.6 seconds
            window_seconds=60,
        )

        # Trigger safe mode via threshold
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")
        assert manager.is_safe_mode_active()
        assert manager.auto_resume_at is not None

        # Wait for auto-resume
        time.sleep(1)

        # Check auto-resume
        manager.check_auto_resume()
        assert not manager.is_safe_mode_active()
        assert manager.state == SafeModeState.NORMAL

    def test_auto_resume_disabled(self):
        """Test auto-resume can be disabled."""
        manager = SafeModeManager(
            rate_limit_threshold=2,
            enable_auto_resume=False,
            window_seconds=60,
        )

        # Trigger safe mode
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        assert manager.auto_resume_at is None  # No auto-resume scheduled

    def test_manual_trigger_no_auto_resume(self):
        """Test manual triggers don't auto-resume."""
        manager = SafeModeManager(auto_resume_minutes=0.01)

        manager.trigger_manual("Test")
        time.sleep(1)

        manager.check_auto_resume()
        assert manager.is_safe_mode_active()  # Still in safe mode

    def test_timeout_recording(self):
        """Test timeout errors are recorded."""
        manager = SafeModeManager()

        manager.record_timeout("alpaca", "Connection timeout")

        status = manager.get_status()
        assert len(status.recent_events) == 1
        assert status.recent_events[0].event_type == "timeout"
        assert status.recent_events[0].provider == "alpaca"

    def test_get_status(self):
        """Test get_status returns complete status."""
        manager = SafeModeManager(rate_limit_threshold=2, window_seconds=60)

        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        status = manager.get_status()

        assert isinstance(status, SafeModeStatus)
        assert status.state == SafeModeState.SAFE_MODE
        assert status.entered_at is not None
        assert "rate limit" in status.reason.lower()
        assert len(status.recent_events) == 2
        assert status.auto_resume_at is not None

    def test_provider_health(self):
        """Test per-provider health tracking."""
        manager = SafeModeManager(window_seconds=60)

        # Record errors for different providers
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_server_error("alpaca", 503, "Error 2")
        manager.record_timeout("public", "Timeout")

        health = manager.get_provider_health()

        assert "tradestation" in health
        assert health["tradestation"]["rate_limit_errors"] == 1
        assert health["tradestation"]["server_errors"] == 0

        assert "alpaca" in health
        assert health["alpaca"]["server_errors"] == 1

        assert "public" in health
        assert health["public"]["timeouts"] == 1

    def test_mixed_provider_errors(self):
        """Test errors from multiple providers are tracked separately."""
        manager = SafeModeManager(rate_limit_threshold=3, window_seconds=60)

        # 2 errors from tradestation, 2 from alpaca - should not trigger
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")
        manager.record_rate_limit_error("alpaca", "Error 3")
        manager.record_rate_limit_error("alpaca", "Error 4")

        # Total is 4, but threshold is per-total, not per-provider
        # So this SHOULD trigger since total >= 3
        assert manager.is_safe_mode_active()

    def test_recent_events_limit(self):
        """Test recent_events limits to last 20."""
        manager = SafeModeManager(window_seconds=3600)  # 1 hour

        # Record 30 events
        for i in range(30):
            manager.record_timeout("test", f"Event {i}")

        status = manager.get_status()
        # Should only return last 20
        assert len(status.recent_events) <= 20

    @patch("fincoll.monitoring.safe_mode.SafeModeManager._send_notification")
    def test_notification_on_trigger(self, mock_notify):
        """Test notification is sent when safe mode triggers."""
        manager = SafeModeManager(rate_limit_threshold=2, window_seconds=60)

        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        # Notification should have been called
        mock_notify.assert_called_once()

    @patch("fincoll.monitoring.safe_mode.SafeModeManager._send_resume_notification")
    def test_notification_on_resume(self, mock_notify):
        """Test notification is sent on manual resume."""
        manager = SafeModeManager()

        manager.trigger_manual("Test")
        manager.resume_manual()

        # Resume notification should have been called with manual=True
        mock_notify.assert_called_once_with(manual=True)

    @patch("fincoll.monitoring.safe_mode.SafeModeManager._send_warning_notification")
    def test_notification_on_warning(self, mock_notify):
        """Test notification is sent on WARNING state."""
        manager = SafeModeManager(rate_limit_threshold=3, window_seconds=60)

        # Trigger WARNING state (threshold - 1)
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        assert manager.state == SafeModeState.WARNING
        mock_notify.assert_called_once()


class TestSafeModeIntegration:
    """Integration tests for safe mode with other components."""

    def test_state_transitions(self):
        """Test full state transition flow."""
        manager = SafeModeManager(
            rate_limit_threshold=3,
            auto_resume_minutes=0.01,
            window_seconds=60,
        )

        # 1. Start in NORMAL
        assert manager.state == SafeModeState.NORMAL

        # 2. Enter WARNING
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")
        assert manager.state == SafeModeState.WARNING

        # 3. Enter SAFE_MODE
        manager.record_rate_limit_error("tradestation", "Error 3")
        assert manager.state == SafeModeState.SAFE_MODE

        # 4. Auto-resume back to NORMAL
        time.sleep(1)
        manager.check_auto_resume()
        assert manager.state == SafeModeState.NORMAL

    def test_event_dataclass(self):
        """Test SafeModeEvent dataclass."""
        event = SafeModeEvent(
            timestamp=time.time(),
            event_type="rate_limit",
            provider="tradestation",
            details="Test error",
        )

        assert event.event_type == "rate_limit"
        assert event.provider == "tradestation"
        assert event.details == "Test error"
        assert isinstance(event.timestamp, float)


class TestSafeModeEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_provider_health(self):
        """Test provider health with no events."""
        manager = SafeModeManager()
        health = manager.get_provider_health()
        assert health == {}

    def test_multiple_manual_triggers(self):
        """Test multiple manual triggers."""
        manager = SafeModeManager()

        manager.trigger_manual("First trigger")
        first_entered = manager.entered_at

        time.sleep(0.1)

        manager.trigger_manual("Second trigger")
        second_entered = manager.entered_at

        # Should update the entered_at time
        assert second_entered > first_entered
        assert manager.reason == "Second trigger"

    def test_resume_clears_state(self):
        """Test resume clears all state properly."""
        manager = SafeModeManager(rate_limit_threshold=2, window_seconds=60)

        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        assert manager.is_safe_mode_active()
        assert manager.entered_at is not None
        assert manager.reason is not None

        manager.resume_manual()

        assert manager.entered_at is None
        assert manager.reason is None
        assert manager.manual_override is False
        assert manager.auto_resume_at is None

    def test_config_validation(self):
        """Test configuration values are respected."""
        manager = SafeModeManager(
            rate_limit_threshold=10,
            server_error_threshold=20,
            window_seconds=600,
            auto_resume_minutes=60,
        )

        assert manager.rate_limit_threshold == 10
        assert manager.server_error_threshold == 20
        assert manager.window_seconds == 600
        assert manager.auto_resume_minutes == 60


class TestNotificationErrorHandling:
    """Test notification and metrics error handling."""

    @patch("fincoll.utils.notifications.send_safe_mode_notification")
    def test_notification_error_handling(self, mock_send):
        """Test safe mode handles notification sending errors gracefully."""
        mock_send.side_effect = Exception("Notification service down")

        manager = SafeModeManager(rate_limit_threshold=2, window_seconds=60)

        # This should trigger safe mode and attempt notification
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        # Should still be in safe mode despite notification failure
        assert manager.is_safe_mode_active()
        assert manager.state == SafeModeState.SAFE_MODE

    @patch("fincoll.utils.notifications.send_safe_mode_resume_notification")
    def test_resume_notification_error_handling(self, mock_send):
        """Test safe mode handles resume notification errors gracefully."""
        mock_send.side_effect = Exception("Notification service down")

        manager = SafeModeManager()
        manager.trigger_manual("Test")

        # Resume should work despite notification failure
        manager.resume_manual()

        # Should be resumed despite notification error
        assert not manager.is_safe_mode_active()
        assert manager.state == SafeModeState.NORMAL

    @patch("fincoll.utils.notifications.send_safe_mode_notification")
    def test_warning_notification_error_handling(self, mock_send):
        """Test safe mode handles warning notification errors gracefully."""
        mock_send.side_effect = Exception("Notification service down")

        manager = SafeModeManager(rate_limit_threshold=3, window_seconds=60)

        # Trigger WARNING state (threshold - 1)
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        # Should still be in warning state despite notification failure
        assert manager.state == SafeModeState.WARNING

    @patch("fincoll.monitoring.metrics.update_safe_mode_metrics")
    def test_state_metrics_error_handling(self, mock_metrics):
        """Test safe mode handles metrics update errors gracefully."""
        mock_metrics.side_effect = Exception("Metrics service down")

        manager = SafeModeManager(rate_limit_threshold=2, window_seconds=60)

        # This should trigger safe mode and attempt metrics update
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        # Should still be in safe mode despite metrics failure
        assert manager.is_safe_mode_active()

    @patch("fincoll.monitoring.metrics.record_safe_mode_trigger")
    def test_trigger_metric_error_handling(self, mock_record):
        """Test safe mode handles trigger metric recording errors gracefully."""
        mock_record.side_effect = Exception("Metrics service down")

        manager = SafeModeManager()

        # Manual trigger should work despite metrics failure
        manager.trigger_manual("Test trigger")

        # Should still be in safe mode despite metrics error
        assert manager.is_safe_mode_active()
        assert manager.state == SafeModeState.MANUAL_PAUSE

    @patch("fincoll.monitoring.metrics.record_safe_mode_event")
    def test_event_metric_error_handling(self, mock_record):
        """Test safe mode handles event metric recording errors gracefully."""
        mock_record.side_effect = Exception("Metrics service down")

        manager = SafeModeManager(rate_limit_threshold=2, window_seconds=60)

        # Record events - should work despite metrics failure
        manager.record_rate_limit_error("tradestation", "Error 1")
        manager.record_rate_limit_error("tradestation", "Error 2")

        # Should still be in safe mode despite metrics error
        assert manager.is_safe_mode_active()
        # Events should still be recorded
        status = manager.get_status()
        assert len(status.recent_events) == 2


class TestGlobalSafeModeManager:
    """Test global safe mode manager singleton."""

    def test_get_safe_mode_manager(self):
        """Test global manager retrieval."""
        from fincoll.monitoring.safe_mode import get_safe_mode_manager

        manager1 = get_safe_mode_manager()
        manager2 = get_safe_mode_manager()

        # Should return same instance
        assert manager1 is manager2

    def test_is_safe_mode_active_shortcut(self):
        """Test global is_safe_mode_active function."""
        from fincoll.monitoring.safe_mode import (
            get_safe_mode_manager,
            is_safe_mode_active,
        )

        manager = get_safe_mode_manager()
        manager.trigger_manual("Test")

        assert is_safe_mode_active() is True

        manager.resume_manual()
        assert is_safe_mode_active() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
