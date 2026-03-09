"""
Safe Mode Circuit Breaker for FinColl.

Auto-pauses trading on repeated failures (429/5xx errors) to prevent:
- Rate limit violations
- Cascading failures
- Account suspension risk
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SafeModeState(str, Enum):
    """Safe mode states."""

    NORMAL = "normal"  # Operating normally
    WARNING = "warning"  # Elevated error rate, close to threshold
    SAFE_MODE = "safe_mode"  # Trading paused, errors exceeded threshold
    MANUAL_PAUSE = "manual_pause"  # Manually triggered pause


@dataclass
class SafeModeEvent:
    """Event that contributes to safe mode decision."""

    timestamp: float
    event_type: str  # "rate_limit", "server_error", "timeout", "manual"
    provider: Optional[str]
    details: Optional[str]


@dataclass
class SafeModeStatus:
    """Current safe mode status."""

    state: SafeModeState
    entered_at: Optional[datetime]
    reason: Optional[str]
    recent_events: List[SafeModeEvent]
    auto_resume_at: Optional[datetime]
    manual_override: bool


class SafeModeManager:
    """
    Manages safe mode state for FinColl.

    Safe mode is triggered when:
    - Too many 429 (rate limit) errors in short window
    - Too many 5xx (server) errors in short window
    - Manual trigger via API
    - Prediction divergence signals (future enhancement)
    """

    def __init__(
        self,
        rate_limit_threshold: int = 3,  # 3 rate limit errors
        server_error_threshold: int = 5,  # 5 server errors
        window_seconds: int = 300,  # 5 minute window
        auto_resume_minutes: int = 30,  # Auto-resume after 30 min
        enable_auto_resume: bool = True,
    ):
        self.rate_limit_threshold = rate_limit_threshold
        self.server_error_threshold = server_error_threshold
        self.window_seconds = window_seconds
        self.auto_resume_minutes = auto_resume_minutes
        self.enable_auto_resume = enable_auto_resume

        self.state = SafeModeState.NORMAL
        self.entered_at: Optional[datetime] = None
        self.reason: Optional[str] = None
        self.events: List[SafeModeEvent] = []
        self.manual_override = False
        self.auto_resume_at: Optional[datetime] = None

    def record_rate_limit_error(
        self, provider: str, details: Optional[str] = None
    ) -> None:
        """Record a 429 rate limit error."""
        event = SafeModeEvent(
            timestamp=time.time(),
            event_type="rate_limit",
            provider=provider,
            details=details,
        )
        self.events.append(event)
        self._update_metrics_for_event(provider, "rate_limit")
        self._check_thresholds()

    def record_server_error(
        self, provider: str, status_code: int, details: Optional[str] = None
    ) -> None:
        """Record a 5xx server error."""
        event = SafeModeEvent(
            timestamp=time.time(),
            event_type="server_error",
            provider=provider,
            details=f"HTTP {status_code}: {details}"
            if details
            else f"HTTP {status_code}",
        )
        self.events.append(event)
        self._update_metrics_for_event(provider, "server_error")
        self._check_thresholds()

    def record_timeout(self, provider: str, details: Optional[str] = None) -> None:
        """Record a timeout error."""
        event = SafeModeEvent(
            timestamp=time.time(),
            event_type="timeout",
            provider=provider,
            details=details,
        )
        self.events.append(event)
        self._update_metrics_for_event(provider, "timeout")
        self._check_thresholds()

    def trigger_manual(self, reason: str) -> None:
        """Manually trigger safe mode."""
        event = SafeModeEvent(
            timestamp=time.time(),
            event_type="manual",
            provider=None,
            details=reason,
        )
        self.events.append(event)
        self._enter_safe_mode("manual", reason)
        self.manual_override = True
        logger.warning(f"Safe mode manually triggered: {reason}")

    def resume_manual(self) -> bool:
        """Manually resume from safe mode."""
        if self.state in [SafeModeState.SAFE_MODE, SafeModeState.MANUAL_PAUSE]:
            self.state = SafeModeState.NORMAL
            self.entered_at = None
            self.reason = None
            self.manual_override = False
            self.auto_resume_at = None
            logger.info("Safe mode manually resumed")
            self._update_state_metrics()
            self._send_resume_notification(manual=True)
            return True
        return False

    def _check_thresholds(self) -> None:
        """Check if error thresholds exceeded."""
        if self.state in [SafeModeState.SAFE_MODE, SafeModeState.MANUAL_PAUSE]:
            return  # Already in safe mode

        # Clean up old events
        cutoff = time.time() - self.window_seconds
        self.events = [e for e in self.events if e.timestamp > cutoff]

        # Count recent errors by type
        rate_limit_count = len([e for e in self.events if e.event_type == "rate_limit"])
        server_error_count = len(
            [e for e in self.events if e.event_type == "server_error"]
        )

        # Check rate limit threshold
        if rate_limit_count >= self.rate_limit_threshold:
            providers = set(
                e.provider for e in self.events if e.event_type == "rate_limit"
            )
            reason = f"Rate limit threshold exceeded: {rate_limit_count} errors from {providers}"
            self._enter_safe_mode("rate_limit", reason)
            return

        # Check server error threshold
        if server_error_count >= self.server_error_threshold:
            providers = set(
                e.provider for e in self.events if e.event_type == "server_error"
            )
            reason = f"Server error threshold exceeded: {server_error_count} errors from {providers}"
            self._enter_safe_mode("server_error", reason)
            return

        # Check for warning state (approaching thresholds)
        if (
            rate_limit_count >= self.rate_limit_threshold - 1
            or server_error_count >= self.server_error_threshold - 2
        ):
            if self.state != SafeModeState.WARNING:
                self.state = SafeModeState.WARNING
                warning_msg = (
                    f"Entering WARNING state: {rate_limit_count} rate limit errors, "
                    f"{server_error_count} server errors in last {self.window_seconds}s"
                )
                logger.warning(warning_msg)
                self._update_state_metrics()
                self._send_warning_notification(warning_msg)
        else:
            if self.state == SafeModeState.WARNING:
                self.state = SafeModeState.NORMAL
                logger.info("Returning to NORMAL state")
                self._update_state_metrics()

    def _enter_safe_mode(self, trigger_type: str, reason: str) -> None:
        """Enter safe mode."""
        self.state = (
            SafeModeState.MANUAL_PAUSE
            if trigger_type == "manual"
            else SafeModeState.SAFE_MODE
        )
        self.entered_at = datetime.now()
        self.reason = reason

        if self.enable_auto_resume and trigger_type != "manual":
            self.auto_resume_at = datetime.now() + timedelta(
                minutes=self.auto_resume_minutes
            )
            logger.critical(
                f"SAFE MODE ACTIVATED: {reason}. "
                f"Auto-resume at {self.auto_resume_at.isoformat()}"
            )
        else:
            self.auto_resume_at = None
            logger.critical(
                f"SAFE MODE ACTIVATED: {reason}. Manual intervention required."
            )

        # Update metrics
        self._update_state_metrics()
        self._record_trigger_metric(trigger_type)

        # Send notification
        self._send_notification()

    def check_auto_resume(self) -> bool:
        """Check if auto-resume conditions met."""
        if (
            self.state == SafeModeState.SAFE_MODE
            and not self.manual_override
            and self.auto_resume_at
            and datetime.now() >= self.auto_resume_at
        ):
            self.state = SafeModeState.NORMAL
            self.entered_at = None
            self.reason = None
            self.auto_resume_at = None
            logger.info("Safe mode auto-resumed")
            self._update_state_metrics()
            self._send_resume_notification(manual=False)
            return True
        return False

    def is_safe_mode_active(self) -> bool:
        """Check if safe mode is currently active."""
        self.check_auto_resume()  # Check for auto-resume first
        return self.state in [SafeModeState.SAFE_MODE, SafeModeState.MANUAL_PAUSE]

    def get_status(self) -> SafeModeStatus:
        """Get current safe mode status."""
        self.check_auto_resume()  # Update state first

        # Get recent events (last hour)
        cutoff = time.time() - 3600
        recent_events = [e for e in self.events if e.timestamp > cutoff]

        return SafeModeStatus(
            state=self.state,
            entered_at=self.entered_at,
            reason=self.reason,
            recent_events=recent_events[-20:],  # Last 20 events
            auto_resume_at=self.auto_resume_at,
            manual_override=self.manual_override,
        )

    def get_provider_health(self) -> Dict[str, dict]:
        """Get per-provider health stats."""
        cutoff = time.time() - self.window_seconds
        recent_events = [e for e in self.events if e.timestamp > cutoff]

        providers = set(e.provider for e in recent_events if e.provider)
        health = {}

        for provider in providers:
            provider_events = [e for e in recent_events if e.provider == provider]
            rate_limit_errors = len(
                [e for e in provider_events if e.event_type == "rate_limit"]
            )
            server_errors = len(
                [e for e in provider_events if e.event_type == "server_error"]
            )
            timeouts = len([e for e in provider_events if e.event_type == "timeout"])

            health[provider] = {
                "total_errors": len(provider_events),
                "rate_limit_errors": rate_limit_errors,
                "server_errors": server_errors,
                "timeouts": timeouts,
                "healthy": rate_limit_errors < self.rate_limit_threshold
                and server_errors < self.server_error_threshold,
            }

        return health

    def _send_notification(self) -> None:
        """Send notification for safe mode activation."""
        try:
            from ..utils.notifications import send_safe_mode_notification

            send_safe_mode_notification(
                state=self.state.value,
                reason=self.reason or "Unknown",
                auto_resume_at=(
                    self.auto_resume_at.isoformat() if self.auto_resume_at else None
                ),
            )
        except Exception as e:
            logger.error(f"Failed to send safe mode notification: {e}")

    def _send_resume_notification(self, manual: bool) -> None:
        """Send notification for safe mode resume."""
        try:
            from ..utils.notifications import send_safe_mode_resume_notification

            send_safe_mode_resume_notification(manual=manual)
        except Exception as e:
            logger.error(f"Failed to send resume notification: {e}")

    def _send_warning_notification(self, message: str) -> None:
        """Send warning notification."""
        try:
            from ..utils.notifications import send_safe_mode_notification

            send_safe_mode_notification(
                state="warning",
                reason=message,
                auto_resume_at=None,
            )
        except Exception as e:
            logger.error(f"Failed to send warning notification: {e}")

    def _update_state_metrics(self) -> None:
        """Update Prometheus metrics for current state."""
        try:
            from .metrics import update_safe_mode_metrics

            update_safe_mode_metrics(
                state=self.state.value,
                is_active=self.is_safe_mode_active(),
            )
        except Exception as e:
            logger.debug(f"Failed to update state metrics: {e}")

    def _record_trigger_metric(self, trigger_type: str) -> None:
        """Record safe mode trigger in Prometheus."""
        try:
            from .metrics import record_safe_mode_trigger

            record_safe_mode_trigger(trigger_type=trigger_type)
        except Exception as e:
            logger.debug(f"Failed to record trigger metric: {e}")

    def _update_metrics_for_event(self, provider: str, event_type: str) -> None:
        """Record event in Prometheus metrics."""
        try:
            from .metrics import record_safe_mode_event

            record_safe_mode_event(provider=provider, event_type=event_type)
        except Exception as e:
            logger.debug(f"Failed to record event metric: {e}")


# Global safe mode manager instance
_safe_mode_manager: Optional[SafeModeManager] = None


def get_safe_mode_manager() -> SafeModeManager:
    """Get global safe mode manager instance."""
    global _safe_mode_manager
    if _safe_mode_manager is None:
        _safe_mode_manager = SafeModeManager()
    return _safe_mode_manager


def is_safe_mode_active() -> bool:
    """Quick check if safe mode is active."""
    return get_safe_mode_manager().is_safe_mode_active()
