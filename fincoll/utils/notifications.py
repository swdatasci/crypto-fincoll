"""
Notification utilities for critical system events

Sends notifications via:
1. Caelum MCP notification service (if available)
2. HTTP POST to PIM notification service (fallback)
"""

import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def send_safe_mode_notification(
    state: str, reason: str, auto_resume_at: Optional[str] = None
):
    """
    Send notification when safe mode is triggered

    Args:
        state: Safe mode state ("safe_mode", "manual_pause", "warning")
        reason: Reason for safe mode activation
        auto_resume_at: ISO timestamp when auto-resume will occur (if applicable)
    """
    # Determine priority and emoji based on state
    if state == "safe_mode":
        priority = "critical"
        emoji = "🛑"
        subject = "SAFE MODE ACTIVATED"
    elif state == "manual_pause":
        priority = "high"
        emoji = "⏸️"
        subject = "SAFE MODE MANUAL PAUSE"
    elif state == "warning":
        priority = "medium"
        emoji = "⚠️"
        subject = "SAFE MODE WARNING"
    else:
        priority = "low"
        emoji = "ℹ️"
        subject = "SAFE MODE UPDATE"

    # Build message
    message_parts = [
        f"{emoji} {subject}",
        "",
        f"Reason: {reason}",
    ]

    if auto_resume_at:
        message_parts.append(f"Auto-resume: {auto_resume_at}")
    else:
        message_parts.append("Manual resume required")

    message_parts.extend(
        [
            "",
            "Actions:",
            "- Check status: curl http://10.32.3.27:8002/api/v1/safe-mode/status",
            "- Manual resume: curl -X POST http://10.32.3.27:8002/api/v1/safe-mode/resume",
            "",
            "See SAFE_MODE_DOCUMENTATION.md for incident response playbook.",
        ]
    )

    notification = {
        "type": "safe_mode",
        "subject": f"{emoji} {subject}",
        "message": "\n".join(message_parts),
        "details": {
            "state": state,
            "reason": reason,
            "auto_resume_at": auto_resume_at,
            "timestamp": datetime.now().isoformat(),
        },
        "priority": priority,
        "channel": "both" if priority == "critical" else "email",
    }

    # Try Caelum MCP first
    caelum_sent = _send_via_caelum_mcp(notification)

    # Try PIM notification service as fallback
    pim_sent = _send_via_pim_service(notification)

    if not caelum_sent and not pim_sent:
        logger.error(f"Failed to send safe mode notification via all channels: {state}")
    else:
        logger.info(f"Safe mode notification sent: {state}")


def send_safe_mode_resume_notification(manual: bool = False):
    """
    Send notification when safe mode is resumed

    Args:
        manual: Whether resume was manual or automatic
    """
    resume_type = "Manual" if manual else "Automatic"

    notification = {
        "type": "safe_mode_resume",
        "subject": f"✅ Safe Mode Resumed ({resume_type})",
        "message": f"Safe mode has been resumed ({resume_type.lower()}).\n\nTrading operations are now active.",
        "details": {
            "resume_type": resume_type,
            "timestamp": datetime.now().isoformat(),
        },
        "priority": "medium",
        "channel": "email",
    }

    # Try Caelum MCP first
    caelum_sent = _send_via_caelum_mcp(notification)

    # Try PIM notification service as fallback
    pim_sent = _send_via_pim_service(notification)

    if not caelum_sent and not pim_sent:
        logger.error("Failed to send safe mode resume notification via all channels")
    else:
        logger.info(f"Safe mode resume notification sent: {resume_type}")


def send_auth_failure_notification(
    service: str, error_message: str, details: Optional[dict] = None
):
    """
    Send critical notification when authentication fails

    Args:
        service: Name of the service (e.g., "TradeStation", "Alpaca")
        error_message: Error message from the auth failure
        details: Additional details about the failure
    """
    notification = {
        "type": "auth_failure",
        "subject": f"🚨 CRITICAL: {service} Authentication Failed",
        "message": f"{service} API authentication has failed. Trading and data collection will be unavailable until resolved.\n\nError: {error_message}",
        "details": {
            "service": service,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            **(details or {}),
        },
        "priority": "critical",
        "channel": "both",  # Email AND SMS
    }

    # Try Caelum MCP first
    caelum_sent = _send_via_caelum_mcp(notification)

    # Try PIM notification service as fallback
    pim_sent = _send_via_pim_service(notification)

    if not caelum_sent and not pim_sent:
        logger.error(
            f"Failed to send {service} auth failure notification via all channels"
        )
    else:
        logger.info(f"Auth failure notification sent for {service}")


def _send_via_caelum_mcp(notification: dict) -> bool:
    """Send notification via Caelum MCP unified service"""
    try:
        # Caelum daemon is at 10.32.3.27:8090
        response = requests.post(
            "http://10.32.3.27:8090/api/notifications/send",
            json={
                "message": notification["message"],
                "priority": notification["priority"],
                "type": "error",
            },
            timeout=5,
        )

        if response.status_code == 200:
            logger.info("Notification sent via Caelum MCP")
            return True
        else:
            logger.warning(f"Caelum MCP returned status {response.status_code}")
            return False

    except Exception as e:
        logger.warning(f"Could not send via Caelum MCP: {e}")
        return False


def _send_via_pim_service(notification: dict) -> bool:
    """Send notification via PIM notification service"""
    try:
        # PIM server notification endpoint
        pim_url = os.getenv(
            "PIM_NOTIFICATION_URL", "http://10.32.3.27:3000/api/notifications"
        )

        response = requests.post(pim_url, json=notification, timeout=5)

        if response.status_code in (200, 201):
            logger.info("Notification sent via PIM service")
            return True
        else:
            logger.warning(f"PIM service returned status {response.status_code}")
            return False

    except Exception as e:
        logger.warning(f"Could not send via PIM service: {e}")
        return False
