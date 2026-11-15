"""Notification service abstraction (Phase 5).

A `NotificationChannel` is anything that can be told "this alert was just
created/updated/resolved/expired". **In-app is implemented for real** — an
in-process feed `GET /api/v1/alerts/notifications` serves — because it
needs nothing external: the alert itself, immediately queryable the moment
it's written, already is the notification. WebSocket, email and SMS are
deliberately **stubs that refuse to send** (raise `NotConfiguredError`)
rather than silently doing nothing or pretending to work — per the explicit
instruction not to send real external notifications without a configured
provider and explicit testing. Wiring a real one later means filling in
that channel's `send()`, not touching any caller of `NotificationService`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class NotConfiguredError(Exception):
    """Raised by a channel that has no real provider wired up yet."""


@dataclass
class AlertNotification:
    alert_id: int
    public_id: str
    event: str  # "created" | "updated" | "resolved" | "expired"
    title: str
    severity: str
    hazard_type: str
    grid_cell_id: str
    message: str
    created_at: datetime


class NotificationChannel(ABC):
    name: str

    @abstractmethod
    def send(self, notification: AlertNotification) -> None: ...


class InAppNotificationChannel(NotificationChannel):
    name = "in_app"

    def __init__(self, max_items: int = 200):
        self._feed: deque[AlertNotification] = deque(maxlen=max_items)

    def send(self, notification: AlertNotification) -> None:
        self._feed.append(notification)
        logger.info(
            "notification.in_app event=%s alert=%s hazard=%s severity=%s",
            notification.event, notification.public_id, notification.hazard_type, notification.severity,
        )

    def recent(self, limit: int = 50) -> list[AlertNotification]:
        items = list(self._feed)
        return list(reversed(items[-limit:]))


class _UnconfiguredChannel(NotificationChannel):
    """Shared base for the channels Phase 5 only prepares an abstraction
    for — see the module docstring. Each one names the real provider it
    would need and the doc section tracking the work.
    """

    reason: str

    def send(self, notification: AlertNotification) -> None:
        raise NotConfiguredError(self.reason)


class WebSocketNotificationChannel(_UnconfiguredChannel):
    name = "websocket"
    reason = (
        "WebSocket push is not implemented yet — no connection manager or "
        "client subscription model exists. See docs/architecture.md §8."
    )


class EmailNotificationChannel(_UnconfiguredChannel):
    name = "email"
    reason = "No email provider (SMTP/SES/SendGrid, etc.) is configured. See docs/architecture.md §8."


class SMSNotificationChannel(_UnconfiguredChannel):
    name = "sms"
    reason = "No SMS gateway is configured. See docs/architecture.md §8."


class NotificationService:
    """Fans one alert event out to every configured channel. A channel
    raising `NotConfiguredError` is logged and skipped — it never stops the
    in-app channel (or any other working one) from still delivering.
    """

    def __init__(self, channels: list[NotificationChannel] | None = None):
        self.in_app = InAppNotificationChannel()
        self._channels: list[NotificationChannel] = channels if channels is not None else [self.in_app]

    def notify(self, notification: AlertNotification) -> None:
        for channel in self._channels:
            try:
                channel.send(notification)
            except NotConfiguredError as exc:
                logger.info("notification.skipped channel=%s reason=%s", channel.name, exc)

    def recent_in_app(self, limit: int = 50) -> list[AlertNotification]:
        return self.in_app.recent(limit)


# Process-wide singleton, same pattern as weather_service.py's module-level
# TTLCache instances — one feed per running process, not shared across
# `--workers N > 1` (same documented caveat as the weather cache).
_service = NotificationService()


def get_notification_service() -> NotificationService:
    return _service
