from datetime import datetime, timezone

import pytest

from app.services.notification_service import (
    AlertNotification,
    EmailNotificationChannel,
    InAppNotificationChannel,
    NotConfiguredError,
    NotificationService,
    SMSNotificationChannel,
    WebSocketNotificationChannel,
)


def _notification(**overrides) -> AlertNotification:
    base = dict(
        alert_id=1, public_id="ALT-0001", event="created", title="Heavy Rainfall Warning",
        severity="high", hazard_type="heavy_rainfall", grid_cell_id="GRID-001",
        message="test", created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return AlertNotification(**base)


def test_in_app_channel_records_and_returns_most_recent_first():
    channel = InAppNotificationChannel()
    channel.send(_notification(alert_id=1))
    channel.send(_notification(alert_id=2))

    recent = channel.recent(limit=10)
    assert [n.alert_id for n in recent] == [2, 1]


def test_in_app_channel_respects_max_items():
    channel = InAppNotificationChannel(max_items=2)
    for i in range(5):
        channel.send(_notification(alert_id=i))
    assert len(channel.recent(limit=100)) == 2


@pytest.mark.parametrize("channel_cls", [WebSocketNotificationChannel, EmailNotificationChannel, SMSNotificationChannel])
def test_unconfigured_channels_refuse_to_send_rather_than_silently_no_op(channel_cls):
    channel = channel_cls()
    with pytest.raises(NotConfiguredError):
        channel.send(_notification())


def test_notification_service_delivers_to_in_app_and_skips_unconfigured_channels():
    service = NotificationService(channels=[InAppNotificationChannel(), WebSocketNotificationChannel()])
    service.notify(_notification())  # must not raise even though one channel refuses

    assert len(service.recent_in_app()) == 0  # the service's own default in_app wasn't used
    assert len(service._channels[0].recent()) == 1  # but the explicit in-app channel got it


def test_default_notification_service_uses_in_app_by_default():
    service = NotificationService()
    service.notify(_notification(event="resolved"))
    recent = service.recent_in_app()
    assert len(recent) == 1
    assert recent[0].event == "resolved"
