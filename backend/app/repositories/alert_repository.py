"""Alert persistence. Lifecycle transitions (active -> resolved / expired,
acknowledgement) happen here as in-place updates on one row — only the
initial creation and the risk_assessments-style append-only tables elsewhere
grow without bound; an alert's own row is the single source of truth for
its current lifecycle state, matching the "Active / Resolved / Expired
alerts" and "Acknowledgement status" requirements directly.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Alert, AlertStatus


def find_active_by_key(db: Session, alert_key: str) -> Alert | None:
    return db.execute(
        select(Alert).where(Alert.alert_key == alert_key, Alert.status == AlertStatus.ACTIVE)
    ).scalar_one_or_none()


def get_by_id(db: Session, alert_id: int) -> Alert | None:
    return db.get(Alert, alert_id)


def create(db: Session, alert: Alert) -> Alert:
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def save(db: Session, alert: Alert) -> Alert:
    """Persist in-place changes to an already-existing row (update,
    acknowledge, resolve, expire) — never re-inserts.
    """
    db.commit()
    db.refresh(alert)
    return alert


def list_all(
    db: Session,
    status: AlertStatus | None = None,
    severity: str | None = None,
    hazard_type: str | None = None,
    limit: int = 200,
) -> list[Alert]:
    stmt = select(Alert)
    if status is not None:
        stmt = stmt.where(Alert.status == status)
    if severity is not None:
        stmt = stmt.where(Alert.severity == severity)
    if hazard_type is not None:
        stmt = stmt.where(Alert.hazard_type == hazard_type)
    stmt = stmt.order_by(Alert.issued_at.desc()).limit(limit)
    return list(db.execute(stmt).scalars())


def list_active(db: Session, limit: int = 200) -> list[Alert]:
    return list_all(db, status=AlertStatus.ACTIVE, limit=limit)


def list_history(db: Session, limit: int = 200) -> list[Alert]:
    """Resolved + expired alerts, most recently concluded first — the
    "Alert history" / "Resolved alerts" / "Expired alerts" requirement.
    """
    stmt = (
        select(Alert)
        .where(Alert.status.in_([AlertStatus.RESOLVED, AlertStatus.EXPIRED]))
        .order_by(Alert.issued_at.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars())


def list_active_expiring_before(db: Session, cutoff: datetime) -> list[Alert]:
    return list(
        db.execute(
            select(Alert).where(Alert.status == AlertStatus.ACTIVE, Alert.expires_at < cutoff)
        ).scalars()
    )
