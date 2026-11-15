from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_weather_service
from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.models import Alert, AlertStatus
from app.repositories import alert_repository
from app.schemas.alert import AlertOut, NotificationOut, PipelineRunOut
from app.services import alert_service, pipeline
from app.services.notification_service import get_notification_service
from app.services.weather_service import WeatherService

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _build_response(alert: Alert) -> AlertOut:
    return AlertOut(
        id=alert_service.public_id(alert),
        grid_cell_id=alert.grid_cell_id,
        location_id=alert.location_id,
        location=alert.location_label,
        title=alert.title,
        hazard_type=alert.hazard_type,
        severity=alert.severity.value,
        status=alert.status.value,
        risk_score=alert.risk_score,
        evidence=alert.evidence,
        description=alert.description,
        recommended_action=alert.recommended_action,
        confidence=alert.confidence,
        source=alert.source,
        issued_at=alert.issued_at,
        expires_at=alert.expires_at,
        resolved_at=alert.resolved_at,
        last_seen_at=alert.last_seen_at,
        acknowledged=alert.acknowledged,
        acknowledged_at=alert.acknowledged_at,
    )


def _resolve_alert_id(raw: str) -> int:
    """Accepts either the public "ALT-0001" form or a bare integer."""
    text = raw.upper().removeprefix("ALT-")
    try:
        return int(text)
    except ValueError:
        raise NotFoundError(f"Unknown alert id: {raw!r}") from None


@router.get("/active", response_model=list[AlertOut])
def get_active_alerts(db: Session = Depends(get_db)) -> list[AlertOut]:
    return [_build_response(a) for a in alert_repository.list_active(db)]


@router.get("/history", response_model=list[AlertOut])
def get_alert_history(db: Session = Depends(get_db)) -> list[AlertOut]:
    return [_build_response(a) for a in alert_repository.list_history(db)]


@router.get("/notifications", response_model=list[NotificationOut])
def get_recent_notifications(limit: int = Query(50, ge=1, le=200)) -> list[NotificationOut]:
    """The in-app notification feed (see notification_service.py) — what a
    dashboard would poll for "what just happened" without re-fetching the
    whole alert list.
    """
    service = get_notification_service()
    return [
        NotificationOut(
            alert_id=n.alert_id, public_id=n.public_id, event=n.event, title=n.title,
            severity=n.severity, hazard_type=n.hazard_type, grid_cell_id=n.grid_cell_id,
            message=n.message, created_at=n.created_at,
        )
        for n in service.recent_in_app(limit)
    ]


@router.post("/process", response_model=PipelineRunOut)
async def trigger_pipeline_run(
    db: Session = Depends(get_db),
    weather_service: WeatherService = Depends(get_weather_service),
) -> PipelineRunOut:
    """Manually runs one pipeline cycle immediately, instead of waiting for
    the next scheduled tick — for testing, demos, and ops ("run it now").
    Shares the same `asyncio.Lock` as the scheduler, so this can never run
    concurrently with (or duplicate) a scheduled tick that's already going.
    """
    result = await pipeline.run_pipeline_once(weather_service, db)
    return PipelineRunOut(**result)


@router.get("", response_model=list[AlertOut])
def get_all_alerts(
    status: str | None = Query(None, description="active | resolved | expired"),
    severity: str | None = Query(None, description="low | moderate | high | severe"),
    hazard_type: str | None = Query(None, alias="hazardType"),
    db: Session = Depends(get_db),
) -> list[AlertOut]:
    status_enum = AlertStatus(status) if status else None
    rows = alert_repository.list_all(db, status=status_enum, severity=severity, hazard_type=hazard_type)
    return [_build_response(a) for a in rows]


@router.post("/{alert_id}/acknowledge", response_model=AlertOut)
def acknowledge_alert(alert_id: str, db: Session = Depends(get_db)) -> AlertOut:
    alert = alert_repository.get_by_id(db, _resolve_alert_id(alert_id))
    if alert is None:
        raise NotFoundError(f"Unknown alert id: {alert_id!r}")
    return _build_response(alert_service.acknowledge(db, alert))


@router.post("/{alert_id}/resolve", response_model=AlertOut)
def resolve_alert(alert_id: str, db: Session = Depends(get_db)) -> AlertOut:
    alert = alert_repository.get_by_id(db, _resolve_alert_id(alert_id))
    if alert is None:
        raise NotFoundError(f"Unknown alert id: {alert_id!r}")
    return _build_response(alert_service.resolve(db, alert))


# Registered after the literal routes above ("/active", "/history", ...) --
# otherwise this catch-all would capture them as alert_id="active" etc.
@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: str, db: Session = Depends(get_db)) -> AlertOut:
    alert = alert_repository.get_by_id(db, _resolve_alert_id(alert_id))
    if alert is None:
        raise NotFoundError(f"Unknown alert id: {alert_id!r}")
    return _build_response(alert)
