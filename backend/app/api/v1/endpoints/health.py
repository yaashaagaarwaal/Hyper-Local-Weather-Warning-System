import logging

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.health import DatabaseHealthResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Application liveness. Always returns 200 if the process is up — does
    NOT touch the database, so it can't report "down" just because the DB is
    slow or unreachable. Use /health/db for that.
    """
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.APP_NAME, environment=settings.APP_ENV)


@router.get("/health/db", response_model=DatabaseHealthResponse)
def health_db(response: Response, db: Session = Depends(get_db)) -> DatabaseHealthResponse:
    """Database readiness — actually executes a query against PostgreSQL.

    Returns HTTP 200 only when the database is truly reachable; HTTP 503
    otherwise, so load balancers / uptime checks treat this as a real
    readiness probe rather than always-200 noise.
    """
    try:
        db.execute(text("SELECT 1"))
        return DatabaseHealthResponse(status="ok", database="connected")
    except Exception as exc:  # noqa: BLE001 — intentionally broad: any DB failure means "not ready"
        logger.error("Database health check failed: %s", exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return DatabaseHealthResponse(status="error", database="unreachable", detail=str(exc))
