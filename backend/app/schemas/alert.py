"""Alert API shapes — camelCase, matching `src/data/alertsData.js`'s mock
`ALERTS` shape field-for-field wherever the two concepts line up (id,
title, severity, location, issuedAt, expiresAt, status, hazardType,
riskScore, description, recommendedAction, confidence), plus the new
fields Phase 5 actually needs (gridCellId, evidence, source, acknowledged)
that the mock never had to model.
"""

from datetime import datetime

from app.schemas.weather import CamelModel


class AlertOut(CamelModel):
    id: str  # "ALT-0001" — see app.services.alert_service.public_id
    grid_cell_id: str
    location_id: str | None
    location: str
    title: str
    hazard_type: str
    severity: str  # low | moderate | high | severe
    status: str  # active | resolved | expired
    risk_score: int
    evidence: str
    description: str
    recommended_action: str
    confidence: float | None
    source: str  # rule-based-v1 | ml-v1 | combined
    issued_at: datetime
    expires_at: datetime
    resolved_at: datetime | None
    last_seen_at: datetime
    acknowledged: bool
    acknowledged_at: datetime | None


class PipelineRunOut(CamelModel):
    skipped: bool
    reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    grids_processed: int | None = None
    alerts_created: int | None = None
    alerts_updated: int | None = None
    alerts_resolved: int | None = None
    alerts_expired: int | None = None


class NotificationOut(CamelModel):
    alert_id: int
    public_id: str
    event: str
    title: str
    severity: str
    hazard_type: str
    grid_cell_id: str
    message: str
    created_at: datetime
