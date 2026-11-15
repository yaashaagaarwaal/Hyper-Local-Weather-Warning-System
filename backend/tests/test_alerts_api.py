"""End-to-end tests for the Phase 5 `/alerts/*` endpoints, through the real
FastAPI app and real database. Most tests seed an Alert row directly (no
network) to test the API/lifecycle surface in isolation; one test
(`test_manual_pipeline_trigger_creates_real_alerts_end_to_end`) is marked
`network` because it exercises the actual ingestion -> risk -> prediction ->
alert pipeline against the real Open-Meteo API, same justification as
test_weather_api.py and test_predictions_api.py.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db.session import SessionLocal
from app.models import Alert, AlertStatus, GridCell, Location
from app.models.risk_assessment import RiskLevel


@pytest.fixture
def seeded_alert():
    session = SessionLocal()
    # A fresh grid id (hence a fresh alert_key) per test invocation — this
    # fixture's writes really commit (plain SessionLocal, not the
    # rolled-back db_session fixture), so a fixed id would collide with the
    # partial-unique "one ACTIVE alert per key" index across tests.
    cell = GridCell(
        id=f"GRID-ALERTAPI-{uuid.uuid4().hex[:8].upper()}", row=0, col=0, resolution="alertapi",
        bounds=[[28.4, 77.0], [28.4, 77.1], [28.5, 77.1], [28.5, 77.0]],
        center_lat=28.45, center_lng=77.05,
    )
    session.merge(cell)
    now = datetime.now(timezone.utc)
    alert = Alert(
        alert_key=f"thunderstorm:{cell.id}", grid_cell_id=cell.id, title="Thunderstorm Warning",
        hazard_type="thunderstorm", severity=RiskLevel.HIGH, status=AlertStatus.ACTIVE,
        location_label="API Test Area", risk_score=72, evidence="CAPE: 3200 J/kg (forecast)",
        description="Thunderstorm Warning for GRID-ALERTAPI-001.",
        recommended_action="Move indoors.", confidence=0.8, source="rule-based-v1",
        issued_at=now, expires_at=now + timedelta(hours=3), last_seen_at=now, acknowledged=False,
    )
    session.add(alert)
    session.commit()
    try:
        yield alert
    finally:
        session.close()


@pytest.fixture
def seeded_pipeline_location():
    session = SessionLocal()
    # A coordinate not used by any other test module's fixtures (see
    # test_predictions_api.py's comment on the same issue) so this
    # pipeline-trigger test's grid unambiguously maps to this location.
    loc = Location(id="loc-alertapi-pipeline", name="Alert Pipeline Test", region="Test", latitude=28.80, longitude=76.95)
    session.merge(loc)
    cell = GridCell(
        id="GRID-ALERTAPI-PIPE", row=0, col=0, resolution="alertapi",
        bounds=[[28.75, 76.90], [28.75, 77.00], [28.85, 77.00], [28.85, 76.90]],
        center_lat=28.80, center_lng=76.95,
    )
    session.merge(cell)
    session.commit()
    try:
        yield {"location_id": loc.id, "grid_id": cell.id}
    finally:
        session.close()


def test_get_active_alerts_returns_seeded_active_alert(client, seeded_alert):
    response = client.get("/api/v1/alerts/active")
    assert response.status_code == 200
    body = response.json()
    ids = {a["id"] for a in body}
    assert alert_public_id(seeded_alert) in ids


def alert_public_id(alert: Alert) -> str:
    return f"ALT-{alert.id:04d}"


def test_get_alert_by_public_id(client, seeded_alert):
    response = client.get(f"/api/v1/alerts/{alert_public_id(seeded_alert)}")
    assert response.status_code == 200
    body = response.json()
    assert body["hazardType"] == "thunderstorm"
    assert body["severity"] == "high"
    assert body["status"] == "active"
    assert body["acknowledged"] is False
    assert body["source"] == "rule-based-v1"


def test_get_alert_by_bare_integer_id_also_works(client, seeded_alert):
    response = client.get(f"/api/v1/alerts/{seeded_alert.id}")
    assert response.status_code == 200
    assert response.json()["id"] == alert_public_id(seeded_alert)


def test_get_unknown_alert_returns_404(client):
    response = client.get("/api/v1/alerts/ALT-9999999")
    assert response.status_code == 404


def test_acknowledge_then_resolve_lifecycle(client, seeded_alert):
    ack = client.post(f"/api/v1/alerts/{alert_public_id(seeded_alert)}/acknowledge")
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] is True

    resolved = client.post(f"/api/v1/alerts/{alert_public_id(seeded_alert)}/resolve")
    assert resolved.status_code == 200
    body = resolved.json()
    assert body["status"] == "resolved"
    assert body["resolvedAt"] is not None
    # acknowledgement must survive resolution rather than being reset
    assert body["acknowledged"] is True

    active = client.get("/api/v1/alerts/active").json()
    assert alert_public_id(seeded_alert) not in {a["id"] for a in active}

    history = client.get("/api/v1/alerts/history").json()
    assert alert_public_id(seeded_alert) in {a["id"] for a in history}


def test_filter_alerts_by_status_and_hazard(client, seeded_alert):
    response = client.get("/api/v1/alerts", params={"status": "active", "hazardType": "thunderstorm"})
    assert response.status_code == 200
    assert any(a["id"] == alert_public_id(seeded_alert) for a in response.json())

    response = client.get("/api/v1/alerts", params={"hazardType": "flood"})
    assert alert_public_id(seeded_alert) not in {a["id"] for a in response.json()}


def test_notifications_endpoint_returns_a_list(client):
    response = client.get("/api/v1/alerts/notifications")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.network
def test_manual_pipeline_trigger_creates_real_alerts_end_to_end(client, seeded_pipeline_location):
    response = client.post("/api/v1/alerts/process")
    assert response.status_code == 200
    body = response.json()
    assert body["skipped"] is False
    assert body["gridsProcessed"] >= 1

    # Whatever the real weather turned out to be, the pipeline must have run
    # to completion and the alerts list must be queryable afterwards without
    # error (this does not assert an alert was actually created — real
    # weather may or may not have crossed an alert threshold today).
    active = client.get("/api/v1/alerts/active")
    assert active.status_code == 200
