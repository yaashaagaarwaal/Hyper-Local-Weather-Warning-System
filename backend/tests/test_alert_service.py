"""Unit + DB tests for the Phase 5 alert-generation engine
(app/services/alert_service.py) — synthetic RiskResult/RawGridPrediction
inputs (no network, no live model loading needed to exercise the
evidence-combination logic), real DB writes for the lifecycle tests.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models import Alert, AlertStatus, GridCell
from app.models.risk_assessment import RiskLevel
from app.repositories import alert_repository
from app.services import alert_service
from app.services.prediction_service import RawGridPrediction, RawTargetPrediction
from app.services.risk_engine.types import EvidenceQuality, HazardAssessment, HazardIndicator, RiskResult


def _risk_result(scores: dict[str, int]) -> RiskResult:
    from app.services.risk_engine.thresholds import score_to_level

    hazards = {
        name: HazardAssessment(
            hazard=name,
            score=score,
            level=score_to_level(score),
            indicators=[HazardIndicator(label="Rainfall rate", value=f"{score} (test)", category="measured")],
            notes="",
        )
        for name, score in scores.items()
    }
    dominant = max(hazards.values(), key=lambda h: h.score)
    return RiskResult(
        overall_score=dominant.score,
        risk_level=dominant.level,
        dominant_hazard=dominant.hazard,
        hazards=hazards,
        evidence_quality=EvidenceQuality.MEASURED,
        confidence=0.9,
        assessed_at=datetime.now(timezone.utc),
    )


def _default_scores(**overrides) -> dict[str, int]:
    base = {"heavy_rainfall": 10, "thunderstorm": 10, "lightning": 10, "flood": 10}
    base.update(overrides)
    return base


def _thunderstorm_prediction(probability: float, low_confidence: bool = False, value: float = 1.0) -> RawGridPrediction:
    now = datetime.now(timezone.utc)
    pred = RawTargetPrediction(
        target="thunderstorm_risk",
        horizon_hours=3,
        forecast_for=now + timedelta(hours=3),
        is_classification=True,
        value=value,
        probability=probability,
        algorithm="xgboost",
        model_version="ml-v1",
        val_metrics={"precision": 0.78, "recall": 0.93, "pr_auc": 0.93},
        low_confidence=low_confidence,
    )
    return RawGridPrediction(
        grid_cell_id="GRID-TEST", generated_at=now, data_available=True,
        nearest_location_id="loc-x", nearest_location_name="Test", distance_km=1.0,
        missing_feature_count=0, predictions=[pred],
    )


# --- Evidence combination (pure logic, no DB) ---

def test_heavy_rainfall_evidence_is_rule_based_only():
    result = _risk_result(_default_scores(heavy_rainfall=90))
    evidence = alert_service._evidence_for("heavy_rainfall", result, prediction=None)
    assert evidence.source == "rule-based-v1"
    assert evidence.level == RiskLevel.SEVERE


def test_flood_evidence_is_rule_based_only():
    result = _risk_result(_default_scores(flood=60))
    evidence = alert_service._evidence_for("flood", result, prediction=None)
    assert evidence.source == "rule-based-v1"
    assert evidence.level == RiskLevel.HIGH


def test_lightning_is_never_alertable():
    assert "lightning" not in alert_service.ALERTABLE_HAZARDS


def test_thunderstorm_combines_rule_and_confident_ml_prediction():
    result = _risk_result(_default_scores(thunderstorm=20))  # rule alone: LOW
    prediction = _thunderstorm_prediction(probability=0.9)  # confident ML risk
    evidence = alert_service._evidence_for("thunderstorm", result, prediction)
    assert evidence.level == RiskLevel.SEVERE  # ML alone pushes this to alertable
    assert evidence.source == "combined"
    assert any("ML thunderstorm risk" in line for line in evidence.evidence_lines)


def test_thunderstorm_ignores_a_low_confidence_ml_prediction():
    result = _risk_result(_default_scores(thunderstorm=20))
    prediction = _thunderstorm_prediction(probability=0.9, low_confidence=True)
    evidence = alert_service._evidence_for("thunderstorm", result, prediction)
    assert evidence.source == "rule-based-v1"  # ML evidence discarded
    assert evidence.level == RiskLevel.LOW


def test_thunderstorm_ignores_ml_when_prediction_unavailable():
    result = _risk_result(_default_scores(thunderstorm=90))
    evidence = alert_service._evidence_for("thunderstorm", result, prediction=None)
    assert evidence.source == "rule-based-v1"
    assert evidence.level == RiskLevel.SEVERE


# --- Lifecycle (real DB) ---

def _make_grid_cell(db_session, suffix="ALERT") -> GridCell:
    cell = GridCell(
        id=f"GRID-TEST-{suffix}", row=0, col=0, resolution="alerttest",
        bounds=[[28.5, 77.1], [28.5, 77.2], [28.6, 77.2], [28.6, 77.1]],
        center_lat=28.55, center_lng=77.15,
    )
    db_session.add(cell)
    db_session.commit()
    return cell


def test_sync_creates_an_alert_when_a_hazard_is_high(db_session):
    cell = _make_grid_cell(db_session, "A")
    result = _risk_result(_default_scores(heavy_rainfall=90))

    outcome = alert_service.sync_alerts_for_grid(db_session, cell, result, prediction=None, location=None)

    assert len(outcome.created) == 1
    alert = outcome.created[0]
    assert alert.hazard_type == "heavy_rainfall"
    assert alert.severity == RiskLevel.SEVERE
    assert alert.status == AlertStatus.ACTIVE
    assert alert.source == "rule-based-v1"
    assert "Rainfall rate" in alert.evidence


def test_sync_does_not_create_an_alert_for_low_or_moderate_risk(db_session):
    cell = _make_grid_cell(db_session, "B")
    result = _risk_result(_default_scores())  # everything at 10 -> LOW

    outcome = alert_service.sync_alerts_for_grid(db_session, cell, result, prediction=None, location=None)

    assert outcome.created == []
    rows = alert_repository.list_all(db_session)
    assert rows == []


def test_repeated_high_risk_updates_the_same_alert_not_a_duplicate(db_session):
    cell = _make_grid_cell(db_session, "C")
    result = _risk_result(_default_scores(heavy_rainfall=90))

    first = alert_service.sync_alerts_for_grid(db_session, cell, result, prediction=None, location=None)
    second = alert_service.sync_alerts_for_grid(db_session, cell, result, prediction=None, location=None)

    assert len(first.created) == 1
    assert len(second.created) == 0
    assert len(second.updated) == 1
    assert first.created[0].id == second.updated[0].id

    active = alert_repository.list_active(db_session)
    assert len(active) == 1  # never duplicated despite two runs with the same conditions


def test_conditions_subsiding_resolves_the_active_alert(db_session):
    cell = _make_grid_cell(db_session, "D")
    hot = alert_service.sync_alerts_for_grid(
        db_session, cell, _risk_result(_default_scores(heavy_rainfall=90)), prediction=None, location=None
    )
    assert len(hot.created) == 1

    cooled = alert_service.sync_alerts_for_grid(
        db_session, cell, _risk_result(_default_scores(heavy_rainfall=5)), prediction=None, location=None
    )
    assert len(cooled.resolved) == 1
    assert cooled.resolved[0].status == AlertStatus.RESOLVED
    assert cooled.resolved[0].resolved_at is not None
    assert alert_repository.list_active(db_session) == []


def test_expire_due_alerts_moves_overdue_active_alerts_to_expired(db_session):
    cell = _make_grid_cell(db_session, "E")
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    alert = Alert(
        alert_key=f"flood:{cell.id}", grid_cell_id=cell.id, title="Flood Risk Warning",
        hazard_type="flood", severity=RiskLevel.HIGH, status=AlertStatus.ACTIVE,
        location_label="Test", risk_score=60, evidence="test", description="test",
        recommended_action="test", confidence=0.8, source="rule-based-v1",
        issued_at=past, expires_at=past, last_seen_at=past, acknowledged=False,
    )
    db_session.add(alert)
    db_session.commit()

    expired = alert_service.expire_due_alerts(db_session)

    assert len(expired) == 1
    assert expired[0].status == AlertStatus.EXPIRED
    assert alert_repository.list_active(db_session) == []


def test_acknowledge_and_resolve_helpers(db_session):
    cell = _make_grid_cell(db_session, "F")
    outcome = alert_service.sync_alerts_for_grid(
        db_session, cell, _risk_result(_default_scores(flood=90)), prediction=None, location=None
    )
    alert = outcome.created[0]
    assert alert.acknowledged is False

    alert_service.acknowledge(db_session, alert)
    assert alert.acknowledged is True
    assert alert.acknowledged_at is not None

    alert_service.resolve(db_session, alert)
    assert alert.status == AlertStatus.RESOLVED
    assert alert.resolved_at is not None


def test_public_id_format():
    cell_alert = Alert(id=42, severity=RiskLevel.LOW, status=AlertStatus.ACTIVE)
    assert alert_service.public_id(cell_alert) == "ALT-0042"
