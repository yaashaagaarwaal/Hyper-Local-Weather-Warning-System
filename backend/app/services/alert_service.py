"""Alert generation engine (Phase 5).

Turns a real `RiskResult` (always available, rule-based — Phase 3) and,
optionally, a real `RawGridPrediction` (Phase 4, when its artifacts loaded
and the grid had usable weather) into `Alert` rows. Never invents a hazard
assessment of its own — every number here traces back to an existing,
already-tested hazard module or trained model.

**Hazard coverage, and exactly why:**

- **heavy_rainfall** — rule-based `risk_engine` score only. The Phase 4 ML
  `heavy_rain_risk` classifier is real but has ~1-3% precision at every
  trained horizon (docs/model_report.md §9 — a direct consequence of how
  rare heavy rain is in the training data, not a bug). Using it to trigger
  warnings would mean 97-99% false alarms, so it is deliberately **not**
  used here. `source` is always `"rule-based-v1"`.
- **thunderstorm** — rule-based `risk_engine` score, OR the Phase 4 ML
  `thunderstorm_risk` classifier (genuinely validated: PR-AUC ~0.88, 66%
  precision / 90% recall at the 3h horizon), whichever evidence is more
  severe. `source` is `"rule-based-v1"`, `"ml-v1"`, or `"combined"`,
  recording exactly which evidence actually fired — never blurred, per the
  explicit instruction not to present a rule-based estimate as an ML
  prediction or vice versa.
- **lightning** — **never generates an alert.** No lightning sensor or
  authoritative lightning-warning feed exists in this system (same
  conclusion as `risk_engine/lightning.py` and the Phase 4 model report);
  the rule-based "lightning" hazard score is a *derived thunderstorm-
  favorability* indicator, not independent evidence. Issuing a "Lightning
  Warning" from it would be exactly the kind of unsupported hazard claim
  this phase's instructions forbid.
- **flood** — rule-based `risk_engine` score only (a rainfall+humidity
  proxy, not a hydrological/drainage model — the evidence text says so
  explicitly). No ML flood model exists (no suitable ground-truth labels —
  docs/model_report.md §9).

Only **HIGH or SEVERE** risk levels generate/keep an alert — LOW/MODERATE
don't warrant a public warning bulletin. Same `score_to_level()` thresholds
the risk engine already uses and cites (app/services/risk_engine/thresholds.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.models import Alert, AlertStatus, Location
from app.models.risk_assessment import RiskLevel
from app.repositories import alert_repository
from app.services.prediction_service import RawGridPrediction
from app.services.risk_engine.thresholds import score_to_level
from app.services.risk_engine.types import RiskResult

# Hazards this engine will ever generate an alert for. Lightning is
# deliberately absent — see the module docstring.
ALERTABLE_HAZARDS = ("heavy_rainfall", "thunderstorm", "flood")

_LEVEL_ORDER = {RiskLevel.LOW: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2, RiskLevel.SEVERE: 3}
_ALERT_LEVEL_FLOOR = RiskLevel.HIGH  # minimum level that actually warrants an alert

# Thunderstorm's ML prediction is only trusted when it isn't the model's own
# documented low-confidence case — see prediction_service.LOW_CONFIDENCE_PRECISION_THRESHOLD.
_ML_HORIZON_HOURS = 3  # the flagship horizon — see docs/model_report.md §5/§6

_HAZARD_TITLES = {
    "heavy_rainfall": "Heavy Rainfall Warning",
    "thunderstorm": "Thunderstorm Warning",
    "flood": "Flood Risk Warning",
}
_HAZARD_ACTIONS = {
    "heavy_rainfall": (
        "Avoid low-lying roads and underpasses; expect localized waterlogging and reduced visibility."
    ),
    "thunderstorm": (
        "Move indoors. Avoid open areas, elevated structures, and unnecessary travel until conditions pass."
    ),
    "flood": (
        "Residents in low-lying areas should move valuables to higher ground and avoid flood-prone roads "
        "and drains."
    ),
}
# How long an alert stays valid before it's swept to EXPIRED if conditions
# haven't been re-confirmed — matches the forecast horizons this system
# actually validates at (docs/model_report.md), not an arbitrary guess.
_VALIDITY_HOURS = {"heavy_rainfall": 3, "thunderstorm": 3, "flood": 6}


def _level_ge(a: RiskLevel, b: RiskLevel) -> bool:
    return _LEVEL_ORDER[a] >= _LEVEL_ORDER[b]


def _max_level(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b


@dataclass
class HazardEvidence:
    level: RiskLevel
    score: int
    source: str  # "rule-based-v1" | "ml-v1" | "combined"
    evidence_lines: list[str]
    confidence: float  # 0.0-1.0


def _rule_evidence(hazard: str, risk_result: RiskResult) -> HazardEvidence:
    assessment = risk_result.hazards[hazard]
    lines = [f"{i.label}: {i.value} ({i.category})" for i in assessment.indicators]
    if assessment.notes:
        lines.append(assessment.notes)
    return HazardEvidence(
        level=assessment.level,
        score=assessment.score,
        source="rule-based-v1",
        evidence_lines=lines,
        confidence=risk_result.confidence,
    )


def _thunderstorm_ml_evidence(prediction: RawGridPrediction | None) -> HazardEvidence | None:
    if prediction is None or not prediction.data_available:
        return None
    match = next(
        (
            p
            for p in prediction.predictions
            if p.target == "thunderstorm_risk" and p.horizon_hours == _ML_HORIZON_HOURS
        ),
        None,
    )
    if match is None or match.low_confidence or not match.value:
        return None
    probability_pct = round((match.probability or 0.0) * 100)
    precision_pct = round((match.val_metrics.get("precision") or 0.0) * 100)
    return HazardEvidence(
        level=RiskLevel.SEVERE if probability_pct >= 85 else RiskLevel.HIGH,
        score=probability_pct,
        source="ml-v1",
        evidence_lines=[
            f"ML thunderstorm risk ({match.algorithm}, {match.horizon_hours}h horizon): "
            f"{probability_pct}% probability (validation precision {precision_pct}%)"
        ],
        confidence=match.probability or 0.0,
    )


def _combine(hazard: str, rule: HazardEvidence, ml: HazardEvidence | None) -> HazardEvidence:
    if ml is None:
        return rule
    if _level_ge(ml.level, rule.level) and ml.level != rule.level:
        source = "combined"
    elif ml.level == rule.level:
        source = "combined"
    else:
        source = rule.source
    return HazardEvidence(
        level=_max_level(rule.level, ml.level),
        score=max(rule.score, ml.score),
        source=source,
        evidence_lines=rule.evidence_lines + ml.evidence_lines,
        confidence=max(rule.confidence, ml.confidence),
    )


def _evidence_for(hazard: str, risk_result: RiskResult, prediction: RawGridPrediction | None) -> HazardEvidence:
    rule = _rule_evidence(hazard, risk_result)
    if hazard == "thunderstorm":
        ml = _thunderstorm_ml_evidence(prediction)
        return _combine(hazard, rule, ml)
    return rule


@dataclass
class SyncOutcome:
    created: list[Alert] = field(default_factory=list)
    updated: list[Alert] = field(default_factory=list)
    resolved: list[Alert] = field(default_factory=list)


def sync_alerts_for_grid(
    db,
    grid_cell,
    risk_result: RiskResult,
    prediction: RawGridPrediction | None,
    location: Location | None,
) -> SyncOutcome:
    """For one grid cell: for every alertable hazard, either create a new
    alert, update an existing active one (duplicate prevention — the same
    ongoing event never produces a second row), or resolve one whose
    conditions have genuinely subsided.

    `location` is whichever real `Location` the risk assessment actually
    used (its `GridWeatherAssociation.location_id`) — passed in by the
    caller rather than re-derived here, so the alert's location label is
    always the exact station the evidence came from, not a second,
    independently-computed "nearest" guess.
    """
    outcome = SyncOutcome()
    now = datetime.now(timezone.utc)

    for hazard in ALERTABLE_HAZARDS:
        evidence = _evidence_for(hazard, risk_result, prediction)
        alert_key = f"{hazard}:{grid_cell.id}"
        existing = alert_repository.find_active_by_key(db, alert_key)

        if not _level_ge(evidence.level, _ALERT_LEVEL_FLOOR):
            if existing is not None:
                existing.status = AlertStatus.RESOLVED
                existing.resolved_at = now
                alert_repository.save(db, existing)
                outcome.resolved.append(existing)
            continue

        title = f"{'Severe ' if evidence.level == RiskLevel.SEVERE else ''}{_HAZARD_TITLES[hazard]}"
        description = (
            f"{_HAZARD_TITLES[hazard]} for grid {grid_cell.id}"
            f"{f' ({location.name})' if location else ''} — risk score {evidence.score}/100."
        )
        evidence_text = "; ".join(evidence.evidence_lines) or "No supporting indicators recorded."
        expires_at = now + timedelta(hours=_VALIDITY_HOURS[hazard])

        if existing is not None:
            existing.severity = evidence.level
            existing.risk_score = evidence.score
            existing.evidence = evidence_text
            existing.description = description
            existing.confidence = evidence.confidence
            existing.source = evidence.source
            existing.last_seen_at = now
            existing.expires_at = expires_at
            alert_repository.save(db, existing)
            outcome.updated.append(existing)
        else:
            alert = Alert(
                alert_key=alert_key,
                grid_cell_id=grid_cell.id,
                location_id=location.id if location else None,
                title=title,
                hazard_type=hazard,
                severity=evidence.level,
                status=AlertStatus.ACTIVE,
                location_label=location.name if location else f"Grid {grid_cell.id}",
                risk_score=evidence.score,
                evidence=evidence_text,
                description=description,
                recommended_action=_HAZARD_ACTIONS[hazard],
                confidence=evidence.confidence,
                source=evidence.source,
                issued_at=now,
                expires_at=expires_at,
                last_seen_at=now,
                acknowledged=False,
            )
            alert_repository.create(db, alert)
            outcome.created.append(alert)

    return outcome


def expire_due_alerts(db, now: datetime | None = None) -> list[Alert]:
    """Sweeps every ACTIVE alert whose `expires_at` has passed without being
    re-confirmed by a later pipeline run into EXPIRED — the "Expired alerts"
    lifecycle bucket, distinct from a hazard that was actively resolved.
    """
    now = now or datetime.now(timezone.utc)
    due = alert_repository.list_active_expiring_before(db, now)
    for alert in due:
        alert.status = AlertStatus.EXPIRED
        alert_repository.save(db, alert)
    return due


def acknowledge(db, alert: Alert) -> Alert:
    alert.acknowledged = True
    alert.acknowledged_at = datetime.now(timezone.utc)
    return alert_repository.save(db, alert)


def resolve(db, alert: Alert) -> Alert:
    alert.status = AlertStatus.RESOLVED
    alert.resolved_at = datetime.now(timezone.utc)
    return alert_repository.save(db, alert)


def public_id(alert: Alert) -> str:
    return f"ALT-{alert.id:04d}"


def to_notification(alert: Alert, event: str):
    from app.services.notification_service import AlertNotification

    return AlertNotification(
        alert_id=alert.id,
        public_id=public_id(alert),
        event=event,
        title=alert.title,
        severity=alert.severity.value,
        hazard_type=alert.hazard_type,
        grid_cell_id=alert.grid_cell_id,
        message=f"{alert.title} — {alert.location_label} (risk {alert.risk_score}/100)",
        created_at=datetime.now(timezone.utc),
    )
