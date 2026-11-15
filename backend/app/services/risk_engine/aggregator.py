"""Combines the 4 hazard modules into one overall RiskResult.

**Aggregation rule: overall score = the single highest hazard score** (not an
average). Documented reasoning: a grid facing severe flood risk but calm
wind should read as severely at-risk overall, not diluted into "moderate" by
averaging against unrelated, low-scoring hazards. This mirrors the frontend's
existing `RISK_LEVELS` severity-bucket philosophy (one dominant hazard drives
the badge/color shown to the user) and matches docs/architecture.md §7's
description of the eventual Risk Fusion Engine's output contract (one
overall score + one dominant/primary hazard label).

**Confidence and evidence quality** are derived from the SAME spatial/
temporal facts the grid-weather mapping already computed (distance to the
nearest contributing station, data freshness) — never from the hazard scores
themselves, so a "severe" reading is never reported with inflated confidence
just because the number is high.
"""

from datetime import datetime, timezone

from app.services.risk_engine import flood, heavy_rainfall, lightning, thunderstorm
from app.services.risk_engine.thresholds import CLOSE_RADIUS_KM, MAX_USABLE_RADIUS_KM, score_to_level
from app.services.risk_engine.types import EvidenceQuality, HazardAssessment, RiskResult, WeatherInput

HAZARD_MODULES = {
    "heavy_rainfall": heavy_rainfall.assess,
    "thunderstorm": thunderstorm.assess,
    "lightning": lightning.assess,
    "flood": flood.assess,
}


def _evidence_quality(weather: WeatherInput) -> EvidenceQuality:
    if not weather.data_available:
        return EvidenceQuality.INSUFFICIENT_DATA
    if weather.is_stale or (weather.distance_km is not None and weather.distance_km > CLOSE_RADIUS_KM):
        return EvidenceQuality.REGIONAL_ESTIMATE
    return EvidenceQuality.MEASURED


def _confidence(weather: WeatherInput) -> float:
    """1.0 = fresh, nearby, real data. Degrades with distance (linear, floor
    at MAX_USABLE_RADIUS_KM) and with staleness (flat penalty). Floored at
    0.1 rather than 0 so a result is never reported as "zero confidence" —
    use `evidence_quality == insufficient_data` for that case instead.
    """
    if not weather.data_available:
        return 0.0
    confidence = 1.0
    if weather.distance_km is not None:
        confidence -= min(weather.distance_km / MAX_USABLE_RADIUS_KM, 1.0) * 0.5
    if weather.is_stale:
        confidence -= 0.25
    return round(max(0.1, min(1.0, confidence)), 2)


def _no_data_result(now: datetime) -> RiskResult:
    hazards = {
        name: HazardAssessment(
            hazard=name,
            score=0,
            level=score_to_level(0),
            indicators=[],
            notes="No usable weather data available for this grid (no sufficiently nearby or fresh observation).",
        )
        for name in HAZARD_MODULES
    }
    return RiskResult(
        overall_score=0,
        risk_level=score_to_level(0),
        dominant_hazard="none",
        hazards=hazards,
        evidence_quality=EvidenceQuality.INSUFFICIENT_DATA,
        confidence=0.0,
        assessed_at=now,
    )


def assess_grid(weather: WeatherInput) -> RiskResult:
    now = datetime.now(timezone.utc)
    if not weather.data_available:
        return _no_data_result(now)

    hazards = {name: fn(weather) for name, fn in HAZARD_MODULES.items()}
    dominant_name, dominant = max(hazards.items(), key=lambda kv: kv[1].score)

    return RiskResult(
        overall_score=dominant.score,
        risk_level=score_to_level(dominant.score),
        dominant_hazard=dominant_name,
        hazards=hazards,
        evidence_quality=_evidence_quality(weather),
        confidence=_confidence(weather),
        assessed_at=now,
    )
