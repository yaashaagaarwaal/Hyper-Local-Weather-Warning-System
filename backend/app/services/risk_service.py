"""Orchestrates: grid → weather mapping → rule-based risk engine →
persistence, for one grid cell or a batch of them.

Computed on-demand (synchronously, per API request) rather than by a
background scheduler — Phase 3 explicitly excludes automated/background
alerting. Each call both returns a fresh assessment AND persists it (append-
only), so `risk_assessments`/`grid_weather_associations` double as a history
log with no separate "run a batch job" step needed.
"""

from app.models import GridCell, GridWeatherAssociation, Location, RiskAssessment, WeatherObservation
from app.repositories import grid_repository
from app.services import grid_mapping_service
from app.services.risk_engine import assess_grid
from app.services.risk_engine import flood_vulnerability
from app.services.risk_engine.thresholds import clamp_score, score_to_level
from app.services.risk_engine.types import HazardIndicator, RiskResult

RISK_ENGINE_SOURCE = "rule-based-v1"

# RiskAssessment.hazard_scores keys — "heavyRain" (not "heavy_rainfall")
# specifically to match the shape already documented on the RiskAssessment
# model and already used by the frontend's mock `getRiskSummary`/
# `getGridHazardProbabilities` (src/data/riskData.js) for the other 3 keys.
_HAZARD_SCORE_KEYS = {
    "heavy_rainfall": "heavyRain",
    "thunderstorm": "thunderstorm",
    "lightning": "lightning",
    "flood": "flood",
}


def _apply_flood_vulnerability(result: RiskResult, grid_cell: GridCell) -> RiskResult:
    """Adds a bounded, explicitly-labeled static-vulnerability modifier to
    the flood hazard, when the grid cell's center falls inside one of the
    20 verified Delhi-NCR districts bundled in
    app/data/ncr_flood_vulnerability.geojson (see flood_vulnerability.py's
    module docstring for full provenance and the honesty rules this
    follows). A grid outside that coverage is returned completely
    unchanged -- never a guessed adjustment.
    """
    vuln = flood_vulnerability.lookup(float(grid_cell.center_lat), float(grid_cell.center_lng))
    if vuln is None or vuln.score_bonus <= 0:
        return result

    flood = result.hazards["flood"]
    new_score = clamp_score(flood.score + vuln.score_bonus)
    new_indicators = list(flood.indicators) + [
        HazardIndicator(
            label="Historical flood vulnerability (District Flood Severity Index)",
            value=f"{vuln.dfsi:.1f}/20 — {vuln.district_name} (source: India Flood Inventory, CC-BY-NC-4.0)",
            category="historical_vulnerability",
        )
    ]
    updated_flood = type(flood)(
        hazard=flood.hazard,
        score=new_score,
        level=score_to_level(new_score),
        indicators=new_indicators,
        notes=flood.notes + f" Adjusted +{vuln.score_bonus:.1f} pts for this district's historical "
        f"flood severity (static, non-meteorological signal — never the sole driver of the score).",
    )
    new_hazards = {**result.hazards, "flood": updated_flood}
    dominant_name, dominant = max(new_hazards.items(), key=lambda kv: kv[1].score)

    return type(result)(
        overall_score=dominant.score,
        risk_level=score_to_level(dominant.score),
        dominant_hazard=dominant_name,
        hazards=new_hazards,
        evidence_quality=result.evidence_quality,
        confidence=result.confidence,
        assessed_at=result.assessed_at,
    )


def assess_and_persist(
    db,
    grid_cell: GridCell,
    locations: list[Location],
    observation_cache: dict[str, WeatherObservation | None] | None = None,
) -> tuple[RiskResult, RiskAssessment, GridWeatherAssociation]:
    mapping = grid_mapping_service.map_grid_weather(db, grid_cell, locations, observation_cache)
    result = assess_grid(mapping.weather_input)
    if mapping.weather_input.data_available:
        result = _apply_flood_vulnerability(result, grid_cell)

    association = GridWeatherAssociation(
        grid_cell_id=grid_cell.id,
        location_id=mapping.nearest_location.id if mapping.nearest_location else None,
        weather_observation_id=mapping.nearest_observation.id if mapping.nearest_observation else None,
        distance_km=mapping.nearest_distance_km,
        observation_count=mapping.observation_count,
        temperature_c=mapping.weather_input.temperature_c,
        humidity_pct=mapping.weather_input.humidity_pct,
        wind_speed_kmh=mapping.weather_input.wind_speed_kmh,
        rainfall_mm=mapping.weather_input.rainfall_mm,
        pressure_hpa=mapping.weather_input.pressure_hpa,
        condition=mapping.weather_input.condition,
        observed_at=mapping.weather_input.observed_at,
        is_stale=mapping.weather_input.is_stale,
        data_available=mapping.data_available,
    )
    grid_repository.save_weather_association(db, association)

    hazard_scores = {_HAZARD_SCORE_KEYS[name]: h.score for name, h in result.hazards.items()}
    assessment = RiskAssessment(
        grid_cell_id=grid_cell.id,
        assessed_at=result.assessed_at,
        overall_score=result.overall_score,
        risk_level=result.risk_level,
        dominant_hazard=result.dominant_hazard,
        hazard_scores=hazard_scores,
        confidence=result.confidence,
        evidence_quality=result.evidence_quality.value,
        source=RISK_ENGINE_SOURCE,
    )
    grid_repository.save_risk_assessment(db, assessment)

    return result, assessment, association


def assess_many(
    db, grid_cells: list[GridCell], locations: list[Location]
) -> list[tuple[RiskResult, RiskAssessment, GridWeatherAssociation]]:
    """Batch version — shares one observation cache across every grid cell,
    so each location's latest observation is only queried once per request
    regardless of how many grids reference it.
    """
    cache: dict[str, WeatherObservation | None] = {}
    return [assess_and_persist(db, cell, locations, cache) for cell in grid_cells]
