from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.models import GridWeatherAssociation, Location, RiskAssessment
from app.repositories import grid_repository, location_repository
from app.schemas.risk import (
    DistrictVulnerabilityOut,
    HazardAssessmentOut,
    HazardIndicatorOut,
    RiskAssessmentOut,
    RiskLevelCounts,
    RiskSummaryOut,
    WeatherBasisOut,
)
from app.services import risk_service
from app.services.risk_engine import flood_vulnerability
from app.services.risk_engine.types import RiskResult

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/districts")
def get_ncr_flood_districts() -> dict:
    """The 20 Delhi-NCR districts with verified India Flood Inventory
    statistics (see app/services/risk_engine/flood_vulnerability.py) as a
    standard GeoJSON FeatureCollection, for the frontend's district map
    layer. Static/historical data -- never a live measurement or forecast.
    Registered before `/grids/{grid_id}` is irrelevant here (distinct path
    segment, "districts" vs "grids"), but kept as a literal route for
    clarity, same convention as the rest of this file.
    """
    return {"type": "FeatureCollection", "features": flood_vulnerability.list_districts()}


@router.get("/districts/lookup", response_model=DistrictVulnerabilityOut)
def get_district_vulnerability_for_point(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> DistrictVulnerabilityOut:
    """Static historical flood-vulnerability for an arbitrary point (e.g. a
    monitored Location, not just a grid cell) -- same lookup the risk
    engine itself uses (flood_vulnerability.lookup), so a location outside
    the 20 verified Delhi-NCR districts correctly reports `available=False`
    rather than a guessed value. Used by the Locations page.
    """
    result = flood_vulnerability.lookup(lat, lon)
    if result is None:
        return DistrictVulnerabilityOut(available=False)
    return DistrictVulnerabilityOut(
        available=True,
        district_name=result.district_name,
        dfsi=result.dfsi,
        percent_flooded_area=result.percent_flooded_area,
        score_bonus=result.score_bonus,
        source=result.source,
    )


def _build_response(
    result: RiskResult,
    assessment: RiskAssessment,
    association: GridWeatherAssociation,
    locations_by_id: dict[str, Location],
) -> RiskAssessmentOut:
    location = locations_by_id.get(association.location_id) if association.location_id else None
    return RiskAssessmentOut(
        grid_cell_id=assessment.grid_cell_id,
        assessed_at=assessment.assessed_at,
        overall_score=assessment.overall_score,
        risk_level=assessment.risk_level.value,
        dominant_hazard=assessment.dominant_hazard,
        hazards=[
            HazardAssessmentOut(
                hazard=h.hazard,
                score=h.score,
                level=h.level.value,
                indicators=[
                    HazardIndicatorOut(label=i.label, value=i.value, category=i.category)
                    for i in h.indicators
                ],
                notes=h.notes,
            )
            for h in result.hazards.values()
        ],
        confidence=assessment.confidence,
        evidence_quality=assessment.evidence_quality,
        source=assessment.source,
        weather_basis=WeatherBasisOut(
            data_available=association.data_available,
            observation_count=association.observation_count,
            nearest_location_id=association.location_id,
            nearest_location_name=location.name if location else None,
            distance_km=association.distance_km,
            observed_at=association.observed_at,
            is_stale=association.is_stale,
            temperature_c=association.temperature_c,
            humidity_pct=association.humidity_pct,
            wind_speed_kmh=association.wind_speed_kmh,
            rainfall_mm=association.rainfall_mm,
        ),
    )


@router.get("/grids", response_model=list[RiskAssessmentOut])
def get_all_grid_risks(
    resolution: str = Query("7x7", description="Grid resolution label, e.g. '7x7'"),
    db: Session = Depends(get_db),
) -> list[RiskAssessmentOut]:
    """Fresh risk assessment for every grid cell at the given resolution —
    computed on-demand (no background job), and persisted as a side effect
    (see app/services/risk_service.py).
    """
    cells = grid_repository.list_grid_cells(db, resolution=resolution)
    locations = location_repository.list_all(db)
    locations_by_id = {loc.id: loc for loc in locations}

    results = risk_service.assess_many(db, cells, locations)
    return [_build_response(r, a, w, locations_by_id) for r, a, w in results]


@router.get("/grids/{grid_id}", response_model=RiskAssessmentOut)
def get_grid_risk(grid_id: str, db: Session = Depends(get_db)) -> RiskAssessmentOut:
    cell = grid_repository.get_grid_cell(db, grid_id)
    if cell is None:
        raise NotFoundError(f"Unknown grid id: {grid_id!r}")

    locations = location_repository.list_all(db)
    locations_by_id = {loc.id: loc for loc in locations}
    result, assessment, association = risk_service.assess_and_persist(db, cell, locations)
    return _build_response(result, assessment, association, locations_by_id)


@router.get("/summary", response_model=RiskSummaryOut)
def get_risk_summary(
    resolution: str = Query("7x7", description="Grid resolution label, e.g. '7x7'"),
    db: Session = Depends(get_db),
) -> RiskSummaryOut:
    """Aggregate counts across every grid cell at the given resolution —
    mirrors the frontend's mock `getRiskGridStats()`, now backed by real
    weather data run through the rule-based engine.
    """
    cells = grid_repository.list_grid_cells(db, resolution=resolution)
    locations = location_repository.list_all(db)

    results = risk_service.assess_many(db, cells, locations)

    level_counts = Counter(assessment.risk_level.value for _, assessment, _ in results)
    hazard_counts = Counter(assessment.dominant_hazard for _, assessment, _ in results)

    return RiskSummaryOut(
        resolution=resolution,
        total_grids=len(cells),
        counts_by_level=RiskLevelCounts(**{k: level_counts.get(k, 0) for k in ("low", "moderate", "high", "severe")}),
        dominant_hazard_counts=dict(hazard_counts),
        generated_at=datetime.now(timezone.utc),
    )
