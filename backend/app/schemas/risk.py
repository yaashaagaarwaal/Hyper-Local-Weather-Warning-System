from datetime import datetime

from app.schemas.weather import CamelModel


class HazardIndicatorOut(CamelModel):
    label: str
    value: str
    category: str  # "measured" | "forecast" | "official_warning"


class HazardAssessmentOut(CamelModel):
    hazard: str
    score: int
    level: str
    indicators: list[HazardIndicatorOut]
    notes: str


class WeatherBasisOut(CamelModel):
    """Transparency block: exactly what the assessment is (and isn't) based
    on — satisfies "do not claim street-level accuracy" by always surfacing
    distance/staleness/station-count alongside the score.
    """

    data_available: bool
    observation_count: int
    nearest_location_id: str | None
    nearest_location_name: str | None
    distance_km: float | None
    observed_at: datetime | None
    is_stale: bool
    # Phase 5: the actual mapped weather values the score was computed from
    # (same numbers as GridWeatherAssociation) — the frontend's grid detail
    # panels need these for display; previously this block only carried the
    # transparency fields, not the readings themselves.
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_kmh: float | None = None
    rainfall_mm: float | None = None


class RiskAssessmentOut(CamelModel):
    grid_cell_id: str
    assessed_at: datetime
    overall_score: int
    risk_level: str
    dominant_hazard: str
    hazards: list[HazardAssessmentOut]
    confidence: float
    evidence_quality: str
    source: str
    weather_basis: WeatherBasisOut


class RiskLevelCounts(CamelModel):
    low: int = 0
    moderate: int = 0
    high: int = 0
    severe: int = 0


class RiskSummaryOut(CamelModel):
    resolution: str
    total_grids: int
    counts_by_level: RiskLevelCounts
    dominant_hazard_counts: dict[str, int]
    generated_at: datetime


class DistrictVulnerabilityOut(CamelModel):
    """Static, historical flood-vulnerability lookup for one lat/lon point
    -- see app/services/risk_engine/flood_vulnerability.py. `available`
    is False outside the 20 verified Delhi-NCR districts; never a guess.
    """

    available: bool
    district_name: str | None = None
    dfsi: float | None = None
    percent_flooded_area: float | None = None
    score_bonus: float | None = None
    source: str | None = None
