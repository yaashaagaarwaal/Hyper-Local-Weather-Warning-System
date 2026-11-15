"""Shared types for the rule-based risk engine.

`WeatherInput` is the one normalized shape every hazard module reads from —
it is deliberately just the fields a grid's mapped weather actually has
(see app/services/grid_mapping_service.py), so a hazard module can never
silently depend on data that isn't really there; every field is explicitly
optional and modules must handle `None`.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Reused, not redefined — RiskLevel already exists on the RiskAssessment model
# (app/models/risk_assessment.py) with the exact low/moderate/high/severe
# bands the frontend's utils/risk.js also uses. Importing it here (rather
# than a second parallel enum) is what keeps "one risk-level vocabulary
# across the whole system" actually true.
from app.models.risk_assessment import RiskLevel  # noqa: F401 (re-exported for risk_engine callers)


class EvidenceQuality(str, Enum):
    """How much to trust a hazard assessment's inputs — surfaced to the API
    so a consumer can tell "real, fresh, nearby data" apart from "real but
    distant/stale data" apart from "no usable data at all". Never means
    "measured" when the underlying variable (e.g. lightning) was never
    actually measured — see LIGHTNING_EVIDENCE_NOTE in thresholds.py.
    """

    MEASURED = "measured"  # fresh (<= STALE_THRESHOLD_MINUTES), nearby (<= CLOSE_RADIUS_KM) real observation
    REGIONAL_ESTIMATE = "regional_estimate"  # real observation, but distant and/or stale
    INSUFFICIENT_DATA = "insufficient_data"  # no usable observation at all


@dataclass
class WeatherInput:
    """What a hazard module is allowed to see. Mirrors
    GridWeatherSummary (grid_mapping_service.py) field-for-field, minus the
    grid/location bookkeeping fields a hazard module has no business reading."""

    temperature_c: float | None
    humidity_pct: float | None
    wind_speed_kmh: float | None
    rainfall_mm: float | None
    rain_probability_pct: float | None
    pressure_hpa: float | None
    condition: str | None
    weather_code: int | None
    cape_j_kg: float | None
    lifted_index: float | None
    observed_at: datetime | None
    is_stale: bool
    distance_km: float | None
    data_available: bool


@dataclass
class HazardIndicator:
    """One named, human-readable supporting fact behind a hazard score —
    e.g. {"label": "Rainfall rate", "value": "12.4 mm/hr", "category": "measured"}.
    `category` is one of "measured", "forecast", "official_warning", or
    "historical_vulnerability" (a static, non-meteorological signal — see
    flood_vulnerability.py) — never implies more certainty than the
    underlying data actually has.
    """

    label: str
    value: str
    category: str  # "measured" | "forecast" | "official_warning" | "historical_vulnerability"


@dataclass
class HazardAssessment:
    hazard: str  # "heavy_rainfall" | "thunderstorm" | "lightning" | "flood"
    score: int  # 0-100
    level: RiskLevel
    indicators: list[HazardIndicator] = field(default_factory=list)
    notes: str = ""


@dataclass
class RiskResult:
    overall_score: int
    risk_level: RiskLevel
    dominant_hazard: str
    hazards: dict[str, HazardAssessment]
    evidence_quality: EvidenceQuality
    confidence: float  # 0.0-1.0
    assessed_at: datetime
