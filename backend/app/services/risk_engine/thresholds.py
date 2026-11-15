"""Documented thresholds for the rule-based risk engine.

Every number here is a cited, standard meteorological convention — nothing
is an invented/arbitrary cutoff. Where two conventions exist (e.g. IMD's are
for 24-hour totals, we only have an hourly rate live), the adapted version is
noted explicitly.

Sources:
  - IMD rainfall intensity classification (24h totals): Light 2.5-15.5mm,
    Moderate 15.6-64.4mm, Heavy 64.5-115.5mm, Very Heavy 115.6-204.4mm,
    Extremely Heavy >204.5mm. (India Meteorological Department, standard
    public rainfall-warning categories.)
  - WMO/IMD hourly rainfall-intensity convention (used here, since live
    ingestion is hourly-rate, not 24h totals): Light <2.5 mm/hr, Moderate
    2.5-7.5 mm/hr, Heavy 7.6-35.5 mm/hr, Very Heavy 35.6-64.4 mm/hr,
    Extreme >64.4 mm/hr.
  - WMO weather codes 95/96/99 = thunderstorm / thunderstorm with hail /
    severe thunderstorm (ww code table, used by Open-Meteo and most NWP
    providers).
  - CAPE (Convective Available Potential Energy) instability classification,
    standard convective-meteorology convention (as used by e.g. the US
    Storm Prediction Center): <1000 J/kg weak, 1000-2500 J/kg moderate,
    2500-4000 J/kg strong, >4000 J/kg extreme instability.
  - Lifted Index (LI) convective-stability convention: LI > 0 stable,
    0 to -3 marginally unstable, -3 to -6 moderately unstable (thunderstorms
    likely), < -6 very unstable (severe thunderstorm potential).
"""

from app.services.risk_engine.types import RiskLevel

# --- Spatial/temporal evidence quality (shared across all hazard modules) ---
STALE_THRESHOLD_MINUTES = 60
CLOSE_RADIUS_KM = 15.0  # "local enough to call this measured, not a regional estimate"
MAX_USABLE_RADIUS_KM = 45.0  # beyond this, don't use the reading at all

# --- Heavy rainfall (hourly rate, mm/hr — WMO/IMD hourly convention) ---
RAINFALL_LIGHT_MM_HR = 2.5
RAINFALL_MODERATE_MM_HR = 7.6
RAINFALL_HEAVY_MM_HR = 35.6
RAINFALL_EXTREME_MM_HR = 64.4

# Rainfall-rate -> 0-100 score curve, shared by the heavy_rainfall and flood
# modules (both are primarily driven by the same physical quantity).
RAINFALL_RATE_SCORE_POINTS = [
    (0.0, 0.0),
    (RAINFALL_LIGHT_MM_HR, 25.0),
    (RAINFALL_MODERATE_MM_HR, 50.0),
    (RAINFALL_HEAVY_MM_HR, 75.0),
    (RAINFALL_EXTREME_MM_HR, 95.0),
    (RAINFALL_EXTREME_MM_HR * 2, 100.0),
]

# --- Thunderstorm: WMO weather codes that directly indicate thunderstorm activity ---
THUNDERSTORM_WEATHER_CODES = {95, 96, 99}
SEVERE_THUNDERSTORM_WEATHER_CODES = {96, 99}  # with hail / severe

# --- CAPE instability bands (J/kg) ---
CAPE_WEAK = 1000.0
CAPE_MODERATE = 2500.0
CAPE_STRONG = 4000.0

# --- Lifted Index instability bands (°C; LOWER = more unstable) ---
LI_STABLE = 0.0
LI_MARGINAL = -3.0
LI_MODERATE = -6.0

# --- Score -> category thresholds — intentionally identical to the
# frontend's src/utils/risk.js (`riskLevelFromScore`) and the RiskLevel enum
# already on the RiskAssessment model, so "severe" means the same 0-100
# number everywhere in the system. ---
LEVEL_MODERATE_AT = 30
LEVEL_HIGH_AT = 55
LEVEL_SEVERE_AT = 80

# Lightning has no independent sensor or authoritative-warning data source in
# this system (see app/services/providers/imd.py — IMD's real warning feed is
# access-gated). The "lightning" hazard module is a *potential/favorability*
# indicator derived entirely from the same thunderstorm proxies (weather
# code + CAPE/LI) as the thunderstorm module, NEVER an independent "detected
# lightning" claim. Every field name, docstring, and API response for this
# hazard says "potential", never "detected".
LIGHTNING_EVIDENCE_NOTE = (
    "Derived from thunderstorm-favorable conditions (weather code, CAPE, "
    "Lifted Index) — not an actual lightning-strike detection or an "
    "authoritative lightning warning. No lightning sensor or official "
    "lightning-warning feed is connected."
)


def score_to_level(score: int) -> RiskLevel:
    """Same bands as src/utils/risk.js's riskLevelFromScore — keep in sync."""
    if score >= LEVEL_SEVERE_AT:
        return RiskLevel.SEVERE
    if score >= LEVEL_HIGH_AT:
        return RiskLevel.HIGH
    if score >= LEVEL_MODERATE_AT:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))
