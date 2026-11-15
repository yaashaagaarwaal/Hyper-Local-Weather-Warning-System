"""Flood risk hazard module.

A rainfall-rate-driven proxy, NOT a hydrological or drainage model: it reuses
the same measured rainfall rate and WMO/IMD intensity bands as the heavy-
rainfall module (the primary physical driver of flash flooding), with a
documented humidity-based modifier (high relative humidity as a proxy for a
saturated, monsoon-type air mass — more of the rain is likely to run off
rather than evaporate). It does NOT model terrain, drainage capacity, or soil/ground saturation
directly — this module's own output is purely the dynamic, rainfall-driven
score. A static historical-vulnerability modifier (docs/architecture.md §7's
"a 70% probability in a well-drained grid is moderate, but the same
probability in a known flood-prone grid is severe") is applied AFTER this
module runs, by `risk_service._apply_flood_vulnerability` — see
`flood_vulnerability.py` — and only for the 20 Delhi-NCR districts that
have been verified against the India Flood Inventory; everywhere else still
has no vulnerability layer.
"""

from app.services.risk_engine.scoring_utils import piecewise_score
from app.services.risk_engine.thresholds import RAINFALL_RATE_SCORE_POINTS, clamp_score, score_to_level
from app.services.risk_engine.types import HazardAssessment, HazardIndicator, WeatherInput

HUMIDITY_MODIFIER_THRESHOLD_PCT = 80.0
HUMIDITY_MODIFIER_MAX_POINTS = 15.0


def assess(weather: WeatherInput) -> HazardAssessment:
    rate = weather.rainfall_mm
    humidity = weather.humidity_pct

    base_score = piecewise_score(rate, RAINFALL_RATE_SCORE_POINTS) if rate is not None else 0.0

    humidity_bonus = 0.0
    if humidity is not None and humidity > HUMIDITY_MODIFIER_THRESHOLD_PCT:
        saturation_fraction = (humidity - HUMIDITY_MODIFIER_THRESHOLD_PCT) / (100 - HUMIDITY_MODIFIER_THRESHOLD_PCT)
        humidity_bonus = saturation_fraction * HUMIDITY_MODIFIER_MAX_POINTS

    score = clamp_score(base_score + humidity_bonus)

    indicators = []
    if rate is not None:
        indicators.append(
            HazardIndicator(label="Rainfall rate", value=f"{rate:.1f} mm/hr", category="measured")
        )
    if humidity is not None:
        indicators.append(
            HazardIndicator(label="Relative humidity", value=f"{humidity:.0f}%", category="measured")
        )

    return HazardAssessment(
        hazard="flood",
        score=score,
        level=score_to_level(score),
        indicators=indicators,
        notes=(
            "Rainfall-rate-based proxy with a humidity (saturated-air) modifier. Does NOT "
            "model terrain, drainage, or soil/ground saturation — see module docstring."
        ),
    )
