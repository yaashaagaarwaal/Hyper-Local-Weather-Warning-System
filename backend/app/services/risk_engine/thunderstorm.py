"""Thunderstorm hazard module.

Three real, documented signals, combined by taking the maximum (a strong
single signal is enough to flag the hazard — it is not diluted by averaging
against a calm co-indicator):

  1. WMO weather code 95/96/99 — the forecast model's own direct
     thunderstorm/severe-thunderstorm flag ("forecast" category).
  2. CAPE (Convective Available Potential Energy) — standard convective-
     instability measure ("forecast"/model-derived category).
  3. Lifted Index — standard convective-stability measure, same category.

No lightning-strike sensor or official thunderstorm warning feed is
connected (see app/services/providers/imd.py) — every indicator here is a
forecast-model variable, never a measured/observed thunderstorm event.
"""

from app.services.risk_engine.scoring_utils import piecewise_score
from app.services.risk_engine.thresholds import (
    CAPE_MODERATE,
    CAPE_STRONG,
    CAPE_WEAK,
    LI_MARGINAL,
    LI_MODERATE,
    LI_STABLE,
    SEVERE_THUNDERSTORM_WEATHER_CODES,
    THUNDERSTORM_WEATHER_CODES,
    clamp_score,
    score_to_level,
)
from app.services.risk_engine.types import HazardAssessment, HazardIndicator, WeatherInput

_CAPE_POINTS = [(0.0, 0.0), (CAPE_WEAK, 30.0), (CAPE_MODERATE, 55.0), (CAPE_STRONG, 80.0), (CAPE_STRONG * 1.5, 100.0)]
# Lifted Index: LOWER = more unstable. piecewise_score requires points sorted
# ascending by x, so these are expressed in NEGATED-LI space (ascending) —
# computed from the named thresholds, not re-typed as separate magic numbers.
_NEGATIVE_LI_POINTS = [
    (-LI_STABLE, 0.0),
    (-LI_MARGINAL, 35.0),
    (-LI_MODERATE, 65.0),
    (-(LI_MODERATE - 3), 100.0),
]


def _weather_code_score(code: int | None) -> float:
    if code is None:
        return 0.0
    if code in SEVERE_THUNDERSTORM_WEATHER_CODES:
        return 90.0
    if code in THUNDERSTORM_WEATHER_CODES:
        return 70.0
    return 0.0


def assess(weather: WeatherInput) -> HazardAssessment:
    code_score = _weather_code_score(weather.weather_code)
    cape_score = piecewise_score(weather.cape_j_kg, _CAPE_POINTS) if weather.cape_j_kg is not None else 0.0
    li_score = (
        piecewise_score(-weather.lifted_index, _NEGATIVE_LI_POINTS)
        if weather.lifted_index is not None
        else 0.0
    )

    score = clamp_score(max(code_score, cape_score, li_score))

    indicators = []
    if weather.weather_code is not None:
        indicators.append(
            HazardIndicator(
                label="Weather code", value=f"WMO {weather.weather_code} ({weather.condition})", category="forecast"
            )
        )
    if weather.cape_j_kg is not None:
        indicators.append(
            HazardIndicator(label="CAPE", value=f"{weather.cape_j_kg:.0f} J/kg", category="forecast")
        )
    if weather.lifted_index is not None:
        indicators.append(
            HazardIndicator(label="Lifted Index", value=f"{weather.lifted_index:.1f}", category="forecast")
        )

    return HazardAssessment(
        hazard="thunderstorm",
        score=score,
        level=score_to_level(score),
        indicators=indicators,
        notes=(
            "Scored from the forecast model's own weather code plus standard CAPE/Lifted "
            "Index convective-instability indicators — not a lightning or storm-cell "
            "detection system."
        ),
    )
