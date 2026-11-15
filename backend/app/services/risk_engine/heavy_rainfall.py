"""Heavy rainfall hazard module.

Primary signal: current rainfall rate (mm/hr), scored against the WMO/IMD
hourly rainfall-intensity bands (thresholds.py). Secondary signal: the
forecast rain probability, so a dry-but-high-probability moment still reads
as elevated risk rather than zero.

Both inputs are real, provider-reported values (Open-Meteo) — "measured"
(current rate) and "forecast" (probability) respectively, never an official
IMD rainfall warning (that integration doesn't exist — see
app/services/providers/imd.py).

Known limitation: this scores the *current hourly rate*, not a rolling
3h/6h/24h accumulation (unlike the historical CSV dataset, which has those —
see docs/dataset_analysis.md §2 — the live Open-Meteo ingestion pipeline does
not fetch/store rolling sums yet). A single intense hour and a sustained
multi-hour event at the same instantaneous rate currently score identically;
a future enhancement is noted in docs/architecture.md.
"""

from app.services.risk_engine.scoring_utils import piecewise_score
from app.services.risk_engine.thresholds import (
    RAINFALL_RATE_SCORE_POINTS,
    clamp_score,
    score_to_level,
)
from app.services.risk_engine.types import HazardAssessment, HazardIndicator, WeatherInput


def assess(weather: WeatherInput) -> HazardAssessment:
    rate = weather.rainfall_mm
    probability = weather.rain_probability_pct

    rate_score = piecewise_score(rate, RAINFALL_RATE_SCORE_POINTS) if rate is not None else 0.0
    probability_score = probability if probability is not None else 0.0

    # 70% current rate, 30% forecast probability — a simple, documented
    # heuristic blend (not a fitted/calibrated weight), chosen so active
    # rainfall dominates but an imminent-rain forecast still registers.
    combined = rate_score * 0.7 + probability_score * 0.3
    score = clamp_score(combined)

    indicators = []
    if rate is not None:
        indicators.append(
            HazardIndicator(label="Rainfall rate", value=f"{rate:.1f} mm/hr", category="measured")
        )
    if probability is not None:
        indicators.append(
            HazardIndicator(
                label="Rain probability", value=f"{probability:.0f}%", category="forecast"
            )
        )

    return HazardAssessment(
        hazard="heavy_rainfall",
        score=score,
        level=score_to_level(score),
        indicators=indicators,
        notes=(
            "Scored from current hourly rainfall rate (WMO/IMD hourly-intensity bands) "
            "and forecast rain probability. No rolling accumulation data in the live "
            "pipeline yet — see module docstring."
        ),
    )
