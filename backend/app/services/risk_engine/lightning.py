"""Lightning hazard module — a POTENTIAL/favorability indicator, never a
detection claim.

No lightning-strike sensor network or authoritative lightning-warning feed is
connected anywhere in this system (IMD's own API, which does publish real
warnings, is access-gated — see app/services/providers/imd.py and
docs/architecture.md §3). This module does not and cannot claim "lightning
detected" or "lightning observed."

Instead, it reuses the thunderstorm module's own instability signals
(weather code + CAPE + Lifted Index) — the same standard meteorological
basis for inferring thunderstorm activity, which by definition is
accompanied by lightning — and applies a documented, deliberately
conservative dampening factor (`LIGHTNING_DAMPENING`) since "a thunderstorm
is forecast" is one inferential step short of "lightning is occurring now".
This keeps the lightning score never exceeding (and typically slightly below)
the thunderstorm score for the same input, which is the honest ordering given
we have direct-er evidence for the thunderstorm hazard than for lightning
specifically.
"""

from app.services.risk_engine import thunderstorm
from app.services.risk_engine.thresholds import LIGHTNING_EVIDENCE_NOTE, clamp_score, score_to_level
from app.services.risk_engine.types import HazardAssessment, HazardIndicator, WeatherInput

LIGHTNING_DAMPENING = 0.9


def assess(weather: WeatherInput) -> HazardAssessment:
    thunderstorm_assessment = thunderstorm.assess(weather)
    score = clamp_score(thunderstorm_assessment.score * LIGHTNING_DAMPENING)

    indicators = [
        HazardIndicator(
            label="Basis",
            value="Derived from thunderstorm indicators (see Thunderstorm hazard)",
            category="forecast",
        ),
    ]

    return HazardAssessment(
        hazard="lightning",
        score=score,
        level=score_to_level(score),
        indicators=indicators,
        notes=LIGHTNING_EVIDENCE_NOTE,
    )
