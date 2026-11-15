"""Transparent, rule-based weather risk assessment engine (Phase 3).

Entry point: `assess_grid(weather: WeatherInput) -> RiskResult`
(app.services.risk_engine.aggregator). Everything here is deterministic and
inspectable — no ML model, no learned weights, no fitted probabilities (see
docs/architecture.md §7 for the scoring formulas and cited thresholds).
"""

from app.services.risk_engine.aggregator import assess_grid
from app.services.risk_engine.types import EvidenceQuality, HazardAssessment, RiskResult, WeatherInput

__all__ = ["assess_grid", "EvidenceQuality", "HazardAssessment", "RiskResult", "WeatherInput"]
