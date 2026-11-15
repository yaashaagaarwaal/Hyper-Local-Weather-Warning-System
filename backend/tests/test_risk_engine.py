from datetime import datetime, timezone

import pytest

from app.services.risk_engine import assess_grid
from app.services.risk_engine.thresholds import score_to_level
from app.services.risk_engine.types import EvidenceQuality, RiskLevel, WeatherInput


def _weather(**overrides) -> WeatherInput:
    base = dict(
        temperature_c=25.0, humidity_pct=50.0, wind_speed_kmh=5.0, rainfall_mm=0.0,
        rain_probability_pct=5.0, pressure_hpa=1013.0, condition="Clear Sky", weather_code=0,
        cape_j_kg=50.0, lifted_index=8.0, observed_at=datetime.now(timezone.utc),
        is_stale=False, distance_km=2.0, data_available=True,
    )
    base.update(overrides)
    return WeatherInput(**base)


# --- score_to_level ---

@pytest.mark.parametrize(
    "score,expected",
    [(0, RiskLevel.LOW), (29, RiskLevel.LOW), (30, RiskLevel.MODERATE), (54, RiskLevel.MODERATE),
     (55, RiskLevel.HIGH), (79, RiskLevel.HIGH), (80, RiskLevel.SEVERE), (100, RiskLevel.SEVERE)],
)
def test_score_to_level_thresholds(score, expected):
    assert score_to_level(score) == expected


# --- calm / baseline scenario ---

def test_calm_clear_weather_scores_low():
    result = assess_grid(_weather())
    assert result.overall_score < 10
    assert result.risk_level == RiskLevel.LOW
    assert result.evidence_quality == EvidenceQuality.MEASURED
    assert result.confidence > 0.9


# --- heavy rainfall module ---

def test_heavy_rainfall_rate_drives_the_hazard_score():
    light = assess_grid(_weather(rainfall_mm=1.0, rain_probability_pct=20))
    heavy = assess_grid(_weather(rainfall_mm=50.0, rain_probability_pct=90))
    assert heavy.hazards["heavy_rainfall"].score > light.hazards["heavy_rainfall"].score
    assert heavy.hazards["heavy_rainfall"].level in (RiskLevel.HIGH, RiskLevel.SEVERE)


def test_zero_rainfall_and_zero_probability_scores_zero_rainfall_hazard():
    result = assess_grid(_weather(rainfall_mm=0.0, rain_probability_pct=0.0))
    assert result.hazards["heavy_rainfall"].score == 0


def test_rainfall_indicators_are_labeled_measured_and_forecast():
    result = assess_grid(_weather(rainfall_mm=10.0, rain_probability_pct=80))
    categories = {i.label: i.category for i in result.hazards["heavy_rainfall"].indicators}
    assert categories["Rainfall rate"] == "measured"
    assert categories["Rain probability"] == "forecast"


# --- thunderstorm module ---

def test_severe_thunderstorm_weather_code_scores_high():
    result = assess_grid(_weather(weather_code=99, cape_j_kg=100, lifted_index=5))
    assert result.hazards["thunderstorm"].score >= 80


def test_clear_sky_with_stable_atmosphere_scores_thunderstorm_zero():
    result = assess_grid(_weather(weather_code=0, cape_j_kg=0, lifted_index=10))
    assert result.hazards["thunderstorm"].score == 0


def test_high_cape_alone_elevates_thunderstorm_score_even_without_code():
    result = assess_grid(_weather(weather_code=2, cape_j_kg=4200, lifted_index=2))
    assert result.hazards["thunderstorm"].score > 50


def test_very_negative_lifted_index_alone_elevates_thunderstorm_score():
    result = assess_grid(_weather(weather_code=1, cape_j_kg=100, lifted_index=-8))
    assert result.hazards["thunderstorm"].score > 50


def test_thunderstorm_indicators_never_claim_measured_category():
    """Thunderstorm has no real sensor — every indicator must be forecast,
    never "measured" (which would overstate certainty)."""
    result = assess_grid(_weather(weather_code=99, cape_j_kg=4000, lifted_index=-8))
    categories = {i.category for i in result.hazards["thunderstorm"].indicators}
    assert categories == {"forecast"}


# --- lightning module: potential, never "detection" ---

def test_lightning_score_never_exceeds_thunderstorm_score_for_same_input():
    weather = _weather(weather_code=99, cape_j_kg=4500, lifted_index=-9)
    result = assess_grid(weather)
    assert result.hazards["lightning"].score <= result.hazards["thunderstorm"].score


def test_lightning_notes_explicitly_disclaim_detection():
    result = assess_grid(_weather(weather_code=99, cape_j_kg=4000, lifted_index=-8))
    notes = result.hazards["lightning"].notes.lower()
    assert "not an actual lightning-strike detection" in notes
    assert "no lightning sensor" in notes


def test_lightning_indicators_never_claim_measured_or_official_warning():
    result = assess_grid(_weather(weather_code=95, cape_j_kg=2000, lifted_index=-5))
    categories = {i.category for i in result.hazards["lightning"].indicators}
    assert "measured" not in categories
    assert "official_warning" not in categories


# --- flood module ---

def test_high_humidity_increases_flood_score_over_same_rainfall_with_low_humidity():
    dry = assess_grid(_weather(rainfall_mm=5.0, humidity_pct=40))
    humid = assess_grid(_weather(rainfall_mm=5.0, humidity_pct=95))
    assert humid.hazards["flood"].score > dry.hazards["flood"].score


def test_flood_notes_disclaim_hydrological_modeling():
    result = assess_grid(_weather(rainfall_mm=10.0))
    assert "does not" in result.hazards["flood"].notes.lower() or "not model" in result.hazards["flood"].notes.lower()


# --- aggregation rule: overall = max of hazards ---

def test_overall_score_equals_the_single_highest_hazard_score():
    result = assess_grid(_weather(weather_code=99, cape_j_kg=4500, lifted_index=-9, rainfall_mm=0))
    assert result.overall_score == max(h.score for h in result.hazards.values())
    assert result.dominant_hazard == max(result.hazards.items(), key=lambda kv: kv[1].score)[0]


def test_severe_single_hazard_is_not_diluted_by_calm_cohazards():
    # Heavy rain, but totally calm thunderstorm/CAPE/LI — overall must still
    # reflect the severe rain, not an averaged-down number.
    result = assess_grid(_weather(rainfall_mm=80.0, rain_probability_pct=98, weather_code=0, cape_j_kg=0, lifted_index=10))
    assert result.risk_level in (RiskLevel.HIGH, RiskLevel.SEVERE)
    assert result.dominant_hazard in ("heavy_rainfall", "flood")


# --- missing data ---

def test_no_data_available_returns_insufficient_data_with_zero_confidence():
    weather = WeatherInput(
        temperature_c=None, humidity_pct=None, wind_speed_kmh=None, rainfall_mm=None,
        rain_probability_pct=None, pressure_hpa=None, condition=None, weather_code=None,
        cape_j_kg=None, lifted_index=None, observed_at=None, is_stale=True,
        distance_km=None, data_available=False,
    )
    result = assess_grid(weather)
    assert result.overall_score == 0
    assert result.risk_level == RiskLevel.LOW
    assert result.evidence_quality == EvidenceQuality.INSUFFICIENT_DATA
    assert result.confidence == 0.0
    assert result.dominant_hazard == "none"


# --- evidence quality / confidence from spatial-temporal facts ---

def test_nearby_fresh_data_is_measured_with_high_confidence():
    result = assess_grid(_weather(distance_km=1.0, is_stale=False))
    assert result.evidence_quality == EvidenceQuality.MEASURED
    assert result.confidence >= 0.9


def test_distant_data_is_regional_estimate_with_lower_confidence():
    result = assess_grid(_weather(distance_km=40.0, is_stale=False))
    assert result.evidence_quality == EvidenceQuality.REGIONAL_ESTIMATE
    assert result.confidence < 0.9


def test_stale_data_is_regional_estimate_even_if_nearby():
    result = assess_grid(_weather(distance_km=1.0, is_stale=True))
    assert result.evidence_quality == EvidenceQuality.REGIONAL_ESTIMATE


def test_distant_and_stale_has_the_lowest_confidence():
    near_fresh = assess_grid(_weather(distance_km=1.0, is_stale=False))
    far_stale = assess_grid(_weather(distance_km=40.0, is_stale=True))
    assert far_stale.confidence < near_fresh.confidence
