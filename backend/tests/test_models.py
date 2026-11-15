from datetime import datetime, timezone

from app.models import (
    Alert,
    AlertStatus,
    GridCell,
    Location,
    ModelPrediction,
    RiskAssessment,
    RiskLevel,
    WeatherForecast,
    WeatherObservation,
)
from app.models.weather_forecast import ForecastHorizon

NOW = datetime.now(timezone.utc)


def _make_location(db_session, suffix="1"):
    loc = Location(
        id=f"loc-test-{suffix}",
        name="Test Location",
        region="Test Region",
        latitude=28.6,
        longitude=77.2,
        is_primary=False,
        saved_as=None,
    )
    db_session.add(loc)
    db_session.flush()
    return loc


def _make_grid_cell(db_session, suffix="1"):
    cell = GridCell(
        id=f"GRID-TEST-{suffix}",
        row=0,
        col=0,
        bounds=[[28.0, 77.0], [28.0, 77.1], [28.1, 77.1], [28.1, 77.0]],
        center_lat=28.05,
        center_lng=77.05,
    )
    db_session.add(cell)
    db_session.flush()
    return cell


def test_create_location(db_session):
    loc = _make_location(db_session)
    db_session.commit()
    fetched = db_session.get(Location, loc.id)
    assert fetched is not None
    assert fetched.name == "Test Location"
    assert fetched.created_at is not None


def test_create_grid_cell(db_session):
    cell = _make_grid_cell(db_session)
    db_session.commit()
    fetched = db_session.get(GridCell, cell.id)
    assert fetched is not None
    assert len(fetched.bounds) == 4


def test_weather_observation_links_to_location(db_session):
    loc = _make_location(db_session)
    obs = WeatherObservation(
        location_id=loc.id,
        recorded_at=NOW,
        temperature_c=21.5,
        humidity_pct=77,
        wind_speed_kmh=3.4,
        rainfall_mm=0,
        condition="Clear Sky",
        source="open-meteo",
    )
    db_session.add(obs)
    db_session.commit()

    assert obs.id is not None
    assert obs.location.id == loc.id
    assert loc.weather_observations[0].temperature_c == 21.5


def test_weather_forecast_hourly_and_daily_rows(db_session):
    loc = _make_location(db_session)
    hourly = WeatherForecast(
        location_id=loc.id,
        horizon=ForecastHorizon.HOURLY,
        forecast_for=NOW,
        generated_at=NOW,
        temperature_c=22.0,
        rainfall_mm=1.2,
        source="open-meteo",
    )
    daily = WeatherForecast(
        location_id=loc.id,
        horizon=ForecastHorizon.DAILY,
        forecast_for=NOW,
        generated_at=NOW,
        temperature_high_c=30.0,
        temperature_low_c=18.0,
        rainfall_mm=5.0,
        source="open-meteo",
    )
    db_session.add_all([hourly, daily])
    db_session.commit()

    assert hourly.horizon == ForecastHorizon.HOURLY
    assert daily.horizon == ForecastHorizon.DAILY
    assert len(loc.weather_forecasts) == 2


def test_risk_assessment_links_to_grid_cell_with_flexible_hazard_scores(db_session):
    cell = _make_grid_cell(db_session)
    assessment = RiskAssessment(
        grid_cell_id=cell.id,
        assessed_at=NOW,
        overall_score=54,
        risk_level=RiskLevel.MODERATE,
        dominant_hazard="Flash Flood",
        hazard_scores={"heavyRain": 22, "thunderstorm": 46, "lightning": 12, "flood": 61},
        confidence=0.87,
        source="mock",
    )
    db_session.add(assessment)
    db_session.commit()

    assert assessment.id is not None
    assert cell.risk_assessments[0].hazard_scores["flood"] == 61
    assert assessment.risk_level == RiskLevel.MODERATE


def test_model_prediction_links_to_grid_cell(db_session):
    cell = _make_grid_cell(db_session)
    prediction = ModelPrediction(
        grid_cell_id=cell.id,
        predicted_at=NOW,
        valid_for=NOW,
        horizon_minutes=15,
        thunderstorm_probability=0.42,
        cloudburst_probability=0.18,
        flash_flood_probability=0.30,
        model_version="unit-test-stub",
    )
    db_session.add(prediction)
    db_session.commit()

    assert prediction.id is not None
    assert cell.model_predictions[0].model_version == "unit-test-stub"


def _make_alert(db_session, cell, suffix="A"):
    alert = Alert(
        alert_key=f"heavy_rainfall:{cell.id}",
        grid_cell_id=cell.id,
        title="Heavy Rainfall Warning",
        hazard_type="heavy_rainfall",
        severity=RiskLevel.HIGH,
        status=AlertStatus.ACTIVE,
        location_label="South Delhi & Saket",
        risk_score=68,
        evidence="Rainfall rate: 12.4 mm/hr (measured)",
        description="Very heavy rainfall expected.",
        recommended_action="Avoid low-lying underpasses.",
        confidence=0.87,
        source="rule-based-v1",
        issued_at=NOW,
        expires_at=NOW,
        last_seen_at=NOW,
        acknowledged=False,
    )
    db_session.add(alert)
    db_session.commit()
    return alert


def test_alert_belongs_to_one_grid_cell(db_session):
    cell = _make_grid_cell(db_session, suffix="A")
    alert = _make_alert(db_session, cell)

    fetched = db_session.get(Alert, alert.id)
    assert fetched.grid_cell_id == cell.id
    assert cell.alerts[0].id == alert.id
    assert fetched.severity == RiskLevel.HIGH


def test_deleting_grid_cell_cascades_to_its_alerts(db_session):
    cell = _make_grid_cell(db_session, suffix="B")
    alert = _make_alert(db_session, cell, suffix="B")
    alert_id = alert.id

    db_session.delete(cell)
    db_session.commit()

    assert db_session.get(Alert, alert_id) is None
