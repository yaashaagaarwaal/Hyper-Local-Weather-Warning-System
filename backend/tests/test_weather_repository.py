from datetime import datetime, timezone

from sqlalchemy import select

from app.models import Location, WeatherForecast, WeatherObservation
from app.models.weather_forecast import ForecastHorizon
from app.repositories import weather_repository
from app.schemas.weather import CurrentWeather, DailyForecast, DailyForecastPoint

NOW = datetime.now(timezone.utc)


def _make_location(db_session, suffix="repo-1"):
    loc = Location(
        id=f"loc-{suffix}", name="Repo Test", region="Test", latitude=28.6, longitude=77.2
    )
    db_session.add(loc)
    db_session.flush()
    return loc


def _current_weather(**overrides):
    base = dict(
        latitude=28.6,
        longitude=77.2,
        temperature=25.0,
        feels_like=26.0,
        humidity=60.0,
        wind_speed=5.0,
        wind_direction="N",
        wind_direction_degrees=0.0,
        pressure=1013.0,
        visibility=8.0,
        rainfall=0.0,
        rain_probability=10.0,
        condition="Clear Sky",
        uv_index=3.0,
        observed_at=NOW,
        retrieved_at=NOW,
        source="open-meteo",
    )
    base.update(overrides)
    return CurrentWeather(**base)


def test_save_observation_inserts_a_row(db_session):
    loc = _make_location(db_session)
    weather_repository.save_observation(db_session, loc.id, _current_weather())

    rows = db_session.execute(
        select(WeatherObservation).where(WeatherObservation.location_id == loc.id)
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].temperature_c == 25.0
    assert rows[0].wind_direction_deg == 0.0


def test_save_observation_upserts_same_timestamp_instead_of_duplicating(db_session):
    loc = _make_location(db_session)
    weather_repository.save_observation(db_session, loc.id, _current_weather(temperature=25.0))
    weather_repository.save_observation(db_session, loc.id, _current_weather(temperature=27.5))

    rows = db_session.execute(
        select(WeatherObservation).where(WeatherObservation.location_id == loc.id)
    ).scalars().all()
    assert len(rows) == 1, "expected an upsert, not a second row"
    assert rows[0].temperature_c == 27.5, "expected the second call's value to win"


def test_save_observation_different_timestamps_create_separate_rows(db_session):
    loc = _make_location(db_session)
    from datetime import timedelta

    weather_repository.save_observation(db_session, loc.id, _current_weather(observed_at=NOW))
    weather_repository.save_observation(
        db_session, loc.id, _current_weather(observed_at=NOW + timedelta(hours=1))
    )

    rows = db_session.execute(
        select(WeatherObservation).where(WeatherObservation.location_id == loc.id)
    ).scalars().all()
    assert len(rows) == 2


def test_save_daily_forecast_upserts_per_point(db_session):
    from datetime import timedelta

    loc = _make_location(db_session, suffix="repo-daily")
    day1, day2 = NOW, NOW + timedelta(days=1)
    forecast = DailyForecast(
        latitude=28.6,
        longitude=77.2,
        source="open-meteo",
        generated_at=NOW,
        retrieved_at=NOW,
        points=[
            DailyForecastPoint(date=day1, day="Thu", temperature_high=30, temperature_low=18, rainfall=0, rain_probability=0, condition="Clear Sky"),
            DailyForecastPoint(date=day2, day="Fri", temperature_high=31, temperature_low=19, rainfall=1, rain_probability=20, condition="Overcast"),
        ],
    )
    n = weather_repository.save_daily_forecast(db_session, loc.id, forecast)
    assert n == 2

    rows = db_session.execute(
        select(WeatherForecast).where(
            WeatherForecast.location_id == loc.id, WeatherForecast.horizon == ForecastHorizon.DAILY
        )
    ).scalars().all()
    assert len(rows) == 2

    # Re-saving the same forecast_for point (day1) with an updated value
    # upserts in place, rather than duplicating.
    forecast.points[0] = DailyForecastPoint(
        date=day1, day="Thu", temperature_high=33, temperature_low=20, rainfall=0, rain_probability=0, condition="Clear Sky"
    )
    weather_repository.save_daily_forecast(db_session, loc.id, forecast)
    rows_after = db_session.execute(
        select(WeatherForecast).where(
            WeatherForecast.location_id == loc.id, WeatherForecast.horizon == ForecastHorizon.DAILY
        )
    ).scalars().all()
    assert len(rows_after) == 2
    updated = next(r for r in rows_after if r.temperature_high_c == 33)
    assert updated is not None
