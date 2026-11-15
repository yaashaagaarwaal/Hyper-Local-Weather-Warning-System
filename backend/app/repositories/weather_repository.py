"""Persistence for real weather readings.

Every write here is an upsert keyed on the unique constraints added to
WeatherObservation / WeatherForecast in Phase 2 (location + timestamp +
source), so re-fetching the same reading updates it in place instead of
accumulating duplicate rows.
"""

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import ForecastHorizon, WeatherForecast, WeatherObservation
from app.schemas.weather import CurrentWeather, DailyForecast, HourlyForecast


def save_observation(db: Session, location_id: str, data: CurrentWeather) -> None:
    stmt = pg_insert(WeatherObservation).values(
        location_id=location_id,
        recorded_at=data.observed_at,
        temperature_c=data.temperature,
        feels_like_c=data.feels_like,
        humidity_pct=data.humidity,
        wind_speed_kmh=data.wind_speed,
        wind_direction_deg=data.wind_direction_degrees,
        pressure_hpa=data.pressure,
        visibility_km=data.visibility,
        rainfall_mm=data.rainfall,
        rain_probability_pct=data.rain_probability,
        condition=data.condition,
        uv_index=data.uv_index,
        weather_code=data.weather_code,
        cape_j_kg=data.cape,
        lifted_index=data.lifted_index,
        source=data.source,
    )
    update_cols = {
        c.name: c
        for c in stmt.excluded
        if c.name not in ("id", "location_id", "recorded_at", "source", "created_at")
    }
    stmt = stmt.on_conflict_do_update(
        constraint="uq_weather_observations_location_recorded_source",
        set_=update_cols,
    )
    db.execute(stmt)
    db.commit()


def save_hourly_forecast(db: Session, location_id: str, data: HourlyForecast) -> int:
    return _save_forecast_points(
        db,
        location_id=location_id,
        horizon=ForecastHorizon.HOURLY,
        source=data.source,
        generated_at=data.generated_at,
        rows=[
            {
                "forecast_for": p.time,
                "temperature_c": p.temperature,
                "humidity_pct": p.humidity,
                "rainfall_mm": p.rainfall,
                "rain_probability_pct": p.rain_probability,
                "condition": None,
            }
            for p in data.points
        ],
    )


def save_daily_forecast(db: Session, location_id: str, data: DailyForecast) -> int:
    return _save_forecast_points(
        db,
        location_id=location_id,
        horizon=ForecastHorizon.DAILY,
        source=data.source,
        generated_at=data.generated_at,
        rows=[
            {
                "forecast_for": p.date,
                "temperature_high_c": p.temperature_high,
                "temperature_low_c": p.temperature_low,
                "rainfall_mm": p.rainfall,
                "rain_probability_pct": p.rain_probability,
                "condition": p.condition,
            }
            for p in data.points
        ],
    )


def _save_forecast_points(
    db: Session,
    *,
    location_id: str,
    horizon: ForecastHorizon,
    source: str,
    generated_at,
    rows: list[dict],
) -> int:
    if not rows:
        return 0

    values = [
        {
            "location_id": location_id,
            "horizon": horizon,
            "generated_at": generated_at,
            "source": source,
            **row,
        }
        for row in rows
    ]
    stmt = pg_insert(WeatherForecast).values(values)
    update_cols = {
        c.name: c
        for c in stmt.excluded
        if c.name
        not in ("id", "location_id", "horizon", "forecast_for", "source", "created_at")
    }
    stmt = stmt.on_conflict_do_update(
        constraint="uq_weather_forecasts_location_horizon_for_source",
        set_=update_cols,
    )
    db.execute(stmt)
    db.commit()
    return len(rows)
