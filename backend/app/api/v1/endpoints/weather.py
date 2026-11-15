from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_weather_service, resolve_location
from app.core.exceptions import AppError
from app.db.session import get_db
from app.models import Location
from app.repositories import location_repository
from app.schemas.weather import CurrentWeather, DailyForecast, HourlyForecast, WeatherLocation
from app.services.weather_service import WeatherService

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/current", response_model=CurrentWeather)
async def get_current_weather(
    location: tuple[Location | None, float, float] = Depends(resolve_location),
    service: WeatherService = Depends(get_weather_service),
) -> CurrentWeather:
    """Real current conditions for one location (by `locationId`) or raw
    `lat`/`lon` coordinates. Backed by Open-Meteo; persisted to
    WeatherObservation when `locationId` is used.
    """
    loc, lat, lon = location
    return await service.get_current(loc, lat, lon)


@router.get("/current/batch", response_model=list[CurrentWeather])
async def get_current_weather_batch(
    location_ids: str = Query(
        ..., alias="locationIds", description="Comma-separated known location ids"
    ),
    db: Session = Depends(get_db),
    service: WeatherService = Depends(get_weather_service),
) -> list[CurrentWeather]:
    """Current conditions for several known locations in one batched
    Open-Meteo request — mirrors the frontend's `fetchAllCurrentWeather()`.
    """
    ids = [i.strip() for i in location_ids.split(",") if i.strip()]
    if not ids:
        raise AppError("locationIds must contain at least one id.", status_code=400)

    locations = []
    for location_id in ids:
        loc = location_repository.get_by_id(db, location_id)
        if loc is None:
            raise AppError(f"Unknown locationId: {location_id!r}", status_code=404)
        locations.append(loc)

    return await service.get_current_batch(locations)


@router.get("/hourly", response_model=HourlyForecast)
async def get_hourly_forecast(
    location: tuple[Location | None, float, float] = Depends(resolve_location),
    hours: int = Query(24, ge=1, le=48, description="How many hourly points to return"),
    service: WeatherService = Depends(get_weather_service),
) -> HourlyForecast:
    loc, lat, lon = location
    return await service.get_hourly(loc, lat, lon, hours)


@router.get("/daily", response_model=DailyForecast)
async def get_daily_forecast(
    location: tuple[Location | None, float, float] = Depends(resolve_location),
    days: int = Query(7, ge=1, le=8, description="How many daily points to return"),
    service: WeatherService = Depends(get_weather_service),
) -> DailyForecast:
    loc, lat, lon = location
    return await service.get_daily(loc, lat, lon, days)


@router.get("/locations", response_model=list[WeatherLocation])
def get_locations(db: Session = Depends(get_db)) -> list[WeatherLocation]:
    """All monitored locations (mirrors the frontend's `src/data/locations.js`)."""
    rows = location_repository.list_all(db)
    return [
        WeatherLocation(
            id=r.id,
            name=r.name,
            region=r.region,
            latitude=float(r.latitude),
            longitude=float(r.longitude),
            is_primary=r.is_primary,
            saved_as=r.saved_as,
        )
        for r in rows
    ]
