"""Shared FastAPI dependencies for the v1 API."""

from fastapi import Depends, Query, Request
from sqlalchemy.orm import Session

from app.core.exceptions import AppError, NotFoundError
from app.db.session import get_db
from app.models import Location
from app.repositories import location_repository
from app.services.providers.open_meteo import OpenMeteoProvider
from app.services.weather_service import WeatherService


def get_weather_service(request: Request, db: Session = Depends(get_db)) -> WeatherService:
    provider = OpenMeteoProvider(client=request.app.state.http_client)
    return WeatherService(provider=provider, db=db)


def resolve_location(
    location_id: str | None = Query(None, alias="locationId", description="A known location id, e.g. 'loc-ndls'"),
    lat: float | None = Query(None, ge=-90, le=90, description="Latitude, if not using locationId"),
    lon: float | None = Query(None, ge=-180, le=180, description="Longitude, if not using locationId"),
    db: Session = Depends(get_db),
) -> tuple[Location | None, float, float]:
    """Resolves a request's target coordinates from either `locationId` or
    raw `lat`/`lon`. Returns (location_or_none, lat, lon).

    Raises 400 if neither is given, 404 if `locationId` doesn't match a known
    Location.
    """
    if location_id is not None:
        location = location_repository.get_by_id(db, location_id)
        if location is None:
            raise NotFoundError(f"Unknown locationId: {location_id!r}")
        return location, float(location.latitude), float(location.longitude)

    if lat is not None and lon is not None:
        return None, lat, lon

    raise AppError("Provide either 'locationId' or both 'lat' and 'lon'.", status_code=400)
