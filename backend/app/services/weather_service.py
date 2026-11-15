"""Centralized weather ingestion service.

The only thing API endpoints talk to for weather data. Orchestrates:
provider call (through the shared cache) → persistence (only when the
request is for a known `Location`, never for an ad-hoc lat/lon) → return the
normalized schema. Endpoints never touch a provider or a repository
directly.
"""

import logging
from contextlib import asynccontextmanager

from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.models import Location
from app.repositories import weather_repository
from app.schemas.weather import CurrentWeather, DailyForecast, HourlyForecast
from app.services.cache import TTLCache
from app.services.providers.base import ProviderUnavailableError, WeatherProvider
from app.services.providers.open_meteo import OpenMeteoError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _translate_provider_errors():
    """Any known provider failure becomes a clean 503 (service unavailable)
    through the centralized error handler, instead of a bare 500 — a weather
    provider being unreachable is an operational condition, not a bug.
    """
    try:
        yield
    except (OpenMeteoError, ProviderUnavailableError) as exc:
        raise AppError(str(exc), status_code=503) from exc


_settings = get_settings()
# One process-wide cache, shared by every request this worker handles (see
# app/services/cache.py's docstring for the multi-worker-process caveat).
_current_cache: TTLCache[CurrentWeather] = TTLCache(_settings.WEATHER_CACHE_TTL_SECONDS)
_current_batch_cache: TTLCache[list[CurrentWeather]] = TTLCache(_settings.WEATHER_CACHE_TTL_SECONDS)
_hourly_cache: TTLCache[HourlyForecast] = TTLCache(_settings.WEATHER_CACHE_TTL_SECONDS)
_daily_cache: TTLCache[DailyForecast] = TTLCache(_settings.WEATHER_CACHE_TTL_SECONDS)
# Phase 4: the ML prediction pipeline's own history fetch, kept as a separate
# cache (not reused from _current_cache/_hourly_cache) because it requests a
# much wider field set and a different time window than those.
_feature_history_cache: TTLCache[list] = TTLCache(_settings.WEATHER_CACHE_TTL_SECONDS)


def _coord_key(prefix: str, lat: float, lon: float, *extra: object) -> str:
    parts = [prefix, f"{lat:.3f}", f"{lon:.3f}", *map(str, extra)]
    return ":".join(parts)


class WeatherService:
    def __init__(self, provider: WeatherProvider, db: Session):
        self.provider = provider
        self.db = db

    async def get_current(
        self, location: Location | None, lat: float, lon: float
    ) -> CurrentWeather:
        key = _coord_key("current", lat, lon)
        async with _translate_provider_errors():
            result, hit = await _current_cache.get_or_fetch(
                key, lambda: self.provider.get_current(lat, lon)
            )
        result = result.model_copy(update={"served_from_cache": hit})
        logger.info(
            "weather.current lat=%s lon=%s source=%s cache_hit=%s location_id=%s",
            lat, lon, self.provider.name, hit, location.id if location else None,
        )

        if location is not None:
            result = result.model_copy(
                update={
                    "location_id": location.id,
                    "location_name": location.name,
                    "region": location.region,
                }
            )
            await run_in_threadpool(
                weather_repository.save_observation, self.db, location.id, result
            )
        return result

    async def get_current_batch(self, locations: list[Location]) -> list[CurrentWeather]:
        if not locations:
            return []
        key = "current_batch:" + ",".join(sorted(loc.id for loc in locations))
        coords = [(float(loc.latitude), float(loc.longitude)) for loc in locations]
        async with _translate_provider_errors():
            results, hit = await _current_batch_cache.get_or_fetch(
                key, lambda: self.provider.get_current_batch(coords)
            )
        logger.info(
            "weather.current_batch count=%d source=%s cache_hit=%s",
            len(locations), self.provider.name, hit,
        )

        enriched = []
        for loc, weather in zip(locations, results, strict=True):
            weather = weather.model_copy(
                update={
                    "served_from_cache": hit,
                    "location_id": loc.id,
                    "location_name": loc.name,
                    "region": loc.region,
                }
            )
            enriched.append(weather)
            await run_in_threadpool(
                weather_repository.save_observation, self.db, loc.id, weather
            )
        return enriched

    async def get_hourly(
        self, location: Location | None, lat: float, lon: float, hours: int
    ) -> HourlyForecast:
        key = _coord_key("hourly", lat, lon, hours)
        async with _translate_provider_errors():
            result, hit = await _hourly_cache.get_or_fetch(
                key, lambda: self.provider.get_hourly(lat, lon, hours)
            )
        result = result.model_copy(update={"served_from_cache": hit})
        logger.info(
            "weather.hourly lat=%s lon=%s hours=%d source=%s cache_hit=%s",
            lat, lon, hours, self.provider.name, hit,
        )

        if location is not None:
            result = result.model_copy(update={"location_id": location.id})
            await run_in_threadpool(
                weather_repository.save_hourly_forecast, self.db, location.id, result
            )
        return result

    async def get_daily(
        self, location: Location | None, lat: float, lon: float, days: int
    ) -> DailyForecast:
        key = _coord_key("daily", lat, lon, days)
        async with _translate_provider_errors():
            result, hit = await _daily_cache.get_or_fetch(
                key, lambda: self.provider.get_daily(lat, lon, days)
            )
        result = result.model_copy(update={"served_from_cache": hit})
        logger.info(
            "weather.daily lat=%s lon=%s days=%d source=%s cache_hit=%s",
            lat, lon, days, self.provider.name, hit,
        )

        if location is not None:
            result = result.model_copy(update={"location_id": location.id})
            await run_in_threadpool(
                weather_repository.save_daily_forecast, self.db, location.id, result
            )
        return result

    async def get_feature_history(self, lat: float, lon: float, hours_back: int) -> list[dict]:
        """Real recent hourly weather for the Phase 4 prediction pipeline's
        lag features — see `OpenMeteoProvider.get_feature_history`. Not
        persisted (it's wide, ephemeral model input, not a weather reading
        the rest of the app needs), only cached in-process like every other
        provider call.
        """
        key = _coord_key("feature_history", lat, lon, hours_back)
        async with _translate_provider_errors():
            result, hit = await _feature_history_cache.get_or_fetch(
                key, lambda: self.provider.get_feature_history(lat, lon, hours_back)
            )
        logger.info(
            "weather.feature_history lat=%s lon=%s hours_back=%d source=%s cache_hit=%s",
            lat, lon, hours_back, self.provider.name, hit,
        )
        return result
