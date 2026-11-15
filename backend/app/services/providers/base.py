"""Weather provider abstraction.

`WeatherService` (app/services/weather_service.py) only ever talks to this
interface — never to a concrete provider's HTTP client directly. That's what
lets a second provider (e.g. IMD, once its API is actually reachable — see
imd.py) be added later without changing the service, the repositories, or any
API endpoint: implement this interface, swap/add it where `WeatherService` is
constructed, done.
"""

import asyncio
from abc import ABC, abstractmethod

from app.schemas.weather import CurrentWeather, DailyForecast, HourlyForecast


class ProviderUnavailableError(Exception):
    """Raised when a provider cannot serve a request at all (e.g. not
    configured, like IMD today) as opposed to a transient network failure.
    """


class WeatherProvider(ABC):
    name: str

    @abstractmethod
    async def get_current(self, lat: float, lon: float) -> CurrentWeather: ...

    @abstractmethod
    async def get_hourly(self, lat: float, lon: float, hours: int) -> HourlyForecast: ...

    @abstractmethod
    async def get_daily(self, lat: float, lon: float, days: int) -> DailyForecast: ...

    async def get_current_batch(self, coords: list[tuple[float, float]]) -> list[CurrentWeather]:
        """Default implementation: fan out individual requests concurrently.

        A provider whose API supports a real batched call (Open-Meteo does —
        comma-separated coordinates in one request) should override this for
        efficiency; providers that don't get correct-but-unbatched behavior
        for free.
        """
        return await asyncio.gather(*(self.get_current(lat, lon) for lat, lon in coords))

    async def get_feature_history(self, lat: float, lon: float, hours_back: int) -> list[dict]:
        """Phase 4: real recent hourly weather (one dict per hour) for the ML
        prediction pipeline's lag features. Not every provider can supply
        this (it needs a documented, stable variable catalog) — the default
        raises `ProviderUnavailableError`, same as an unconfigured provider,
        rather than silently returning nothing. Only `OpenMeteoProvider`
        overrides it today.
        """
        raise ProviderUnavailableError(f"{self.name} does not support ML feature history")
