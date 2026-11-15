"""IMD (India Meteorological Department) provider — NOT implemented.

Research done for AtmoGuard Phase 2 (2026-11):

- IMD runs a documented API portal at https://api.imd.gov.in/ with a public
  reference page (https://api.imd.gov.in/public/api_reference.html) listing
  real endpoints: city forecasts (`/api/v1/cityforecast`), current weather
  (`/api/v1/current_wx`), district/station nowcasts, district/subdivision
  *warnings* (`/api/v1/districtwarning`, `/api/v1/subdivisionwarning`),
  AWS/ARG station data, and cyclone tracks — i.e. exactly the official
  warnings data this project would eventually want.
- Empirically confirmed (direct HTTP requests, this session): every one of
  those endpoints returns `HTTP 401 {"error": "API key missing"}`. There is
  no self-serve signup — IMD's own documentation says access requires
  registering with IMD and going through their nodal officer, i.e. it is
  **documented but not accessible** without an approval process this project
  cannot complete on its own.
- Separately, IMD closed public access to its AWS/ARG station-data portal in
  2025 over commercial-use concerns (reported by Down To Earth), reinforcing
  that this is a gated, not an open, data source right now.

Per the Phase 2 instructions — "only implement an actual IMD integration if
an accessible, documented data source is available" and "do not invent
official warnings" — this file is a stub, not a working provider. It exists
so the *shape* of an IMD integration is already decided (implement
WeatherProvider, same as OpenMeteoProvider) and nothing else needs to change
when/if a real API key is obtained: fill in the three methods below, wire an
instance into `app/services/weather_service.py`'s provider selection, done.
"""

from app.core.config import get_settings
from app.schemas.weather import CurrentWeather, DailyForecast, HourlyForecast
from app.services.providers.base import ProviderUnavailableError, WeatherProvider


class IMDProvider(WeatherProvider):
    name = "imd"

    def __init__(self) -> None:
        self._api_key = get_settings().IMD_API_KEY

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _unavailable(self) -> ProviderUnavailableError:
        return ProviderUnavailableError(
            "IMD provider is not configured — IMD's API "
            "(https://api.imd.gov.in/) requires an approved API key that "
            "this project does not have. Set IMD_API_KEY once one is "
            "obtained; see this module's docstring for what was checked."
        )

    async def get_current(self, lat: float, lon: float) -> CurrentWeather:
        raise self._unavailable()

    async def get_hourly(self, lat: float, lon: float, hours: int) -> HourlyForecast:
        raise self._unavailable()

    async def get_daily(self, lat: float, lon: float, days: int) -> DailyForecast:
        raise self._unavailable()
