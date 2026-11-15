"""Normalized weather schemas — the internal contract every provider
(Open-Meteo today, IMD later) must return, and what the API serializes.

Wire format is camelCase (via `alias_generator`) to match the frontend's
existing internal object shape (src/services/weatherService.js's
`normalizeCurrent()` etc.) field-for-field, so the frontend adapter added in
Phase 2 is a near-passthrough rather than a reshaping layer.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class WeatherLocation(CamelModel):
    """Mirrors src/data/locations.js exactly."""

    id: str
    name: str
    region: str
    latitude: float
    longitude: float
    is_primary: bool = False
    saved_as: str | None = None


class CurrentWeather(CamelModel):
    location_id: str | None = None
    location_name: str | None = None
    region: str | None = None
    latitude: float
    longitude: float

    temperature: float | None = None
    feels_like: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    wind_direction: str = "--"
    # Raw degrees, preserved alongside the compass label above — provider
    # metadata we'd otherwise lose (the frontend contract only wants the
    # compass string, but storage/analytics may want the precise angle).
    wind_direction_degrees: float | None = None
    pressure: float | None = None
    visibility: float | None = None
    rainfall: float | None = None
    rain_probability: float | None = None
    condition: str = "Humid & Hazy"
    uv_index: float | None = None
    # Raw WMO weather code backing `condition` — Phase 3's thunderstorm/
    # lightning-potential hazard modules classify on this directly (95/96/99
    # = thunderstorm codes) rather than re-parsing the text label.
    weather_code: int | None = None
    # Convective Available Potential Energy (J/kg) and Lifted Index (°C) —
    # standard meteorological instability indicators, not available on
    # Open-Meteo's `current` block so sourced from the nearest hourly point
    # (see OpenMeteoProvider._normalize_current). Used by the thunderstorm/
    # lightning hazard modules (app/services/risk_engine/) — see
    # docs/architecture.md §7 for the documented threshold bands.
    cape: float | None = None
    lifted_index: float | None = None

    # The provider's own observation timestamp — NOT when we served this
    # response. Stays correct even when `served_from_cache` is true, so a
    # cache hit never silently hides how old the underlying reading is.
    observed_at: datetime
    # When THIS response was produced (now, for a fresh fetch; the original
    # fetch time, for a cache hit).
    retrieved_at: datetime
    served_from_cache: bool = False

    source: str
    units: dict[str, str] = {
        "temperature": "celsius",
        "windSpeed": "km/h",
        "pressure": "hPa",
        "visibility": "km",
        "rainfall": "mm",
    }


class HourlyForecastPoint(CamelModel):
    time: datetime
    hour: str  # "HH:00" label, for direct chart-axis compatibility
    temperature: float | None = None
    humidity: float | None = None
    rainfall: float | None = None
    rain_probability: float | None = None
    wind_speed: float | None = None
    visibility: float | None = None
    uv_index: float | None = None


class HourlyForecast(CamelModel):
    location_id: str | None = None
    latitude: float
    longitude: float
    source: str
    generated_at: datetime
    retrieved_at: datetime
    served_from_cache: bool = False
    points: list[HourlyForecastPoint]


class DailyForecastPoint(CamelModel):
    date: datetime
    day: str  # "Mon" label, for direct chart-axis compatibility
    temperature_high: float | None = None
    temperature_low: float | None = None
    rainfall: float | None = None
    rain_probability: float | None = None
    condition: str = "Humid & Hazy"


class DailyForecast(CamelModel):
    location_id: str | None = None
    latitude: float
    longitude: float
    source: str
    generated_at: datetime
    retrieved_at: datetime
    served_from_cache: bool = False
    points: list[DailyForecastPoint]
