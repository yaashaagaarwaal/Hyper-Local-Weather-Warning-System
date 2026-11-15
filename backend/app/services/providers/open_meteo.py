"""Open-Meteo provider — the only concrete, working WeatherProvider today.

Async (httpx.AsyncClient), with timeouts, retry-with-backoff, and structured
(key=value) logging around every outbound call. All normalization (WMO code
→ text, degrees → compass, unit bookkeeping) happens here so every other
layer only ever sees the shared `CurrentWeather` / `HourlyForecast` /
`DailyForecast` schemas — never a raw Open-Meteo response.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

import httpx

from app.core.config import get_settings
from app.schemas.weather import (
    CurrentWeather,
    DailyForecast,
    DailyForecastPoint,
    HourlyForecast,
    HourlyForecastPoint,
)
from app.services.providers.base import WeatherProvider
from app.utils.weather_codes import degrees_to_compass, describe_weather_code

logger = logging.getLogger(__name__)

_CURRENT_FIELDS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,"
    "rain,weather_code,pressure_msl,wind_speed_10m,wind_direction_10m,is_day"
)
_HOURLY_FIELDS = (
    "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,"
    "weather_code,visibility,uv_index,wind_speed_10m,cape,lifted_index"
)
_DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,precipitation_probability_max"
)

# Phase 4: the full set of hourly variables the ML prediction pipeline's
# lag features need, mapped to the same column names used in
# dataset/delhi_ncr_labeled_weather.csv (and backend/ml/config.py's
# BASE_WEATHER_COLUMNS) so a value fetched live and a value read from the
# training CSV mean exactly the same thing. Verified live against the
# Open-Meteo API (see docs/model_report.md "Serving-time feature parity") --
# every one of these variables, including the three soil-moisture depth
# bins, is returned by the /v1/forecast endpoint with no special access.
# Kept as its own field list (not merged into _HOURLY_FIELDS) because it's
# only needed by the prediction endpoints, not by every /weather/* call.
_ML_FEATURE_FIELD_MAP = {
    "temperature_c": "temperature_2m",
    "apparent_temperature_c": "apparent_temperature",
    "relative_humidity_pct": "relative_humidity_2m",
    "dew_point_c": "dew_point_2m",
    "precipitation_mm_hour": "precipitation",
    "rainfall_mm_hour": "rain",
    "showers_mm_hour": "showers",
    "weather_code": "weather_code",
    "wind_speed_kmh": "wind_speed_10m",
    "wind_gust_kmh": "wind_gusts_10m",
    "wind_direction_deg": "wind_direction_10m",
    "visibility_m": "visibility",
    "pressure_hpa": "pressure_msl",
    "surface_pressure_hpa": "surface_pressure",
    "cloud_cover_pct": "cloud_cover",
    "soil_moisture_0_7cm": "soil_moisture_0_to_7cm",
    "soil_moisture_7_28cm": "soil_moisture_7_to_28cm",
    "soil_moisture_28_100cm": "soil_moisture_28_to_100cm",
    "cape_j_kg": "cape",
    "lifted_index": "lifted_index",
    "wet_bulb_temperature_c": "wet_bulb_temperature_2m",
}


class OpenMeteoError(Exception):
    pass


class OpenMeteoProvider(WeatherProvider):
    name = "open-meteo"

    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        settings = get_settings()
        self._base_url = settings.OPEN_METEO_BASE_URL
        self._timeout = settings.HTTP_TIMEOUT_SECONDS
        self._retry_attempts = settings.HTTP_RETRY_ATTEMPTS
        self._backoff = settings.HTTP_RETRY_BACKOFF_SECONDS

    def _build_params(
        self, latitude: float | list[float], longitude: float | list[float], *, light: bool = False
    ) -> dict:
        # Open-Meteo wants comma-joined coordinates for a batched request
        # (e.g. "28.1,28.2"), not repeated query params — join explicitly
        # rather than letting httpx serialize a list its own way.
        lat_str = ",".join(str(v) for v in latitude) if isinstance(latitude, list) else str(latitude)
        lon_str = ",".join(str(v) for v in longitude) if isinstance(longitude, list) else str(longitude)
        params = {
            "latitude": lat_str,
            "longitude": lon_str,
            "current": _CURRENT_FIELDS,
            "timezone": "auto",
        }
        if not light:
            params["hourly"] = _HOURLY_FIELDS
            params["daily"] = _DAILY_FIELDS
            params["past_hours"] = 6
            params["forecast_days"] = 8
        return params

    async def _get_json(self, params: dict) -> dict | list:
        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts + 1):
            start = time.monotonic()
            try:
                response = await self._client.get(
                    self._base_url, params=params, timeout=self._timeout
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("error"):
                    raise OpenMeteoError(data.get("reason", "Open-Meteo returned an error"))
                elapsed_ms = round((time.monotonic() - start) * 1000)
                logger.info(
                    "open_meteo.request_ok attempt=%d latency_ms=%d url=%s",
                    attempt, elapsed_ms, response.url,
                )
                return data
            except Exception as exc:  # noqa: BLE001 — any failure triggers the same retry path
                last_exc = exc
                elapsed_ms = round((time.monotonic() - start) * 1000)
                logger.warning(
                    "open_meteo.request_failed attempt=%d latency_ms=%d error=%s",
                    attempt, elapsed_ms, exc,
                )
                if attempt < self._retry_attempts:
                    await asyncio.sleep(self._backoff * (attempt + 1))

        logger.error("open_meteo.request_exhausted error=%s", last_exc)
        raise OpenMeteoError(
            f"Unable to reach Open-Meteo after {self._retry_attempts + 1} attempt(s): {last_exc}"
        ) from last_exc

    @staticmethod
    def _nearest_hourly_index(hourly: dict, iso_time: str | None) -> int:
        times = hourly.get("time") or []
        if not times or not iso_time:
            return 0
        target = iso_time[:13]
        for idx, t in enumerate(times):
            if t[:13] == target:
                return idx
        return 0

    def _normalize_current(self, payload: dict, retrieved_at: datetime) -> CurrentWeather:
        current = payload.get("current") or {}
        hourly = payload.get("hourly") or {}
        idx = self._nearest_hourly_index(hourly, current.get("time"))

        def _at(key: str):
            values = hourly.get(key) or []
            return values[idx] if idx < len(values) else None

        visibility_m = _at("visibility")
        observed_at = (
            datetime.fromisoformat(current["time"]).replace(tzinfo=timezone.utc)
            if current.get("time")
            else retrieved_at
        )

        return CurrentWeather(
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude"),
            temperature=current.get("temperature_2m"),
            feels_like=current.get("apparent_temperature", current.get("temperature_2m")),
            humidity=current.get("relative_humidity_2m"),
            wind_speed=current.get("wind_speed_10m"),
            wind_direction=degrees_to_compass(current.get("wind_direction_10m")),
            wind_direction_degrees=current.get("wind_direction_10m"),
            pressure=round(current["pressure_msl"]) if current.get("pressure_msl") is not None else None,
            visibility=round(visibility_m / 1000, 1) if visibility_m is not None else None,
            rainfall=current.get("precipitation", current.get("rain", 0)) or 0,
            rain_probability=_at("precipitation_probability"),
            condition=describe_weather_code(current.get("weather_code")),
            uv_index=round(_at("uv_index")) if _at("uv_index") is not None else None,
            weather_code=current.get("weather_code"),
            cape=_at("cape"),
            lifted_index=_at("lifted_index"),
            observed_at=observed_at,
            retrieved_at=retrieved_at,
            source=self.name,
        )

    async def get_current(self, lat: float, lon: float) -> CurrentWeather:
        payload = await self._get_json(self._build_params(lat, lon))
        return self._normalize_current(payload, datetime.now(timezone.utc))

    async def get_current_batch(self, coords: list[tuple[float, float]]) -> list[CurrentWeather]:
        if not coords:
            return []
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        payload = await self._get_json(self._build_params(lats, lons, light=True))
        results = payload if isinstance(payload, list) else [payload]
        now = datetime.now(timezone.utc)
        return [self._normalize_current(entry, now) for entry in results]

    async def get_hourly(self, lat: float, lon: float, hours: int) -> HourlyForecast:
        payload = await self._get_json(self._build_params(lat, lon))
        hourly = payload.get("hourly") or {}
        current = payload.get("current") or {}
        times = hourly.get("time") or []
        now = datetime.now(timezone.utc)

        if not times:
            return HourlyForecast(
                latitude=payload.get("latitude"), longitude=payload.get("longitude"),
                source=self.name, generated_at=now, retrieved_at=now, points=[],
            )

        start_idx = self._nearest_hourly_index(hourly, current.get("time"))
        points = []
        for i, t in enumerate(times[start_idx : start_idx + hours]):
            idx = start_idx + i
            dt = datetime.fromisoformat(t)
            points.append(
                HourlyForecastPoint(
                    time=dt,
                    hour=f"{dt.hour:02d}:00",
                    temperature=hourly["temperature_2m"][idx],
                    humidity=hourly["relative_humidity_2m"][idx],
                    rainfall=hourly["precipitation"][idx],
                    rain_probability=hourly.get("precipitation_probability", [None] * len(times))[idx],
                    wind_speed=hourly.get("wind_speed_10m", [None] * len(times))[idx],
                    visibility=(
                        round(hourly["visibility"][idx] / 1000, 1)
                        if hourly.get("visibility") and hourly["visibility"][idx] is not None
                        else None
                    ),
                    uv_index=hourly.get("uv_index", [None] * len(times))[idx],
                )
            )

        return HourlyForecast(
            latitude=payload.get("latitude"), longitude=payload.get("longitude"),
            source=self.name, generated_at=now, retrieved_at=now, points=points,
        )

    async def get_daily(self, lat: float, lon: float, days: int) -> DailyForecast:
        payload = await self._get_json(self._build_params(lat, lon))
        daily = payload.get("daily") or {}
        times = daily.get("time") or []
        now = datetime.now(timezone.utc)

        if not times:
            return DailyForecast(
                latitude=payload.get("latitude"), longitude=payload.get("longitude"),
                source=self.name, generated_at=now, retrieved_at=now, points=[],
            )

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        points = []
        for idx, date_str in enumerate(times[:days]):
            dt = datetime.fromisoformat(date_str)
            rain_prob = daily.get("precipitation_probability_max", [None] * len(times))[idx]
            points.append(
                DailyForecastPoint(
                    date=dt,
                    day=day_names[dt.weekday()],
                    temperature_high=daily["temperature_2m_max"][idx],
                    temperature_low=daily["temperature_2m_min"][idx],
                    rainfall=daily["precipitation_sum"][idx],
                    rain_probability=rain_prob,
                    condition=describe_weather_code(daily["weather_code"][idx]),
                )
            )

        return DailyForecast(
            latitude=payload.get("latitude"), longitude=payload.get("longitude"),
            source=self.name, generated_at=now, retrieved_at=now, points=points,
        )

    async def get_feature_history(self, lat: float, lon: float, hours_back: int) -> list[dict]:
        """Real hourly weather for the last `hours_back` hours up to now, one
        dict per hour with keys matching the ML pipeline's BASE_WEATHER_COLUMNS
        names (plus `"time"`). Used only by the Phase 4 prediction service to
        build lag features -- never persisted, never shown to the frontend
        directly. Explicitly drops any entry whose timestamp is in the future
        relative to `retrieved_at`: Open-Meteo's `hourly` block always also
        contains forecast hours, and using one of those as if it were a past
        observation would quietly turn a real forecast into a fabricated
        "measurement", defeating the whole point of lag features.
        """
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "hourly": ",".join(_ML_FEATURE_FIELD_MAP.values()),
            "timezone": "auto",
            "past_hours": max(hours_back, 1),
            "forecast_days": 1,
        }
        payload = await self._get_json(params)
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        retrieved_at = datetime.now(timezone.utc)

        rows = []
        for idx, iso_time in enumerate(times):
            dt = datetime.fromisoformat(iso_time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > retrieved_at:
                continue
            row = {"time": dt}
            for dataset_name, om_name in _ML_FEATURE_FIELD_MAP.items():
                values = hourly.get(om_name) or []
                row[dataset_name] = values[idx] if idx < len(values) else None
            rows.append(row)

        return rows[-hours_back:] if hours_back > 0 else rows
