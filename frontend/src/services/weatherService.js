// Centralized weather client.
//
// This is the ONLY file that makes real network calls for weather data.
// Every app-facing function (fetchCurrentWeather, fetchHourlyTrend, …) tries
// the FastAPI backend FIRST (api/v1/weather/*), with a short timeout, and
// falls back to calling Open-Meteo directly — unchanged from before the
// backend existed — the instant the backend is unreachable, errors, or times
// out. The two paths are never blended within one response: a given call
// either fully resolves via the backend or fully resolves via the direct
// path, and the returned object's `source` field says which (`"open-meteo"`
// for direct, `"open-meteo-backend"` when the FastAPI backend served it) so
// it's always possible to tell which one actually answered.
import { LOCATIONS } from "../data/locations";
import { deriveRiskScore } from "../utils/risk";
import { describeWeatherCode, degreesToCompass } from "../utils/weatherCodes";
import { BACKEND_ENABLED, fetchBackendJson, warnFallback } from "./apiClient";

export const WEATHER_SOURCE = "Open-Meteo";
const BACKEND_SOURCE = "open-meteo-backend";

// ---- Backend response -> frontend shape adapters ----
// (The backend's field names already match these almost exactly — camelCase
// aliases chosen specifically for this — these adapters only rename the
// handful of fields that differ and compute the client-side-only riskScore.)

function adaptBackendCurrent(data) {
  return {
    locationId: data.locationId ?? null,
    locationName: data.locationName ?? null,
    region: data.region ?? null,
    temperature: data.temperature,
    feelsLike: data.feelsLike,
    humidity: data.humidity,
    windSpeed: data.windSpeed,
    windDirection: data.windDirection,
    pressure: data.pressure,
    visibility: data.visibility,
    rainfall: data.rainfall,
    rainProbability: data.rainProbability,
    condition: data.condition,
    uvIndex: data.uvIndex,
    // True provider observation time (not fetch time) — preserved as
    // `lastUpdated` so existing components (e.g. CurrentWeatherPanel) don't
    // need to change, and now read the real observation timestamp instead
    // of "whenever we happened to fetch this."
    lastUpdated: data.observedAt,
    source: BACKEND_SOURCE,
  };
}

function adaptBackendHourly(data) {
  return (data.points || []).map((p) => ({
    hour: p.hour,
    label: p.hour,
    temperature: p.temperature,
    rainfall: p.rainfall ?? 0,
    humidity: p.humidity,
    riskScore: deriveRiskScore(p.rainfall ?? 0, p.humidity ?? 0),
  }));
}

function adaptBackendDaily(data) {
  return (data.points || []).map((p) => ({
    day: p.day,
    high: Math.round(p.temperatureHigh),
    low: Math.round(p.temperatureLow),
    rainfall: p.rainfall ?? 0,
    riskScore: Math.min(97, Math.round((p.rainfall ?? 0) * 1.4 + (p.rainProbability ?? 0) * 0.4)),
    condition: p.condition,
  }));
}

function adaptBackendLocation(loc) {
  return {
    id: loc.id,
    name: loc.name,
    region: loc.region,
    lat: loc.latitude,
    lng: loc.longitude,
    isPrimary: loc.isPrimary,
    savedAs: loc.savedAs ?? undefined,
  };
}

const FORECAST_URL = "https://api.open-meteo.com/v1/forecast";
const CACHE_TTL_MS = 5 * 60 * 1000; // avoid re-fetching the same location too often
const RETRY_ATTEMPTS = 2;
const RETRY_DELAY_MS = 800;
const REQUEST_TIMEOUT_MS = 12000;

const CURRENT_FIELDS = [
  "temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation",
  "rain", "weather_code", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "is_day",
].join(",");

const HOURLY_FIELDS = [
  "temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation",
  "weather_code", "visibility", "uv_index",
].join(",");

const DAILY_FIELDS = [
  "weather_code", "temperature_2m_max", "temperature_2m_min",
  "precipitation_sum", "precipitation_probability_max",
].join(",");

// Request de-dupe + short-lived cache, keyed by rounded coordinate(s).
// Concurrent callers for the same location share one in-flight request, and
// a resolved response is reused for CACHE_TTL_MS instead of re-fetching.
const requestCache = new Map();

function roundCoord(n) {
  return Number(n).toFixed(3);
}

function cacheKeyFor(latitude, longitude) {
  return Array.isArray(latitude)
    ? `multi:${latitude.map(roundCoord).join("_")}|${longitude.map(roundCoord).join("_")}`
    : `${roundCoord(latitude)},${roundCoord(longitude)}`;
}

function buildForecastUrl(latitude, longitude, { light = false } = {}) {
  const params = new URLSearchParams({
    latitude: Array.isArray(latitude) ? latitude.join(",") : String(latitude),
    longitude: Array.isArray(longitude) ? longitude.join(",") : String(longitude),
    current: CURRENT_FIELDS,
    timezone: "auto",
  });
  if (!light) {
    params.set("hourly", HOURLY_FIELDS);
    params.set("daily", DAILY_FIELDS);
    params.set("past_hours", "6");
    params.set("forecast_days", "8");
  }
  return `${FORECAST_URL}?${params.toString()}`;
}

async function fetchJsonWithRetry(url) {
  let lastError;
  for (let attempt = 0; attempt <= RETRY_ATTEMPTS; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (!response.ok) throw new Error(`Weather service responded with ${response.status}`);
      const json = await response.json();
      if (json?.error) throw new Error(json.reason || "Weather service returned an error");
      return json;
    } catch (err) {
      clearTimeout(timeoutId);
      lastError = err;
      if (attempt < RETRY_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS * (attempt + 1)));
      }
    }
  }
  throw new Error(lastError?.message || "Unable to reach the weather service. Check your connection and retry.");
}

/**
 * Raw Open-Meteo bundle (current + hourly + daily) for one location.
 * Cached for CACHE_TTL_MS and de-duplicated so concurrent callers share one
 * network request instead of firing one each.
 */
export function getWeatherData(latitude, longitude) {
  const key = cacheKeyFor(latitude, longitude);
  const entry = requestCache.get(key);
  const now = Date.now();

  if (entry?.pending) return entry.pending;
  if (entry && now - entry.timestamp < CACHE_TTL_MS) return Promise.resolve(entry.data);

  const pending = fetchJsonWithRetry(buildForecastUrl(latitude, longitude))
    .then((data) => {
      requestCache.set(key, { data, timestamp: Date.now() });
      return data;
    })
    .catch((err) => {
      requestCache.delete(key);
      throw err;
    });

  requestCache.set(key, { pending });
  return pending;
}

function nearestHourlyIndex(hourly, isoTime) {
  if (!hourly?.time?.length || !isoTime) return 0;
  const target = isoTime.slice(0, 13);
  const idx = hourly.time.findIndex((t) => t.slice(0, 13) === target);
  return idx === -1 ? 0 : idx;
}

function normalizeCurrent(json) {
  const { current, hourly } = json;
  const idx = nearestHourlyIndex(hourly, current?.time);
  const visibilityMeters = hourly?.visibility?.[idx];
  const uvIndex = hourly?.uv_index?.[idx];
  const rainProbability = hourly?.precipitation_probability?.[idx];

  return {
    temperature: current?.temperature_2m ?? null,
    feelsLike: current?.apparent_temperature ?? current?.temperature_2m ?? null,
    humidity: current?.relative_humidity_2m ?? null,
    windSpeed: current?.wind_speed_10m ?? null,
    windDirection: degreesToCompass(current?.wind_direction_10m),
    pressure: current?.pressure_msl != null ? Math.round(current.pressure_msl) : null,
    visibility: visibilityMeters != null ? Number((visibilityMeters / 1000).toFixed(1)) : null,
    rainfall: current?.precipitation ?? current?.rain ?? 0,
    rainProbability: rainProbability ?? null,
    condition: describeWeatherCode(current?.weather_code),
    uvIndex: uvIndex != null ? Math.round(uvIndex) : null,
    lastUpdated: new Date().toISOString(),
    source: WEATHER_SOURCE,
  };
}

/** Normalized current weather for one coordinate pair. REAL data. */
export async function getCurrentWeather(latitude, longitude) {
  const json = await getWeatherData(latitude, longitude);
  return normalizeCurrent(json);
}

/** Next ~24 hours of real hourly forecast, shaped for the trend charts. */
export async function getHourlyForecast(latitude, longitude) {
  const json = await getWeatherData(latitude, longitude);
  const { hourly, current } = json;
  if (!hourly?.time?.length) return [];
  const startIdx = nearestHourlyIndex(hourly, current?.time);
  return hourly.time.slice(startIdx, startIdx + 24).map((time, i) => {
    const idx = startIdx + i;
    const date = new Date(time);
    const label = `${String(date.getHours()).padStart(2, "0")}:00`;
    const rainfall = hourly.precipitation[idx] ?? 0;
    const humidity = hourly.relative_humidity_2m[idx];
    return {
      hour: label,
      label,
      temperature: hourly.temperature_2m[idx],
      rainfall,
      humidity,
      riskScore: deriveRiskScore(rainfall, humidity),
    };
  });
}

/**
 * Trailing "recent observations" for the 1h / 6h analytics filters, sliced
 * from the real hourly series' `past_hours` window.
 */
export async function getRecentSeries(latitude, longitude, rangeKey) {
  const json = await getWeatherData(latitude, longitude);
  const { hourly, current } = json;
  if (!hourly?.time?.length) return [];
  const nowIdx = nearestHourlyIndex(hourly, current?.time);
  const hoursBack = rangeKey === "1h" ? 1 : 6;
  const startIdx = Math.max(0, nowIdx - hoursBack);
  return hourly.time.slice(startIdx, nowIdx + 1).map((time, i) => {
    const idx = startIdx + i;
    const minutesAgo = (nowIdx - idx) * 60;
    const rainfall = hourly.precipitation[idx] ?? 0;
    const humidity = hourly.relative_humidity_2m[idx];
    return {
      label: minutesAgo === 0 ? "Now" : `-${minutesAgo}m`,
      temperature: hourly.temperature_2m[idx],
      rainfall,
      riskScore: deriveRiskScore(rainfall, humidity),
    };
  });
}

/** Real 7-day outlook, shaped for the weekly bar chart & predictions page. */
export async function getDailyForecast(latitude, longitude) {
  const json = await getWeatherData(latitude, longitude);
  const { daily } = json;
  if (!daily?.time?.length) return [];
  const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  return daily.time.slice(0, 7).map((date, idx) => {
    const rainfall = Number((daily.precipitation_sum[idx] ?? 0).toFixed(1));
    const rainProbability = daily.precipitation_probability_max[idx] ?? 0;
    return {
      day: dayNames[new Date(date).getDay()],
      high: Math.round(daily.temperature_2m_max[idx]),
      low: Math.round(daily.temperature_2m_min[idx]),
      rainfall,
      riskScore: Math.min(97, Math.round(rainfall * 1.4 + rainProbability * 0.4)),
      condition: describeWeatherCode(daily.weather_code[idx]),
    };
  });
}

/** Full real weather bundle for a known location object (adds id/name/region). */
export async function getWeatherByLocation(location) {
  const current = await getCurrentWeather(location.lat, location.lng);
  return { ...current, locationId: location.id, locationName: location.name, region: location.region };
}

function findLocation(locationId) {
  return LOCATIONS.find((l) => l.id === locationId) ?? LOCATIONS[0];
}

// ---- App-facing functions (what pages/components actually call) ----
// Each tries the FastAPI backend first (when VITE_USE_BACKEND !== "false"),
// then falls back to calling Open-Meteo directly — unmodified from before
// the backend existed — on any backend failure. See the module docstring at
// the top of this file for the "never blend the two" rule.

// GET /api/v1/weather/current?locationId=...
export async function fetchCurrentWeather(locationId) {
  if (BACKEND_ENABLED) {
    try {
      const data = await fetchBackendJson("/weather/current", { locationId });
      return adaptBackendCurrent(data);
    } catch (err) {
      warnFallback("current weather", err);
    }
  }
  return getWeatherByLocation(findLocation(locationId));
}

// GET /api/v1/weather/hourly?locationId=...
export async function fetchHourlyTrend(locationId) {
  if (BACKEND_ENABLED) {
    try {
      const data = await fetchBackendJson("/weather/hourly", { locationId, hours: 24 });
      return adaptBackendHourly(data);
    } catch (err) {
      warnFallback("hourly trend", err);
    }
  }
  const location = findLocation(locationId);
  return getHourlyForecast(location.lat, location.lng);
}

// GET /api/v1/weather/daily?locationId=...
export async function fetchWeeklyOutlook(locationId) {
  if (BACKEND_ENABLED) {
    try {
      const data = await fetchBackendJson("/weather/daily", { locationId, days: 7 });
      return adaptBackendDaily(data);
    } catch (err) {
      warnFallback("weekly outlook", err);
    }
  }
  const location = findLocation(locationId);
  return getDailyForecast(location.lat, location.lng);
}

// GET /analytics/series?locationId=...&range=...
// The backend has no dedicated analytics/short-range endpoint yet (Phase 2
// only ships current/hourly/daily/locations) — 1h/6h always use the direct
// path; 24h/7d reuse the hourly/daily backend-or-fallback functions above
// (still "never blended": each call is fully backend or fully direct).
export async function fetchAnalyticsSeries(locationId, rangeKey) {
  const location = findLocation(locationId);
  if (rangeKey === "1h" || rangeKey === "6h") {
    return getRecentSeries(location.lat, location.lng, rangeKey);
  }
  if (rangeKey === "7d") {
    const days = await fetchWeeklyOutlook(locationId);
    return days.map((d) => ({
      label: d.day,
      temperature: Math.round((d.high + d.low) / 2),
      rainfall: d.rainfall,
      riskScore: d.riskScore,
    }));
  }
  const hours = await fetchHourlyTrend(locationId);
  return hours.map((h) => ({ label: h.label, temperature: h.temperature, rainfall: h.rainfall, riskScore: h.riskScore }));
}

const ALL_CURRENT_CACHE_KEY = "all-current";

// Direct-path implementation (unchanged from before the backend existed) —
// current weather for every monitored location in a single batched
// Open-Meteo request (comma-separated lat/lng), keyed by location id.
function fetchAllCurrentWeatherDirect() {
  const entry = requestCache.get(ALL_CURRENT_CACHE_KEY);
  const now = Date.now();
  if (entry?.pending) return entry.pending;
  if (entry && now - entry.timestamp < CACHE_TTL_MS) return Promise.resolve(entry.data);

  const lats = LOCATIONS.map((l) => l.lat);
  const lngs = LOCATIONS.map((l) => l.lng);
  const url = buildForecastUrl(lats, lngs, { light: true });

  const pending = fetchJsonWithRetry(url)
    .then((results) => {
      const list = Array.isArray(results) ? results : [results];
      const byId = {};
      LOCATIONS.forEach((loc, idx) => {
        const json = list[idx];
        byId[loc.id] = json?.current
          ? { ...normalizeCurrent(json), locationId: loc.id, locationName: loc.name, region: loc.region }
          : null;
      });
      requestCache.set(ALL_CURRENT_CACHE_KEY, { data: byId, timestamp: Date.now() });
      return byId;
    })
    .catch((err) => {
      requestCache.delete(ALL_CURRENT_CACHE_KEY);
      throw err;
    });

  requestCache.set(ALL_CURRENT_CACHE_KEY, { pending });
  return pending;
}

// GET /api/v1/weather/current/batch?locationIds=... — current weather for
// every monitored location, batched into one request either way.
export async function fetchAllCurrentWeather() {
  if (BACKEND_ENABLED) {
    try {
      const locationIds = LOCATIONS.map((l) => l.id).join(",");
      const data = await fetchBackendJson("/weather/current/batch", { locationIds });
      const byId = {};
      for (const entry of data) {
        byId[entry.locationId] = adaptBackendCurrent(entry);
      }
      return byId;
    } catch (err) {
      warnFallback("all-locations current weather", err);
    }
  }
  return fetchAllCurrentWeatherDirect();
}

// GET /api/v1/weather/locations
export async function fetchLocations() {
  if (BACKEND_ENABLED) {
    try {
      const data = await fetchBackendJson("/weather/locations");
      return data.map(adaptBackendLocation);
    } catch (err) {
      warnFallback("locations list", err);
    }
  }
  return LOCATIONS;
}
