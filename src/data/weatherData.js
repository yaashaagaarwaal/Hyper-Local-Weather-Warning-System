import { seededRandom, randomInRange, pick } from "../utils/seededRandom";
import { deriveRiskScore } from "../utils/risk";
import { LOCATIONS } from "./locations";

const CONDITIONS = [
  "Partly Cloudy",
  "Overcast",
  "Light Rain",
  "Thunderstorms",
  "Clear Sky",
  "Humid & Hazy",
];

function buildCurrentWeather(location) {
  const rng = seededRandom(`current-${location.id}`);
  const temperature = Number(randomInRange(rng, 24, 39).toFixed(1));
  const humidity = Math.round(randomInRange(rng, 38, 92));
  const windSpeed = Number(randomInRange(rng, 4, 38).toFixed(1));
  const rainfall = Number(randomInRange(rng, 0, 42).toFixed(1));

  return {
    locationId: location.id,
    locationName: location.name,
    region: location.region,
    temperature,
    feelsLike: Number((temperature + randomInRange(rng, -1.5, 4)).toFixed(1)),
    humidity,
    windSpeed,
    windDirection: pick(rng, ["NW", "N", "NE", "E", "SE", "S", "SW", "W"]),
    pressure: Math.round(randomInRange(rng, 998, 1016)),
    visibility: Number(randomInRange(rng, 1.5, 10).toFixed(1)),
    rainfall,
    rainProbability: Math.round(randomInRange(rng, 10, 95)),
    condition: pick(rng, CONDITIONS),
    uvIndex: Math.round(randomInRange(rng, 1, 11)),
    lastUpdated: new Date().toISOString(),
  };
}

export const CURRENT_WEATHER_BY_LOCATION = LOCATIONS.reduce((acc, loc) => {
  acc[loc.id] = buildCurrentWeather(loc);
  return acc;
}, {});

export function getCurrentWeather(locationId) {
  return CURRENT_WEATHER_BY_LOCATION[locationId] ?? CURRENT_WEATHER_BY_LOCATION[LOCATIONS[0].id];
}

// 60-minute short-term nowcast used by the AI Prediction panel.
export function getShortTermForecast(locationId) {
  const base = getCurrentWeather(locationId);
  const rng = seededRandom(`forecast-${locationId}`);
  const steps = [0, 15, 30, 45, 60];
  let rainfall = base.rainfall;
  let risk = Number(randomInRange(rng, 15, 45).toFixed(0));

  return steps.map((minute, idx) => {
    if (idx > 0) {
      rainfall = Math.max(0, rainfall + randomInRange(rng, -3, 9));
      risk = Math.min(98, Math.max(4, risk + randomInRange(rng, -6, 14)));
    }
    return {
      minute,
      label: minute === 0 ? "Now" : `+${minute} min`,
      rainfall: Number(rainfall.toFixed(1)),
      riskScore: Math.round(risk),
      temperature: Number((base.temperature - minute * 0.02 + randomInRange(rng, -0.4, 0.4)).toFixed(1)),
    };
  });
}

// 24-hour hourly trend for the Analytics page.
export function getHourlyTrend(locationId) {
  const rng = seededRandom(`hourly-${locationId}`);
  const base = getCurrentWeather(locationId);
  const hours = [];
  let temp = base.temperature - 6;
  let rain = 0;

  for (let h = 0; h < 24; h++) {
    temp += randomInRange(rng, -1.2, 1.6);
    rain = Math.max(0, rain + randomInRange(rng, -4, 6));
    const rainfall = Number(Math.min(rain, 60).toFixed(1));
    const humidity = Math.round(randomInRange(rng, 35, 95));
    hours.push({
      hour: `${h.toString().padStart(2, "0")}:00`,
      label: `${h.toString().padStart(2, "0")}:00`,
      temperature: Number(temp.toFixed(1)),
      rainfall,
      humidity,
      riskScore: deriveRiskScore(rainfall, humidity),
    });
  }
  return hours;
}

// 7-day outlook for the Analytics / Predictions pages.
export function getWeeklyOutlook(locationId) {
  const rng = seededRandom(`weekly-${locationId}`);
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  return days.map((day) => ({
    day,
    high: Math.round(randomInRange(rng, 30, 41)),
    low: Math.round(randomInRange(rng, 20, 28)),
    rainfall: Number(randomInRange(rng, 0, 55).toFixed(1)),
    riskScore: Math.round(randomInRange(rng, 10, 90)),
    condition: pick(rng, CONDITIONS),
  }));
}

// Recent "observed" points for the Analytics short-range filters (1h / 6h).
// Kept separate from the forward-looking nowcast so past vs. predicted data
// never share a random seed.
function getRecentObservations(locationId, rangeKey) {
  const config = {
    "1h": { points: 12, stepMin: 5 },
    "6h": { points: 24, stepMin: 15 },
  }[rangeKey];
  const rng = seededRandom(`recent-${locationId}-${rangeKey}`);
  const base = getCurrentWeather(locationId);
  let temp = base.temperature - randomInRange(rng, 0.5, 2);
  let rain = Math.max(0, base.rainfall - randomInRange(rng, 2, 8));

  const points = [];
  for (let i = config.points; i >= 0; i--) {
    temp += randomInRange(rng, -0.8, 0.9);
    rain = Math.max(0, rain + randomInRange(rng, -3, 4));
    const minutesAgo = i * config.stepMin;
    const humidity = Math.round(randomInRange(rng, 35, 95));
    const rainfall = Number(Math.min(rain, 60).toFixed(1));
    points.push({
      label: minutesAgo === 0 ? "Now" : `-${minutesAgo}m`,
      temperature: Number(temp.toFixed(1)),
      rainfall,
      riskScore: deriveRiskScore(rainfall, humidity),
    });
  }
  return points;
}

// Unified trend series consumed by the Analytics page's range filter
// (1h / 6h / 24h / 7d), always returning { label, temperature, rainfall, riskScore }.
export function getAnalyticsSeries(locationId, rangeKey) {
  if (rangeKey === "1h" || rangeKey === "6h") {
    return getRecentObservations(locationId, rangeKey);
  }
  if (rangeKey === "7d") {
    return getWeeklyOutlook(locationId).map((d) => ({
      label: d.day,
      temperature: Math.round((d.high + d.low) / 2),
      rainfall: d.rainfall,
      riskScore: d.riskScore,
    }));
  }
  return getHourlyTrend(locationId).map((h) => ({
    label: h.hour,
    temperature: h.temperature,
    rainfall: h.rainfall,
    riskScore: h.riskScore,
  }));
}

export { LOCATIONS };
