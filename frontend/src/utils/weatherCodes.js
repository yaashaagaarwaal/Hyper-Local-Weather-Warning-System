// Open-Meteo reports conditions as WMO weather codes and wind direction in
// degrees. These helpers turn both into the human-readable text the UI uses.
const WEATHER_CODE_LABELS = {
  0: "Clear Sky",
  1: "Mainly Clear",
  2: "Partly Cloudy",
  3: "Overcast",
  45: "Foggy",
  48: "Freezing Fog",
  51: "Light Drizzle",
  53: "Drizzle",
  55: "Dense Drizzle",
  56: "Freezing Drizzle",
  57: "Freezing Drizzle",
  61: "Light Rain",
  63: "Rain",
  65: "Heavy Rain",
  66: "Freezing Rain",
  67: "Freezing Rain",
  71: "Light Snow",
  73: "Snow",
  75: "Heavy Snow",
  77: "Snow Grains",
  80: "Light Rain Showers",
  81: "Rain Showers",
  82: "Violent Rain Showers",
  85: "Snow Showers",
  86: "Heavy Snow Showers",
  95: "Thunderstorms",
  96: "Thunderstorms with Hail",
  99: "Severe Thunderstorms",
};

export function describeWeatherCode(code) {
  return WEATHER_CODE_LABELS[code] ?? "Humid & Hazy";
}

const COMPASS_DIRECTIONS = [
  "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
];

export function degreesToCompass(degrees) {
  if (degrees === null || degrees === undefined || Number.isNaN(degrees)) return "--";
  return COMPASS_DIRECTIONS[Math.round(degrees / 22.5) % 16];
}
