"""WMO weather-code → text and wind-degrees → compass helpers.

Deliberate 1:1 port of src/utils/weatherCodes.js on the frontend, so a
location's `condition` / `windDirection` text reads identically whether it
came from the frontend calling Open-Meteo directly or from this backend.
"""

_WEATHER_CODE_LABELS = {
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
}


def describe_weather_code(code: int | None) -> str:
    if code is None:
        return "Humid & Hazy"
    return _WEATHER_CODE_LABELS.get(code, "Humid & Hazy")


_COMPASS_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def degrees_to_compass(degrees: float | None) -> str:
    if degrees is None:
        return "--"
    return _COMPASS_DIRECTIONS[round(degrees / 22.5) % 16]
