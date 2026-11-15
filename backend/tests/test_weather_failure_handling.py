"""Confirms a provider failure surfaces as a clean, centrally-handled 503 —
never a raw 500/stack trace — all the way through the real HTTP stack.
"""

import httpx
import respx


@respx.mock
def test_current_weather_returns_503_when_open_meteo_is_down(client):
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(500)
    )
    response = client.get("/api/v1/weather/current", params={"lat": 28.6, "lon": 77.2})
    assert response.status_code == 503
    body = response.json()
    assert "error" in body
    assert "path" in body


@respx.mock
def test_hourly_forecast_returns_503_when_open_meteo_is_down(client):
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(500)
    )
    response = client.get(
        "/api/v1/weather/hourly", params={"lat": 28.6, "lon": 77.2, "hours": 6}
    )
    assert response.status_code == 503
