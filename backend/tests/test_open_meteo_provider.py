"""Tests the Open-Meteo provider two ways:

1. Against the REAL Open-Meteo API (network required) — proves the actual
   integration works, not just that our mocks agree with our own code.
2. Against a mocked transport (respx) — proves retry/backoff/failure-handling
   behavior that would be impractical to trigger reliably against the real,
   usually-healthy API.
"""

import httpx
import pytest
import respx

from app.services.providers.open_meteo import OpenMeteoError, OpenMeteoProvider

# Coordinates for the 5 locations the task explicitly requires coverage for.
REQUIRED_LOCATIONS = {
    "Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Sector 62, Noida": (28.6280, 77.3649),
    "Ghaziabad": (28.6692, 77.4538),
    "Agra": (27.1767, 78.0081),
}


@pytest.fixture
async def provider():
    async with httpx.AsyncClient() as client:
        yield OpenMeteoProvider(client=client)


# ---- Real network integration tests ----


@pytest.mark.network
async def test_get_current_real_request_for_each_required_location(provider):
    for name, (lat, lon) in REQUIRED_LOCATIONS.items():
        result = await provider.get_current(lat, lon)
        assert result.temperature is not None, f"{name}: no temperature returned"
        assert result.source == "open-meteo"
        assert result.condition != ""
        assert result.wind_direction in {
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "--",
        }


@pytest.mark.network
async def test_real_locations_return_distinct_readings(provider):
    results = {}
    for name, (lat, lon) in REQUIRED_LOCATIONS.items():
        results[name] = await provider.get_current(lat, lon)

    # Not a strict guarantee in nature, but with 5 spread-out NCR + Agra
    # points this should essentially never collapse to one shared value —
    # a real regression (e.g. always hitting the same cell) would fail this.
    temps = {r.temperature for r in results.values()}
    assert len(temps) > 1, "all locations returned the identical temperature — suspicious"


@pytest.mark.network
async def test_get_current_batch_real_request_matches_individual_calls(provider):
    coords = list(REQUIRED_LOCATIONS.values())
    batch_results = await provider.get_current_batch(coords)
    assert len(batch_results) == len(coords)
    for (lat, lon), result in zip(coords, batch_results, strict=True):
        assert abs(result.latitude - lat) < 0.2
        assert abs(result.longitude - lon) < 0.2
        assert result.temperature is not None


@pytest.mark.network
async def test_get_hourly_real_request(provider):
    result = await provider.get_hourly(*REQUIRED_LOCATIONS["Delhi"], hours=6)
    assert len(result.points) == 6
    assert all(p.temperature is not None for p in result.points)


@pytest.mark.network
async def test_get_daily_real_request(provider):
    result = await provider.get_daily(*REQUIRED_LOCATIONS["Agra"], days=5)
    assert len(result.points) == 5
    assert all(p.temperature_high is not None for p in result.points)
    assert len({p.day for p in result.points}) == 5  # 5 distinct day labels


# ---- Mocked-transport tests (retry / backoff / failure handling) ----


@respx.mock
async def test_retries_then_succeeds_after_transient_failures(provider):
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"latitude": 28.6, "longitude": 77.2, "current": {"temperature_2m": 25.0, "time": "2026-01-01T00:00"}}),
        ]
    )
    result = await provider.get_current(28.6, 77.2)
    assert route.call_count == 3
    assert result.temperature == 25.0


@respx.mock
async def test_exhausts_retries_and_raises_open_meteo_error(provider):
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(500)
    )
    with pytest.raises(OpenMeteoError):
        await provider.get_current(28.6, 77.2)
    # provider.retry_attempts defaults to settings.HTTP_RETRY_ATTEMPTS (2) + the
    # initial attempt = 3 total calls.
    assert route.call_count == 3


@respx.mock
async def test_open_meteo_error_payload_is_treated_as_a_failure(provider):
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={"error": True, "reason": "Latitude must be in range of -90 to 90°"})
    )
    with pytest.raises(OpenMeteoError, match="Latitude must be in range"):
        await provider.get_current(999, 77.2)


@respx.mock
async def test_batch_request_comma_joins_coordinates(provider):
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=[
            {"latitude": 28.6, "longitude": 77.2, "current": {"temperature_2m": 20.0, "time": "2026-01-01T00:00"}},
            {"latitude": 27.1, "longitude": 78.0, "current": {"temperature_2m": 19.0, "time": "2026-01-01T00:00"}},
        ])
    )
    results = await provider.get_current_batch([(28.6, 77.2), (27.1, 78.0)])
    assert len(results) == 2
    request = route.calls.last.request
    assert "latitude=28.6%2C27.1" in str(request.url)
    assert "longitude=77.2%2C78.0" in str(request.url)


@respx.mock
async def test_normalize_handles_missing_hourly_block_gracefully(provider):
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(
            200,
            json={"latitude": 28.6, "longitude": 77.2, "current": {"temperature_2m": 25.0, "time": "2026-01-01T00:00"}},
        )
    )
    result = await provider.get_current(28.6, 77.2)
    assert result.temperature == 25.0
    assert result.visibility is None
    assert result.uv_index is None
    assert result.rain_probability is None
