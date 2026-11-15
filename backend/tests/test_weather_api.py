"""End-to-end tests through the real FastAPI app, the real database
(atmoguard_test), and the real Open-Meteo API — these are integration tests,
not unit tests, by design (the task explicitly asks for "actual Open-Meteo
API requests" and "database storage" to be tested, not mocked).
"""

import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Location, WeatherForecast, WeatherObservation

pytestmark = pytest.mark.network


REQUIRED_LOCATIONS = {
    "loc-ndls": "Delhi",
    "loc-noida": "Noida",
    "loc-sec62": "Sector 62, Noida",
    "loc-ghz": "Ghaziabad",
    "loc-agra": "Agra",
}


@pytest.fixture
def seeded_locations():
    """Seeds via a plain, independently-committed `SessionLocal()` (NOT the
    `db_session` fixture) — `db_session` deliberately joins an external
    transaction that's rolled back at teardown (see conftest.py), so writes
    through it are invisible to the separate connection the `client` fixture
    uses internally. These rows need a real commit so the TestClient's own
    request can see them.
    """
    session = SessionLocal()
    rows = [
        Location(id="loc-ndls", name="New Delhi", region="Delhi", latitude=28.6139, longitude=77.209, is_primary=True),
        Location(id="loc-noida", name="Noida", region="Uttar Pradesh", latitude=28.5355, longitude=77.391),
        Location(id="loc-sec62", name="Sector 62", region="Noida", latitude=28.628, longitude=77.3649, saved_as="home"),
        Location(id="loc-ghz", name="Ghaziabad", region="Uttar Pradesh", latitude=28.6692, longitude=77.4538),
        Location(id="loc-agra", name="Agra", region="Uttar Pradesh", latitude=27.1767, longitude=78.0081),
    ]
    for row in rows:
        session.merge(row)
    session.commit()
    try:
        yield session
    finally:
        session.close()


def test_current_weather_for_each_required_location(client, seeded_locations):
    for location_id, name in REQUIRED_LOCATIONS.items():
        response = client.get("/api/v1/weather/current", params={"locationId": location_id})
        assert response.status_code == 200, f"{name}: {response.text}"
        body = response.json()
        assert body["locationId"] == location_id
        assert body["temperature"] is not None
        assert body["source"] == "open-meteo"
        assert body["units"]["temperature"] == "celsius"


def test_current_weather_by_raw_coordinates(client):
    response = client.get("/api/v1/weather/current", params={"lat": 28.6139, "lon": 77.209})
    assert response.status_code == 200
    body = response.json()
    assert body["locationId"] is None
    assert body["temperature"] is not None


def test_current_weather_missing_params_returns_400(client):
    response = client.get("/api/v1/weather/current")
    assert response.status_code == 400
    assert "locationId" in response.json()["error"]


def test_current_weather_unknown_location_returns_404(client):
    response = client.get("/api/v1/weather/current", params={"locationId": "loc-does-not-exist"})
    assert response.status_code == 404


def test_current_weather_invalid_latitude_returns_422(client):
    response = client.get("/api/v1/weather/current", params={"lat": 999, "lon": 77})
    assert response.status_code == 422


def test_batch_current_weather_returns_distinct_values(client, seeded_locations):
    response = client.get(
        "/api/v1/weather/current/batch",
        params={"locationIds": "loc-ndls,loc-agra,loc-ghz"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3
    assert {w["locationId"] for w in body} == {"loc-ndls", "loc-agra", "loc-ghz"}
    assert len({w["temperature"] for w in body}) > 1


def test_batch_current_weather_unknown_id_returns_404(client, seeded_locations):
    response = client.get(
        "/api/v1/weather/current/batch", params={"locationIds": "loc-ndls,loc-bogus"}
    )
    assert response.status_code == 404


def test_hourly_forecast_respects_hours_param(client, seeded_locations):
    response = client.get(
        "/api/v1/weather/hourly", params={"locationId": "loc-sec62", "hours": 6}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 6


def test_hourly_forecast_hours_out_of_range_returns_422(client):
    response = client.get(
        "/api/v1/weather/hourly", params={"lat": 28.6, "lon": 77.2, "hours": 100}
    )
    assert response.status_code == 422


def test_daily_forecast_respects_days_param(client, seeded_locations):
    response = client.get(
        "/api/v1/weather/daily", params={"locationId": "loc-ghz", "days": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 3
    assert len({p["day"] for p in body["points"]}) == 3


def test_locations_endpoint_returns_seeded_rows(client, seeded_locations):
    response = client.get("/api/v1/weather/locations")
    assert response.status_code == 200
    body = response.json()
    ids = {loc["id"] for loc in body}
    # Subset, not strict equality: the shared test database legitimately
    # accumulates other real Location rows from other test modules (e.g.
    # Phase 3's grid/risk tests) within the same test session — this test's
    # job is only to confirm the 5 required locations are present/returned
    # correctly, not that they're the only rows in the table.
    assert set(REQUIRED_LOCATIONS.keys()).issubset(ids)


def test_current_weather_persists_to_database(client, seeded_locations):
    client.get("/api/v1/weather/current", params={"locationId": "loc-ndls"})
    rows = seeded_locations.execute(
        select(WeatherObservation).where(WeatherObservation.location_id == "loc-ndls")
    ).scalars().all()
    assert len(rows) >= 1
    assert rows[0].source == "open-meteo"
    assert rows[0].temperature_c is not None


def test_repeated_current_weather_request_does_not_duplicate_rows(client, seeded_locations):
    client.get("/api/v1/weather/current", params={"locationId": "loc-noida"})
    first_count = len(
        seeded_locations.execute(
            select(WeatherObservation).where(WeatherObservation.location_id == "loc-noida")
        ).scalars().all()
    )
    client.get("/api/v1/weather/current", params={"locationId": "loc-noida"})
    second_count = len(
        seeded_locations.execute(
            select(WeatherObservation).where(WeatherObservation.location_id == "loc-noida")
        ).scalars().all()
    )
    # Same observation timestamp (same provider timestep) -> upsert, not a
    # second row. (A real new row would only appear once the hour rolls over.)
    assert second_count == first_count


def test_hourly_forecast_persists_to_database(client, seeded_locations):
    client.get("/api/v1/weather/hourly", params={"locationId": "loc-agra", "hours": 4})
    rows = seeded_locations.execute(
        select(WeatherForecast).where(WeatherForecast.location_id == "loc-agra")
    ).scalars().all()
    assert len(rows) == 4
