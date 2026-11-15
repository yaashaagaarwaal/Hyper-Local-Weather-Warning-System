from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import GridCell, GridWeatherAssociation, Location, RiskAssessment, WeatherObservation


@pytest.fixture
def seeded_grid_and_locations():
    """Real, independently-committed rows (see test_weather_api.py's
    `seeded_locations` fixture for why `db_session` can't be used here: the
    TestClient's own DB connection must see this data)."""
    now = datetime.now(timezone.utc)  # fresh per test invocation — a shared module-level
    # constant previously collided with the unique (location_id, recorded_at, source)
    # constraint once more than one test using this fixture had run.
    session = SessionLocal()
    loc_near = Location(id="loc-api-near", name="API Near", region="Test", latitude=28.60, longitude=77.20)
    loc_far = Location(id="loc-api-far", name="API Far", region="Test", latitude=28.90, longitude=77.70)
    session.merge(loc_near)
    session.merge(loc_far)

    cell = GridCell(
        id="GRID-APITEST-001", row=0, col=0, resolution="apitest",
        bounds=[[28.55, 77.15], [28.55, 77.25], [28.65, 77.25], [28.65, 77.15]],
        center_lat=28.60, center_lng=77.20,
    )
    session.merge(cell)

    obs = WeatherObservation(
        location_id="loc-api-near", recorded_at=now, temperature_c=30.0, humidity_pct=70.0,
        wind_speed_kmh=12.0, rainfall_mm=20.0, rain_probability_pct=85.0, pressure_hpa=1005.0,
        condition="Thunderstorms", weather_code=95, cape_j_kg=3000.0, lifted_index=-7.0,
        source="open-meteo",
    )
    session.add(obs)
    session.commit()
    try:
        yield {"grid_id": cell.id, "resolution": "apitest", "location_near": loc_near, "location_far": loc_far}
    finally:
        session.close()


# --- /grids ---

def test_list_grids_filters_by_resolution(client, seeded_grid_and_locations):
    response = client.get("/api/v1/grids", params={"resolution": "apitest"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "GRID-APITEST-001"
    assert body[0]["resolution"] == "apitest"


def test_list_grids_default_resolution_returns_the_seeded_7x7_grid(client):
    response = client.get("/api/v1/grids")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 49
    assert all(g["resolution"] == "7x7" for g in body)


def test_list_grids_bbox_filter_narrows_results(client):
    full = client.get("/api/v1/grids").json()
    narrowed = client.get(
        "/api/v1/grids", params={"south": 28.32, "north": 28.5, "west": 76.82, "east": 77.56}
    ).json()
    assert len(narrowed) < len(full)
    assert len(narrowed) > 0


def test_get_single_grid(client, seeded_grid_and_locations):
    response = client.get(f"/api/v1/grids/{seeded_grid_and_locations['grid_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["centerLat"] == pytest.approx(28.60)
    assert len(body["bounds"]) == 4


def test_get_unknown_grid_returns_404(client):
    response = client.get("/api/v1/grids/GRID-DOES-NOT-EXIST")
    assert response.status_code == 404


# --- /risk/grids/{grid_id} ---

def test_grid_risk_reflects_real_seeded_severe_weather(client, seeded_grid_and_locations):
    response = client.get(f"/api/v1/risk/grids/{seeded_grid_and_locations['grid_id']}")
    assert response.status_code == 200
    body = response.json()

    assert body["gridCellId"] == "GRID-APITEST-001"
    assert body["overallScore"] >= 55  # severe thunderstorm + heavy rain seeded above
    assert body["riskLevel"] in ("high", "severe")
    assert body["evidenceQuality"] == "measured"
    assert body["weatherBasis"]["dataAvailable"] is True
    assert body["weatherBasis"]["nearestLocationName"] == "API Near"
    assert len(body["hazards"]) == 4
    hazard_names = {h["hazard"] for h in body["hazards"]}
    assert hazard_names == {"heavy_rainfall", "thunderstorm", "lightning", "flood"}


def test_grid_risk_persists_a_risk_assessment_row(client, seeded_grid_and_locations):
    grid_id = seeded_grid_and_locations["grid_id"]
    client.get(f"/api/v1/risk/grids/{grid_id}")

    session = SessionLocal()
    try:
        rows = session.execute(
            select(RiskAssessment).where(RiskAssessment.grid_cell_id == grid_id)
        ).scalars().all()
        assert len(rows) >= 1
        assert rows[0].source == "rule-based-v1"
        assert rows[0].evidence_quality in ("measured", "regional_estimate")
        assert set(rows[0].hazard_scores.keys()) == {"heavyRain", "thunderstorm", "lightning", "flood"}
    finally:
        session.close()


def test_grid_risk_persists_a_weather_association_row(client, seeded_grid_and_locations):
    grid_id = seeded_grid_and_locations["grid_id"]
    client.get(f"/api/v1/risk/grids/{grid_id}")

    session = SessionLocal()
    try:
        rows = session.execute(
            select(GridWeatherAssociation).where(GridWeatherAssociation.grid_cell_id == grid_id)
        ).scalars().all()
        assert len(rows) >= 1
        assert rows[0].data_available is True
        assert rows[0].location_id == "loc-api-near"
    finally:
        session.close()


def test_repeated_risk_requests_accumulate_history_not_overwrite(client, seeded_grid_and_locations):
    grid_id = seeded_grid_and_locations["grid_id"]
    session = SessionLocal()
    try:
        before = len(
            session.execute(select(RiskAssessment).where(RiskAssessment.grid_cell_id == grid_id)).scalars().all()
        )
    finally:
        session.close()

    client.get(f"/api/v1/risk/grids/{grid_id}")
    client.get(f"/api/v1/risk/grids/{grid_id}")

    session = SessionLocal()
    try:
        after = len(
            session.execute(select(RiskAssessment).where(RiskAssessment.grid_cell_id == grid_id)).scalars().all()
        )
    finally:
        session.close()
    assert after == before + 2  # append-only history, not an upsert


def test_get_risk_for_unknown_grid_returns_404(client):
    response = client.get("/api/v1/risk/grids/GRID-DOES-NOT-EXIST")
    assert response.status_code == 404


# --- /risk/grids (all) and /risk/summary ---

def test_get_all_grid_risks_for_resolution(client, seeded_grid_and_locations):
    response = client.get("/api/v1/risk/grids", params={"resolution": "apitest"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["gridCellId"] == "GRID-APITEST-001"


def test_risk_summary_counts_match_total_grids(client):
    response = client.get("/api/v1/risk/summary", params={"resolution": "7x7"})
    assert response.status_code == 200
    body = response.json()
    assert body["totalGrids"] == 49
    counts = body["countsByLevel"]
    assert counts["low"] + counts["moderate"] + counts["high"] + counts["severe"] == 49


# --- /risk/districts ---

def test_ncr_flood_districts_is_valid_geojson_with_no_nan(client):
    response = client.get("/api/v1/risk/districts")
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 20
    for feature in body["features"]:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
        # Delhi's "East" district has a genuine NaN DFSI in the source data
        # (see test_flood_vulnerability.py) -- must come through as valid
        # JSON null, never a raw NaN token (which response.json() would
        # fail to parse at all if it were present).
        for key in ("DFSI", "Percent_Flooded_Area"):
            value = feature["properties"].get(key)
            assert value is None or isinstance(value, (int, float))


def test_district_vulnerability_lookup_inside_coverage(client):
    response = client.get("/api/v1/risk/districts/lookup", params={"lat": 28.6139, "lon": 77.2090})  # New Delhi
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["districtName"] == "New Delhi"
    assert 0 < body["scoreBonus"] <= 15.0


def test_district_vulnerability_lookup_outside_coverage(client):
    response = client.get("/api/v1/risk/districts/lookup", params={"lat": 19.0760, "lon": 72.8777})  # Mumbai
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["dfsi"] is None


def test_district_vulnerability_lookup_rejects_invalid_coordinates(client):
    response = client.get("/api/v1/risk/districts/lookup", params={"lat": 999, "lon": 77.0})
    assert response.status_code == 422


def test_risk_summary_for_apitest_resolution_reflects_severe_weather(client, seeded_grid_and_locations):
    response = client.get("/api/v1/risk/summary", params={"resolution": "apitest"})
    assert response.status_code == 200
    body = response.json()
    assert body["totalGrids"] == 1
    assert body["countsByLevel"]["high"] + body["countsByLevel"]["severe"] == 1
