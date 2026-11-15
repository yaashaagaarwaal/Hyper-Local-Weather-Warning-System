"""End-to-end tests for the Phase 4 `/predictions/*` endpoints, through the
real FastAPI app and real database. The "data available" happy path hits
the real Open-Meteo API (there's no local fixture for "the last 73 hours of
real weather"), same justification as test_weather_api.py; the no-data and
404 paths need no network and aren't marked.
"""

import pytest

from app.db.session import SessionLocal
from app.models import GridCell, Location


@pytest.fixture
def seeded_prediction_grid():
    session = SessionLocal()
    # Deliberately not 28.60/77.20 — test_grid_risk_api.py's "API Near"
    # fixture commits a real Location at exactly that point, and since
    # seeded Locations accumulate (committed, never rolled back) across the
    # shared test database, this module's own point must stay far enough
    # from every other test module's coordinates that it can never tie with
    # one on nearest-distance.
    loc = Location(
        id="loc-pred-near", name="Prediction Test Near", region="Test",
        latitude=28.70, longitude=77.03,
    )
    session.merge(loc)

    near_cell = GridCell(
        id="GRID-PREDTEST-NEAR", row=0, col=0, resolution="predtest",
        bounds=[[28.65, 76.98], [28.65, 77.08], [28.75, 77.08], [28.75, 76.98]],
        center_lat=28.70, center_lng=77.03,
    )
    session.merge(near_cell)

    # Far out over open ocean -- guaranteed to be >45km from every real
    # seeded Location regardless of what other test modules have committed.
    far_cell = GridCell(
        id="GRID-PREDTEST-FAR", row=0, col=1, resolution="predtest",
        bounds=[[9.95, 69.95], [9.95, 70.05], [10.05, 70.05], [10.05, 69.95]],
        center_lat=10.0, center_lng=70.0,
    )
    session.merge(far_cell)

    session.commit()
    try:
        yield {"near": near_cell.id, "far": far_cell.id, "resolution": "predtest"}
    finally:
        session.close()


def test_unknown_grid_returns_404(client):
    response = client.get("/api/v1/predictions/GRID-DOES-NOT-EXIST")
    assert response.status_code == 404


def test_grid_with_no_nearby_station_returns_no_data_not_a_crash(client, seeded_prediction_grid):
    response = client.get(f"/api/v1/predictions/{seeded_prediction_grid['far']}")
    assert response.status_code == 200
    body = response.json()
    assert body["dataAvailable"] is False
    assert body["temperature"] == []
    assert body["rainfall"] == []
    assert body["heavyRainRisk"] == []
    assert body["thunderstormRisk"] == []
    assert body["unsupportedHazards"] == ["flood", "lightning"]


@pytest.mark.network
def test_grid_with_real_nearby_station_returns_real_predictions(client, seeded_prediction_grid):
    response = client.get(f"/api/v1/predictions/{seeded_prediction_grid['near']}")
    assert response.status_code == 200
    body = response.json()
    assert body["dataAvailable"] is True
    assert body["basis"]["nearestLocationId"] == "loc-pred-near"

    # 3 horizons (1h, 3h, 6h) trained for every target.
    assert len(body["temperature"]) == 3
    assert len(body["rainfall"]) == 3
    assert len(body["heavyRainRisk"]) == 3
    assert len(body["thunderstormRisk"]) == 3

    for point in body["temperature"]:
        assert isinstance(point["value"], (int, float))
        assert point["horizonHours"] in (1, 3, 6)
        assert point["algorithm"]

    for point in body["heavyRainRisk"]:
        assert isinstance(point["risk"], bool)
        assert 0.0 <= point["probability"] <= 1.0
        # Documented, expected limitation (see docs/model_report.md) --
        # the heavy-rain classifier's own validation precision is too low
        # to call it actionable on its own at any trained horizon.
        assert point["lowConfidence"] is True


@pytest.mark.network
def test_prediction_summary_endpoint(client, seeded_prediction_grid):
    response = client.get(
        "/api/v1/predictions/summary",
        params={"resolution": seeded_prediction_grid["resolution"], "horizon": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["totalGrids"] == 2  # the near + far test cells
    assert body["horizonHours"] == 3
    assert 0 <= body["gridsWithPredictions"] <= body["totalGrids"]


def test_prediction_summary_rejects_an_untrained_horizon(client, seeded_prediction_grid):
    """Phase 3 integration fix: horizon=99 (no model trained for it) used to
    return 200 with a structurally-valid-looking but meaningless body
    (gridsWithPredictions counting data-availability, not real predictions,
    and heavy_rain/thunderstorm counts silently 0) -- it must now be a
    clear 422, never a quiet "checked and safe" response.
    """
    response = client.get(
        "/api/v1/predictions/summary",
        params={"resolution": seeded_prediction_grid["resolution"], "horizon": 99},
    )
    assert response.status_code == 422

    for horizon in (1, 3, 6):
        ok = client.get(
            "/api/v1/predictions/summary",
            params={"resolution": seeded_prediction_grid["resolution"], "horizon": horizon},
        )
        assert ok.status_code == 200
