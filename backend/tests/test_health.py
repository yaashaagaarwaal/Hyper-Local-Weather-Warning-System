def test_health_liveness(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "AtmoGuard API"
    assert body["environment"] == "test"


def test_health_versioned_alias_matches_root(client):
    root = client.get("/health")
    versioned = client.get("/api/v1/health")
    assert root.status_code == versioned.status_code == 200
    assert root.json() == versioned.json()


def test_health_db_reports_connected(client):
    """Exercises the real DB connectivity check against atmoguard_test."""
    response = client.get("/health/db")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"


def test_health_db_versioned_alias(client):
    response = client.get("/api/v1/health/db")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unknown_route_returns_centralized_404_shape(client):
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert "error" in body
    assert body["path"] == "/this-route-does-not-exist"
