from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.session import engine, get_db
from app.main import app


def test_engine_connects_to_test_database():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_engine_points_at_atmoguard_test_not_dev_db():
    assert "atmoguard_test" in str(engine.url)


def test_health_db_reports_503_when_database_unreachable(client):
    """Proves /health/db is a REAL check, not a hardcoded 200 — point the
    dependency at an engine that can't connect and confirm it fails loudly.
    """
    broken_engine = create_engine(
        "postgresql+psycopg2://nouser:nopass@localhost:59999/nonexistent",
        pool_pre_ping=False,
    )
    BrokenSession = sessionmaker(bind=broken_engine)

    def broken_get_db():
        db = BrokenSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = broken_get_db
    try:
        response = client.get("/health/db")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
        assert body["database"] == "unreachable"
    finally:
        app.dependency_overrides.pop(get_db, None)
