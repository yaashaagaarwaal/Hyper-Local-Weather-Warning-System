"""Pytest fixtures.

IMPORTANT: this file sets DATABASE_URL to a dedicated test database *before*
anything under `app` is imported, so the app's engine (created at import
time in app/db/session.py) binds to the test DB, never the dev database.
"""

import os

os.environ["DATABASE_URL"] = "postgresql+psycopg2://yashagarwal@localhost:5432/atmoguard_test"
os.environ["APP_ENV"] = "test"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()  # ensure the env vars above are actually picked up

from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _test_schema():
    """Drops and recreates all tables at the START of the test session (not
    just creating at the start and dropping at the end) — so the test
    database is always left in a real, queryable state after a pytest run
    finishes, and a stale `alembic_version` row never gets out of sync with
    "the tables don't actually exist" (which previously broke running
    `alembic upgrade head` against this same database after a test run).
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(scope="session", autouse=True)
def _seed_default_grid(_test_schema):
    """Seeds the default 49-cell "7x7" grid once per test session (real,
    independently-committed rows — plain SessionLocal, not the rolled-back
    `db_session` fixture — so every connection, including the TestClient's
    own, can see them). Phase 3 tests that exercise the real default
    resolution (not a disposable per-test one) rely on this.
    """
    from app.repositories.grid_repository import save_grid_cells
    from app.utils.grid import build_grid_geometry

    session = SessionLocal()
    try:
        save_grid_cells(session, build_grid_geometry())
    finally:
        session.close()


@pytest.fixture
def db_session() -> Session:
    """A DB session wrapped in a transaction that's rolled back after the
    test, so tests never leave data behind for each other.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client() -> TestClient:
    # Used as a context manager so the app's lifespan actually runs (Phase 2
    # needs this: it's what creates `app.state.http_client`, the shared
    # httpx.AsyncClient every weather provider call depends on).
    with TestClient(app) as test_client:
        yield test_client
