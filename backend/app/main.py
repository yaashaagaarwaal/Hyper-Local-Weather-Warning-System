import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints import health as health_endpoints
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging

settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)


async def _scheduler_loop(app: FastAPI, interval_seconds: int) -> None:
    """Phase 5's automated processing pipeline, run on a fixed interval
    inside this one process (see app/services/pipeline.py's docstring for
    why this is a plain asyncio loop and not a separate worker/queue).
    Never raises out of the loop — a failed tick is logged and the loop
    waits for the next interval rather than dying silently.
    """
    from app.db.session import SessionLocal
    from app.services import pipeline
    from app.services.providers.open_meteo import OpenMeteoProvider
    from app.services.weather_service import WeatherService

    while True:
        db = SessionLocal()
        try:
            provider = OpenMeteoProvider(client=app.state.http_client)
            weather_service = WeatherService(provider=provider, db=db)
            await pipeline.run_pipeline_once(weather_service, db)
        except Exception:
            logger.exception("pipeline.scheduler_tick_failed")
        finally:
            db.close()
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s starting up (env=%s)", settings.APP_NAME, settings.APP_ENV)
    # One shared httpx.AsyncClient for the whole process lifetime — reused
    # by every weather provider call instead of opening a new connection
    # (and TCP handshake / TLS negotiation) per request.
    app.state.http_client = httpx.AsyncClient()

    # Phase 2 Step 5: load all 12 prediction artifacts now rather than on
    # each one's first request. Same test-suite exception as the scheduler
    # below — a TestClient's lifespan shouldn't pay this cost on every test.
    if settings.MODEL_WARMUP_ENABLED and settings.APP_ENV != "test":
        import time

        from app.ml import model_registry

        t0 = time.perf_counter()
        warmup_result = model_registry.warm_up()
        elapsed = time.perf_counter() - t0
        logger.info(
            "model_registry.warmup_complete loaded=%d failed=%s elapsed_seconds=%.2f",
            len(warmup_result["loaded"]), warmup_result["failed"], elapsed,
        )

    scheduler_task = None
    # Never started under the test suite — tests use their own lifespan-run
    # TestClient and must not have a real, repeating Open-Meteo-polling
    # background loop firing during assertions (see tests/conftest.py).
    if settings.PIPELINE_ENABLED and settings.APP_ENV != "test":
        scheduler_task = asyncio.create_task(
            _scheduler_loop(app, settings.PIPELINE_INTERVAL_SECONDS)
        )
        logger.info("pipeline.scheduler_started interval_seconds=%d", settings.PIPELINE_INTERVAL_SECONDS)

    yield

    if scheduler_task is not None:
        scheduler_task.cancel()
    await app.state.http_client.aclose()
    logger.info("%s shutting down", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AtmoGuard — Hyper-Local Weather Intelligence & Early Warning System API. "
        "Weather ingestion, spatial grid + risk engine, ML predictions, and an "
        "automated alert-generation pipeline. See docs/architecture.md for the "
        "full target pipeline and what's still interim/planned."
    ),
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

# Versioned API.
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Bare, unversioned /health and /health/db too — conventional location for
# infra liveness/readiness probes (load balancers, container orchestrators)
# that expect a root-level health check rather than a versioned one.
app.include_router(health_endpoints.router)

# Imagery module (satellite/precipitation browse samples + user uploads +
# processing outputs) — served as plain static files rather than through a
# DB-backed endpoint, since these are just PNG/MP4 bytes with a JSON
# sidecar (see app/services/imagery/storage.py). Two distinct roots:
# the curated, real GIBS sample set (read-only) and the runtime
# upload/processed scratch area.
from app.services.imagery import storage as _imagery_storage  # noqa: E402

app.mount("/media/imagery", StaticFiles(directory=_imagery_storage.RUNTIME_DIR), name="imagery-runtime-media")
if _imagery_storage.DATASET_DIR.exists():
    app.mount("/media/imagery-dataset", StaticFiles(directory=_imagery_storage.DATASET_DIR), name="imagery-dataset-media")
