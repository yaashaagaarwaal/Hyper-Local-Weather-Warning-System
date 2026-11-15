"""The automated end-to-end processing pipeline (Phase 5):

    Weather Data Ingestion -> Data Validation -> Spatial Grid Mapping ->
    Risk Assessment -> ML Prediction (where available) -> Alert Generation
    -> Database Storage -> Frontend Notification

Runs on a schedule (a single `asyncio` loop started in `app/main.py`'s
lifespan — see that module for why not a separate worker process) and can
also be triggered on demand (`POST /api/v1/alerts/process`) for testing and
demos. A module-level `asyncio.Lock` makes a tick that fires while a run is
still in progress skip itself (logged, not queued) rather than overlap with
it — "avoid duplicate concurrent jobs".

Step-by-step mapping onto the pipeline above:
  1-2. Ingestion + validation: `WeatherService.get_current_batch()` — the
       same Phase 2 path every `/weather/current` call already uses;
       Pydantic (`CurrentWeather`) is what "validation" means here, exactly
       as it already did before this phase.
  3-4. Spatial grid mapping + risk assessment: `risk_service.assess_many()`
       (Phase 3, unchanged).
  5.   ML prediction: `prediction_service.predict_for_many()` (Phase 4,
       unchanged) — wrapped in its own try/except so a model-loading or
       provider problem can never block the rule-based alerting this
       pipeline must still produce.
  6-7. Alert generation + storage: `alert_service.sync_alerts_for_grid()`
       (new this phase), one grid cell at a time, each call committing its
       own changes.
  8.   Frontend notification: `notification_service.NotificationService`
       (new this phase) — in-app only, see that module's docstring.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.repositories import grid_repository, location_repository
from app.services import alert_service, prediction_service, risk_service
from app.services.notification_service import get_notification_service
from app.services.weather_service import WeatherService

logger = logging.getLogger(__name__)

_pipeline_lock = asyncio.Lock()


async def run_pipeline_once(weather_service: WeatherService, db, resolution: str = "7x7") -> dict:
    if _pipeline_lock.locked():
        logger.warning("pipeline.skipped reason=already_running")
        return {"skipped": True, "reason": "already_running"}

    async with _pipeline_lock:
        started = datetime.now(timezone.utc)

        locations = location_repository.list_all(db)
        if locations:
            try:
                await weather_service.get_current_batch(locations)
            except Exception:
                # A provider outage here is exactly what Phase 2's 503
                # handling already covers for direct API calls; the
                # pipeline keeps going on whatever observations already
                # exist in the database rather than aborting entirely.
                logger.exception("pipeline.ingestion_step_failed")

        cells = grid_repository.list_grid_cells(db, resolution=resolution)
        risk_results = risk_service.assess_many(db, cells, locations)

        predictions_by_grid: dict[str, object] = {}
        try:
            raw_predictions = await prediction_service.predict_for_many(weather_service, cells, locations)
            predictions_by_grid = {p.grid_cell_id: p for p in raw_predictions}
        except Exception:
            logger.exception("pipeline.prediction_step_failed")

        locations_by_id = {loc.id: loc for loc in locations}
        notifier = get_notification_service()
        created = updated = resolved = 0

        for result, assessment, association in risk_results:
            cell = next(c for c in cells if c.id == assessment.grid_cell_id)
            prediction = predictions_by_grid.get(cell.id)
            location = locations_by_id.get(association.location_id) if association.location_id else None

            outcome = alert_service.sync_alerts_for_grid(db, cell, result, prediction, location)
            created += len(outcome.created)
            updated += len(outcome.updated)
            resolved += len(outcome.resolved)

            for alert in outcome.created:
                notifier.notify(alert_service.to_notification(alert, "created"))
            for alert in outcome.updated:
                notifier.notify(alert_service.to_notification(alert, "updated"))
            for alert in outcome.resolved:
                notifier.notify(alert_service.to_notification(alert, "resolved"))

        expired = alert_service.expire_due_alerts(db)
        for alert in expired:
            notifier.notify(alert_service.to_notification(alert, "expired"))

        finished = datetime.now(timezone.utc)
        summary = {
            "skipped": False,
            "startedAt": started.isoformat(),
            "finishedAt": finished.isoformat(),
            "gridsProcessed": len(cells),
            "alertsCreated": created,
            "alertsUpdated": updated,
            "alertsResolved": resolved,
            "alertsExpired": len(expired),
        }
        logger.info("pipeline.run_complete %s", summary)
        return summary
