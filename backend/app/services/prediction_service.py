"""Orchestrates: grid -> nearest real `Location` -> real recent weather
history -> feature construction -> trained-model inference, for the Phase 4
prediction endpoints. Uses the same `MAX_USABLE_RADIUS_KM` threshold as
Phase 3's risk engine (same "too far to be meaningful" line), but resolves
the nearest location purely by distance rather than by stored
`WeatherObservation` rows -- see `_nearest_location` below for why.

Computed on demand, same as Phase 3's risk engine -- no background
scheduler (out of scope for this phase, same as Phase 3). Unlike risk
assessments, predictions are NOT persisted: `ModelPrediction`
(app/models/model_prediction.py) was defined in Phase 1 for the *original*
fixed 3-head spec (thunderstorm/cloudburst/flash-flood probabilities, see
docs/architecture.md §5) and its columns don't match what this phase's
baseline+XGBoost pipeline actually produces (temperature and rainfall
regression, plus heavy-rain/thunderstorm classification) -- writing this
phase's output into that table would misrepresent the data. See
docs/model_report.md "What's not in Phase 4" for the full reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.ml import model_registry
from app.ml.feature_builder import build_feature_values, order_values
from app.models import GridCell, Location
from app.services.risk_engine.thresholds import MAX_USABLE_RADIUS_KM
from app.services.weather_service import WeatherService
from app.utils.geo import haversine_km
from ml import config as ml_config

PREDICTION_MODEL_SOURCE = "ml-v1"
UNSUPPORTED_HAZARDS = ["flood", "lightning"]

# Covers the longest lag (24h) plus the longest trailing-rain window (72h)
# plus the current hour itself -- see app/ml/feature_builder.py.
FEATURE_HISTORY_HOURS_BACK = 73

# A classification model whose validation precision falls below this is
# flagged `low_confidence` in the API response rather than withheld --
# see docs/model_report.md "Known limitations" (heavy_rain_risk at every
# horizon falls well below this given the target's ~0.3-1% positive rate).
LOW_CONFIDENCE_PRECISION_THRESHOLD = 0.10


@dataclass
class RawTargetPrediction:
    target: str
    horizon_hours: int
    forecast_for: datetime
    is_classification: bool
    value: float
    probability: float | None
    algorithm: str
    model_version: str
    val_metrics: dict
    low_confidence: bool = False


@dataclass
class RawGridPrediction:
    grid_cell_id: str
    generated_at: datetime
    data_available: bool
    nearest_location_id: str | None
    nearest_location_name: str | None
    distance_km: float | None
    missing_feature_count: int
    predictions: list[RawTargetPrediction] = field(default_factory=list)
    unsupported_hazards: list[str] = field(default_factory=lambda: list(UNSUPPORTED_HAZARDS))


def _run_one(target: str, horizon: int, now: datetime, values: dict) -> RawTargetPrediction:
    loaded = model_registry.get_model(target, horizon)
    metadata = loaded.metadata
    ordered = order_values(values, metadata["feature_columns"])
    # A one-row DataFrame with the training column names (not a bare numpy
    # array) -- the scaler was fit on named columns, and scikit-learn warns
    # on every call otherwise.
    X = pd.DataFrame([ordered], columns=metadata["feature_columns"])
    X_scaled = loaded.scaler.transform(X)

    is_classification = bool(metadata["is_classification"])
    if is_classification:
        probability = float(loaded.model.predict_proba(X_scaled)[0, 1])
        # Phase 2 Step 1: a per-target, per-horizon threshold tuned on the
        # validation PR curve (ml/threshold_tuning.py), falling back to the
        # naive 0.5 cutoff for any model that predates that tuning pass.
        decision_threshold = metadata.get("decision_threshold", 0.5)
        value = 1.0 if probability >= decision_threshold else 0.0
        low_confidence = metadata["val_metrics"].get("precision", 1.0) < LOW_CONFIDENCE_PRECISION_THRESHOLD
    else:
        probability = None
        value = float(loaded.model.predict(X_scaled)[0])
        if target == "rainfall_mm_hour":
            # Rainfall cannot be negative; linear regression (the selected
            # model at some horizons -- see model_comparison.json) has no
            # such constraint built in. Clamping a known physical bound is
            # not the same as fabricating a value -- the model's raw signal
            # (how close to zero) is preserved, just not allowed below it.
            value = max(0.0, value)
        low_confidence = False

    return RawTargetPrediction(
        target=target,
        horizon_hours=horizon,
        forecast_for=now + timedelta(hours=horizon),
        is_classification=is_classification,
        value=value,
        probability=probability,
        algorithm=metadata["algorithm"],
        model_version=metadata["model_version"],
        val_metrics=metadata["val_metrics"],
        low_confidence=low_confidence,
    )


def _nearest_location(grid_cell: GridCell, locations: list[Location]) -> tuple[Location | None, float | None]:
    """Nearest real `Location` by great-circle distance, with no dependency
    on whether that location has any `WeatherObservation` row in the
    database -- unlike `grid_mapping_service.map_grid_weather` (which Phase 3
    built around *stored* observations), Phase 4 always fetches weather
    live (see `get_feature_history`), so a location's DB history is
    irrelevant to whether it can serve as this grid's nearest station.
    Same `MAX_USABLE_RADIUS_KM` threshold as the risk engine, so both
    pipelines draw the "too far to be meaningful" line in the same place.
    """
    if not locations:
        return None, None
    center_lat, center_lng = float(grid_cell.center_lat), float(grid_cell.center_lng)
    nearest = min(
        locations,
        key=lambda loc: haversine_km(center_lat, center_lng, float(loc.latitude), float(loc.longitude)),
    )
    distance = haversine_km(center_lat, center_lng, float(nearest.latitude), float(nearest.longitude))
    return nearest, round(distance, 2)


async def predict_for_grid(
    weather_service: WeatherService,
    grid_cell: GridCell,
    locations: list[Location],
) -> RawGridPrediction:
    now = datetime.now(timezone.utc)
    loc, distance_km = _nearest_location(grid_cell, locations)

    if loc is None or distance_km is None or distance_km > MAX_USABLE_RADIUS_KM:
        return RawGridPrediction(
            grid_cell_id=grid_cell.id,
            generated_at=now,
            data_available=False,
            nearest_location_id=loc.id if loc else None,
            nearest_location_name=loc.name if loc else None,
            distance_km=distance_km,
            missing_feature_count=0,
        )

    history = await weather_service.get_feature_history(
        float(loc.latitude), float(loc.longitude), FEATURE_HISTORY_HOURS_BACK
    )
    values, debug = build_feature_values(history, float(loc.latitude), float(loc.longitude), None, now)

    if values is None:
        return RawGridPrediction(
            grid_cell_id=grid_cell.id,
            generated_at=now,
            data_available=False,
            nearest_location_id=loc.id,
            nearest_location_name=loc.name,
            distance_km=distance_km,
            missing_feature_count=len(debug.get("missing", [])),
        )

    predictions = [
        _run_one(target, horizon, now, values)
        for target in model_registry.all_targets()
        for horizon in model_registry.available_horizons(target)
    ]

    return RawGridPrediction(
        grid_cell_id=grid_cell.id,
        generated_at=now,
        data_available=True,
        nearest_location_id=loc.id,
        nearest_location_name=loc.name,
        distance_km=distance_km,
        missing_feature_count=len(debug.get("missing", [])),
        predictions=predictions,
    )


async def predict_for_many(
    weather_service: WeatherService, grid_cells: list[GridCell], locations: list[Location]
) -> list[RawGridPrediction]:
    """Batch version. Each call's own `get_feature_history` request is
    already de-duplicated per (lat, lon) by `WeatherService`'s TTL cache, so
    grid cells sharing a nearest station only trigger one real provider
    call per request, same as risk_service.assess_many's observation cache
    achieves for Phase 3.
    """
    return [await predict_for_grid(weather_service, cell, locations) for cell in grid_cells]
