"""Unit tests for the inference-only serving layer (app/ml/) -- synthetic
history, no network, no database. Model-registry tests load the real
artifacts backend/ml/train.py produced, since those files are the actual
contract the API endpoints depend on.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.ml import feature_builder, model_registry
from ml import config as ml_config


def _history_row(hours_ago: float, now: datetime, **overrides) -> dict:
    row = {
        "time": now - timedelta(hours=hours_ago),
        "temperature_c": 25.0,
        "apparent_temperature_c": 26.0,
        "relative_humidity_pct": 55.0,
        "dew_point_c": 15.0,
        "precipitation_mm_hour": 0.0,
        "rainfall_mm_hour": 0.0,
        "showers_mm_hour": 0.0,
        "weather_code": 0,
        "wind_speed_kmh": 8.0,
        "wind_gust_kmh": 12.0,
        "wind_direction_deg": 200.0,
        "visibility_m": 20000.0,
        "pressure_hpa": 1012.0,
        "surface_pressure_hpa": 998.0,
        "cloud_cover_pct": 5.0,
        "soil_moisture_0_7cm": 0.2,
        "soil_moisture_7_28cm": 0.2,
        "soil_moisture_28_100cm": 0.2,
        "cape_j_kg": 0.0,
        "lifted_index": 8.0,
        "wet_bulb_temperature_c": 19.0,
    }
    row.update(overrides)
    return row


def _full_history(now: datetime, hours_back: int = 73) -> list[dict]:
    return [_history_row(h, now) for h in range(hours_back, -1, -1)]


def test_build_feature_values_succeeds_with_complete_history():
    now = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    history = _full_history(now)
    values, debug = feature_builder.build_feature_values(history, 28.6, 77.2, 216.0, now)
    assert values is not None
    assert debug["missing"] == []
    assert values["temperature_c"] == 25.0
    assert values["temperature_c_lag1h"] == 25.0
    assert values["latitude"] == 28.6
    assert values["is_monsoon_season"] == 1.0  # July


def test_build_feature_values_refuses_when_mostly_missing():
    now = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    # Only one real data point, far from "now" -- nowhere near enough to
    # build current + 4 lags for 21 base columns.
    history = [_history_row(1000, now)]
    values, debug = feature_builder.build_feature_values(history, 28.6, 77.2, 216.0, now)
    assert values is None
    assert len(debug["missing"]) > 0


def test_build_feature_values_empty_history_returns_none():
    now = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    values, debug = feature_builder.build_feature_values([], 28.6, 77.2, 216.0, now)
    assert values is None
    assert debug["reason"] == "no_history"


def test_trailing_rain_window_sums_only_the_past():
    now = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    history = _full_history(now)
    # Put 5mm of rain 2 hours ago (inside the 3h window) and 10mm 50 hours
    # ago (inside the 72h window but outside the 24h one).
    for row in history:
        if row["time"] == now - timedelta(hours=2):
            row["rainfall_mm_hour"] = 5.0
        if row["time"] == now - timedelta(hours=50):
            row["rainfall_mm_hour"] = 10.0

    values, _ = feature_builder.build_feature_values(history, 28.6, 77.2, 216.0, now)
    assert values["rain_3h_mm"] == pytest.approx(5.0)
    assert values["rain_24h_mm"] == pytest.approx(5.0)  # 50h-ago rain must NOT be in the 24h window
    assert values["rain_72h_mm"] == pytest.approx(15.0)  # both rains are in the 72h window


def test_order_values_matches_requested_column_order():
    values = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert feature_builder.order_values(values, ["c", "a", "b"]) == [3.0, 1.0, 2.0]
    # A name the dict doesn't have must not crash inference — default to 0.0
    # rather than raising, since model_registry's saved metadata is the only
    # thing that decides the real column list.
    assert feature_builder.order_values(values, ["a", "missing"]) == [1.0, 0.0]


@pytest.mark.parametrize(
    "target,horizon",
    [(t, h) for t in model_registry.all_targets() for h in ml_config.HORIZONS],
)
def test_every_trained_artifact_loads_and_matches_its_declared_feature_count(target, horizon):
    loaded = model_registry.get_model(target, horizon)
    assert loaded is not None, f"missing artifact for {target} h={horizon} — run `python -m ml.train` first"
    assert loaded.metadata["target"] == target
    assert loaded.metadata["horizon_hours"] == horizon
    assert len(loaded.metadata["feature_columns"]) == loaded.metadata["feature_columns"].__len__()
    n_features = len(loaded.metadata["feature_columns"])

    dummy = np.zeros((1, n_features))
    scaled = loaded.scaler.transform(dummy)
    if loaded.metadata["is_classification"]:
        proba = loaded.model.predict_proba(scaled)
        assert proba.shape == (1, 2)
    else:
        pred = loaded.model.predict(scaled)
        assert pred.shape == (1,)


def test_get_model_returns_none_for_an_unknown_horizon():
    assert model_registry.get_model("temperature_c", 999) is None


def test_get_model_returns_none_for_an_unknown_target():
    assert model_registry.get_model("not_a_real_target", 3) is None


@pytest.mark.parametrize(
    "target,horizon",
    [(t, h) for t in ml_config.TARGET_CLASSIFICATION for h in ml_config.HORIZONS],
)
def test_classification_artifacts_have_a_tuned_decision_threshold(target, horizon):
    """Phase 2 Step 1: every classification model's metadata must carry a
    validation-selected `decision_threshold` in [0, 1] — the naive 0.5
    cutoff is only a fallback for an artifact that predates tuning.
    """
    loaded = model_registry.get_model(target, horizon)
    assert loaded is not None
    threshold = loaded.metadata.get("decision_threshold")
    assert threshold is not None, f"{target} h={horizon} metadata is missing a tuned decision_threshold"
    assert 0.0 <= threshold <= 1.0


def test_run_one_applies_the_models_own_tuned_threshold(monkeypatch):
    """A model whose tuned threshold is far from 0.5 must actually flip the
    binary `value` at that threshold, not at 0.5 — guards against
    prediction_service silently ignoring `decision_threshold`.
    """
    from datetime import datetime, timezone

    from app.ml.model_registry import LoadedModel
    from app.services import prediction_service

    class _StubModel:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])  # probability of the positive class = 0.9

    stub = LoadedModel(
        model=_StubModel(),
        scaler=_IdentityScaler(),
        metadata={
            "feature_columns": ["a"],
            "is_classification": True,
            "algorithm": "xgboost",
            "model_version": "ml-v1",
            "val_metrics": {"precision": 0.5},
            "decision_threshold": 0.95,  # higher than the stub's 0.9 probability
        },
    )
    monkeypatch.setattr(prediction_service.model_registry, "get_model", lambda t, h: stub)

    result = prediction_service._run_one("heavy_rain_risk", 3, datetime.now(timezone.utc), {"a": 1.0})
    assert result.probability == pytest.approx(0.9)
    assert result.value == 0.0  # 0.9 < 0.95 tuned threshold -> negative, despite being > 0.5


class _IdentityScaler:
    def transform(self, X):
        return X


def test_warm_up_loads_every_real_artifact_with_no_failures():
    """Phase 2 Step 5: warm_up() must successfully populate the cache for
    all 12 deployed artifacts against the real artifacts directory.
    """
    model_registry.get_model.cache_clear()
    result = model_registry.warm_up()
    assert result["failed"] == []
    assert len(result["loaded"]) == 12
