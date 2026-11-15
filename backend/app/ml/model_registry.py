"""Loads `backend/ml/artifacts/*` once per process and caches them.

If an artifact for a given (target, horizon) doesn't exist, `get_model`
returns `None` -- callers must treat that as "no prediction available",
never fabricate one. This is the only place in the running app that reads
from `backend/ml/artifacts/`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

import joblib

from ml import config as ml_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    model: object
    scaler: object
    metadata: dict


@lru_cache(maxsize=None)
def get_model(target: str, horizon: int) -> LoadedModel | None:
    metadata_path = ml_config.ARTIFACTS_DIR / f"{target}_h{horizon}_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    model = joblib.load(ml_config.ARTIFACTS_DIR / metadata["model_path"])
    scaler = joblib.load(ml_config.ARTIFACTS_DIR / metadata["scaler_path"])
    return LoadedModel(model=model, scaler=scaler, metadata=metadata)


def available_horizons(target: str) -> list[int]:
    return [h for h in ml_config.HORIZONS if get_model(target, h) is not None]


def all_targets() -> list[str]:
    return list(ml_config.TARGET_REGRESSION) + list(ml_config.TARGET_CLASSIFICATION)


def warm_up() -> dict[str, list[str]]:
    """Phase 2 Step 5: eagerly populates `get_model`'s lru_cache for every
    (target, horizon) combination -- moves the joblib/xgboost deserialization
    cost (measured empirically at ~1.5s per artifact the first time it's
    loaded; ~0.1ms on every cached call after) from a user's first prediction
    request to application startup. Does not change get_model's caching at
    all -- this just calls it early.

    A missing artifact (`get_model` returns None) or a corrupted one (raises
    during `joblib.load`) is logged and skipped, never raised past this
    function -- one bad file must not crash startup. Callers get back which
    (target, horizon) keys loaded and which didn't, for a visible startup log
    line rather than a silent partial warmup.
    """
    loaded: list[str] = []
    failed: list[str] = []
    for target in all_targets():
        for horizon in ml_config.HORIZONS:
            key = f"{target}_h{horizon}"
            try:
                result = get_model(target, horizon)
            except Exception:
                logger.exception("model_registry.warmup_failed target=%s horizon=%d", target, horizon)
                failed.append(key)
                continue
            (loaded if result is not None else failed).append(key)
    return {"loaded": loaded, "failed": failed}
