"""Builds the named feature values a trained model expects, from real live
weather history (see `OpenMeteoProvider.get_feature_history`) plus a
location's static coordinates.

This is deliberately a *named dict*, not a fixed-order vector: the actual
column order used to call `.predict()` always comes from that specific
model's own saved `metadata["feature_columns"]` (see model_registry.py /
prediction_service.py), not a second, hand-maintained list here -- so a
future change to the training feature set can never silently desync from
serving. The lists this module reads from `ml.config` (BASE_WEATHER_COLUMNS,
LAG_HOURS, TRAILING_RAIN_COLUMNS) only decide which values are *computed*,
not the order they're fed to a model in.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from ml import config as ml_config

# How much of a feature row can be missing (no live value close enough in
# time) before we refuse to predict rather than guess. Lag/trailing-window
# features are allowed to be absent this often (e.g. a brand-new location
# with little history yet) -- if more than this fraction is missing, the
# prediction would be built mostly from filled-in zeros, which is exactly
# the kind of fabrication this project's instructions explicitly forbid.
MAX_MISSING_FRACTION = 0.3

# An hourly series should have a point within half an hour of any target
# time; anything further off is treated as "not actually available" rather
# than silently reused.
_MATCH_TOLERANCE_SECONDS = 90 * 60

_TRAILING_RAIN_WINDOWS_HOURS = [3, 6, 12, 24, 48, 72]


def _nearest_value(history: list[dict], target_time: datetime, col: str) -> float | None:
    if not history:
        return None
    best = min(history, key=lambda row: abs((row["time"] - target_time).total_seconds()))
    if abs((best["time"] - target_time).total_seconds()) > _MATCH_TOLERANCE_SECONDS:
        return None
    return best.get(col)


def build_feature_values(
    history: list[dict],
    latitude: float,
    longitude: float,
    elevation_m: float | None,
    now: datetime,
) -> tuple[dict[str, float] | None, dict]:
    """Returns (values, debug) where `values` maps every feature name the
    training pipeline can produce to a float, or `(None, debug)` if too much
    of the row is unavailable to trust a prediction built from it.
    `debug["missing"]` always lists what was unavailable, for the response's
    transparency fields even on a successful build.
    """
    if not history:
        return None, {"missing": ["__all__"], "reason": "no_history"}

    values: dict[str, float | None] = {}
    missing: list[str] = []

    for col in ml_config.BASE_WEATHER_COLUMNS:
        current = _nearest_value(history, now, col)
        values[col] = current
        if current is None:
            missing.append(col)
        for lag in ml_config.LAG_HOURS:
            lag_val = _nearest_value(history, now - timedelta(hours=lag), col)
            key = f"{col}_lag{lag}h"
            values[key] = lag_val
            if lag_val is None:
                missing.append(key)

    for col, window_hours in zip(ml_config.TRAILING_RAIN_COLUMNS, _TRAILING_RAIN_WINDOWS_HOURS):
        window_vals = [
            row.get("rainfall_mm_hour")
            for row in history
            if 0 <= (now - row["time"]).total_seconds() / 3600 <= window_hours
            and row.get("rainfall_mm_hour") is not None
        ]
        # Same fallback as training (ml/preprocessing.py clean()): no
        # accumulated data yet for this window means 0, not unknown.
        values[col] = float(sum(window_vals)) if window_vals else 0.0

    values["latitude"] = latitude
    values["longitude"] = longitude
    values["elevation_m_api"] = elevation_m if elevation_m is not None else 0.0

    hour = now.hour
    day_of_year = now.timetuple().tm_yday
    values["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    values["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    values["day_of_year_sin"] = math.sin(2 * math.pi * day_of_year / 365.25)
    values["day_of_year_cos"] = math.cos(2 * math.pi * day_of_year / 365.25)
    values["is_monsoon_season"] = 1.0 if now.month in (6, 7, 8, 9) else 0.0

    debug = {"missing": missing}
    total_checkable = len(ml_config.BASE_WEATHER_COLUMNS) * (1 + len(ml_config.LAG_HOURS))
    if len(missing) > total_checkable * MAX_MISSING_FRACTION:
        return None, debug

    # Any still-missing current/lag values at this point are below the
    # refuse-to-predict threshold -- fill with 0.0 only so the numeric model
    # can run; `debug["missing"]` already records exactly which inputs these
    # were, so the API response can surface that this specific prediction
    # rested on partial data instead of presenting it as fully measured.
    for key in missing:
        if values.get(key) is None:
            values[key] = 0.0

    return values, debug


def order_values(values: dict[str, float], feature_columns: list[str]) -> list[float]:
    return [float(values.get(col, 0.0)) for col in feature_columns]
