"""Forecast target construction.

All targets look strictly FORWARD from each row's timestamp (never at the
row's own time) -- this is what makes this a prediction/forecasting task
rather than a same-instant nowcast that would just reproduce the dataset's
own rule-derived proxy columns. See docs/model_report.md "Target
construction" for the full rationale, including why hazard targets use a
window-max over the horizon rather than the single value at t+h.
"""

from __future__ import annotations

import pandas as pd

from ml import config


def _forward_shifts(df: pd.DataFrame, col: str, horizon: int, group_col: str = "area_name") -> pd.DataFrame:
    """The h columns shift(-1)..shift(-horizon), one per future hour, each
    scoped within `group_col` so a station's forecast never reaches into
    another station's rows.
    """
    grouped = df.groupby(group_col, sort=False)
    return pd.concat(
        [grouped[col].shift(-k) for k in range(1, horizon + 1)],
        axis=1,
        keys=[f"_h{k}" for k in range(1, horizon + 1)],
    )


def future_value(df: pd.DataFrame, col: str, horizon: int) -> pd.Series:
    """The single value of `col` exactly `horizon` hours ahead. Used for the
    continuous regression targets (temperature, rainfall): a warning system
    forecasting "the temperature at 3pm" genuinely means the value AT 3pm,
    not a window maximum.
    """
    return df.groupby("area_name", sort=False)[col].shift(-horizon)


def forward_window_max(df: pd.DataFrame, col: str, horizon: int) -> pd.Series:
    """max(col) over the next `horizon` hours (t+1 .. t+horizon), NaN if any
    of those hours is missing (end of a station's series) -- never claims
    "no hazard in the window" from a partially-observed window.
    """
    shifts = _forward_shifts(df, col, horizon)
    return shifts.max(axis=1, skipna=False)


def forward_window_any(df: pd.DataFrame, bool_col: pd.Series, horizon: int, area_name: pd.Series) -> pd.Series:
    """True if a boolean/0-1 condition holds at any point in the next
    `horizon` hours; NaN (not False) if the window is incomplete.
    """
    tmp = pd.DataFrame({"area_name": area_name, "_flag": bool_col})
    shifts = _forward_shifts(tmp, "_flag", horizon)
    any_true = shifts.eq(1.0).any(axis=1)
    all_present = shifts.notna().all(axis=1)
    out = any_true.astype(float)
    out[~all_present] = float("nan")
    return out


def thunderstorm_indicator(df: pd.DataFrame) -> pd.Series:
    """Same-row convective indicator, built directly from raw instrument
    values (weather_code, CAPE, Lifted Index) -- deliberately NOT read from
    the dataset's own pre-computed `thunderstorm_proxy_class` column, so the
    target definition stays fully transparent and reproducible from raw
    columns alone, and consistent with the CAPE/LI thresholds already used by
    the Phase 3 rule-based thunderstorm hazard module
    (app/services/risk_engine/thunderstorm.py): CAPE >= 1000 J/kg (the "weak
    instability" band) combined with Lifted Index <= -3 (the "marginal"
    band), OR an actual WMO thunderstorm weather code.
    """
    code_hit = df["weather_code"].isin(config.THUNDERSTORM_WEATHER_CODES)
    cape_li_hit = (df["cape_j_kg"] >= 1000) & (df["lifted_index"] <= -3)
    return (code_hit | cape_li_hit).astype(float)


def build_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Adds all target columns for one horizon to a copy of `df`. Must be
    called on a frame already sorted by (area_name, timestamp_local) --
    preprocessing.add_lag_features already sorts this way and all callers in
    this pipeline run targets after that step.
    """
    out = df.copy()
    out[f"target_temperature_c_h{horizon}"] = future_value(df, "temperature_c", horizon)
    out[f"target_rainfall_mm_hour_h{horizon}"] = future_value(df, "rainfall_mm_hour", horizon)

    rain_window_max = forward_window_max(df, "rainfall_mm_hour", horizon)
    out[f"target_heavy_rain_risk_h{horizon}"] = (
        rain_window_max >= config.HEAVY_RAINFALL_MM_HR
    ).astype(float)
    out.loc[rain_window_max.isna(), f"target_heavy_rain_risk_h{horizon}"] = float("nan")

    storm_flag = thunderstorm_indicator(df)
    out[f"target_thunderstorm_risk_h{horizon}"] = forward_window_any(
        df, storm_flag, horizon, df["area_name"]
    )
    return out


def target_column(target_name: str, horizon: int) -> str:
    return f"target_{target_name}_h{horizon}"
