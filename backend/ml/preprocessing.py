"""Loading, cleaning, feature engineering and splitting for the Phase 4
pipeline. Pure pandas/numpy -- no model code here (see train_*.py), and no
FastAPI imports (this must run standalone from the training venv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml import config


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.DATASET_PATH)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value handling. Every decision here is a documented choice,
    not a default imputer call -- see docs/model_report.md "Preprocessing
    decisions".
    """
    df = df.drop(columns=config.HIGH_MISSINGNESS_COLUMNS, errors="ignore")

    # Trailing rain-accumulation columns are NaN only in the first few hours
    # of each station's series, before enough history exists to accumulate --
    # not a measurement gap. 0.0 is the physically correct value in that
    # situation (no accumulated rain data yet == no rain accumulated yet).
    for col in config.TRAILING_RAIN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp_local"]
    hour = ts.dt.hour
    day_of_year = ts.dt.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # Delhi's monsoon season (roughly June-September) is when the dataset's
    # own analysis showed rain and thunderstorm proxies concentrate -- a
    # genuine domain feature, not an invented one.
    df["is_monsoon_season"] = ts.dt.month.isin([6, 7, 8, 9]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-location lag features -- the "sequence generation" step: a fixed
    tabular encoding of each station's own recent history. Grouped by
    `area_name` so a lag never crosses a station boundary (each station's
    history is actually contiguous hourly data; see dataset_analysis.md).
    """
    df = df.sort_values(["area_name", "timestamp_local"]).reset_index(drop=True)
    grouped = df.groupby("area_name", sort=False)
    lagged = pd.concat(
        {
            f"{col}_lag{lag}h": grouped[col].shift(lag)
            for col in config.BASE_WEATHER_COLUMNS
            for lag in config.LAG_HOURS
        },
        axis=1,
    )
    return pd.concat([df, lagged], axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The full input feature list for modeling: base weather + their lags +
    trailing rain windows + static spatial + calendar. Excludes identifiers,
    the raw timestamp, and everything in LEAKAGE_OR_LABEL_COLUMNS.
    """
    lag_cols = [
        f"{col}_lag{lag}h"
        for col in config.BASE_WEATHER_COLUMNS
        for lag in config.LAG_HOURS
    ]
    calendar_cols = [
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "is_monsoon_season",
    ]
    cols = (
        config.BASE_WEATHER_COLUMNS
        + lag_cols
        + [c for c in config.TRAILING_RAIN_COLUMNS if c in df.columns]
        + config.STATIC_SPATIAL_COLUMNS
        + calendar_cols
    )
    return [c for c in cols if c in df.columns]


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split strictly by calendar date, never by row shuffling, so no model
    is ever evaluated on a timestamp earlier than any timestamp it trained on
    -- the "chronological splitting" the prompt explicitly requires.
    """
    train_end = pd.Timestamp(config.TRAIN_END_DATE) + pd.Timedelta(hours=23)
    val_end = pd.Timestamp(config.VALIDATION_END_DATE) + pd.Timedelta(hours=23)
    train = df[df["timestamp_local"] <= train_end]
    val = df[(df["timestamp_local"] > train_end) & (df["timestamp_local"] <= val_end)]
    test = df[df["timestamp_local"] > val_end]
    return train, val, test
