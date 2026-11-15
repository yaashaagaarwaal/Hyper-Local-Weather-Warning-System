"""Unit tests for the offline Phase 4 pipeline (backend/ml/) -- pure pandas,
no database, no network, small synthetic frames instead of the real 82MB
CSV so these run in milliseconds as part of the normal suite.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml import config, preprocessing, targets


def _synthetic_rows(area: str, n: int, start: datetime, **overrides) -> pd.DataFrame:
    base = {
        "timestamp_local": [start + pd.Timedelta(hours=i) for i in range(n)],
        "area_name": area,
        "temperature_c": np.linspace(20, 20 + n - 1, n),
        "apparent_temperature_c": np.linspace(20, 20 + n - 1, n),
        "relative_humidity_pct": [60.0] * n,
        "dew_point_c": [15.0] * n,
        "precipitation_mm_hour": [0.0] * n,
        "rainfall_mm_hour": [0.0] * n,
        "showers_mm_hour": [0.0] * n,
        "weather_code": [0] * n,
        "wind_speed_kmh": [10.0] * n,
        "wind_gust_kmh": [15.0] * n,
        "wind_direction_deg": [180.0] * n,
        "visibility_m": [24000.0] * n,
        "pressure_hpa": [1013.0] * n,
        "surface_pressure_hpa": [1000.0] * n,
        "cloud_cover_pct": [10.0] * n,
        "soil_moisture_0_7cm": [0.2] * n,
        "soil_moisture_7_28cm": [0.2] * n,
        "soil_moisture_28_100cm": [0.2] * n,
        "cape_j_kg": [0.0] * n,
        "lifted_index": [10.0] * n,
        "boundary_layer_height_m": [np.nan] * n,
        "wet_bulb_temperature_c": [18.0] * n,
        "latitude": [28.6] * n,
        "longitude": [77.2] * n,
        "elevation_m_api": [216.0] * n,
        "rain_3h_mm": [np.nan] * n,
        "rain_6h_mm": [np.nan] * n,
        "rain_12h_mm": [np.nan] * n,
        "rain_24h_mm": [np.nan] * n,
        "rain_48h_mm": [np.nan] * n,
        "rain_72h_mm": [np.nan] * n,
        "rainfall_daily_total_mm": [999.0] * n,  # deliberately absurd — must never be used as a feature
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_rainfall_daily_total_is_excluded_from_features():
    df = _synthetic_rows("A", 30, datetime(2024, 1, 1))
    engineered = preprocessing.engineer_features(df)
    cols = preprocessing.feature_columns(engineered)
    assert "rainfall_daily_total_mm" not in cols
    assert "date" not in cols
    for leak_col in config.LEAKAGE_OR_LABEL_COLUMNS:
        assert leak_col not in cols


def test_high_missingness_column_is_dropped_not_imputed():
    df = _synthetic_rows("A", 10, datetime(2024, 1, 1))
    cleaned = preprocessing.clean(df)
    assert "boundary_layer_height_m" not in cleaned.columns


def test_trailing_rain_nan_filled_with_zero_not_dropped():
    df = _synthetic_rows("A", 10, datetime(2024, 1, 1))
    cleaned = preprocessing.clean(df)
    assert (cleaned["rain_3h_mm"] == 0.0).all()
    assert cleaned["rain_3h_mm"].isna().sum() == 0


def test_lag_features_never_cross_station_boundary():
    df_a = _synthetic_rows("A", 5, datetime(2024, 1, 1), temperature_c=[10, 11, 12, 13, 14])
    df_b = _synthetic_rows("B", 5, datetime(2024, 1, 1), temperature_c=[100, 101, 102, 103, 104])
    df = pd.concat([df_a, df_b], ignore_index=True)
    lagged = preprocessing.add_lag_features(df)

    b_first_row = lagged[(lagged.area_name == "B")].iloc[0]
    # Station B's very first row must not see station A's temperature as a
    # "lag" value, even though A's rows sort immediately before B's.
    assert pd.isna(b_first_row["temperature_c_lag1h"])

    a_last_row = lagged[(lagged.area_name == "A")].iloc[-1]
    assert a_last_row["temperature_c_lag1h"] == 13  # one hour before the last row


def test_chronological_split_has_no_overlap_and_is_time_ordered():
    # Span from 10 days before TRAIN_END_DATE to 10 days after
    # VALIDATION_END_DATE, so the synthetic frame actually crosses both of
    # the pipeline's real (fixed-calendar-date) split boundaries.
    start = pd.Timestamp(config.TRAIN_END_DATE) - pd.Timedelta(days=10)
    end = pd.Timestamp(config.VALIDATION_END_DATE) + pd.Timedelta(days=10)
    n_hours = int((end - start).total_seconds() // 3600)
    df = _synthetic_rows("A", n_hours, start.to_pydatetime())
    train, val, test = preprocessing.chronological_split(df)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert train["timestamp_local"].max() < val["timestamp_local"].min()
    assert val["timestamp_local"].max() < test["timestamp_local"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_future_value_uses_strictly_later_rows_only():
    df = _synthetic_rows("A", 10, datetime(2024, 1, 1), temperature_c=list(range(10)))
    out = targets.future_value(df, "temperature_c", horizon=3)
    assert out.iloc[0] == 3  # row 0's target is row 3's value
    assert pd.isna(out.iloc[-1])  # last 3 rows can't have a +3h future value
    assert pd.isna(out.iloc[-3])


def test_forward_window_max_requires_a_complete_window():
    df = _synthetic_rows(
        "A", 6, datetime(2024, 1, 1),
        rainfall_mm_hour=[0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    )
    out = targets.forward_window_max(df, "rainfall_mm_hour", horizon=3)
    # row 0 window = rows 1,2,3 -> max(0, 10, 0) = 10
    assert out.iloc[0] == 10.0
    # last 3 rows: window runs past the end of the series -> NaN, never 0
    assert out.iloc[-1:].isna().all()
    assert out.iloc[-3:].isna().all()


def test_heavy_rain_risk_target_matches_documented_threshold():
    df = _synthetic_rows(
        "A", 6, datetime(2024, 1, 1),
        rainfall_mm_hour=[0.0, 0.0, config.HEAVY_RAINFALL_MM_HR + 1, 0.0, 0.0, 0.0],
    )
    out = targets.build_targets(df, horizon=2)
    col = targets.target_column("heavy_rain_risk", 2)
    # row 0's window (rows 1,2) includes the heavy value -> risk = 1
    assert out[col].iloc[0] == 1.0
    # row 2's window (rows 3,4) does not -> risk = 0
    assert out[col].iloc[2] == 0.0


def test_thunderstorm_indicator_true_for_convective_weather_code_or_cape_li():
    df = _synthetic_rows(
        "A", 1, datetime(2024, 1, 1),
        weather_code=[95], cape_j_kg=[0.0], lifted_index=[10.0],
    )
    assert targets.thunderstorm_indicator(df).iloc[0] == 1.0

    df2 = _synthetic_rows(
        "A", 1, datetime(2024, 1, 1),
        weather_code=[0], cape_j_kg=[1500.0], lifted_index=[-4.0],
    )
    assert targets.thunderstorm_indicator(df2).iloc[0] == 1.0

    df3 = _synthetic_rows(
        "A", 1, datetime(2024, 1, 1),
        weather_code=[0], cape_j_kg=[0.0], lifted_index=[10.0],
    )
    assert targets.thunderstorm_indicator(df3).iloc[0] == 0.0


def test_feature_columns_order_is_stable_across_calls():
    df = _synthetic_rows("A", 30, datetime(2024, 1, 1))
    engineered = preprocessing.engineer_features(df)
    assert preprocessing.feature_columns(engineered) == preprocessing.feature_columns(engineered)
