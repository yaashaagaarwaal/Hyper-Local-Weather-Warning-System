"""Constants for the Phase 4 ML pipeline. Every threshold here is either a
value already established in Phase 3 (`app/services/risk_engine/thresholds.py`,
kept identical so the ML layer and the rule-based risk engine never disagree
on what "heavy rain" means) or a documented, justified choice for this
pipeline specifically. See docs/model_report.md for the full rationale.
"""

from pathlib import Path

RANDOM_SEED = 42

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BACKEND_DIR.parent / "dataset" / "delhi_ncr_labeled_weather.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# --- Columns known, from dataset_analysis.md / Phase 4 leakage review, to be
# unsafe or meaningless as model *inputs* ---
#
# `rainfall_daily_total_mm`: verified (see docs/model_report.md "Data leakage
# review") to be the FIXED total for the whole calendar day, identical across
# all 24 hourly rows of that day -- i.e. at 00:00 it already encodes rain that
# has not happened yet. Using it as a feature would leak the future. Dropped.
#
# `date`: redundant with `timestamp_local` once calendar features are derived
# from the latter; kept only transiently during feature engineering.
#
# `rain_class`, `heat_class`, `wind_class`, `visibility_class`,
# `fog_risk_proxy_class`, `thunderstorm_proxy_class`,
# `flood_risk_rain_soil_proxy_class`, `label_source`, `label_quality`: these
# are the dataset's own proxy labels, computed by a rule FROM the same-row raw
# weather values. Using the current-row version as an input feature would be
# circular (the model would just be re-learning the labeling rule, not
# predicting anything). We only ever use *future* rows of `rain_class` /
# `thunderstorm_proxy_class` / raw rainfall & weather_code to CONSTRUCT
# forecast targets (see targets.py) -- never as same-row inputs.
LEAKAGE_OR_LABEL_COLUMNS = [
    "rainfall_daily_total_mm",
    "date",
    "rain_class",
    "heat_class",
    "wind_class",
    "visibility_class",
    "fog_risk_proxy_class",
    "thunderstorm_proxy_class",
    "flood_risk_rain_soil_proxy_class",
    "label_source",
    "label_quality",
]

# Dropped outright: >66% missing (see dataset_analysis.md). Imputing most of a
# column's values would mean fabricating data, not handling missingness.
HIGH_MISSINGNESS_COLUMNS = ["boundary_layer_height_m"]

# Backward-looking trailing-accumulation columns (verified empirically against
# a rolling backward sum, corr=0.998, vs. a forward sum, corr=0.635 -- see
# docs/model_report.md). Safe as inputs: they only summarize the past.
TRAILING_RAIN_COLUMNS = [
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_12h_mm",
    "rain_24h_mm",
    "rain_48h_mm",
    "rain_72h_mm",
]

# Raw instantaneous weather columns used as the base feature set (plus their
# lagged versions -- see preprocessing.py).
BASE_WEATHER_COLUMNS = [
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "dew_point_c",
    "precipitation_mm_hour",
    "rainfall_mm_hour",
    "showers_mm_hour",
    "weather_code",
    "wind_speed_kmh",
    "wind_gust_kmh",
    "wind_direction_deg",
    "visibility_m",
    "pressure_hpa",
    "surface_pressure_hpa",
    "cloud_cover_pct",
    "soil_moisture_0_7cm",
    "soil_moisture_7_28cm",
    "soil_moisture_28_100cm",
    "cape_j_kg",
    "lifted_index",
    "wet_bulb_temperature_c",
]

STATIC_SPATIAL_COLUMNS = ["latitude", "longitude", "elevation_m_api"]

# Lag hours applied to BASE_WEATHER_COLUMNS to give the model real (not
# fabricated) history -- this is the "sequence generation" step: a fixed-width
# tabular encoding of the recent sequence, rather than a 3D tensor, because the
# chosen models (see docs/model_report.md Architecture Decision) are tabular,
# not recurrent/convolutional.
LAG_HOURS = [1, 3, 6, 24]

# Forecast horizons (hours ahead) the pipeline trains and serves.
HORIZONS = [1, 3, 6]
# The horizon used for the full 5-model comparison sweep and for the written
# model-development report's headline numbers. Other horizons are evaluated
# with {persistence, xgboost} only -- see docs/model_report.md "Scope" for why
# that's a deliberate, documented choice rather than a shortcut.
FLAGSHIP_HORIZON = 3

# Same numeric thresholds as app/services/risk_engine/thresholds.py, duplicated
# here (not imported) because this package must run standalone with only the
# training venv, without importing the FastAPI app. Keep these two files in
# sync by hand if the risk engine's thresholds ever change.
HEAVY_RAINFALL_MM_HR = 7.6  # IMD/WMO "heavy rain" hourly intensity band
THUNDERSTORM_WEATHER_CODES = {95, 96, 99}  # WMO codes: thunderstorm (+ hail)

# Chronological split boundaries (dates, inclusive on the lower bound of the
# following split). Chosen to give the test split one full calendar year
# (2025), so evaluation covers a full seasonal cycle including monsoon,
# without ever training on future data relative to validation or test.
TRAIN_END_DATE = "2024-06-30"
VALIDATION_END_DATE = "2024-12-31"
# test split: 2025-01-01 .. 2025-12-31 (end of dataset)

TARGET_REGRESSION = ["temperature_c", "rainfall_mm_hour"]
TARGET_CLASSIFICATION = ["heavy_rain_risk", "thunderstorm_risk"]
