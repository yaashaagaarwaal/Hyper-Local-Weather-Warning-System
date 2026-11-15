"""Phase 4 prediction response shapes.

Deliberately a *separate* vocabulary from `app/schemas/risk.py`: a
prediction here is a trained model's own output (a probability or a
continuous value, with validation-set metrics attached) forecasting the
future, while `/risk/*` is a transparent, rule-based score describing the
present. Callers should not assume they're on the same 0-100/LOW-SEVERE
scale — see docs/model_report.md.
"""

from datetime import datetime

from app.schemas.weather import CamelModel


class RegressionPredictionOut(CamelModel):
    horizon_hours: int
    forecast_for: datetime
    value: float
    algorithm: str
    model_version: str
    val_mae: float | None = None
    val_rmse: float | None = None
    val_r2: float | None = None


class ClassificationPredictionOut(CamelModel):
    horizon_hours: int
    forecast_for: datetime
    risk: bool
    probability: float
    algorithm: str
    model_version: str
    val_precision: float | None = None
    val_recall: float | None = None
    val_pr_auc: float | None = None
    # True when this specific model's own validation precision is too low
    # (see prediction_service.LOW_CONFIDENCE_PRECISION_THRESHOLD) to treat
    # the "risk" flag as actionable on its own -- see docs/model_report.md
    # "Known limitations". Still returned (never suppressed), just labeled.
    low_confidence: bool


class PredictionBasisOut(CamelModel):
    """Mirrors `WeatherBasisOut` in app/schemas/risk.py — same honesty
    fields (nearest real station, distance, data availability), because
    predictions rest on exactly the same limited station density the risk
    engine already has to be transparent about.
    """

    data_available: bool
    nearest_location_id: str | None
    nearest_location_name: str | None
    distance_km: float | None
    missing_feature_count: int


class GridPredictionOut(CamelModel):
    grid_cell_id: str
    generated_at: datetime
    data_available: bool
    temperature: list[RegressionPredictionOut]
    rainfall: list[RegressionPredictionOut]
    heavy_rain_risk: list[ClassificationPredictionOut]
    thunderstorm_risk: list[ClassificationPredictionOut]
    basis: PredictionBasisOut
    # Hazards this ML pipeline does not forecast at all (see
    # ml/config.py / docs/model_report.md "Targets not modeled") — the
    # frontend's "prototype"/"simulated" badge pattern (root CLAUDE.md)
    # should be kept for these; `/risk/grids/{id}` has the Phase 3
    # rule-based equivalent in the meantime.
    unsupported_hazards: list[str]
    source: str = "ml-v1"


class PredictionSummaryOut(CamelModel):
    resolution: str
    horizon_hours: int
    total_grids: int
    grids_with_predictions: int
    heavy_rain_risk_grid_count: int
    thunderstorm_risk_grid_count: int
    mean_temperature: float | None
    mean_rainfall: float | None
    generated_at: datetime
    unsupported_hazards: list[str]
