from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_weather_service
from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.repositories import grid_repository, location_repository
from app.schemas.prediction import (
    ClassificationPredictionOut,
    GridPredictionOut,
    PredictionBasisOut,
    PredictionSummaryOut,
    RegressionPredictionOut,
)
from app.services import prediction_service
from app.services.prediction_service import RawGridPrediction, UNSUPPORTED_HAZARDS
from app.services.weather_service import WeatherService
from ml import config as ml_config

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _build_response(raw: RawGridPrediction) -> GridPredictionOut:
    by_target: dict[str, list] = {"temperature_c": [], "rainfall_mm_hour": [], "heavy_rain_risk": [], "thunderstorm_risk": []}
    for p in raw.predictions:
        by_target.setdefault(p.target, []).append(p)

    def regression(target: str) -> list[RegressionPredictionOut]:
        return [
            RegressionPredictionOut(
                horizon_hours=p.horizon_hours,
                forecast_for=p.forecast_for,
                value=round(p.value, 2),
                algorithm=p.algorithm,
                model_version=p.model_version,
                val_mae=p.val_metrics.get("mae"),
                val_rmse=p.val_metrics.get("rmse"),
                val_r2=p.val_metrics.get("r2"),
            )
            for p in sorted(by_target.get(target, []), key=lambda x: x.horizon_hours)
        ]

    def classification(target: str) -> list[ClassificationPredictionOut]:
        return [
            ClassificationPredictionOut(
                horizon_hours=p.horizon_hours,
                forecast_for=p.forecast_for,
                risk=bool(p.value),
                probability=round(p.probability, 4) if p.probability is not None else 0.0,
                algorithm=p.algorithm,
                model_version=p.model_version,
                val_precision=p.val_metrics.get("precision"),
                val_recall=p.val_metrics.get("recall"),
                val_pr_auc=p.val_metrics.get("pr_auc"),
                low_confidence=p.low_confidence,
            )
            for p in sorted(by_target.get(target, []), key=lambda x: x.horizon_hours)
        ]

    return GridPredictionOut(
        grid_cell_id=raw.grid_cell_id,
        generated_at=raw.generated_at,
        data_available=raw.data_available,
        temperature=regression("temperature_c"),
        rainfall=regression("rainfall_mm_hour"),
        heavy_rain_risk=classification("heavy_rain_risk"),
        thunderstorm_risk=classification("thunderstorm_risk"),
        basis=PredictionBasisOut(
            data_available=raw.data_available,
            nearest_location_id=raw.nearest_location_id,
            nearest_location_name=raw.nearest_location_name,
            distance_km=raw.distance_km,
            missing_feature_count=raw.missing_feature_count,
        ),
        unsupported_hazards=raw.unsupported_hazards,
    )


@router.get("/summary", response_model=PredictionSummaryOut)
async def get_prediction_summary(
    resolution: str = Query("7x7", description="Grid resolution label, e.g. '7x7'"),
    horizon: int = Query(3, description="Forecast horizon in hours (1, 3, or 6)"),
    db: Session = Depends(get_db),
    weather_service: WeatherService = Depends(get_weather_service),
) -> PredictionSummaryOut:
    # Phase 3 integration fix: an unsupported horizon (e.g. 99) used to
    # silently fall through to a structurally-valid-looking response
    # (gridsWithPredictions: 49, heavyRainRiskGridCount: 0) that actually
    # reflected zero real predictions -- reading as "checked and safe"
    # rather than "no model exists for this horizon". `Literal[1, 3, 6]` was
    # tried first but Pydantic v2 doesn't coerce a query string against an
    # int Literal, which rejected the valid `horizon=3` too -- an explicit
    # check against the real trained horizons is simple and correct.
    if horizon not in ml_config.HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"horizon must be one of {ml_config.HORIZONS} (no model is trained for horizon={horizon})",
        )
    cells = grid_repository.list_grid_cells(db, resolution=resolution)
    locations = location_repository.list_all(db)

    results = await prediction_service.predict_for_many(weather_service, cells, locations)

    available = [r for r in results if r.data_available]
    temps = [
        p.value for r in available for p in r.predictions
        if p.target == "temperature_c" and p.horizon_hours == horizon
    ]
    rains = [
        p.value for r in available for p in r.predictions
        if p.target == "rainfall_mm_hour" and p.horizon_hours == horizon
    ]
    heavy_count = sum(
        1 for r in available for p in r.predictions
        if p.target == "heavy_rain_risk" and p.horizon_hours == horizon and p.value >= 0.5
    )
    storm_count = sum(
        1 for r in available for p in r.predictions
        if p.target == "thunderstorm_risk" and p.horizon_hours == horizon and p.value >= 0.5
    )

    return PredictionSummaryOut(
        resolution=resolution,
        horizon_hours=horizon,
        total_grids=len(cells),
        grids_with_predictions=len(available),
        heavy_rain_risk_grid_count=heavy_count,
        thunderstorm_risk_grid_count=storm_count,
        mean_temperature=round(sum(temps) / len(temps), 2) if temps else None,
        mean_rainfall=round(sum(rains) / len(rains), 2) if rains else None,
        generated_at=datetime.now(timezone.utc),
        unsupported_hazards=list(UNSUPPORTED_HAZARDS),
    )


# Registered after "/summary" deliberately -- Starlette matches routes in
# registration order, and "/summary" would otherwise be captured by this
# catch-all as grid_id="summary".
@router.get("/{grid_id}", response_model=GridPredictionOut)
async def get_grid_prediction(
    grid_id: str,
    db: Session = Depends(get_db),
    weather_service: WeatherService = Depends(get_weather_service),
) -> GridPredictionOut:
    cell = grid_repository.get_grid_cell(db, grid_id)
    if cell is None:
        raise NotFoundError(f"Unknown grid id: {grid_id!r}")

    locations = location_repository.list_all(db)
    raw = await prediction_service.predict_for_grid(weather_service, cell, locations)
    return _build_response(raw)
