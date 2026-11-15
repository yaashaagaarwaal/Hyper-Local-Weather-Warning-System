from fastapi import APIRouter

from app.api.v1.endpoints import alerts, grids, health, imagery, predictions, risk, weather

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(weather.router)
api_router.include_router(grids.router)
api_router.include_router(risk.router)
api_router.include_router(predictions.router)
api_router.include_router(alerts.router)
api_router.include_router(imagery.router)
