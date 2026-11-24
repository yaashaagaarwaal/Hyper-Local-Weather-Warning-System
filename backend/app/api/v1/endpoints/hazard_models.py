"""Trained hazard-imagery models: flood mapping (Sentinel-1 radar) and
lightning detection (infrared cloud tops).

Both are PyTorch U-Nets served from the isolated inference worker (see
app/services/imagery/nowcast_worker.py). Every response says which engine
produced it ("trained-ml" vs "classical-fallback"), whether the input was a
REAL observation or a SYNTHETIC simulation, and -- for real labelled samples
-- how the result scores against ground truth.

Neither model feeds the risk engine or alerts; flood and lightning hazards
there remain rule-based. The lightning model is not validated for India.
See docs/flood_lightning_model_report.md.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from app.core.exceptions import AppError
from app.schemas.imagery import HazardSampleRequest
from app.services.imagery import flood_service, lightning_service
from app.services.imagery.hazard_common import hazard_model_dir

router = APIRouter(prefix="/hazard-models", tags=["hazard-models"])

MODELS = ("flood", "lightning")
EVALUATION_FILES = {"flood": "flood_evaluation.json", "lightning": "lightning_evaluation.json"}
FIGURES = {
    "flood": ("flood_examples_real_test.png", "flood_examples_synthetic_test.png", "flood_variant_comparison.png"),
    "lightning": ("lightning_example_real_test.png", "lightning_examples_synthetic_test.png", "lightning_variant_comparison.png"),
}


def _check(model: str) -> None:
    if model not in MODELS:
        raise AppError(f"Unknown model {model!r}. Valid: {list(MODELS)}", status_code=404)


@router.get("/status")
def all_status() -> dict:
    """Both models at once. Never errors: an unavailable model is reported
    as `{"available": false, "reason": ...}`."""
    return {"flood": flood_service.status(), "lightning": lightning_service.status()}


@router.get("/{model}/status")
def model_status(model: str) -> dict:
    _check(model)
    return flood_service.status() if model == "flood" else lightning_service.status()


@router.get("/{model}/metrics")
def model_metrics(model: str) -> dict:
    """The stored evaluation report written by `python -m ml.hazards.<model>_evaluate`:
    real-data test scores, baselines, the training-variant comparison, and --
    separately -- scores on synthetic India samples."""
    _check(model)
    path = hazard_model_dir() / EVALUATION_FILES[model]
    if not path.exists():
        return {"available": False, "reason": f"No evaluation report yet ({path.name}); run the evaluation script."}
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"available": False, "reason": f"Evaluation report {path.name} is unreadable."}
    figures = [{"name": name, "url": f"/api/v1/hazard-models/{model}/figures/{name}"}
               for name in FIGURES[model] if (hazard_model_dir() / "figures" / name).exists()]
    return {"available": True, "model": model, "figures": figures, **report}


@router.get("/{model}/figures/{name}")
def model_figure(model: str, name: str) -> FileResponse:
    _check(model)
    if name not in FIGURES[model]:          # fixed whitelist: never a caller-chosen path
        raise AppError(f"Unknown figure {name!r}.", status_code=404)
    path = hazard_model_dir() / "figures" / name
    if not path.exists():
        raise AppError(f"Figure {name!r} has not been generated yet.", status_code=404)
    return FileResponse(path, media_type="image/png")


# ------------------------------------------------------------------ flood --
@router.get("/flood/samples")
def flood_samples() -> dict:
    """Radar chips the flood model can be run on: real Sen1Floods11 chips
    (with their official train/valid/test split) and synthetic India chips."""
    return flood_service.list_samples()


@router.post("/flood/predict")
def flood_predict(body: HazardSampleRequest) -> dict:
    """Map water in one catalogued chip with the trained U-Net, alongside
    the classical threshold and -- where a label exists -- error metrics."""
    return flood_service.predict_sample(body.sample_id)


@router.post("/flood/predict/upload")
async def flood_predict_upload(file: UploadFile = File(...)) -> dict:
    """Map water in an uploaded 2-band Sentinel-1 GeoTIFF (VV, VH in dB)."""
    return flood_service.predict_upload(await file.read(), file.filename or "upload.tif")


# -------------------------------------------------------------- lightning --
@router.get("/lightning/samples")
def lightning_samples() -> dict:
    """Infrared image pairs the lightning model can be run on: real GOES-16
    (Americas, with GLM truth), synthetic India scenes (simulated truth) and
    real INSAT-3DR frames over India (no truth)."""
    return lightning_service.list_samples()


@router.post("/lightning/predict")
def lightning_predict(body: HazardSampleRequest) -> dict:
    """Estimate where lightning is likely for one catalogued sample with the
    trained U-Net, alongside the classical cold-cloud threshold."""
    return lightning_service.predict_sample(body.sample_id, body.region)
