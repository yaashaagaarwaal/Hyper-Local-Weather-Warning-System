"""Inference engine for the two trained hazard-imagery models (flood mapping,
lightning detection): load once, cache, pick a device, validate, predict.

Like the nowcast engine (ml_inference.py) this runs inside the dedicated
PyTorch worker process, never in the API process -- see nowcast_worker.py.
Importing this module is safe anywhere: torch is imported lazily.

Failure philosophy is the same too: a missing, corrupt or mismatched
artifact (or no PyTorch) raises `ModelUnavailableError` with a plain reason;
callers then serve the classical threshold method, labelled as such. A model
result is never invented and a classical result is never labelled as ML.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.imagery import flood_data, lightning_data
from app.services.imagery.hazard_common import apply_platt, hazard_model_dir
from app.services.imagery.ml_inference import ModelUnavailableError, _import_torch, select_device

SPECS = {
    "flood": {
        "checkpoint": "flood_unet_v1.pt", "metadata": "flood_unet_v1_metadata.json",
        "in_channels": 2, "out_stride": 1,
        "task": "Surface-water (flood + permanent water) mapping from Sentinel-1 VV/VH radar, 10 m pixels",
        "classicalFallback": "Backscatter threshold on the VH band (always available)",
        "train": "python -m ml.hazards.flood_train --all && python -m ml.hazards.flood_evaluate",
    },
    "lightning": {
        "checkpoint": "lightning_unet_v1.pt", "metadata": "lightning_unet_v1_metadata.json",
        "in_channels": 2, "out_stride": 2,
        "task": "Lightning-flash probability per 8 km cell from two infrared cloud-top images 30 min apart",
        "classicalFallback": "Cold-cloud-top temperature threshold (always available)",
        "train": "python -m ml.hazards.lightning_train --all && python -m ml.hazards.lightning_evaluate",
    },
}


@dataclass
class LoadedHazardModel:
    key: str
    model: object
    device: object
    checkpoint: dict
    metadata: dict
    path: Path
    size_bytes: int


@dataclass
class HazardPrediction:
    probability: np.ndarray   # (rows, cols) float32 in [0, 1]
    threshold: float          # decision threshold chosen on real validation data
    device: str
    inference_ms: float


_lock = threading.Lock()
_cache: dict[str, tuple[tuple, LoadedHazardModel]] = {}


def _spec(key: str) -> dict:
    if key not in SPECS:
        raise ModelUnavailableError(f"Unknown hazard model {key!r}.")
    return SPECS[key]


def checkpoint_path(key: str) -> Path:
    return hazard_model_dir() / _spec(key)["checkpoint"]


def load_metadata(key: str) -> dict:
    path = hazard_model_dir() / _spec(key)["metadata"]
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_model(key: str, force_reload: bool = False) -> LoadedHazardModel:
    spec = _spec(key)
    torch = _import_torch()
    path = checkpoint_path(key)
    if not path.exists():
        raise ModelUnavailableError(f"Model artifact not found: {path.name} (train it with `{spec['train']}`).")
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns, os.environ.get("NOWCAST_DEVICE"))
    with _lock:
        cached = _cache.get(key)
        if not force_reload and cached is not None and cached[0] == cache_key:
            return cached[1]

        from app.services.imagery.hazard_models import SmallUNet  # noqa: PLC0415

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # tensors + plain containers only
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(f"Model artifact is unreadable or corrupt ({type(exc).__name__}).") from exc
        try:
            model = SmallUNet(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["state_dict"])
            threshold = float(checkpoint["threshold"])
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"Model artifact does not match the expected architecture ({type(exc).__name__}: {str(exc)[:120]})."
            ) from exc
        if model.in_channels != spec["in_channels"] or model.out_stride != spec["out_stride"]:
            raise ModelUnavailableError("Model artifact was trained for a different input/output layout.")
        if not 0.0 < threshold < 1.0:
            raise ModelUnavailableError("Model artifact has an invalid decision threshold.")
        calibration = checkpoint.get("calibration")
        if calibration is not None and not (float(calibration.get("a", 0)) > 0 and np.isfinite(float(calibration.get("b", np.nan)))):
            raise ModelUnavailableError("Model artifact has invalid probability-calibration parameters.")
        if key == "flood":
            norm = checkpoint.get("normalisation") or {}
            mean, std = np.asarray(norm.get("mean", []), np.float32), np.asarray(norm.get("std", []), np.float32)
            if mean.shape != (2,) or std.shape != (2,) or not (std > 0).all():
                raise ModelUnavailableError("Model artifact has invalid normalisation statistics.")

        device = select_device(torch)
        try:
            model = model.to(device).eval()
        except Exception:  # noqa: BLE001 -- a GPU that is present but unusable
            device = torch.device("cpu")
            model = model.to(device).eval()
        metadata = load_metadata(key)
        metadata.setdefault("name", checkpoint.get("model_name", key))
        metadata.setdefault("version", checkpoint.get("model_version", "unknown"))
        loaded = LoadedHazardModel(key, model, device, checkpoint, metadata, path, stat.st_size)
        _cache[key] = (cache_key, loaded)
        return loaded


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def model_status(key: str) -> dict:
    """Never raises."""
    spec = SPECS.get(key)
    if spec is None:
        return {"model": key, "available": False, "reason": f"Unknown hazard model {key!r}."}
    base = {"model": key, "task": spec["task"], "classicalFallback": spec["classicalFallback"]}
    try:
        loaded = load_model(key)
    except ModelUnavailableError as exc:
        return {**base, "available": False, "reason": exc.reason}

    from app.services.imagery.hazard_models import ARCHITECTURE_NAME, count_parameters  # noqa: PLC0415

    torch = _import_torch()
    meta = loaded.metadata
    return {
        **base, "available": True,
        "name": meta.get("name"), "version": meta.get("version"),
        "architecture": meta.get("architecture", ARCHITECTURE_NAME),
        "parameters": count_parameters(loaded.model),
        "device": str(loaded.device), "torchVersion": str(torch.__version__),
        "artifactFile": loaded.path.name, "artifactSizeBytes": loaded.size_bytes,
        "threshold": round(decision_threshold(loaded), 4),
        "probabilityCalibration": ("Platt scaling fitted on real validation data" if loaded.checkpoint.get("calibration") else "none"),
        "trainedAt": meta.get("trainedAt"), "trainedOn": meta.get("trainedOn"),
        "trainingData": meta.get("trainingData"),
        "testMetrics": meta.get("testMetrics"), "syntheticMetrics": meta.get("syntheticMetrics"),
        "limitations": meta.get("limitations", []),
    }


def _calibrate(loaded: LoadedHazardModel, value):
    """Raw network score -> calibrated probability (identity if the artifact
    carries no calibration). Fitted on real validation data only."""
    calibration = loaded.checkpoint.get("calibration")
    if calibration is None:
        return value
    return apply_platt(value, float(calibration["a"]), float(calibration["b"]))


def decision_threshold(loaded: LoadedHazardModel) -> float:
    """The validation-tuned threshold, on the same (calibrated) scale as the
    probabilities this engine returns."""
    return float(_calibrate(loaded, float(loaded.checkpoint["threshold"])))


def _prepare(key: str, array: np.ndarray, loaded: LoadedHazardModel) -> np.ndarray:
    if key == "flood":
        flood_data.validate_s1(array)
        norm = loaded.checkpoint["normalisation"]
        return flood_data.normalise(array, np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32))
    lightning_data.validate_pair(array)
    return lightning_data.features(array)


def predict(key: str, array: np.ndarray, loaded: LoadedHazardModel | None = None) -> HazardPrediction:
    """`array`: flood -> (2, H, W) VV/VH backscatter in dB;
    lightning -> (2, H, W) brightness temperature in K (earlier, now).

    Raises the data module's 422 error for bad input and
    `ModelUnavailableError` if there is no usable model."""
    if key == "flood":
        flood_data.validate_s1(array)          # cheap, and before any model is loaded
    elif key == "lightning":
        lightning_data.validate_pair(array)
    loaded = loaded or load_model(key)
    torch = _import_torch()
    x = _prepare(key, array, loaded)
    started = time.perf_counter()

    def run(device):
        with torch.no_grad():
            return torch.sigmoid(loaded.model(torch.from_numpy(x)[None].to(device)))[0, 0].float().cpu().numpy()

    device = loaded.device
    try:
        probability = run(device)
    except RuntimeError:
        if str(device) == "cpu":
            raise
        device = torch.device("cpu")           # GPU out of memory / unsupported op: finish on CPU
        loaded.model.to(device)
        loaded.device = device
        probability = run(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not np.isfinite(probability).all():
        raise ModelUnavailableError("Model produced non-finite output; refusing to return it.")
    probability = np.asarray(_calibrate(loaded, probability), dtype=np.float32)
    return HazardPrediction(probability, decision_threshold(loaded), str(device), round(elapsed_ms, 1))
