"""The inference engine for the trained INSAT-3DR nowcast model: load once,
cache, pick a device, validate input, predict.

In the running backend this engine executes inside a dedicated worker
process (`nowcast_worker.py`, driven by `nowcast_client.py`), never in the
API process itself -- PyTorch and XGBoost cannot safely share a process on
macOS (see nowcast_worker.py). Importing this module is safe anywhere: torch
is only imported inside the functions that need it.

Failure philosophy -- the classical pipeline must keep working no matter
what state the model is in:
- torch not installed, artifact missing, artifact corrupt, artifact from an
  incompatible architecture -> `ModelUnavailableError` with a plain reason.
  `model_status()` reports it; the nowcast endpoint then serves the classical
  OpenCV optical-flow extrapolation instead and says so.
- A model is never silently substituted, and a classical result is never
  labelled as a model prediction.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.exceptions import AppError
from app.services.imagery.nowcast_common import (
    INPUT_CHANNELS,
    LEAD_MINUTES,
    LEAD_STEPS,
    N_INPUT_FRAMES,
    fill_nan,
)

BACKEND_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = BACKEND_ROOT / "ml" / "artifacts" / "imagery"
CHECKPOINT_NAME = "insat_nowcast_v1.pt"
METADATA_NAME = "insat_nowcast_v1_metadata.json"

MIN_SIDE_PX = 32
# Physically plausible brightness temperatures for these channels. Anything
# outside is not a calibrated INSAT thermal image (e.g. raw counts, or an
# 8-bit picture) and the model was never trained on it.
PLAUSIBLE_K = (150.0, 350.0)
MAX_NAN_FRACTION = 0.2


class ModelUnavailableError(Exception):
    """The trained model cannot be used right now; `reason` says why."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class InvalidNowcastInputError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=422)


@dataclass
class LoadedModel:
    model: object
    device: object
    mean: np.ndarray
    std: np.ndarray
    metadata: dict
    checkpoint_path: Path
    size_bytes: int
    loaded_at: float


@dataclass
class NowcastPrediction:
    mean_k: np.ndarray   # (n_leads, H, W) predicted TIR1 brightness temperature
    scale_k: np.ndarray  # (n_leads, H, W) Laplace scale = predicted mean absolute error
    device: str
    inference_ms: float


_lock = threading.Lock()
_cache: dict[tuple, LoadedModel] = {}


def checkpoint_path() -> Path:
    return Path(os.environ.get("NOWCAST_MODEL_DIR", MODEL_DIR)) / CHECKPOINT_NAME


def metadata_path() -> Path:
    return Path(os.environ.get("NOWCAST_MODEL_DIR", MODEL_DIR)) / METADATA_NAME


def _import_torch():
    try:
        import torch  # noqa: PLC0415 -- heavy optional dependency, imported lazily
    except ImportError as exc:
        raise ModelUnavailableError("PyTorch is not installed in this environment.") from exc
    return torch


def select_device(torch):
    """GPU when there is one (CUDA, then Apple MPS), CPU otherwise.
    `NOWCAST_DEVICE=cpu|mps|cuda` forces a choice."""
    forced = os.environ.get("NOWCAST_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_metadata() -> dict:
    path = metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_model(force_reload: bool = False) -> LoadedModel:
    """Load (or return the cached) model. Cache key includes the file's size
    and mtime (so replacing the artifact on disk is picked up without a
    restart) and any forced device."""
    torch = _import_torch()
    path = checkpoint_path()
    if not path.exists():
        raise ModelUnavailableError(f"Model artifact not found: {path.name} (train it with `python -m ml.imagery.train`).")
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, os.environ.get("NOWCAST_DEVICE"))

    with _lock:
        if not force_reload and key in _cache:
            return _cache[key]

        from app.services.imagery.nowcast_model import InsatNowcastNet  # noqa: PLC0415

        try:
            # weights_only=True: the file is tensors + plain containers, so
            # loading it can never execute pickled code.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(f"Model artifact is unreadable or corrupt ({type(exc).__name__}).") from exc

        try:
            model = InsatNowcastNet(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["state_dict"])
            mean = np.asarray(checkpoint["normalisation"]["mean"], dtype=np.float32)
            std = np.asarray(checkpoint["normalisation"]["std"], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"Model artifact does not match the expected architecture ({type(exc).__name__}: {str(exc)[:120]})."
            ) from exc
        if mean.shape != (len(INPUT_CHANNELS),) or std.shape != mean.shape or not (std > 0).all():
            raise ModelUnavailableError("Model artifact has invalid normalisation statistics.")
        if model.n_leads != len(LEAD_STEPS) or model.in_channels != len(INPUT_CHANNELS):
            raise ModelUnavailableError("Model artifact was trained for a different input/output layout.")

        device = select_device(torch)
        try:
            model = model.to(device).eval()
        except Exception:  # noqa: BLE001 -- e.g. a GPU that is present but unusable
            device = torch.device("cpu")
            model = model.to(device).eval()

        metadata = load_metadata()
        metadata.setdefault("name", checkpoint.get("model_name", "insat_nowcast"))
        metadata.setdefault("version", checkpoint.get("model_version", "unknown"))
        loaded = LoadedModel(model, device, mean, std, metadata, path, stat.st_size, time.time())
        _cache.clear()
        _cache[key] = loaded
        return loaded


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def model_status() -> dict:
    """Never raises -- this is what the UI polls to decide what to show."""
    base = {
        "task": "INSAT-3DR TIR1 (10.8 um) brightness-temperature nowcast, +30/+60/+90 min",
        "leadMinutes": list(LEAD_MINUTES),
        "inputChannels": list(INPUT_CHANNELS),
        "inputFrames": N_INPUT_FRAMES,
        "classicalFallback": "OpenCV Farneback optical-flow extrapolation (always available)",
    }
    try:
        loaded = load_model()
    except ModelUnavailableError as exc:
        return {**base, "available": False, "reason": exc.reason}

    from app.services.imagery.nowcast_model import ARCHITECTURE_NAME, count_parameters  # noqa: PLC0415

    torch = _import_torch()
    meta = loaded.metadata
    return {
        **base,
        "available": True,
        "name": meta.get("name"),
        "version": meta.get("version"),
        "architecture": meta.get("architecture", ARCHITECTURE_NAME),
        "parameters": count_parameters(loaded.model),
        "device": str(loaded.device),
        "torchVersion": str(torch.__version__),  # plain str: TorchVersion would drag torch into whoever unpickles it
        "artifactFile": loaded.checkpoint_path.name,
        "artifactSizeBytes": loaded.size_bytes,
        "trainedAt": meta.get("trainedAt"),
        "trainingData": meta.get("trainingData"),
        "testMetrics": meta.get("testMetrics"),
        "calibration": meta.get("calibration"),
        "limitations": meta.get("limitations", []),
    }


def validate_frames(frames_k: np.ndarray) -> None:
    """`frames_k`: (T, C, H, W) brightness temperature in kelvin."""
    if not isinstance(frames_k, np.ndarray) or frames_k.ndim != 4:
        raise InvalidNowcastInputError("Nowcast input must be a 4-D array (frames, channels, rows, cols).")
    steps, channels, height, width = frames_k.shape
    if steps != N_INPUT_FRAMES:
        raise InvalidNowcastInputError(f"Exactly {N_INPUT_FRAMES} consecutive frames are required, got {steps}.")
    if channels != len(INPUT_CHANNELS):
        raise InvalidNowcastInputError(f"Each frame needs {len(INPUT_CHANNELS)} channels {INPUT_CHANNELS}, got {channels}.")
    if height < MIN_SIDE_PX or width < MIN_SIDE_PX:
        raise InvalidNowcastInputError(f"Image is too small ({height}x{width}); at least {MIN_SIDE_PX}x{MIN_SIDE_PX} pixels are needed.")
    if not np.issubdtype(frames_k.dtype, np.floating):
        raise InvalidNowcastInputError("Nowcast input must be floating-point brightness temperature in kelvin.")
    nan_fraction = float(np.isnan(frames_k).mean())
    if nan_fraction > MAX_NAN_FRACTION:
        raise InvalidNowcastInputError(f"{100 * nan_fraction:.0f}% of the input is missing data; too much to predict from.")
    if np.isinf(frames_k).any():
        raise InvalidNowcastInputError("Nowcast input contains infinite values.")
    lo, hi = float(np.nanmin(frames_k)), float(np.nanmax(frames_k))
    if lo < PLAUSIBLE_K[0] or hi > PLAUSIBLE_K[1]:
        raise InvalidNowcastInputError(
            f"Values span {lo:.0f}-{hi:.0f}, outside the plausible brightness-temperature range "
            f"{PLAUSIBLE_K[0]:.0f}-{PLAUSIBLE_K[1]:.0f} K. The model only accepts calibrated INSAT-3DR thermal imagery."
        )


def normalise(frames_k: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Identical arithmetic to training (`ml.imagery.dataset.FrameStore`)."""
    filled = np.stack([np.stack([fill_nan(channel) for channel in frame]) for frame in frames_k]).astype(np.float32)
    return (filled - mean[None, :, None, None]) / std[None, :, None, None]


def predict(frames_k: np.ndarray, loaded: LoadedModel | None = None) -> NowcastPrediction:
    """Predict TIR1 brightness temperature at each lead time.

    Raises `InvalidNowcastInputError` for bad input and
    `ModelUnavailableError` if there is no usable model."""
    validate_frames(frames_k)
    loaded = loaded or load_model()
    torch = _import_torch()
    from app.services.imagery.nowcast_model import tiled_forward  # noqa: PLC0415

    x = normalise(frames_k, loaded.mean, loaded.std)
    started = time.perf_counter()
    device = loaded.device
    try:
        mean, scale = tiled_forward(loaded.model, x, device)
    except RuntimeError:
        if str(device) == "cpu":
            raise
        # GPU ran out of memory or hit an unsupported op: finish on CPU.
        device = torch.device("cpu")
        loaded.model.to(device)
        loaded.device = device
        mean, scale = tiled_forward(loaded.model, x, device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    mean_k = mean * loaded.std[0] + loaded.mean[0]
    scale_k = scale * loaded.std[0]
    if not (np.isfinite(mean_k).all() and np.isfinite(scale_k).all()):
        raise ModelUnavailableError("Model produced non-finite output; refusing to return it.")
    return NowcastPrediction(mean_k.astype(np.float32), scale_k.astype(np.float32), str(device), round(elapsed_ms, 1))
