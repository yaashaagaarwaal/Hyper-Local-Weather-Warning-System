"""Main-process handle on the trained nowcast model.

The API process calls `model_status()` / `predict()` here; the actual
PyTorch work happens in a child process (`nowcast_worker.py`) that is
started on first use and kept alive, so the model is loaded once and stays
cached between requests. This module never imports torch -- see
nowcast_worker.py for why that matters.

Every failure (worker cannot start, crashes, times out, model missing or
corrupt) surfaces as `ModelUnavailableError`, which callers turn into the
classical OpenCV fallback. A dead worker is replaced on the next call.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from app.services.imagery.ml_inference import (
    InvalidNowcastInputError,
    ModelUnavailableError,
    NowcastPrediction,
    validate_frames,
)
from app.services.imagery.nowcast_worker import read_message, write_message

BACKEND_ROOT = Path(__file__).resolve().parents[3]
STATUS_TIMEOUT_S = 120.0   # includes the one-off cost of importing torch and loading the model
PREDICT_TIMEOUT_S = 240.0

_lock = threading.Lock()
_process: subprocess.Popen | None = None


def _start() -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(BACKEND_ROOT), "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(  # noqa: S603 -- fixed argv, our own module
        [sys.executable, "-m", "app.services.imagery.nowcast_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=str(BACKEND_ROOT), env=env,
    )


def _stop_locked() -> None:
    global _process
    if _process is None:
        return
    try:
        _process.kill()
        _process.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass
    for stream in (_process.stdin, _process.stdout):
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass
    _process = None


def shutdown() -> None:
    with _lock:
        _stop_locked()


atexit.register(shutdown)


def worker_pid() -> int | None:
    return _process.pid if _process is not None and _process.poll() is None else None


def _request(message: dict, timeout: float) -> dict:
    """One request/response round trip; serialised, with a hard timeout."""
    global _process
    message = {**message, "env": {key: os.environ.get(key) for key in ("NOWCAST_MODEL_DIR", "NOWCAST_DEVICE")}}
    with _lock:
        if _process is None or _process.poll() is not None:
            _stop_locked()
            try:
                _process = _start()
            except OSError as exc:
                raise ModelUnavailableError(f"Could not start the inference worker ({exc}).") from exc
        process = _process
        result: dict = {}

        def exchange() -> None:
            try:
                write_message(process.stdin, message)
                result["reply"] = read_message(process.stdout)
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        thread = threading.Thread(target=exchange, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            _stop_locked()
            raise ModelUnavailableError(f"The inference worker did not answer within {timeout:.0f} s and was restarted.")
        if "error" in result or result.get("reply") is None:
            _stop_locked()
            raise ModelUnavailableError("The inference worker stopped unexpectedly; it will be restarted on the next request.")
        return result["reply"]


def _raise_for(reply: dict) -> None:
    if reply.get("ok"):
        return
    if reply.get("kind") == "invalid":
        raise InvalidNowcastInputError(reply.get("message", "invalid input"))
    raise ModelUnavailableError(reply.get("message", "model unavailable"))


def model_status() -> dict:
    """Never raises."""
    try:
        reply = _request({"op": "status"}, STATUS_TIMEOUT_S)
        _raise_for(reply)
        return {**reply["status"], "isolatedWorker": True}
    except ModelUnavailableError as exc:
        return {
            "available": False, "reason": exc.reason,
            "classicalFallback": "OpenCV Farneback optical-flow extrapolation (always available)",
        }


def predict(frames_k: np.ndarray) -> tuple[NowcastPrediction, dict]:
    """Returns (prediction, {name, version, architecture}). Input is
    validated here first, so bad input never costs a round trip."""
    validate_frames(frames_k)
    reply = _request({"op": "predict", "frames": np.ascontiguousarray(frames_k, dtype=np.float32)}, PREDICT_TIMEOUT_S)
    _raise_for(reply)
    prediction = NowcastPrediction(reply["mean_k"], reply["scale_k"], reply["device"], reply["inference_ms"])
    return prediction, reply["metadata"]
