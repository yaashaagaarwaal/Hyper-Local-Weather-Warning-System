"""Inference worker: the ONLY process in the running backend that imports
PyTorch.

    python -m app.services.imagery.nowcast_worker     (started by nowcast_client)

Why a separate process: PyTorch and XGBoost (which serves AtmoGuard's
weather-prediction models in the main process) each load their own copy of
the OpenMP runtime on macOS. Verified on this project: using both in one
process segfaults (torch first) or deadlocks (xgboost first). Keeping torch
out of the API process removes the conflict entirely rather than relying on
import order.

Protocol: length-prefixed pickles over this process's stdin/stdout. Both
ends are this project's own code on the same machine, and nothing from an
HTTP request is ever unpickled. Messages may contain only builtins and numpy
arrays -- enforced by a restricted unpickler, which also guarantees that
reading a reply can never import torch into the API process.
"""
from __future__ import annotations

import io
import os
import pickle
import struct
import sys

_HEADER = struct.Struct("<Q")


class _PlainUnpickler(pickle.Unpickler):
    """Builtins + numpy only."""

    def find_class(self, module: str, name: str):
        if module == "builtins" or module == "numpy" or module.startswith("numpy."):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}: only builtins and numpy are allowed")


def read_message(stream):
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        return None
    (length,) = _HEADER.unpack(header)
    payload = stream.read(length)
    if len(payload) < length:
        return None
    return _PlainUnpickler(io.BytesIO(payload)).load()


def write_message(stream, message) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _apply_env(request: dict) -> None:
    """The parent forwards its own model-dir/device settings with every
    request, so the worker always acts on the caller's current config."""
    for key in ("NOWCAST_MODEL_DIR", "NOWCAST_DEVICE"):
        value = request.get("env", {}).get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def handle(request: dict) -> dict:
    from app.services.imagery import ml_inference  # noqa: PLC0415

    _apply_env(request)
    op = request.get("op")
    try:
        if op == "ping":
            return {"ok": True, "pid": os.getpid()}
        if op == "status":
            return {"ok": True, "status": ml_inference.model_status()}
        if op == "predict":
            loaded = ml_inference.load_model()
            prediction = ml_inference.predict(request["frames"], loaded)
            return {
                "ok": True,
                "mean_k": prediction.mean_k, "scale_k": prediction.scale_k,
                "device": prediction.device, "inference_ms": prediction.inference_ms,
                "metadata": {k: loaded.metadata.get(k) for k in ("name", "version", "architecture")},
            }
        return {"ok": False, "kind": "error", "message": f"unknown op {op!r}"}
    except ml_inference.ModelUnavailableError as exc:
        return {"ok": False, "kind": "unavailable", "message": exc.reason}
    except ml_inference.InvalidNowcastInputError as exc:
        return {"ok": False, "kind": "invalid", "message": exc.message}
    except Exception as exc:  # noqa: BLE001 -- report, don't die; the parent decides what to do
        return {"ok": False, "kind": "error", "message": f"{type(exc).__name__}: {str(exc)[:300]}"}


def main() -> None:
    # Keep the protocol channel clean: anything a library prints goes to stderr.
    channel_in = sys.stdin.buffer
    channel_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    sys.stdout = sys.stderr
    while True:
        request = read_message(channel_in)
        if request is None or request.get("op") == "shutdown":
            break
        write_message(channel_out, handle(request))


if __name__ == "__main__":
    main()
