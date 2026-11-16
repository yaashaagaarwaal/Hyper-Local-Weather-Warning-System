"""Shared, torch-free pieces of the INSAT-3DR infrared nowcast: the task
definition, normalisation, the two non-learned baselines, and the metrics.

Imported by both the training/evaluation code (`backend/ml/imagery/`) and
the serving code (`ml_inference.py`) so that "what the task is" and "how it
is scored" cannot drift apart between the two.

The task
--------
Given the last `N_INPUT_FRAMES` half-hourly frames of TIR1 (10.8 um) and WV
(6.9 um) brightness temperature, predict the TIR1 brightness-temperature
image 30, 60 and 90 minutes later. The "label" is simply the satellite's
own later observation, so this is self-supervised: no human annotation is
needed, and none exists in this dataset.

This is short-range extrapolation of *satellite imagery*. It is not a
weather forecast, a rainfall estimate, or a storm warning.
"""
from __future__ import annotations

import cv2
import numpy as np

INPUT_CHANNELS = ("TIR1", "WV")
TARGET_CHANNEL = "TIR1"
N_INPUT_FRAMES = 3  # t-60, t-30, t
LEAD_STEPS = (1, 2, 3)  # x 30 minutes
LEAD_MINUTES = tuple(step * 30 for step in LEAD_STEPS)

# Cold-cloud-top threshold on 10.8 um brightness temperature. 235 K is the
# long-established threshold of the GOES Precipitation Index (Arkin &
# Meisner 1987) for "cold cloud" associated with deep convection. Used here
# only to turn a temperature image into a binary mask so that a prediction
# can be scored with IoU / precision / recall. It is a cloud-top-temperature
# criterion, not a rainfall or storm detection.
COLD_CLOUD_THRESHOLD_K = 235.0

# Farneback parameters -- deliberately the same call as the existing
# classical pipeline's `processing.optical_flow`.
_FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
_BT_RENDER_RANGE = (180.0, 320.0)


def bt_to_uint8(bt: np.ndarray) -> np.ndarray:
    lo, hi = _BT_RENDER_RANGE
    return np.clip((np.nan_to_num(bt, nan=hi) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def fill_nan(frame: np.ndarray) -> np.ndarray:
    """Replace NaN (fill) pixels with the frame's own median so that
    convolution/optical flow have finite input. Fill pixels are essentially
    absent from this dataset (0 of ~165 M inspected), so this is a guard,
    not a routine imputation."""
    if not np.isnan(frame).any():
        return frame
    out = frame.copy()
    out[np.isnan(out)] = np.nanmedian(frame) if np.isfinite(frame).any() else 273.15
    return out


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def persistence_forecast(last: np.ndarray, n_leads: int = len(LEAD_STEPS)) -> list[np.ndarray]:
    """'Nothing changes': the last observed frame, repeated."""
    return [last for _ in range(n_leads)]


def farneback_flow(prev: np.ndarray, last: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(bt_to_uint8(prev), bt_to_uint8(last), None, **_FARNEBACK)


def optical_flow_forecast(prev: np.ndarray, last: np.ndarray, lead_steps=LEAD_STEPS) -> list[np.ndarray]:
    """Classical OpenCV nowcast: estimate dense motion between the last two
    frames (Farneback), then move the last frame further along that motion
    field (semi-Lagrangian backward warp), one field for all lead times."""
    prev, last = fill_nan(prev).astype(np.float32), fill_nan(last).astype(np.float32)
    flow = farneback_flow(prev, last)
    height, width = last.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    forecasts = []
    for step in lead_steps:
        forecasts.append(cv2.remap(
            last, grid_x - step * flow[..., 0], grid_y - step * flow[..., 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        ))
    return forecasts


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    valid = np.isfinite(pred) & np.isfinite(truth)
    if not valid.any():
        return {"maeK": None, "rmseK": None, "biasK": None, "pixels": 0}
    err = (pred[valid] - truth[valid]).astype(np.float64)
    return {
        "maeK": float(np.abs(err).mean()),
        "rmseK": float(np.sqrt((err ** 2).mean())),
        "biasK": float(err.mean()),
        "pixels": int(valid.sum()),
    }


def mask_counts(pred_mask: np.ndarray, truth_mask: np.ndarray) -> dict:
    tp = int(np.logical_and(pred_mask, truth_mask).sum())
    fp = int(np.logical_and(pred_mask, ~truth_mask).sum())
    fn = int(np.logical_and(~pred_mask, truth_mask).sum())
    tn = int(np.logical_and(~pred_mask, ~truth_mask).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def mask_metrics_from_counts(counts: dict) -> dict:
    """IoU (identical to the Critical Success Index used in nowcast
    verification), Dice (identical to F1), precision, recall, accuracy.
    A metric whose denominator is zero is reported as None, never as 0 or 1."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

    def ratio(num: float, den: float):
        return float(num / den) if den > 0 else None

    return {
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "truthPositivePct": ratio(100.0 * (tp + fn), tp + fp + fn + tn),
        **counts,
    }


def cold_cloud_metrics(pred_bt: np.ndarray, truth_bt: np.ndarray, threshold: float = COLD_CLOUD_THRESHOLD_K) -> dict:
    valid = np.isfinite(pred_bt) & np.isfinite(truth_bt)
    return mask_metrics_from_counts(mask_counts((pred_bt < threshold) & valid, (truth_bt < threshold) & valid))


def laplace_cdf(threshold: float, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """P(value < threshold) under Laplace(mean, scale) -- the model's
    predictive distribution, so this is its probability that a pixel is
    colder than `threshold`."""
    z = (threshold - mean) / np.maximum(scale, 1e-6)
    tail = 0.5 * np.exp(-np.abs(z))  # always <= 0.5, never overflows
    return np.where(z < 0, tail, 1.0 - tail).astype(np.float32)
