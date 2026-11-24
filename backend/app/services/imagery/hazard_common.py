"""Torch-free pieces shared by the flood and lightning image models:
paths, binary-mask metrics, and the provenance vocabulary that keeps real
observations and synthetic simulations apart everywhere (catalogs, API
responses, evaluation reports).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BACKEND_ROOT.parent
HAZARD_MODEL_DIR = BACKEND_ROOT / "ml" / "artifacts" / "hazards"
SYNTHETIC_ROOT = REPO_ROOT / "dataset" / "synthetic"

SOURCE_REAL = "real"
SOURCE_SYNTHETIC = "synthetic"

SYNTHETIC_NOTICE = (
    "SYNTHETIC SIMULATION -- computer-generated, not a satellite observation. "
    "Scores on synthetic samples only show agreement with the simulator's own rules."
)


def hazard_model_dir() -> Path:
    return Path(os.environ.get("HAZARD_MODEL_DIR", HAZARD_MODEL_DIR))


def confusion_counts(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """`pred`, `truth`: boolean arrays. `valid`: pixels that count."""
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    if valid is None:
        valid = np.ones(pred.shape, dtype=bool)
    return {
        "tp": int((pred & truth & valid).sum()), "fp": int((pred & ~truth & valid).sum()),
        "fn": int((~pred & truth & valid).sum()), "tn": int((~pred & ~truth & valid).sum()),
    }


def add_counts(total: dict, part: dict) -> dict:
    for key in ("tp", "fp", "fn", "tn"):
        total[key] = total.get(key, 0) + part[key]
    return total


def metrics_from_counts(c: dict) -> dict:
    """Positive-class scores. `None` where a ratio is undefined (no
    positives predicted / present) rather than a made-up 0 or 1."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    total = tp + fp + fn + tn

    def ratio(a: int, b: int):
        return a / b if b else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "iou": ratio(tp, tp + fp + fn),             # = Critical Success Index
        "f1": ratio(2 * tp, 2 * tp + fp + fn),      # = Dice
        "precision": precision,
        "recall": recall,                           # = probability of detection
        "falseAlarmRatio": (1 - precision) if precision is not None else None,
        "omissionRate": (1 - recall) if recall is not None else None,
        "accuracy": ratio(tp + tn, total),
        "positiveFraction": ratio(tp + fn, total),
        "pixels": total, **c,
    }


def rounded(metrics: dict, digits: int = 4) -> dict:
    return {k: (round(v, digits) if isinstance(v, float) else v) for k, v in metrics.items()}


def roc_pr_auc(scores: np.ndarray, truth: np.ndarray, bins: int = 2000) -> dict:
    """ROC-AUC and average precision from a score histogram -- exact enough
    for millions of pixels without sorting them all."""
    scores = np.clip(np.asarray(scores, dtype=np.float64).ravel(), 0.0, 1.0)
    truth = np.asarray(truth, dtype=bool).ravel()
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    if n_pos == 0 or n_neg == 0:
        return {"rocAuc": None, "prAuc": None}
    edges = np.linspace(0.0, 1.0, bins + 1)
    pos = np.histogram(scores[truth], edges)[0][::-1].cumsum()
    neg = np.histogram(scores[~truth], edges)[0][::-1].cumsum()
    tpr = np.concatenate([[0.0], pos / n_pos])
    fpr = np.concatenate([[0.0], neg / n_neg])
    precision = np.concatenate([[1.0], pos / np.maximum(pos + neg, 1)])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return {
        "rocAuc": float(trapezoid(tpr, fpr)),
        "prAuc": float(np.sum(np.diff(tpr) * precision[1:])),
    }


def brier_score(scores: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.asarray(scores, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) ** 2))


def reliability_table(scores: np.ndarray, truth: np.ndarray, bins: int = 10) -> list[dict]:
    scores, truth = np.asarray(scores).ravel(), np.asarray(truth, dtype=bool).ravel()
    index = np.minimum((scores * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        members = index == b
        count = int(members.sum())
        rows.append({
            "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "count": count,
            "meanPredicted": float(scores[members].mean()) if count else None,
            "observedFrequency": float(truth[members].mean()) if count else None,
        })
    return rows


def apply_platt(probability: np.ndarray | float, a: float, b: float):
    """Platt scaling: p' = sigmoid(a * logit(p) + b). Strictly increasing in
    p (for a > 0), so it changes how confident a score claims to be without
    changing which pixels rank above which."""
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    z = a * np.log(p / (1 - p)) + b
    return 1.0 / (1.0 + np.exp(-z))
