"""Evaluation metrics, kept separate from training so the same functions
score every algorithm (including the non-fitted persistence baseline)
identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,  # [[tn, fp], [fn, tp]]
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    # ROC-AUC/PR-AUC need both classes present and a real probability score
    # (not persistence's degenerate 0/1 "probability") to be meaningful.
    if len(np.unique(y_true)) == 2 and y_prob is not None and len(np.unique(y_prob)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        # PR-AUC is reported alongside ROC-AUC specifically because these
        # hazard targets are rare-event (<=12% positive, heavy rain <1%) --
        # ROC-AUC alone can look deceptively good under severe imbalance.
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics
