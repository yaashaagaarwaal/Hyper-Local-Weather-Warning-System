"""Phase 2 Step 1: validation-only decision-threshold tuning for the two
classification targets (heavy_rain_risk, thunderstorm_risk), independently
per horizon. Never fits a new model -- loads the existing selected
artifact and only chooses the probability cutoff applied to its output.

Never touches the test split: thresholds are selected purely from each
model's own validation-set precision-recall curve, then written into that
model's existing metadata JSON as `decision_threshold` (read by
app/services/prediction_service.py at serving time instead of the
previous hardcoded 0.5).

Selection criterion: maximize F-beta with beta=2 (recall weighted above
precision) on the validation PR curve -- a missed severe-weather warning is
a worse operational outcome than a false alarm, consistent with how
alert_service.py already treats these two hazards. Both the tuned operating
point and the previous naive-0.5 point are recorded, so the before/after is
always visible in the metadata, not just asserted.

Run as `python -m ml.threshold_tuning` from `backend/`.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

from ml import config, preprocessing, targets
from ml.train import prepare_xy

BETA = 2.0


def fbeta_from_pr(precision: np.ndarray, recall: np.ndarray, beta: float = BETA) -> np.ndarray:
    beta2 = beta * beta
    denom = beta2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denom > 0, (1 + beta2) * precision * recall / denom, 0.0)
    return score


def best_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_val, prob_val)
    # sklearn appends one extra (precision=1, recall=0) point with no
    # matching threshold -- drop it so the arrays line up with `thresholds`.
    precision, recall = precision[:-1], recall[:-1]
    scores = fbeta_from_pr(precision, recall)
    best_idx = int(np.argmax(scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f2": float(scores[best_idx]),
    }


def naive_threshold_metrics(y_val: np.ndarray, prob_val: np.ndarray, cutoff: float = 0.5) -> dict:
    pred = (prob_val >= cutoff).astype(int)
    return {
        "threshold": cutoff,
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
    }


def main() -> list[dict]:
    print("Loading and engineering features...", flush=True)
    raw = preprocessing.load_raw()
    engineered = preprocessing.engineer_features(raw)
    feature_cols = preprocessing.feature_columns(engineered)

    summary = []
    for horizon in config.HORIZONS:
        with_targets = targets.build_targets(engineered, horizon)
        _, val_df, _ = preprocessing.chronological_split(with_targets)

        for target_name in config.TARGET_CLASSIFICATION:
            target_col = targets.target_column(target_name, horizon)
            X_val, y_val = prepare_xy(val_df, feature_cols, target_col)

            meta_path = config.ARTIFACTS_DIR / f"{target_name}_h{horizon}_metadata.json"
            metadata = json.loads(meta_path.read_text())
            model = joblib.load(config.ARTIFACTS_DIR / metadata["model_path"])
            scaler = joblib.load(config.ARTIFACTS_DIR / metadata["scaler_path"])

            prob_val = model.predict_proba(scaler.transform(X_val))[:, 1]
            y_val_np = y_val.to_numpy()

            tuned = best_threshold(y_val_np, prob_val)
            naive = naive_threshold_metrics(y_val_np, prob_val)

            metadata["decision_threshold"] = tuned["threshold"]
            metadata["threshold_selection"] = {
                "method": "f2_max_on_validation_pr_curve",
                "beta": BETA,
                "val_metrics_at_tuned_threshold": tuned,
                "val_metrics_at_naive_0.5": naive,
            }
            meta_path.write_text(json.dumps(metadata, indent=2))

            row = {"target": target_name, "horizon": horizon, "naive": naive, "tuned": tuned}
            summary.append(row)
            print(
                f"{target_name} h={horizon}: naive@0.5 P={naive['precision']:.3f} R={naive['recall']:.3f} "
                f"F1={naive['f1']:.3f}  ->  tuned@{tuned['threshold']:.3f} P={tuned['precision']:.3f} "
                f"R={tuned['recall']:.3f} F2={tuned['f2']:.3f}",
                flush=True,
            )

    return summary


if __name__ == "__main__":
    main()
