"""Training orchestrator. Run as `python -m ml.train` from `backend/`
(inside the venv with pandas/scikit-learn/xgboost installed).

Scope (documented, not a shortcut -- see docs/model_report.md "Scope"): the
full 5-algorithm sweep (persistence, linear/logistic, random_forest,
xgboost, mlp) runs at the flagship 3-hour horizon for all four targets. The
1h and 6h horizons run only {persistence, xgboost} -- the mandatory naive
baseline and the model the flagship comparison already identifies as the
best candidate -- since a full repeat sweep at every horizon would not
change the architecture conclusion, only cost more compute.

For each (target, horizon), the best non-persistence model by validation
score is saved as the deployable artifact; every algorithm's val/test
metrics (persistence included) are kept in model_comparison.json as the
full comparison record the report is built from.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.preprocessing import StandardScaler

from ml import config, evaluate, models, preprocessing, targets


def prepare_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    subset = df[feature_cols + [target_col]].dropna()
    return subset[feature_cols], subset[target_col]


def persistence_predict(X: pd.DataFrame, target_name: str) -> np.ndarray:
    """The naive baseline: assume whatever is true right now stays true at
    t+horizon. No fitting, no parameters -- this is exactly what every other
    model must outperform to justify existing.
    """
    if target_name == "temperature_c":
        return X["temperature_c"].to_numpy()
    if target_name == "rainfall_mm_hour":
        return X["rainfall_mm_hour"].to_numpy()
    if target_name == "heavy_rain_risk":
        return (X["rainfall_mm_hour"].to_numpy() >= config.HEAVY_RAINFALL_MM_HR).astype(float)
    if target_name == "thunderstorm_risk":
        code_hit = X["weather_code"].isin(config.THUNDERSTORM_WEATHER_CODES).to_numpy()
        cape_li_hit = (X["cape_j_kg"].to_numpy() >= 1000) & (X["lifted_index"].to_numpy() <= -3)
        return (code_hit | cape_li_hit).astype(float)
    raise ValueError(target_name)


def algorithms_for_horizon(horizon: int) -> list[str]:
    return models.ALL_ALGORITHMS if horizon == config.FLAGSHIP_HORIZON else models.REDUCED_ALGORITHMS


def run_one(
    algorithm: str,
    target_name: str,
    is_classification: bool,
    X_train, y_train, X_val, y_val, X_test, y_test,
    scaler: StandardScaler,
):
    if algorithm == "persistence":
        val_pred = persistence_predict(X_val, target_name)
        test_pred = persistence_predict(X_test, target_name)
        val_prob = val_pred if is_classification else None
        test_prob = test_pred if is_classification else None
        fitted = None
    else:
        Xtr = scaler.transform(X_train)
        Xv = scaler.transform(X_val)
        Xte = scaler.transform(X_test)
        if is_classification:
            pos = float(y_train.sum())
            neg = float(len(y_train) - pos)
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0
            if algorithm == "xgboost":
                model = models.get_classifier(algorithm, scale_pos_weight=scale_pos_weight)
            else:
                model = models.get_classifier(algorithm)
            model.fit(Xtr, y_train)
            val_pred = model.predict(Xv)
            test_pred = model.predict(Xte)
            val_prob = model.predict_proba(Xv)[:, 1]
            test_prob = model.predict_proba(Xte)[:, 1]
        else:
            model = models.get_regressor(algorithm)
            model.fit(Xtr, y_train)
            val_pred = model.predict(Xv)
            test_pred = model.predict(Xte)
            val_prob = None
            test_prob = None
        fitted = model

    if is_classification:
        val_metrics = evaluate.classification_metrics(y_val, val_pred, val_prob)
        test_metrics = evaluate.classification_metrics(y_test, test_pred, test_prob)
    else:
        val_metrics = evaluate.regression_metrics(y_val, val_pred)
        test_metrics = evaluate.regression_metrics(y_test, test_pred)

    return fitted, val_metrics, test_metrics


def selection_score(is_classification: bool, val_metrics: dict) -> float:
    """Higher is better, for picking the deployable model. PR-AUC for
    classification (meaningful under the heavy imbalance here, unlike raw
    accuracy); negative RMSE for regression.
    """
    if is_classification:
        pr_auc = val_metrics.get("pr_auc")
        return pr_auc if pr_auc is not None else val_metrics["f1"]
    return -val_metrics["rmse"]


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and engineering features...", flush=True)
    raw = preprocessing.load_raw()
    engineered = preprocessing.engineer_features(raw)
    feature_cols = preprocessing.feature_columns(engineered)
    print(f"{len(engineered)} rows, {len(feature_cols)} features", flush=True)

    comparison: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "random_seed": config.RANDOM_SEED,
        "split": {
            "train_end_date": config.TRAIN_END_DATE,
            "validation_end_date": config.VALIDATION_END_DATE,
        },
        "results": {},
    }

    all_targets = [(t, False) for t in config.TARGET_REGRESSION] + [
        (t, True) for t in config.TARGET_CLASSIFICATION
    ]

    for horizon in config.HORIZONS:
        print(f"\n=== horizon = {horizon}h ===", flush=True)
        with_targets = targets.build_targets(engineered, horizon)
        train_df, val_df, test_df = preprocessing.chronological_split(with_targets)

        for target_name, is_classification in all_targets:
            target_col = targets.target_column(target_name, horizon)
            X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
            X_val, y_val = prepare_xy(val_df, feature_cols, target_col)
            X_test, y_test = prepare_xy(test_df, feature_cols, target_col)

            scaler = StandardScaler().fit(X_train)

            algo_results = {}
            best = {"algorithm": None, "score": -np.inf, "model": None, "val": None, "test": None}
            for algorithm in algorithms_for_horizon(horizon):
                print(f"  [{target_name} h={horizon}] fitting {algorithm}...", flush=True)
                fitted, val_metrics, test_metrics = run_one(
                    algorithm, target_name, is_classification,
                    X_train, y_train, X_val, y_val, X_test, y_test, scaler,
                )
                algo_results[algorithm] = {"val": val_metrics, "test": test_metrics}
                if algorithm != "persistence":
                    score = selection_score(is_classification, val_metrics)
                    if score > best["score"]:
                        best = {
                            "algorithm": algorithm, "score": score, "model": fitted,
                            "val": val_metrics, "test": test_metrics,
                        }

            comparison["results"].setdefault(target_name, {})[str(horizon)] = {
                "algorithms": algo_results,
                "train_rows": int(len(X_train)),
                "val_rows": int(len(X_val)),
                "test_rows": int(len(X_test)),
                "selected_model": best["algorithm"],
            }

            if best["model"] is not None:
                model_path = config.ARTIFACTS_DIR / f"{target_name}_h{horizon}_model.joblib"
                scaler_path = config.ARTIFACTS_DIR / f"{target_name}_h{horizon}_scaler.joblib"
                joblib.dump(best["model"], model_path)
                joblib.dump(scaler, scaler_path)

                metadata = {
                    "target": target_name,
                    "horizon_hours": horizon,
                    "is_classification": is_classification,
                    "algorithm": best["algorithm"],
                    "model_version": "ml-v1",
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "feature_columns": feature_cols,
                    "val_metrics": best["val"],
                    "test_metrics": best["test"],
                    "train_rows": int(len(X_train)),
                    "val_rows": int(len(X_val)),
                    "test_rows": int(len(X_test)),
                    "model_path": model_path.name,
                    "scaler_path": scaler_path.name,
                    "random_seed": config.RANDOM_SEED,
                    "sklearn_version": sklearn.__version__,
                    "xgboost_version": xgboost.__version__,
                }
                metadata_path = config.ARTIFACTS_DIR / f"{target_name}_h{horizon}_metadata.json"
                metadata_path.write_text(json.dumps(metadata, indent=2))
                print(
                    f"  -> selected {best['algorithm']} for {target_name} h={horizon} "
                    f"(val score={best['score']:.4f}); saved artifacts",
                    flush=True,
                )

    comparison_path = config.ARTIFACTS_DIR / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nWrote {comparison_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
