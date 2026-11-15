"""Phase 2 Step 4: feature selection for rainfall_mm_hour (regression) and
heavy_rain_risk (classification) only -- temperature_c and thunderstorm_risk
are untouched, per instruction.

Method: permutation importance (`sklearn.inspection.permutation_importance`,
scored on VALIDATION predictions, never training predictions) is used as the
primary ranking signal rather than raw model-based importance -- Step 3's
own finding (multicollinear features split or inflate a linear model's
coefficients) means a model's native coefficient/feature_importances_ can
rank a feature highly for reasons that have nothing to do with how much it
actually helps predict the held-out validation rows. Native importance
(XGBoost's .feature_importances_, or |coefficient| for the linear model) is
still computed and reported alongside, for comparison, never as the
selection criterion.

Every candidate reduced feature set is refit with EXACTLY the currently
deployed algorithm and its default hyperparameters (`ml.models.get_regressor`
/ `get_classifier`, the same calls train.py itself uses) -- this step changes
which inputs a model sees, never which algorithm or hyperparameters it uses,
per "preserve the existing model algorithms and tuned thresholds".

Time-bounding: only 3 candidate sizes are tried per (target, horizon)
(roughly a quarter, a third, and two-thirds of the full feature count) --
not a dense sweep -- and permutation importance uses n_repeats=5 on the
validation split only (~20-44k rows depending on horizon), never training or
test.

A reduced set is only promoted if it beats the full-119-feature baseline's
OWN validation score by IMPROVEMENT_MARGIN -- parity is not enough ("only
promote if it improves ... reliably"). If promoted for heavy_rain_risk, the
decision threshold is re-tuned (same F2-max method as Step 1/3) for the new
feature set; rainfall_mm_hour has no threshold concept. The test split is
read only for a promoted candidate, after selection.

Run as `python -m ml.feature_selection` from `backend/`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from ml import config, evaluate, models, preprocessing, targets
from ml.threshold_tuning import best_threshold
from ml.train import prepare_xy, selection_score

TARGETS = [("rainfall_mm_hour", False), ("heavy_rain_risk", True)]
IMPROVEMENT_MARGIN = 0.01
N_REPEATS = 5
CANDIDATE_FRACTIONS = [0.25, 0.4, 0.65]

EXPERIMENTS_DIR = config.ARTIFACTS_DIR.parent / "artifacts_experiments" / "feature_selection"


def _fit_with_algorithm(algorithm, is_classification, X_train, y_train, scale_pos_weight=1.0):
    if is_classification:
        model = models.get_classifier(algorithm, scale_pos_weight=scale_pos_weight) if algorithm == "xgboost" \
            else models.get_classifier(algorithm)
    else:
        model = models.get_regressor(algorithm)
    model.fit(X_train, y_train)
    return model


def main() -> dict:
    print("Loading and engineering features...", flush=True)
    raw = preprocessing.load_raw()
    engineered = preprocessing.engineer_features(raw)
    feature_cols = preprocessing.feature_columns(engineered)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for target_name, is_classification in TARGETS:
        results[target_name] = {}
        for horizon in config.HORIZONS:
            print(f"\n{'='*20} {target_name} h={horizon} {'='*20}", flush=True)
            with_targets = targets.build_targets(engineered, horizon)
            train_df, val_df, test_df = preprocessing.chronological_split(with_targets)
            target_col = targets.target_column(target_name, horizon)
            X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
            X_val, y_val = prepare_xy(val_df, feature_cols, target_col)

            meta_path = config.ARTIFACTS_DIR / f"{target_name}_h{horizon}_metadata.json"
            current_meta = json.loads(meta_path.read_text())
            algorithm = current_meta["algorithm"]
            current_score = selection_score(is_classification, current_meta["val_metrics"])
            print(f"current: {algorithm}, val_score={current_score:.4f}, "
                  f"{len(current_meta['feature_columns'])} features", flush=True)

            scaler = StandardScaler().fit(X_train)
            Xtr, Xv = scaler.transform(X_train), scaler.transform(X_val)

            spw = 1.0
            if is_classification:
                pos = float(y_train.sum())
                spw = (float(len(y_train)) - pos) / pos if pos > 0 else 1.0

            # Fit once on the FULL feature set (same algorithm as deployed)
            # purely to compute permutation importance against validation.
            full_model = _fit_with_algorithm(algorithm, is_classification, Xtr, y_train, spw)
            scoring = "average_precision" if is_classification else "neg_root_mean_squared_error"
            perm = permutation_importance(
                full_model, Xv, y_val, scoring=scoring, n_repeats=N_REPEATS,
                random_state=config.RANDOM_SEED, n_jobs=-1,
            )
            importance_order = np.argsort(perm.importances_mean)[::-1]
            ranked_cols = [feature_cols[i] for i in importance_order]
            ranked_importance = [float(perm.importances_mean[i]) for i in importance_order]

            native_importance = None
            if hasattr(full_model, "feature_importances_"):
                native_importance = dict(zip(feature_cols, (float(v) for v in full_model.feature_importances_)))
            elif hasattr(full_model, "coef_"):
                # LinearRegression.coef_ is 1D (n_features,); LogisticRegression.coef_
                # is 2D (1, n_features) -- np.ravel handles both uniformly.
                coef = np.ravel(full_model.coef_)
                native_importance = dict(zip(feature_cols, (float(abs(v)) for v in coef)))

            top10 = list(zip(ranked_cols[:10], [round(v, 5) for v in ranked_importance[:10]]))
            bottom10 = list(zip(ranked_cols[-10:], [round(v, 5) for v in ranked_importance[-10:]]))
            print(f"  top-10 by permutation importance: {top10}", flush=True)
            print(f"  bottom-10 (candidates to drop): {bottom10}", flush=True)

            horizon_results = {
                "current": {"algorithm": algorithm, "val_score": current_score,
                            "n_features": len(current_meta["feature_columns"])},
                "permutation_importance_top10": top10,
                "permutation_importance_bottom10": bottom10,
            }

            best = {"name": None, "cols": feature_cols, "score": current_score, "model": None, "scaler": scaler,
                    "val_metrics": current_meta["val_metrics"]}
            for frac in CANDIDATE_FRACTIONS:
                k = max(5, int(len(feature_cols) * frac))
                cols = ranked_cols[:k]
                name = f"top{k}"

                scaler_k = StandardScaler().fit(X_train[cols])
                Xtr_k, Xv_k = scaler_k.transform(X_train[cols]), scaler_k.transform(X_val[cols])
                model_k = _fit_with_algorithm(algorithm, is_classification, Xtr_k, y_train, spw)

                if is_classification:
                    val_pred = model_k.predict(Xv_k)
                    val_prob = model_k.predict_proba(Xv_k)[:, 1]
                    val_metrics = evaluate.classification_metrics(y_val, val_pred, val_prob)
                else:
                    val_pred = model_k.predict(Xv_k)
                    val_metrics = evaluate.regression_metrics(y_val, val_pred)

                score = selection_score(is_classification, val_metrics)
                horizon_results[name] = {"n_features": k, "val_score": score, "val_metrics": val_metrics}
                print(f"  {name}: val_score={score:.4f} (full-119 baseline={current_score:.4f}) "
                      f"metrics={ {k2: v for k2, v in val_metrics.items() if k2 not in ('confusion_matrix',)} }",
                      flush=True)

                joblib.dump(model_k, EXPERIMENTS_DIR / f"{target_name}_h{horizon}_{name}_model.joblib")
                joblib.dump(scaler_k, EXPERIMENTS_DIR / f"{target_name}_h{horizon}_{name}_scaler.joblib")

                if score > best["score"] + IMPROVEMENT_MARGIN:
                    best = {"name": name, "cols": cols, "score": score, "model": model_k, "scaler": scaler_k,
                            "val_metrics": val_metrics}

            promoted = best["name"] is not None
            horizon_results["promoted_variant"] = best["name"]
            print(f"  -> {'promoted: ' + best['name'] if promoted else 'NONE (full feature set retained)'}",
                  flush=True)

            if promoted:
                new_metadata = dict(current_meta)
                new_metadata.update({
                    "model_version": "ml-v2-feature-selected",
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "feature_columns": best["cols"],
                    "val_metrics": best["val_metrics"],
                    "sklearn_version": sklearn.__version__,
                    "xgboost_version": xgboost.__version__,
                    "phase2_feature_selection": {
                        "variant": best["name"], "baseline_val_score": current_score,
                        "baseline_n_features": len(feature_cols),
                    },
                })
                if is_classification:
                    prob_val = best["model"].predict_proba(best["scaler"].transform(X_val[best["cols"]]))[:, 1]
                    tuned = best_threshold(y_val.to_numpy(), prob_val)
                    new_metadata["decision_threshold"] = tuned["threshold"]
                    new_metadata["threshold_selection"] = {
                        "method": "f2_max_on_validation_pr_curve", "beta": 2.0,
                        "val_metrics_at_tuned_threshold": tuned,
                    }

                model_path = config.ARTIFACTS_DIR / f"{target_name}_h{horizon}_model.joblib"
                scaler_path = config.ARTIFACTS_DIR / f"{target_name}_h{horizon}_scaler.joblib"
                joblib.dump(best["model"], model_path)
                joblib.dump(best["scaler"], scaler_path)
                meta_path.write_text(json.dumps(new_metadata, indent=2))

                X_test, y_test = prepare_xy(test_df, best["cols"], target_col)
                Xte = best["scaler"].transform(X_test)
                if is_classification:
                    test_prob = best["model"].predict_proba(Xte)[:, 1]
                    thr = new_metadata["decision_threshold"]
                    test_pred = (test_prob >= thr).astype(int)
                    test_metrics = evaluate.classification_metrics(y_test, test_pred, test_prob)
                else:
                    test_pred = best["model"].predict(Xte)
                    test_metrics = evaluate.regression_metrics(y_test, test_pred)
                horizon_results["promoted_test_metrics"] = test_metrics
                print(f"  -> TEST metrics: {test_metrics}", flush=True)
            else:
                horizon_results["promoted_test_metrics"] = None

            results[target_name][str(horizon)] = horizon_results

    out_path = config.ARTIFACTS_DIR.parent / "feature_selection_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
