"""Phase 2 Step 3: investigate and attempt to stabilize heavy_rain_risk,
particularly the 3h flagship model (Logistic Regression, val PR-AUC 0.1788,
test PR-AUC 0.034 -- Phase 1 audit found its coefficients swinging +-7..10
across correlated lagged features).

Part 1 -- root cause: inspects the CURRENTLY DEPLOYED model's own
coefficients (never refits anything for this part), and separates EXACT
duplicate feature columns from merely-highly-correlated ones using
`ml/duplicate_feature_analysis.py`'s findings (re-run here if its output is
missing). Verified finding: `precipitation_mm_hour` and `rainfall_mm_hour`
(+ their matching lags) are bit-identical across the ENTIRE training window
-- confirmed on the raw CSV directly, not a feature-engineering bug -- and
a first attempt at this step hung for 87 CPU-minutes fitting
L1/liblinear and Elastic Net/saga directly on that exact collinearity
before being killed; this version drops those 5 exact-duplicate columns
before fitting ANYTHING, which is also the fix for the hang (an
unbounded ridge direction in the loss surface, not just a slow solver).

Part 2 -- controlled experiments, all fit on the training split / scored on
the validation split only, on the 114-column EXACT-DEDUPED feature set
(DEDUP_DROP_COLUMNS) unless noted:
  A. existing deployed model, re-scored for a like-for-like reference row.
  B. L1 logistic regression (two C values).
  C. L2 logistic regression (three C values, the current penalty, varied
     strength).
  D. Elastic Net logistic regression.
  E. The BLUNT >0.95 correlation-threshold prune (on top of the exact
     dedup) + L2 -- kept only as a labeled "aggressive" comparison point,
     per the explicit instruction not to blindly treat every >0.95 pair as
     redundant (most of them, e.g. a soil-moisture depth vs. its own 1h lag,
     are legitimate slow-changing signals, not duplicates).
Every variant uses class_weight="balanced" (the existing imbalance
handling); none use synthetic oversampling, inside or outside folds.

Part 3 -- cross-period check: validation (2024-07-01..2024-12-31) is split
at 2024-10-01 into two chronological halves and every variant is re-scored
on each half separately, so "improved on the whole validation set" can't
hide a model that only works in one season.

Promotion rule (none of these alone is sufficient): a candidate must have
PR-AUC within PR_AUC_TOLERANCE of the current model (better OR comparable,
never worse by more than the tolerance), AND a max-coefficient at least
STABILITY_IMPROVEMENT_FACTOR smaller (genuinely more stable), AND not
collapse on either chronological half relative to the baseline's own
half-scores. All fitted variants are saved to
`ml/artifacts_experiments/heavy_rain_stabilization/` regardless of outcome,
for inspection; only a promoted candidate touches the deployed artifact
(backed up already in Step 0). The test split is read only if a candidate
is actually promoted -- never during selection.

Run as `python -m ml.heavy_rain_stabilization` from `backend/`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from ml import config, evaluate, preprocessing, targets
from ml.threshold_tuning import best_threshold
from ml.train import prepare_xy

TARGET = "heavy_rain_risk"
PR_AUC_TOLERANCE = 0.02
STABILITY_IMPROVEMENT_FACTOR = 1.5
PERIOD_SPLIT_DATE = "2024-10-01"
AGGRESSIVE_CORR_THRESHOLD = 0.95

# Confirmed by ml/duplicate_feature_analysis.py: bit-identical to
# rainfall_mm_hour (+lags) across the entire training window. rainfall_mm_hour
# is kept because it's the name used everywhere else in this codebase (the
# risk engine's thresholds, the heavy_rain_risk target itself).
DEDUP_DROP_COLUMNS = [
    "precipitation_mm_hour",
    "precipitation_mm_hour_lag1h",
    "precipitation_mm_hour_lag3h",
    "precipitation_mm_hour_lag6h",
    "precipitation_mm_hour_lag24h",
]

EXPERIMENTS_DIR = config.ARTIFACTS_DIR.parent / "artifacts_experiments" / "heavy_rain_stabilization"


def _coef_stats(model: LogisticRegression) -> dict:
    coef = model.coef_[0]
    return {
        "max_abs_coef": float(np.max(np.abs(coef))),
        "l2_norm": float(np.linalg.norm(coef)),
        "n_nonzero": int(np.count_nonzero(np.abs(coef) > 1e-6)),
        "n_coef": int(len(coef)),
    }


def _aggressive_prune(X_train: pd.DataFrame, threshold: float = AGGRESSIVE_CORR_THRESHOLD) -> list[str]:
    """The blunt >0.95 threshold prune, applied ONLY within variant E, on
    top of the already-exact-deduped column set -- kept separate from the
    default feature set used by every other variant.
    """
    corr = X_train.corr().abs()
    cols = list(X_train.columns)
    to_drop: set[str] = set()
    for i, col_i in enumerate(cols):
        if col_i in to_drop:
            continue
        for col_j in cols[i + 1 :]:
            if col_j in to_drop:
                continue
            if corr.loc[col_i, col_j] > threshold:
                to_drop.add(col_j)
    return [c for c in cols if c not in to_drop]


LR_MAX_ITER = 300  # saga/lbfgs cost ~0.09s/iter on this data -- liblinear's
# dual coordinate descent was found (isolated diagnostic, Step 3 deep-dive)
# to be pathologically slow specifically under this target's extreme
# imbalance (0.137% positive at h=1) combined with class_weight="balanced"
# -- even a 100x looser tolerance didn't finish in 2 minutes. saga solves
# the identical problem in 8.7s for 100 iterations, so every L1/elasticnet
# variant below uses saga, never liblinear.


def _fit_variant(X_train, y_train, X_val, y_val, **lr_kwargs):
    scaler = StandardScaler().fit(X_train)
    Xtr, Xv = scaler.transform(X_train), scaler.transform(X_val)
    model = LogisticRegression(
        max_iter=LR_MAX_ITER, class_weight="balanced", random_state=config.RANDOM_SEED, **lr_kwargs
    )
    model.fit(Xtr, y_train)
    val_pred = model.predict(Xv)
    val_prob = model.predict_proba(Xv)[:, 1]
    val_metrics = evaluate.classification_metrics(y_val, val_pred, val_prob)
    val_metrics["brier"] = float(brier_score_loss(y_val, val_prob))
    n_iter = int(np.asarray(model.n_iter_).max())
    val_metrics["solver_n_iter"] = n_iter
    val_metrics["solver_converged"] = n_iter < LR_MAX_ITER
    return model, scaler, val_metrics


def _score_on_period(model, scaler, cols: list[str], period_df: pd.DataFrame, target_col: str):
    X, y = prepare_xy(period_df, cols, target_col)
    if len(y) == 0 or y.sum() == 0 or len(np.unique(y)) < 2:
        return None
    prob = model.predict_proba(scaler.transform(X))[:, 1]
    return float(average_precision_score(y, prob))


def main() -> dict:
    print("Loading and engineering features...", flush=True)
    raw = preprocessing.load_raw()
    engineered = preprocessing.engineer_features(raw)
    all_feature_cols = preprocessing.feature_columns(engineered)
    base_cols = [c for c in all_feature_cols if c not in DEDUP_DROP_COLUMNS]
    print(f"Dropped {len(all_feature_cols) - len(base_cols)} exact-duplicate columns "
          f"({DEDUP_DROP_COLUMNS}); {len(base_cols)} features remain for every variant below.", flush=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for horizon in config.HORIZONS:
        print(f"\n{'='*20} heavy_rain_risk h={horizon} {'='*20}", flush=True)
        with_targets = targets.build_targets(engineered, horizon)
        train_df, val_df, test_df = preprocessing.chronological_split(with_targets)
        target_col = targets.target_column(TARGET, horizon)
        X_train, y_train = prepare_xy(train_df, base_cols, target_col)
        X_val, y_val = prepare_xy(val_df, base_cols, target_col)

        val_h1 = val_df[val_df["timestamp_local"] < PERIOD_SPLIT_DATE]
        val_h2 = val_df[val_df["timestamp_local"] >= PERIOD_SPLIT_DATE]

        meta_path = config.ARTIFACTS_DIR / f"{TARGET}_h{horizon}_metadata.json"
        current_meta = json.loads(meta_path.read_text())
        current_model = joblib.load(config.ARTIFACTS_DIR / current_meta["model_path"])
        current_scaler = joblib.load(config.ARTIFACTS_DIR / current_meta["scaler_path"])
        current_cols = current_meta["feature_columns"]
        current_score = current_meta["val_metrics"].get("pr_auc") or 0.0

        has_coef = hasattr(current_model, "coef_")
        current_stats = _coef_stats(current_model) if has_coef else None
        print(f"Root cause: current algorithm={current_meta['algorithm']}, "
              f"has_coefficients={has_coef}, coef_stats={current_stats}", flush=True)

        baseline_h1 = _score_on_period(current_model, current_scaler, current_cols, val_h1, target_col)
        baseline_h2 = _score_on_period(current_model, current_scaler, current_cols, val_h2, target_col)
        print(f"  current model cross-period PR-AUC: H1(pre-Oct)={baseline_h1} H2(post-Oct)={baseline_h2}",
              flush=True)

        horizon_results = {
            "current": {
                "algorithm": current_meta["algorithm"], "val_pr_auc": current_score,
                "coef_stats": current_stats, "period_h1_pr_auc": baseline_h1, "period_h2_pr_auc": baseline_h2,
            },
            "exact_duplicates_dropped": DEDUP_DROP_COLUMNS,
        }

        variants = [
            ("B_l1_C1.0", dict(penalty="l1", C=1.0, solver="saga"), base_cols),
            ("B_l1_C0.1", dict(penalty="l1", C=0.1, solver="saga"), base_cols),
            ("C_l2_C1.0", dict(penalty="l2", C=1.0, solver="lbfgs"), base_cols),
            ("C_l2_C0.1", dict(penalty="l2", C=0.1, solver="lbfgs"), base_cols),
            ("C_l2_C0.01", dict(penalty="l2", C=0.01, solver="lbfgs"), base_cols),
            ("D_elasticnet_C1.0_l0.5", dict(penalty="elasticnet", C=1.0, solver="saga", l1_ratio=0.5), base_cols),
        ]
        aggressive_cols = _aggressive_prune(X_train)
        variants.append(("E_aggressive_corr_prune", dict(penalty="l2", C=1.0, solver="lbfgs"), aggressive_cols))
        print(f"  variant E (aggressive >0.95 prune): {len(aggressive_cols)}/{len(base_cols)} features kept",
              flush=True)

        candidates = []
        for name, kwargs, cols in variants:
            Xtr_v, Xval_v = X_train[cols], X_val[cols]
            model, scaler, val_metrics = _fit_variant(Xtr_v, y_train, Xval_v, y_val, **kwargs)
            stats = _coef_stats(model)
            h1 = _score_on_period(model, scaler, cols, val_h1, target_col)
            h2 = _score_on_period(model, scaler, cols, val_h2, target_col)

            joblib.dump(model, EXPERIMENTS_DIR / f"{TARGET}_h{horizon}_{name}_model.joblib")
            joblib.dump(scaler, EXPERIMENTS_DIR / f"{TARGET}_h{horizon}_{name}_scaler.joblib")

            horizon_results[name] = {
                "val_metrics": val_metrics, "coef_stats": stats, "n_features": len(cols),
                "period_h1_pr_auc": h1, "period_h2_pr_auc": h2,
            }
            candidates.append({"name": name, "model": model, "scaler": scaler, "cols": cols,
                                "val_metrics": val_metrics, "coef_stats": stats,
                                "period_h1_pr_auc": h1, "period_h2_pr_auc": h2})

            pr_auc = val_metrics.get("pr_auc") or 0.0
            print(f"  {name}: PR-AUC={pr_auc:.4f} precision={val_metrics['precision']:.3f} "
                  f"recall={val_metrics['recall']:.3f} brier={val_metrics['brier']:.4f} "
                  f"max|coef|={stats['max_abs_coef']:.2f} nonzero={stats['n_nonzero']}/{stats['n_coef']} "
                  f"H1={h1} H2={h2} converged={val_metrics['solver_converged']}({val_metrics['solver_n_iter']}it)",
                  flush=True)

        promoted = None
        for c in candidates:
            pr_auc = c["val_metrics"].get("pr_auc") or 0.0
            comparable_or_better = pr_auc >= current_score - PR_AUC_TOLERANCE
            more_stable = (
                current_stats is None
                or c["coef_stats"]["max_abs_coef"] * STABILITY_IMPROVEMENT_FACTOR <= current_stats["max_abs_coef"]
            )
            h1_ok = c["period_h1_pr_auc"] is None or baseline_h1 is None or c["period_h1_pr_auc"] >= baseline_h1 - PR_AUC_TOLERANCE
            h2_ok = c["period_h2_pr_auc"] is None or baseline_h2 is None or c["period_h2_pr_auc"] >= baseline_h2 - PR_AUC_TOLERANCE
            c["qualifies"] = comparable_or_better and more_stable and h1_ok and h2_ok
            if c["qualifies"] and (promoted is None or pr_auc > (promoted["val_metrics"].get("pr_auc") or 0.0)):
                promoted = c

        horizon_results["promoted_variant"] = promoted["name"] if promoted else None
        print(f"  -> promotion: {promoted['name'] if promoted else 'NONE (existing model retained)'}", flush=True)

        if promoted is not None:
            prob_val = promoted["model"].predict_proba(promoted["scaler"].transform(X_val[promoted["cols"]]))[:, 1]
            tuned = best_threshold(y_val.to_numpy(), prob_val)

            model_path = config.ARTIFACTS_DIR / f"{TARGET}_h{horizon}_model.joblib"
            scaler_path = config.ARTIFACTS_DIR / f"{TARGET}_h{horizon}_scaler.joblib"
            joblib.dump(promoted["model"], model_path)
            joblib.dump(promoted["scaler"], scaler_path)

            new_metadata = dict(current_meta)
            new_metadata.update({
                "algorithm": f"logistic_{promoted['name']}",
                "model_version": "ml-v2-stabilized",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_columns": promoted["cols"],
                "val_metrics": promoted["val_metrics"],
                "sklearn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
                "decision_threshold": tuned["threshold"],
                "threshold_selection": {
                    "method": "f2_max_on_validation_pr_curve", "beta": 2.0,
                    "val_metrics_at_tuned_threshold": tuned,
                },
                "phase2_stabilization": {
                    "variant": promoted["name"], "baseline_val_pr_auc": current_score,
                    "baseline_coef_stats": current_stats, "exact_duplicates_dropped": DEDUP_DROP_COLUMNS,
                },
            })
            meta_path.write_text(json.dumps(new_metadata, indent=2))

            X_test, y_test = prepare_xy(test_df, promoted["cols"], target_col)
            test_prob = promoted["model"].predict_proba(promoted["scaler"].transform(X_test))[:, 1]
            test_pred = (test_prob >= tuned["threshold"]).astype(int)
            test_metrics = evaluate.classification_metrics(y_test, test_pred, test_prob)
            horizon_results["promoted_test_metrics"] = test_metrics
            print(f"  -> promoted; recalibrated threshold={tuned['threshold']:.3f}; "
                  f"TEST PR-AUC={test_metrics['pr_auc']}", flush=True)
        else:
            horizon_results["promoted_test_metrics"] = None

        results[str(horizon)] = horizon_results

    out_path = config.ARTIFACTS_DIR.parent / "heavy_rain_stabilization_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
