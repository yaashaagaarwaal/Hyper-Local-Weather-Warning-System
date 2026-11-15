"""Phase 2 Step 2: time-aware hyperparameter search for XGBoost and Random
Forest, at the flagship 3h horizon only (expand to 1h/6h only if these
results justify it -- see docs/model_report.md Phase 2 write-up).

Uses RandomizedSearchCV with TimeSeriesSplit on the TRAINING split only --
never shuffled, so no fold ever trains on a row chronologically after one of
its own test rows. Training rows are re-sorted by `timestamp_local` (not the
area-then-time order `preprocessing` uses for lag construction) so the CV
folds are genuinely chronological across all 5 stations together, the same
"evaluate on the future relative to training" principle the original
train/val/test split already applies. Validation and test splits are
untouched by the search; the test split is never read by this script.

A tuned candidate only replaces the existing Phase 4 artifact if it beats the
current winner's own validation score by at least IMPROVEMENT_MARGIN -- a
tie or marginal change keeps the original model, per the explicit "do not
replace without demonstrated improvement" instruction. This script only
searches and records results (`ml/hyperparam_search_results.json`); promotion
is a separate, explicit step after reviewing the results.

Run as `python -m ml.hyperparam_search` from `backend/`.
"""

from __future__ import annotations

import json
import time

from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ml import config, evaluate, preprocessing, targets
from ml.config import RANDOM_SEED
from ml.train import prepare_xy, selection_score

HORIZON = config.FLAGSHIP_HORIZON  # 3
N_ITER = 8
CV_SPLITS = 2

# First pass: RandomForestRegressor's default max_features=1.0 (all 119
# features considered at every split) made an initial, unconstrained run
# take >2 CPU-hours on a single target without finishing -- killed and
# redesigned. The search itself now runs on a trailing subsample of the
# training rows (SEARCH_SUBSAMPLE_ROWS, still strictly chronological, still
# only training-split data) purely to make the search tractable; the
# winning hyperparameters are then refit on the FULL training split before
# any validation comparison, so the number being compared against the
# current winner is never computed from the subsample.
SEARCH_SUBSAMPLE_ROWS = 40_000

IMPROVEMENT_MARGIN = {
    "temperature_c": 0.01,  # -RMSE units (degC) -- higher (less negative) is better
    "rainfall_mm_hour": 0.01,
    "heavy_rain_risk": 0.01,  # PR-AUC units
    "thunderstorm_risk": 0.01,
}

XGB_PARAMS = {
    "n_estimators": randint(100, 350),
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.02, 0.18),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}
# max_features is capped explicitly for BOTH classifier and regressor --
# sklearn's RandomForestRegressor default (max_features=1.0, i.e. every
# feature considered at every split) is what made the first attempt this
# slow; "sqrt"/"log2" bound the per-split cost regardless of estimator type.
RF_PARAMS = {
    "n_estimators": randint(80, 200),
    "max_depth": randint(5, 14),
    "min_samples_leaf": randint(2, 12),
    "max_features": ["sqrt", "log2", 0.3],
}


def _base_estimator(algorithm: str, is_classification: bool, scale_pos_weight: float = 1.0):
    if is_classification:
        if algorithm == "xgboost":
            return XGBClassifier(
                n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            )
        return RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced")
    if algorithm == "xgboost":
        return XGBRegressor(n_jobs=-1, random_state=RANDOM_SEED)
    return RandomForestRegressor(n_jobs=-1, random_state=RANDOM_SEED)


def _param_space(algorithm: str) -> dict:
    return XGB_PARAMS if algorithm == "xgboost" else RF_PARAMS


def _build_search(algorithm: str, is_classification: bool, scale_pos_weight: float = 1.0):
    base = _base_estimator(algorithm, is_classification, scale_pos_weight)
    scoring = "average_precision" if is_classification else "neg_root_mean_squared_error"
    # n_jobs=1 at the search level -- the estimator itself already
    # parallelizes (n_jobs=-1); nesting both would oversubscribe cores.
    return RandomizedSearchCV(
        base, _param_space(algorithm), n_iter=N_ITER, cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring=scoring, random_state=RANDOM_SEED, n_jobs=1,
    )


def main() -> dict:
    print("Loading and engineering features...", flush=True)
    raw = preprocessing.load_raw()
    engineered = preprocessing.engineer_features(raw)
    feature_cols = preprocessing.feature_columns(engineered)

    with_targets = targets.build_targets(engineered, HORIZON)
    train_df, val_df, test_df = preprocessing.chronological_split(with_targets)
    train_df_sorted = train_df.sort_values("timestamp_local")

    all_targets = [(t, False) for t in config.TARGET_REGRESSION] + [
        (t, True) for t in config.TARGET_CLASSIFICATION
    ]

    results: dict = {}
    for target_name, is_classification in all_targets:
        target_col = targets.target_column(target_name, HORIZON)
        X_train, y_train = prepare_xy(train_df_sorted, feature_cols, target_col)
        X_val, y_val = prepare_xy(val_df, feature_cols, target_col)

        scaler = StandardScaler().fit(X_train)
        Xtr, Xv = scaler.transform(X_train), scaler.transform(X_val)
        # Trailing subsample (still chronological, still training-split-only)
        # used ONLY to make the search itself tractable -- see module
        # docstring / SEARCH_SUBSAMPLE_ROWS.
        n_search = min(SEARCH_SUBSAMPLE_ROWS, len(Xtr))
        Xtr_search, y_train_search = Xtr[-n_search:], y_train.iloc[-n_search:]

        meta_path = config.ARTIFACTS_DIR / f"{target_name}_h{HORIZON}_metadata.json"
        current_meta = json.loads(meta_path.read_text())
        current_score = selection_score(is_classification, current_meta["val_metrics"])

        target_results = {"current": {"algorithm": current_meta["algorithm"], "val_score": current_score}}

        for algorithm in ["xgboost", "random_forest"]:
            t0 = time.perf_counter()
            if is_classification:
                pos = float(y_train_search.sum())
                neg = float(len(y_train_search) - pos)
                spw = (neg / pos) if pos > 0 else 1.0
                search = _build_search(algorithm, True, spw)
            else:
                spw = 1.0
                search = _build_search(algorithm, False)
            search.fit(Xtr_search, y_train_search)
            search_elapsed = time.perf_counter() - t0

            # Refit the winning hyperparameters on the FULL training split --
            # the number compared against the current winner must never come
            # from the search subsample.
            final_model = _base_estimator(algorithm, is_classification, spw)
            final_model.set_params(**search.best_params_)
            final_model.fit(Xtr, y_train)
            elapsed = time.perf_counter() - t0

            if is_classification:
                val_pred = final_model.predict(Xv)
                val_prob = final_model.predict_proba(Xv)[:, 1]
                val_metrics = evaluate.classification_metrics(y_val, val_pred, val_prob)
            else:
                val_pred = final_model.predict(Xv)
                val_metrics = evaluate.regression_metrics(y_val, val_pred)

            score = selection_score(is_classification, val_metrics)
            margin = IMPROVEMENT_MARGIN[target_name]
            improved = score > current_score + margin
            target_results[algorithm] = {
                "best_params": search.best_params_,
                "cv_best_score_on_subsample": float(search.best_score_),
                "val_score": score,
                "val_metrics": val_metrics,
                "search_seconds": round(search_elapsed, 1),
                "total_seconds": round(elapsed, 1),
                "improved_over_current": improved,
            }
            print(
                f"[{target_name}] {algorithm}: cv_subsample={search.best_score_:.4f} val_score={score:.4f} "
                f"(current {current_meta['algorithm']}={current_score:.4f}) "
                f"improved={improved} [{elapsed:.0f}s]",
                flush=True,
            )

        results[target_name] = target_results

    out_path = config.ARTIFACTS_DIR.parent / "hyperparam_search_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
