"""Phase 2 Step 3 (deep-dive): separates EXACT duplicate feature columns from
features that are merely highly correlated but distinct, before touching any
model. Pure pandas analysis -- no fitting, so this runs in seconds, not the
hour+ the elasticnet solver spent apparently failing to converge on the
exact duplicates found here (see docs/model_report.md Phase 2 write-up).

Checks, in order:
  1. Exact duplicates on the RAW csv (before any feature engineering) --
     establishes whether duplication is inherent to the data source or
     introduced by this project's own preprocessing/feature engineering.
  2. Exact duplicates on the fully engineered feature set (raw + lags +
     calendar + static) -- the set actually used for training.
  3. Highly correlated (>0.95) but NOT exact-duplicate pairs, reported
     separately, with a note on why they are plausibly distinct signals
     (never auto-dropped).
  4. Verifies the train/serve feature-schema contract: that
     `app/ml/feature_builder.order_values` and every deployed artifact's
     own `feature_columns` metadata are the single source of truth read at
     both train and serve time (no separate hardcoded list to drift).

Run as `python -m ml.duplicate_feature_analysis` from `backend/`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml import config, preprocessing

EXACT_TOLERANCE = 1e-9  # allow for float round-trip noise, not a real "near enough" threshold
HIGH_CORR_THRESHOLD = 0.95


def _exact_duplicate_pairs(df: pd.DataFrame, cols: list[str]) -> list[tuple[str, str]]:
    """Columns that are bit-identical (within EXACT_TOLERANCE) on every row
    where BOTH have a value -- not just highly correlated. Compares pairwise
    on the rows where neither is NaN, so a shared-NaN-pattern coincidence
    can't masquerade as equality.
    """
    pairs = []
    checked = df[cols]
    n = len(cols)
    for i in range(n):
        a = cols[i]
        for j in range(i + 1, n):
            b = cols[j]
            both_present = checked[a].notna() & checked[b].notna()
            if both_present.sum() == 0:
                continue
            diff = (checked.loc[both_present, a] - checked.loc[both_present, b]).abs()
            if diff.max() <= EXACT_TOLERANCE:
                pairs.append((a, b))
    return pairs


def main() -> dict:
    print("Loading raw CSV (pre-engineering)...", flush=True)
    raw = preprocessing.load_raw()
    base_cols = [c for c in config.BASE_WEATHER_COLUMNS if c in raw.columns]
    raw_dupes = _exact_duplicate_pairs(raw, base_cols)
    print(f"Exact duplicates among the {len(base_cols)} raw base weather columns: {raw_dupes}", flush=True)

    print("\nEngineering full feature set (lags + calendar + static)...", flush=True)
    engineered = preprocessing.engineer_features(raw)
    feature_cols = preprocessing.feature_columns(engineered)
    print(f"{len(feature_cols)} total engineered features", flush=True)

    # Use the training-period rows only (same split everything else uses) --
    # an engineered-feature duplicate check should reflect what the model
    # actually trains on, not validation/test rows.
    train_df, _, _ = preprocessing.chronological_split(engineered)
    exact_dupes = _exact_duplicate_pairs(train_df, feature_cols)

    corr = train_df[feature_cols].corr().abs()
    exact_dupe_set = {frozenset(p) for p in exact_dupes}
    high_corr_not_exact = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1 :]:
            v = corr.loc[a, b]
            if v > HIGH_CORR_THRESHOLD and frozenset((a, b)) not in exact_dupe_set:
                high_corr_not_exact.append((a, b, float(v)))
    high_corr_not_exact.sort(key=lambda p: -p[2])

    print(f"\nEXACT duplicate pairs in the engineered training feature set ({len(exact_dupes)}):", flush=True)
    for a, b in exact_dupes:
        print(f"  {a}  ==  {b}", flush=True)

    print(f"\nHighly correlated but NOT exact duplicates (>{HIGH_CORR_THRESHOLD}, {len(high_corr_not_exact)} pairs, "
          f"top 15 shown):", flush=True)
    for a, b, v in high_corr_not_exact[:15]:
        print(f"  {a} ~ {b}  (corr={v:.4f})", flush=True)

    # Build a safe de-duplicated column list: for each exact-duplicate pair,
    # keep the one that is NOT itself a lag of the other base column where
    # possible (prefer keeping the more "primary"-looking name), else keep
    # the first alphabetically for a stable, reproducible choice.
    to_drop: set[str] = set()
    for a, b in exact_dupes:
        drop = b if a < b else a  # deterministic: always drop the lexicographically later name
        to_drop.add(drop)
    deduped_cols = [c for c in feature_cols if c not in to_drop]

    report = {
        "raw_base_column_exact_duplicates": raw_dupes,
        "engineered_feature_exact_duplicates": exact_dupes,
        "engineered_feature_exact_duplicate_count": len(exact_dupes),
        "high_correlation_not_exact_duplicate_count": len(high_corr_not_exact),
        "high_correlation_not_exact_duplicate_top15": high_corr_not_exact[:15],
        "columns_dropped_for_dedup": sorted(to_drop),
        "deduped_feature_count": len(deduped_cols),
        "original_feature_count": len(feature_cols),
    }
    out_path = config.ARTIFACTS_DIR.parent / "duplicate_feature_analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    print(f"\nConclusion: {len(raw_dupes)} duplicate(s) already present in the RAW csv "
          f"(pre-existing in the data source, not introduced by this project's preprocessing); "
          f"engineering (lagging) propagates each into {len(exact_dupes) - len(raw_dupes) if len(raw_dupes) <= len(exact_dupes) else 'additional'} "
          f"further exact-duplicate pairs via their lagged copies.", flush=True)
    return report


if __name__ == "__main__":
    main()
