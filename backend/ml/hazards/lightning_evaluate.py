"""Evaluate every trained lightning run, promote one, write the report inputs.

    python -m ml.hazards.lightning_evaluate

- REAL (GOES-16/GLM, Americas) and SYNTHETIC (India simulation) results are
  computed and stored separately, always. Neither is validation over India.
- Promotion uses REAL VALIDATION cells only. Real test cells (test anchors x
  test blocks) are scored once, here, after that choice is fixed.
- Baseline: lightning where the cloud top is colder than one threshold,
  tuned on the real training cells.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from app.services.imagery import lightning_data as ld  # noqa: E402
from app.services.imagery.hazard_common import (  # noqa: E402
    apply_platt, brier_score, confusion_counts, metrics_from_counts, reliability_table, roc_pr_auc, rounded,
)
from app.services.imagery.hazard_models import ARCHITECTURE_NAME, SmallUNet, count_parameters  # noqa: E402
from ml.hazards import config  # noqa: E402
from ml.hazards.lightning_train import GoesCache, SyntheticSet, pick_device, predict_probability, real_scores, run_path  # noqa: E402


def load_run(variant: str, seed: int, device):
    path = run_path(variant, seed)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint, json.loads(path.with_suffix(".json").read_text())


def scored(scores: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    out = rounded(metrics_from_counts(confusion_counts(scores >= threshold, truth)))
    out.update({k: (round(v, 4) if v is not None else None) for k, v in roc_pr_auc(scores, truth).items()})
    out["brier"] = round(brier_score(scores, truth), 5)
    return out


def tolerant(pred: np.ndarray, truth: np.ndarray, member: np.ndarray) -> dict:
    """Scores that forgive a one-cell (8 km) displacement: GLM flash positions
    carry a few km of parallax error relative to the infrared image."""
    near_pred, near_truth = ndi.binary_dilation(pred, iterations=1), ndi.binary_dilation(truth, iterations=1)
    hits = int((truth & near_pred & member).sum())
    misses = int((truth & ~near_pred & member).sum())
    false_alarms = int((pred & ~near_truth & member).sum())
    return {"hits": hits, "misses": misses, "falseAlarms": false_alarms}


def tolerant_metrics(total: dict) -> dict:
    hits, misses, fa = total["hits"], total["misses"], total["falseAlarms"]
    return {"pod": round(hits / max(hits + misses, 1), 4), "falseAlarmRatio": round(fa / max(hits + fa, 1), 4),
            "csi": round(hits / max(hits + misses + fa, 1), 4), **total}


def synthetic_scores(model, items, device) -> tuple[np.ndarray, np.ndarray]:
    scores, truths = [], []
    for pair, truth, _ in items:
        scores.append(predict_probability(model, pair, device).ravel())
        truths.append(truth.ravel())
    return np.concatenate(scores), np.concatenate(truths)


def baseline_real(cache: GoesCache, anchors, block, threshold_k: float) -> tuple[np.ndarray, np.ndarray]:
    preds, truths = [], []
    for anchor in anchors:
        pair = cache.pair(anchor)
        member = ld.cell_valid(pair) & ((cache.blocks == block) if block is not None else True)
        preds.append(ld.threshold_baseline(pair[1], threshold_k)[member])
        truths.append(cache.truth(anchor)[member])
    return np.concatenate(preds), np.concatenate(truths)


def mean_std(values) -> dict:
    values = [v for v in values if v is not None]
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4), "n": len(values)} if values else {}


def main() -> None:
    device = pick_device()
    cache = GoesCache()
    syn_test = SyntheticSet("test")
    print(f"real anchors train/val/test = {len(ld.TRAIN_ANCHORS)}/{len(ld.VAL_ANCHORS)}/{len(ld.TEST_ANCHORS)}  "
          f"synthetic test pairs = {len(syn_test)}")

    # ---------------- baseline: one temperature threshold, tuned on real TRAIN cells ----------------
    best = (235.0, -1.0)
    for threshold_k in np.arange(195.0, 250.1, 1.0):
        pred, truth = baseline_real(cache, ld.TRAIN_ANCHORS, 0, threshold_k)
        csi = metrics_from_counts(confusion_counts(pred, truth))["iou"] or 0.0
        if csi > best[1]:
            best = (float(threshold_k), float(csi))
    baseline_k = best[0]
    pred, truth = baseline_real(cache, ld.TEST_ANCHORS, 2, baseline_k)
    baseline_test = rounded(metrics_from_counts(confusion_counts(pred, truth)))
    cold_score, cold_truth = [], []        # the baseline as a ranking: colder = more likely
    for anchor in ld.TEST_ANCHORS:
        pair = cache.pair(anchor)
        member = ld.cell_valid(pair) & (cache.blocks == 2)
        cold_score.append(np.clip((300.0 - ld.cell_min_bt(pair[1])[member]) / 120.0, 0, 1))
        cold_truth.append(cache.truth(anchor)[member])
    baseline_test.update({k: round(v, 4) for k, v in roc_pr_auc(np.concatenate(cold_score), np.concatenate(cold_truth)).items()})
    pred_all, truth_all = baseline_real(cache, ld.TEST_ANCHORS, None, baseline_k)
    syn_pred = np.concatenate([ld.threshold_baseline(pair[1], baseline_k).ravel() for pair, _, _ in syn_test.items])
    syn_truth = np.concatenate([t.ravel() for _, t, _ in syn_test.items])
    baseline_report = {
        "rule": f"flash where the coldest pixel of the 8 km cell is below {baseline_k:.0f} K",
        "thresholdK": baseline_k, "tunedOn": "real training anchors x training blocks only", "trainCsi": round(best[1], 4),
        "realTest": baseline_test,
        "realTestAllBlocks": rounded(metrics_from_counts(confusion_counts(pred_all, truth_all))),
        "syntheticTest": rounded(metrics_from_counts(confusion_counts(syn_pred, syn_truth))),
    }

    # ---------------- every trained run ----------------
    runs, loaded = [], {}
    for variant in config.LIGHTNING_VARIANTS:
        for seed in config.LIGHTNING_SEEDS:
            run = load_run(variant, seed, device)
            if run is None:
                continue
            model, checkpoint, summary = run
            threshold = checkpoint["threshold"]
            s_test, t_test = real_scores(model, cache, ld.TEST_ANCHORS, 2, device)
            s_all, t_all = real_scores(model, cache, ld.TEST_ANCHORS, None, device)
            s_syn, t_syn = synthetic_scores(model, syn_test.items, device)
            row = {"variant": variant, "seed": seed, "threshold": threshold, "bestEpoch": summary["bestEpoch"],
                   "realValPrAuc": summary["realValPrAuc"], "realValCsi": summary["realValCsiAtThreshold"],
                   "realTest": scored(s_test, t_test, threshold), "realTestAllBlocks": scored(s_all, t_all, threshold),
                   "syntheticTest": scored(s_syn, t_syn, threshold)}
            runs.append(row)
            loaded[(variant, seed)] = (model, checkpoint, summary, s_test, t_test)
            print(f"  {variant:<15} seed {seed}  real-val PR-AUC {row['realValPrAuc']:.4f}  real-test CSI {row['realTest']['iou']:.4f} "
                  f"PR-AUC {row['realTest']['prAuc']:.4f}  synthetic-test CSI {row['syntheticTest']['iou']:.4f} PR-AUC {row['syntheticTest']['prAuc']:.4f}")
    if not runs:
        raise SystemExit("No trained lightning runs found -- run `python -m ml.hazards.lightning_train --all` first.")

    comparison = {}
    for variant in config.LIGHTNING_VARIANTS:
        rows = [r for r in runs if r["variant"] == variant]
        if rows:
            comparison[variant] = {
                "runs": len(rows), "realValPrAuc": mean_std([r["realValPrAuc"] for r in rows]),
                "realTestCsi": mean_std([r["realTest"]["iou"] for r in rows]),
                "realTestPrAuc": mean_std([r["realTest"]["prAuc"] for r in rows]),
                "realTestRocAuc": mean_std([r["realTest"]["rocAuc"] for r in rows]),
                "syntheticTestCsi": mean_std([r["syntheticTest"]["iou"] for r in rows]),
                "syntheticTestPrAuc": mean_std([r["syntheticTest"]["prAuc"] for r in rows]),
            }

    # ---------------- promotion: REAL VALIDATION only ----------------
    promoted_variant = max(comparison, key=lambda v: comparison[v]["realValPrAuc"]["mean"])
    promoted = max((r for r in runs if r["variant"] == promoted_variant), key=lambda r: r["realValPrAuc"])
    model, checkpoint, summary, s_test, t_test = loaded[(promoted["variant"], promoted["seed"])]
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_path(promoted["variant"], promoted["seed"]), config.LIGHTNING_CHECKPOINT)
    threshold = promoted["threshold"]

    # ---------------- probability calibration: REAL VALIDATION cells only ----------------
    # Training up-weights the rare flash class, which makes raw scores far too
    # confident. Platt scaling (2 numbers) is fitted on the validation cells and
    # stored in the served artifact. It is monotonic: which cells are flagged,
    # and every threshold-based score, are unchanged.
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    s_val, t_val = real_scores(model, cache, ld.VAL_ANCHORS, 1, device)
    clipped = np.clip(s_val.astype(np.float64), 1e-6, 1 - 1e-6)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(clipped / (1 - clipped)).reshape(-1, 1), t_val)
    platt = {"a": float(fit.coef_[0, 0]), "b": float(fit.intercept_[0]),
             "fittedOn": "real validation anchors x validation blocks", "method": "Platt scaling"}
    served = torch.load(config.LIGHTNING_CHECKPOINT, map_location="cpu", weights_only=True)
    served["calibration"] = {"a": platt["a"], "b": platt["b"]}
    torch.save(served, config.LIGHTNING_CHECKPOINT)
    calibrated_test = apply_platt(s_test, platt["a"], platt["b"])
    calibration_report = {**platt, "servedThreshold": round(float(apply_platt(threshold, platt["a"], platt["b"])), 4),
                          "realTestBrierRaw": round(brier_score(s_test, t_test), 5),
                          "realTestBrierCalibrated": round(brier_score(calibrated_test, t_test), 5),
                          "realTestBrierClimatology": round(float(t_test.mean() * (1 - t_test.mean())), 5)}

    tol_model, tol_base = {"hits": 0, "misses": 0, "falseAlarms": 0}, {"hits": 0, "misses": 0, "falseAlarms": 0}
    for anchor in ld.TEST_ANCHORS:
        pair = cache.pair(anchor)
        member = ld.cell_valid(pair) & (cache.blocks == 2)
        truth = cache.truth(anchor)
        for total, pred in ((tol_model, predict_probability(model, pair, device) >= threshold),
                            (tol_base, ld.threshold_baseline(pair[1], baseline_k))):
            for key, value in tolerant(pred, truth, member).items():
                total[key] += value
    per_kind = {}
    for kind in sorted({meta["stormKind"] for _, _, meta in syn_test.items}):
        items = [item for item in syn_test.items if item[2]["stormKind"] == kind]
        s, t = synthetic_scores(model, items, device)
        base = np.concatenate([ld.threshold_baseline(pair[1], baseline_k).ravel() for pair, _, _ in items])
        per_kind[kind] = {"pairs": len(items), "lightningCells": int(t.sum()),
                          "model": rounded(metrics_from_counts(confusion_counts(s >= threshold, t))),
                          "baseline": rounded(metrics_from_counts(confusion_counts(base, t)))}

    # ---------------- figures ----------------
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    anchor = ld.TEST_ANCHORS[1]
    pair = cache.pair(anchor)
    prob = predict_probability(model, pair, device)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    axes[0, 0].imshow(pair[1], cmap="gray_r", vmin=190, vmax=310)
    axes[0, 0].set_title(f"REAL GOES-16 infrared (band 13), {cache.frame_ids[anchor]}  [Americas]", fontsize=9)
    axes[0, 1].imshow(cache.flashes[anchor] > 0, cmap="hot")
    axes[0, 1].contour(cache.blocks == 2, levels=[0.5], colors="cyan", linewidths=0.4)
    axes[0, 1].set_title("REAL GLM: cells with a flash within +/-5 min (cyan = held-out test blocks)", fontsize=9)
    axes[1, 0].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title(f"U-Net flash probability (threshold {threshold})", fontsize=9)
    axes[1, 1].imshow(ld.threshold_baseline(pair[1], baseline_k), cmap="hot")
    axes[1, 1].set_title(f"baseline: cloud top colder than {baseline_k:.0f} K", fontsize=9)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "lightning_example_real_test.png", dpi=75)
    plt.close(fig)

    picks, seen = [], set()
    for item in syn_test.items[2::3]:               # the last pair (t-30, t) of each sample
        kind = item[2]["stormKind"]
        if kind not in seen and kind in ("squall", "mcs", "isolated", "orographic", "coastal", "wd", "cirrus"):
            seen.add(kind)
            picks.append(item)
    fig, axes = plt.subplots(len(picks), 3, figsize=(10, 3.2 * len(picks)), squeeze=False)
    for ax, (pair, truth, meta) in zip(axes, picks):
        ax[0].imshow(pair[1], cmap="gray_r", vmin=190, vmax=310)
        ax[0].set_title(f"SYNTHETIC infrared\n{meta['place']}, {meta['state']} ({meta['monthName']})", fontsize=8)
        ax[1].imshow(truth, cmap="hot")
        ax[1].set_title(f"SIMULATED lightning cells\n{meta['scenario']}", fontsize=8)
        ax[2].imshow(predict_probability(model, pair, device), cmap="magma", vmin=0, vmax=1)
        ax[2].set_title("U-Net flash probability", fontsize=8)
        for a in ax:
            a.axis("off")
    fig.suptitle("Lightning U-Net on SYNTHETIC India test scenes -- simulations, not observations", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "lightning_examples_synthetic_test.png", dpi=75)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels, width = list(comparison), 0.27
    for k, (key, name) in enumerate((("realTestCsi", "real test CSI"), ("realTestPrAuc", "real test PR-AUC"),
                                     ("syntheticTestCsi", "synthetic India test CSI"))):
        ax.bar(np.arange(len(labels)) + (k - 1) * width, [comparison[v][key]["mean"] for v in labels], width,
               yerr=[comparison[v][key]["std"] for v in labels], capsize=3, label=name)
    ax.axhline(baseline_test["iou"], color="grey", ls="--", lw=1, label=f"{baseline_k:.0f} K threshold, real test CSI")
    ax.set_xticks(np.arange(len(labels)), [f"trained on\n{v}" for v in labels])
    ax.set_title("Lightning U-Net: training-data variants (mean of 3 seeds, bars = std)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "lightning_variant_comparison.png", dpi=80)
    plt.close(fig)

    # ---------------- outputs ----------------
    reliability = reliability_table(calibrated_test, t_test)
    evaluation = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "note": "REAL = GOES-16 infrared + GLM flashes over the AMERICAS (2 hours, 2 Aug 2024). SYNTHETIC = simulated "
                "Indian storm scenes with simulated flashes. Neither is a validation over India.",
        "split": {"trainAnchors": [cache.frame_ids[i] for i in ld.TRAIN_ANCHORS], "valAnchors": [cache.frame_ids[i] for i in ld.VAL_ANCHORS],
                  "testAnchors": [cache.frame_ids[i] for i in ld.TEST_ANCHORS], "blockCells": ld.BLOCK_CELLS,
                  "blockShares": [float((cache.blocks == k).mean()) for k in range(3)],
                  "rule": "a cell is scored only if BOTH its time and its 384 km block belong to the split"},
        "promotion": {"rule": "best mean REAL validation PR-AUC across seeds, then that variant's best seed; test never consulted",
                      "variant": promoted["variant"], "seed": promoted["seed"]},
        "variantComparison": comparison, "runs": runs, "baseline": baseline_report,
        "promoted": {"realTest": promoted["realTest"], "realTestAllBlocks": promoted["realTestAllBlocks"],
                     "realTestOneCellTolerance": {"model": tolerant_metrics(tol_model), "baseline": tolerant_metrics(tol_base)},
                     "probabilityCalibration": calibration_report, "realTestReliabilityRaw": reliability_table(s_test, t_test),
                     "realTestReliability": reliability, "syntheticTest": promoted["syntheticTest"],
                     "syntheticTestPerStormKind": per_kind},
    }
    config.LIGHTNING_EVALUATION.write_text(json.dumps(evaluation, indent=2))
    metadata = {
        "name": config.LIGHTNING_MODEL_NAME, "version": config.LIGHTNING_MODEL_VERSION, "architecture": ARCHITECTURE_NAME,
        "task": "Probability of at least one lightning flash per 8 km cell within +/-5 min, from two infrared "
                "cloud-top images 30 min apart (4 km grid)",
        "parameters": count_parameters(model), "modelConfig": checkpoint["model_config"], "input": checkpoint["input"],
        "threshold": calibration_report["servedThreshold"], "rawThreshold": threshold, "probabilityCalibration": calibration_report,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainedOn": promoted["variant"], "seed": promoted["seed"], "bestEpoch": summary["bestEpoch"],
        "device": summary["device"], "torchVersion": summary["torchVersion"],
        "trainingData": {
            "real": "GOES-16 ABI band 13 infrared + GLM flashes, CONUS sector (Americas), 2 Aug 2024 20:00-22:00 UTC: "
                    f"{len(ld.TRAIN_ANCHORS)} training times x 60 % of the 384 km blocks",
            "synthetic": (f"{summary['syntheticTrainPairs']} synthetic India image pairs (ml.hazards.synth_lightning)"
                          if summary["syntheticTrainPairs"] else "none"),
            "validation": f"{len(ld.VAL_ANCHORS)} later times x 20 % of blocks (real) -- model selection and threshold",
            "test": f"{len(ld.TEST_ANCHORS)} latest times x 20 % of blocks (real) -- scored once, after selection",
        },
        "anchorSplits": {cache.frame_ids[i]: ld.split_of_anchor(i) for i in range(len(cache.frame_ids))},
        "testMetrics": {"real": promoted["realTest"], "realOneCellTolerance": tolerant_metrics(tol_model),
                        "baseline": baseline_report["realTest"], "baselineOneCellTolerance": tolerant_metrics(tol_base)},
        "syntheticMetrics": {"note": "synthetic India test scenes -- agreement with the simulator only",
                             "model": promoted["syntheticTest"], "baseline": baseline_report["syntheticTest"]},
        "variantComparison": comparison, "baselineThresholdK": baseline_k,
        "limitations": [
            "Trained and tested on TWO HOURS of one August afternoon over the Americas. The test cells are later in "
            "the same afternoon (in different 384 km blocks) -- not another day, season or continent.",
            "NOT validated for India. No Indian lightning ground truth is in this project; GOES does not see India.",
            "Infers lightning from cloud-top temperature and its 30-minute change. It does not detect flashes: "
            "there is no lightning sensor in the loop.",
            "Scores on synthetic India scenes measure agreement with a simulator whose flash rule was written by hand "
            "(and scaled to one GOES statistic). They are not accuracy over India.",
            "Cold land surfaces (snow, high winter terrain) were never seen in training and can be mistaken for cloud.",
            "Not connected to the risk engine or alerts.",
        ],
    }
    config.LIGHTNING_METADATA.write_text(json.dumps(metadata, indent=2))
    print(f"\npromoted: {promoted['variant']} seed {promoted['seed']} -> {config.LIGHTNING_CHECKPOINT.name} "
          f"({config.LIGHTNING_CHECKPOINT.stat().st_size / 1e6:.2f} MB)")
    print(json.dumps({"variantComparison": comparison, "baseline": baseline_report, "promotedRealTest": promoted["realTest"],
                      "tolerant": {"model": tolerant_metrics(tol_model), "baseline": tolerant_metrics(tol_base)},
                      "promotedSyntheticTest": promoted["syntheticTest"], "calibration": calibration_report}, indent=1))


if __name__ == "__main__":
    main()
