"""Evaluate every trained flood run, promote one, write the report inputs.

    python -m ml.hazards.flood_evaluate

- REAL and SYNTHETIC results are computed and stored separately, always.
- Promotion uses REAL VALIDATION scores only (variant with the best mean
  validation IoU across seeds, then its best seed). The real test chips are
  scored once, here, after that choice is fixed.
- Baselines: the Otsu mask shipped with Sen1Floods11, a per-image Otsu
  threshold, and a single global threshold tuned on the real training chips.
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

from app.services.imagery import flood_data  # noqa: E402
from app.services.imagery.hazard_common import (  # noqa: E402
    add_counts, brier_score, confusion_counts, metrics_from_counts, reliability_table, roc_pr_auc, rounded,
)
from app.services.imagery.hazard_models import ARCHITECTURE_NAME, SmallUNet, count_parameters  # noqa: E402
from ml.hazards import config  # noqa: E402
from ml.hazards.flood_train import ChipSet, pick_device, predict_probability, run_path  # noqa: E402


def load_run(variant: str, seed: int, device):
    path = run_path(variant, seed)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint, json.loads(path.with_suffix(".json").read_text())


def score_masks(chips: ChipSet, masks: list[np.ndarray | None]) -> dict:
    """Pooled-pixel scores plus the mean of per-chip IoU."""
    counts: dict = {}
    per_chip = []
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        s1, label = chips.arrays(i)
        c = confusion_counts(mask, label == 1, flood_data.valid_mask(s1, label))
        add_counts(counts, c)
        iou = metrics_from_counts(c)["iou"]
        if iou is not None:
            per_chip.append(iou)
    out = rounded(metrics_from_counts(counts))
    out["meanPerChipIou"] = round(float(np.mean(per_chip)), 4) if per_chip else None
    out["chips"] = sum(m is not None for m in masks)
    return out


def model_probabilities(model, chips: ChipSet, checkpoint, device) -> list[np.ndarray]:
    mean = np.asarray(checkpoint["normalisation"]["mean"], np.float32)
    std = np.asarray(checkpoint["normalisation"]["std"], np.float32)
    return [predict_probability(model, chips.arrays(i)[0], mean, std, device) for i in range(len(chips))]


def tune_global_threshold(train: ChipSet, band: str) -> tuple[float, float]:
    best = (0.0, -1.0)
    for threshold in np.arange(-30.0, -7.9, 0.5):
        iou = score_masks(train, [flood_data.threshold_water(train.arrays(i)[0], band, threshold) for i in range(len(train))])["iou"] or 0
        if iou > best[1]:
            best = (float(threshold), float(iou))
    return best


def subset(chips: ChipSet, masks: list, keep) -> tuple[ChipSet, list]:
    index = [i for i, c in enumerate(chips.chips) if keep(c)]
    part = ChipSet.__new__(ChipSet)
    part.chips = [chips.chips[i] for i in index]
    part.s1 = [chips.s1[i] for i in index]
    part.label = [chips.label[i] for i in index]
    return part, [masks[i] for i in index]


def mean_std(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4), "n": len(values)} if values else {}


def figure_examples(chips: ChipSet, probabilities, threshold, baseline_masks, path, title, pick: list[int]) -> None:
    fig, axes = plt.subplots(len(pick), 4, figsize=(13, 3.3 * len(pick)), squeeze=False)
    for ax, i in zip(axes, pick):
        s1, label = chips.arrays(i)
        valid = flood_data.valid_mask(s1, label)
        truth = np.ma.masked_where(~valid, label == 1)
        model_mask = probabilities[i] >= threshold
        iou_model = metrics_from_counts(confusion_counts(model_mask, label == 1, valid))["iou"]
        iou_base = metrics_from_counts(confusion_counts(baseline_masks[i], label == 1, valid))["iou"]
        ax[0].imshow(s1[1], cmap="gray", vmin=-32, vmax=-7)
        ax[0].set_title(f"{chips.chips[i].chip_id}\nradar VH (dB)", fontsize=8)
        ax[1].imshow(truth, cmap="Blues", vmin=0, vmax=1)
        ax[1].set_title("label (white = not labelled)", fontsize=8)
        ax[2].imshow(probabilities[i], cmap="Blues", vmin=0, vmax=1)
        ax[2].contour(model_mask, levels=[0.5], colors="crimson", linewidths=0.5)
        ax[2].set_title(f"U-Net water probability\nIoU {iou_model:.2f}" if iou_model is not None else "U-Net water probability", fontsize=8)
        ax[3].imshow(baseline_masks[i], cmap="Blues", vmin=0, vmax=1)
        ax[3].set_title(f"threshold baseline\nIoU {iou_base:.2f}" if iou_base is not None else "threshold baseline", fontsize=8)
        for a in ax:
            a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def main() -> None:
    device = pick_device()
    real = flood_data.real_catalog()
    synthetic = flood_data.synthetic_catalog()
    real_train = ChipSet([c for c in real if c.split == "train"])
    real_val = ChipSet([c for c in real if c.split == "valid"])
    real_test = ChipSet([c for c in real if c.split == "test"])
    syn_test = ChipSet([c for c in synthetic if c.split == "test"])
    print(f"real train/val/test = {len(real_train)}/{len(real_val)}/{len(real_test)}  synthetic test = {len(syn_test)}")

    # ---------------- baselines (no learning beyond one number) ----------------
    vh_threshold, _ = tune_global_threshold(real_train, "VH")
    vv_threshold, _ = tune_global_threshold(real_train, "VV")

    def baselines_for(chips: ChipSet) -> dict:
        s1s = [chips.arrays(i)[0] for i in range(len(chips))]
        return {
            "datasetOtsuReference": [flood_data.read_otsu_reference(c) for c in chips.chips],
            "perImageOtsuVH": [flood_data.otsu_water(s, "VH")[0] for s in s1s],
            "globalThresholdVH": [flood_data.threshold_water(s, "VH", vh_threshold) for s in s1s],
            "globalThresholdVV": [flood_data.threshold_water(s, "VV", vv_threshold) for s in s1s],
        }

    base_real, base_syn = baselines_for(real_test), baselines_for(syn_test)
    is_india = lambda c: c.region == "India"  # noqa: E731
    baseline_report = {
        "tunedOn": "real training chips only", "globalThresholdVHdB": vh_threshold, "globalThresholdVVdB": vv_threshold,
        "realTest": {k: score_masks(real_test, v) for k, v in base_real.items()},
        "realTestIndiaOnly": {k: score_masks(*subset(real_test, v, is_india)) for k, v in base_real.items()},
        "syntheticTest": {k: score_masks(syn_test, v) for k, v in base_syn.items() if any(m is not None for m in v)},
    }

    # ---------------- every trained run ----------------
    runs, loaded = [], {}
    for variant in config.FLOOD_VARIANTS:
        for seed in config.FLOOD_SEEDS:
            run = load_run(variant, seed, device)
            if run is None:
                continue
            model, checkpoint, summary = run
            threshold = checkpoint["threshold"]
            p_real = model_probabilities(model, real_test, checkpoint, device)
            p_syn = model_probabilities(model, syn_test, checkpoint, device)
            masks_real = [p >= threshold for p in p_real]
            row = {
                "variant": variant, "seed": seed, "threshold": threshold, "bestEpoch": summary["bestEpoch"],
                "realValIou": summary["realValIouAtThreshold"],
                "realTest": score_masks(real_test, masks_real),
                "realTestIndiaOnly": score_masks(*subset(real_test, masks_real, is_india)),
                "syntheticTest": score_masks(syn_test, [p >= threshold for p in p_syn]),
            }
            runs.append(row)
            loaded[(variant, seed)] = (model, checkpoint, summary, p_real, p_syn)
            print(f"  {variant:<15} seed {seed}  real-val IoU {row['realValIou']:.4f}  real-test IoU {row['realTest']['iou']:.4f}  "
                  f"India-test IoU {row['realTestIndiaOnly']['iou']:.4f}  synthetic-test IoU {row['syntheticTest']['iou']:.4f}")
    if not runs:
        raise SystemExit("No trained flood runs found -- run `python -m ml.hazards.flood_train --all` first.")

    comparison = {}
    for variant in config.FLOOD_VARIANTS:
        rows = [r for r in runs if r["variant"] == variant]
        if rows:
            comparison[variant] = {
                "runs": len(rows), "realValIou": mean_std([r["realValIou"] for r in rows]),
                "realTestIou": mean_std([r["realTest"]["iou"] for r in rows]),
                "realTestF1": mean_std([r["realTest"]["f1"] for r in rows]),
                "realTestIndiaIou": mean_std([r["realTestIndiaOnly"]["iou"] for r in rows]),
                "syntheticTestIou": mean_std([r["syntheticTest"]["iou"] for r in rows]),
            }

    # ---------------- promotion: REAL VALIDATION only ----------------
    promoted_variant = max(comparison, key=lambda v: comparison[v]["realValIou"]["mean"])
    promoted = max((r for r in runs if r["variant"] == promoted_variant), key=lambda r: r["realValIou"])
    model, checkpoint, summary, p_real, p_syn = loaded[(promoted["variant"], promoted["seed"])]
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_path(promoted["variant"], promoted["seed"]), config.FLOOD_CHECKPOINT)

    # threshold-free scores for the promoted model, real test
    scores, truths = [], []
    for i in range(len(real_test)):
        s1, label = real_test.arrays(i)
        valid = flood_data.valid_mask(s1, label)
        scores.append(p_real[i][valid])
        truths.append((label == 1)[valid])
    scores, truths = np.concatenate(scores), np.concatenate(truths)
    probabilistic = {**{k: round(v, 4) for k, v in roc_pr_auc(scores, truths).items()},
                     "brier": round(brier_score(scores, truths), 4), "reliability": reliability_table(scores, truths)}

    per_event = {}
    masks_real = [p >= promoted["threshold"] for p in p_real]
    for event in sorted({c.region for c in real_test.chips}):
        keep = lambda c, e=event: c.region == e  # noqa: E731
        per_event[event] = {"model": score_masks(*subset(real_test, masks_real, keep)),
                            "globalThresholdVH": score_masks(*subset(real_test, base_real["globalThresholdVH"], keep)),
                            "datasetOtsuReference": score_masks(*subset(real_test, base_real["datasetOtsuReference"], keep))}
    per_kind = {}
    masks_syn = [p >= promoted["threshold"] for p in p_syn]
    for kind in sorted({c.meta["floodKind"] for c in syn_test.chips}):
        keep = lambda c, k=kind: c.meta["floodKind"] == k  # noqa: E731
        per_kind[kind] = {"model": score_masks(*subset(syn_test, masks_syn, keep)),
                          "globalThresholdVH": score_masks(*subset(syn_test, base_syn["globalThresholdVH"], keep))}

    # ---------------- figures ----------------
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    india = [i for i, c in enumerate(real_test.chips) if c.region == "India"]
    other = [i for i, c in enumerate(real_test.chips) if c.region != "India"]
    figure_examples(real_test, p_real, promoted["threshold"], base_real["globalThresholdVH"],
                    config.FIGURES_DIR / "flood_examples_real_test.png",
                    "Flood U-Net on REAL held-out test chips (Sen1Floods11) -- first three India, then Pakistan / Sri Lanka",
                    india[:3] + other[:2])
    kinds_seen, pick = set(), []
    for i, c in enumerate(syn_test.chips):
        if c.meta["floodKind"] not in kinds_seen and c.meta["floodLabel"] == "flood":
            kinds_seen.add(c.meta["floodKind"])
            pick.append(i)
    figure_examples(syn_test, p_syn, promoted["threshold"], base_syn["globalThresholdVH"],
                    config.FIGURES_DIR / "flood_examples_synthetic_test.png",
                    "Flood U-Net on SYNTHETIC India test chips -- simulations, not satellite data", pick[:6])
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(comparison)
    width = 0.27
    for k, (key, name) in enumerate((("realTestIou", "real test (all 29)"), ("realTestIndiaIou", "real test (India, 14)"),
                                     ("syntheticTestIou", "synthetic India test (40)"))):
        ax.bar(np.arange(len(labels)) + (k - 1) * width, [comparison[v][key]["mean"] for v in labels], width,
               yerr=[comparison[v][key]["std"] for v in labels], capsize=3, label=name)
    ax.axhline(baseline_report["realTest"]["globalThresholdVH"]["iou"], color="grey", ls="--", lw=1, label="VH threshold, real test")
    ax.set_xticks(np.arange(len(labels)), [f"trained on\n{v}" for v in labels])
    ax.set_ylabel("water IoU")
    ax.set_title("Flood U-Net: training-data variants (mean of 3 seeds, bars = std)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "flood_variant_comparison.png", dpi=80)
    plt.close(fig)

    # ---------------- outputs ----------------
    evaluation = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "note": "REAL = Sen1Floods11 human-labelled chips. SYNTHETIC = simulator output; scores there only show "
                "agreement with the simulator and are never evidence of real-world accuracy.",
        "promotion": {"rule": "best mean REAL validation IoU across seeds, then that variant's best seed; test never consulted",
                      "variant": promoted["variant"], "seed": promoted["seed"]},
        "variantComparison": comparison, "runs": runs, "baselines": baseline_report,
        "promoted": {"realTest": promoted["realTest"], "realTestIndiaOnly": promoted["realTestIndiaOnly"],
                     "realTestProbabilistic": probabilistic, "realTestPerEvent": per_event,
                     "syntheticTest": promoted["syntheticTest"], "syntheticTestPerFloodKind": per_kind},
    }
    config.FLOOD_EVALUATION.write_text(json.dumps(evaluation, indent=2))

    metadata = {
        "name": config.FLOOD_MODEL_NAME, "version": config.FLOOD_MODEL_VERSION, "architecture": ARCHITECTURE_NAME,
        "task": "Surface-water (flood + permanent water) segmentation from Sentinel-1 VV/VH backscatter, 10 m pixels",
        "parameters": count_parameters(model), "modelConfig": checkpoint["model_config"],
        "inputChannels": list(flood_data.CHANNELS), "normalisation": checkpoint["normalisation"],
        "threshold": promoted["threshold"], "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainedOn": promoted["variant"], "seed": promoted["seed"], "bestEpoch": summary["bestEpoch"],
        "device": summary["device"], "torchVersion": summary["torchVersion"],
        "trainingData": {
            "real": f"{len(real_train)} Sen1Floods11 hand-labelled chips (official train split; India, Pakistan, Sri Lanka)",
            "synthetic": (f"{summary['syntheticTrainChips']} synthetic India chips (ml.hazards.synth_flood)"
                          if summary["syntheticTrainChips"] else "none"),
            "validation": f"{len(real_val)} real chips (official valid split) -- model selection and threshold",
            "test": f"{len(real_test)} real chips (official test split) -- scored once, after selection",
        },
        "chipSplits": {c.chip_id: c.split for c in real},
        "testMetrics": {"real": promoted["realTest"], "realIndiaOnly": promoted["realTestIndiaOnly"],
                        "realProbabilistic": {k: v for k, v in probabilistic.items() if k != "reliability"},
                        "baselineGlobalThresholdVH": baseline_report["realTest"]["globalThresholdVH"],
                        "baselineDatasetOtsu": baseline_report["realTest"]["datasetOtsuReference"]},
        "syntheticMetrics": {"note": "synthetic India test chips -- agreement with the simulator only",
                             "model": promoted["syntheticTest"],
                             "baselineGlobalThresholdVH": baseline_report["syntheticTest"]["globalThresholdVH"]},
        "variantComparison": comparison,
        "baselineThresholdsDb": {"VH": vh_threshold, "VV": vv_threshold},
        "limitations": [
            "Trained on 80 real chips from three flood events (Assam 2016, Pakistan 2017, Sri Lanka 2017); tested on 29. "
            "This is a small-data flood-mapping model, not a nationally validated one.",
            "Maps where water IS in one radar image. It does not forecast floods, estimate depth, or tell flood water "
            "from a permanent river without a reference water layer.",
            "Radar confusions remain: wind-roughened water and flooded vegetation are missed; smooth dry surfaces, "
            "sand bars and radar shadow can be mistaken for water.",
            "No live Sentinel-1 feed is connected: it runs on the chips on disk or an uploaded GeoTIFF. "
            "Not connected to the risk engine or alerts.",
            "Scores on synthetic India chips measure agreement with a simulator, not accuracy over India.",
        ],
    }
    config.FLOOD_METADATA.write_text(json.dumps(metadata, indent=2))
    print(f"\npromoted: {promoted['variant']} seed {promoted['seed']} -> {config.FLOOD_CHECKPOINT.name} "
          f"({config.FLOOD_CHECKPOINT.stat().st_size / 1e6:.2f} MB)")
    print(json.dumps({"variantComparison": comparison, "baselinesRealTest": {k: v["iou"] for k, v in baseline_report["realTest"].items()},
                      "promotedRealTest": promoted["realTest"], "probabilistic": {k: v for k, v in probabilistic.items() if k != "reliability"}}, indent=1))


if __name__ == "__main__":
    main()
