"""Held-out TEST-set evaluation of the trained nowcast model against the two
non-learned baselines, plus figures and the model's metadata file.

    python -m ml.imagery.evaluate

The model is loaded and run through the exact serving code path
(`app.services.imagery.ml_inference`), so what is measured here is what the
API serves. Nothing in this script feeds back into training.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from app.services.imagery import insat, ml_inference  # noqa: E402
from app.services.imagery.nowcast_common import (  # noqa: E402
    COLD_CLOUD_THRESHOLD_K,
    INPUT_CHANNELS,
    LEAD_MINUTES,
    LEAD_STEPS,
    N_INPUT_FRAMES,
    laplace_cdf,
    mask_counts,
    mask_metrics_from_counts,
    optical_flow_forecast,
)
from app.services.imagery.nowcast_model import ARCHITECTURE_NAME, count_parameters  # noqa: E402
from ml.imagery import config  # noqa: E402
from ml.imagery.dataset import FrameStore  # noqa: E402

METHODS = ("model", "opticalFlow", "persistence")
METHOD_LABELS = {"model": "ConvLSTM (trained)", "opticalFlow": "OpenCV optical flow", "persistence": "Persistence"}
AUC_PIXELS_PER_FRAME = 150_000
RELIABILITY_BINS = 10


class Accumulator:
    def __init__(self):
        self.abs = self.sq = self.err = 0.0
        self.n = 0
        self.counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.per_frame_mae: list[float] = []

    def add(self, pred: np.ndarray, truth: np.ndarray) -> None:
        err = (pred - truth).astype(np.float64)
        self.abs += np.abs(err).sum()
        self.sq += (err ** 2).sum()
        self.err += err.sum()
        self.n += err.size
        self.per_frame_mae.append(float(np.abs(err).mean()))
        for key, value in mask_counts(pred < COLD_CLOUD_THRESHOLD_K, truth < COLD_CLOUD_THRESHOLD_K).items():
            self.counts[key] += value

    def summary(self) -> dict:
        return {
            "maeK": self.abs / self.n, "rmseK": float(np.sqrt(self.sq / self.n)), "biasK": self.err / self.n,
            "pixels": self.n, "frames": len(self.per_frame_mae), "perFrameMaeK": self.per_frame_mae,
            "coldCloud": mask_metrics_from_counts(self.counts),
        }


def serving_path_check(store: FrameStore) -> float:
    """Max |difference| (K) between brightness temperature as the API reads
    it from HDF5 and as training read it from the cache. Must be 0."""
    anchor = store.anchors_for("test")[0]
    window = insat.PixelWindow(700, 828, 800, 928)
    worst = 0.0
    for frame in anchor.inputs:
        info = insat.get_frame(store.index["frameIds"][frame], config.INSAT_DIR)
        for channel in INPUT_CHANNELS:
            served = insat.read_bt(info.path, channel, window)
            trained = store.bt(channel, frame, window.row0, window.col0, *window.shape)
            worst = max(worst, float(np.abs(served - trained).max()))
    return worst


def evaluate(store: FrameStore, loaded, split: str = "test") -> tuple[dict, dict]:
    rng = np.random.default_rng(config.RANDOM_SEED)
    north = insat.region_window("north_india")
    region_slice = (slice(north.row0, north.row1), slice(north.col0, north.col1))
    acc = {m: [Accumulator() for _ in LEAD_STEPS] for m in METHODS}
    acc_region = {m: [Accumulator() for _ in LEAD_STEPS] for m in METHODS}
    auc_truth = [[] for _ in LEAD_STEPS]
    auc_scores = {key: [[] for _ in LEAD_STEPS] for key in ("modelProbability", "model", "opticalFlow", "persistence")}
    brier = np.zeros(len(LEAD_STEPS))
    cover50 = np.zeros(len(LEAD_STEPS))
    cover90 = np.zeros(len(LEAD_STEPS))
    scale_sum = np.zeros(len(LEAD_STEPS))
    pixels = np.zeros(len(LEAD_STEPS))
    rel_pred = np.zeros((len(LEAD_STEPS), RELIABILITY_BINS))
    rel_obs = np.zeros_like(rel_pred)
    rel_n = np.zeros_like(rel_pred)
    example = {}
    anchors = store.anchors_for(split)
    inference_ms = []

    for a_i, anchor in enumerate(anchors):
        frames_k = np.stack([
            np.stack([store.bt(c, f) for c in INPUT_CHANNELS]) for f in anchor.inputs
        ]).astype(np.float32)
        prediction = ml_inference.predict(frames_k, loaded)
        inference_ms.append(prediction.inference_ms)
        last = frames_k[-1, 0]
        flow = optical_flow_forecast(frames_k[-2, 0], last)

        for k, target in enumerate(anchor.targets):
            if target < 0:
                continue
            truth = store.bt("TIR1", target)
            forecasts = {"model": prediction.mean_k[k], "opticalFlow": flow[k], "persistence": last}
            for method, pred in forecasts.items():
                acc[method][k].add(pred, truth)
                acc_region[method][k].add(pred[region_slice], truth[region_slice])

            scale = prediction.scale_k[k]
            prob = laplace_cdf(COLD_CLOUD_THRESHOLD_K, prediction.mean_k[k], scale)
            cold = truth < COLD_CLOUD_THRESHOLD_K
            abs_err = np.abs(prediction.mean_k[k] - truth)
            brier[k] += ((prob - cold) ** 2).sum()
            cover50[k] += (abs_err <= scale * np.log(2.0)).sum()
            cover90[k] += (abs_err <= scale * np.log(10.0)).sum()
            scale_sum[k] += scale.sum()
            pixels[k] += truth.size
            bins = np.minimum((prob * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
            rel_n[k] += np.bincount(bins.ravel(), minlength=RELIABILITY_BINS)
            rel_pred[k] += np.bincount(bins.ravel(), weights=prob.ravel(), minlength=RELIABILITY_BINS)
            rel_obs[k] += np.bincount(bins.ravel(), weights=cold.ravel().astype(np.float64), minlength=RELIABILITY_BINS)

            pick = rng.choice(truth.size, AUC_PIXELS_PER_FRAME, replace=False)
            auc_truth[k].append(cold.ravel()[pick])
            auc_scores["modelProbability"][k].append(prob.ravel()[pick])
            for method, pred in forecasts.items():
                auc_scores[method][k].append(-pred.ravel()[pick])  # colder forecast = stronger "cold cloud" score

            if a_i == len(anchors) // 2 and k == 1:
                example = {"anchor": anchor, "lead": k, "last": last, "truth": truth, **forecasts, "scale": scale, "prob": prob}

    leads = {}
    for k, minutes in enumerate(LEAD_MINUTES):
        truth_flat = np.concatenate(auc_truth[k])
        per_method = {m: acc[m][k].summary() for m in METHODS}
        for key in auc_scores:
            target = per_method["model"] if key in ("modelProbability", "model") else per_method[key]
            name = "rocAucProbability" if key == "modelProbability" else "rocAucTemperatureScore"
            target["coldCloud"][name] = float(roc_auc_score(truth_flat, np.concatenate(auc_scores[key][k])))
        model, flow_m, pers = per_method["model"], per_method["opticalFlow"], per_method["persistence"]
        model["coldCloud"]["brier"] = float(brier[k] / pixels[k])
        for baseline in (flow_m, pers):  # a yes/no forecast's Brier score is its error rate
            c = baseline["coldCloud"]
            c["brier"] = (c["fp"] + c["fn"]) / (c["tp"] + c["fp"] + c["fn"] + c["tn"])
        frames = model["frames"]
        leads[str(minutes)] = {
            "frames": frames,
            **per_method,
            "skill": {
                "maeReductionVsPersistencePct": 100 * (1 - model["maeK"] / pers["maeK"]),
                "maeReductionVsOpticalFlowPct": 100 * (1 - model["maeK"] / flow_m["maeK"]),
                "opticalFlowMaeReductionVsPersistencePct": 100 * (1 - flow_m["maeK"] / pers["maeK"]),
                "framesModelBeatsOpticalFlow": int(sum(a < b for a, b in zip(model["perFrameMaeK"], flow_m["perFrameMaeK"]))),
                "framesModelBeatsPersistence": int(sum(a < b for a, b in zip(model["perFrameMaeK"], pers["perFrameMaeK"]))),
            },
            "uncertainty": {
                "meanPredictedScaleK": float(scale_sum[k] / pixels[k]),
                "actualMaeK": model["maeK"],
                "coverage50Pct": float(100 * cover50[k] / pixels[k]),
                "coverage90Pct": float(100 * cover90[k] / pixels[k]),
                "reliability": [
                    {"bin": f"{b / RELIABILITY_BINS:.1f}-{(b + 1) / RELIABILITY_BINS:.1f}",
                     "meanPredicted": float(rel_pred[k, b] / rel_n[k, b]) if rel_n[k, b] else None,
                     "observedFrequency": float(rel_obs[k, b] / rel_n[k, b]) if rel_n[k, b] else None,
                     "pixels": int(rel_n[k, b])}
                    for b in range(RELIABILITY_BINS)
                ],
            },
            "northIndia": {m: {key: acc_region[m][k].summary()[key] for key in ("maeK", "rmseK")} for m in METHODS},
        }

    report = {
        "split": split,
        "anchorTimes": [store.timestamps[a.index].isoformat() for a in anchors],
        "sequences": len(anchors),
        "coldCloudThresholdK": COLD_CLOUD_THRESHOLD_K,
        "evaluatedOn": "full 1616x1737 sector, every pixel, native 4 km resolution",
        "device": str(loaded.device),
        "meanInferenceMsFullSector": float(np.mean(inference_ms)),
        "leads": leads,
    }
    return report, example


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figure_mae(report: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colours = {"model": "#7c3aed", "opticalFlow": "#0891b2", "persistence": "#9ca3af"}
    width = 0.26
    for i, method in enumerate(METHODS):
        values = [report["leads"][str(m)][method]["maeK"] for m in LEAD_MINUTES]
        bars = ax.bar(np.arange(len(LEAD_MINUTES)) + (i - 1) * width, values, width, label=METHOD_LABELS[method], color=colours[method])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(LEAD_MINUTES)), [f"+{m} min" for m in LEAD_MINUTES])
    ax.set_ylabel("mean absolute error (K) -- lower is better")
    ax.set_title(f"Held-out test set ({report['sequences']} sequences, full sector)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_confusion(report: dict, path) -> None:
    fig, axes = plt.subplots(2, len(LEAD_MINUTES), figsize=(3.6 * len(LEAD_MINUTES), 7))
    for row, method in enumerate(("model", "opticalFlow")):
        for col, minutes in enumerate(LEAD_MINUTES):
            c = report["leads"][str(minutes)][method]["coldCloud"]
            matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=np.float64)
            share = matrix / matrix.sum(axis=1, keepdims=True)
            ax = axes[row, col]
            ax.imshow(share, cmap="Purples" if method == "model" else "GnBu", vmin=0, vmax=1)
            for (i, j), value in np.ndenumerate(share):
                ax.text(j, i, f"{100 * value:.1f}%\n{int(matrix[i, j]):,}", ha="center", va="center",
                        color="white" if value > 0.5 else "black", fontsize=8)
            ax.set_xticks([0, 1], ["pred warm", "pred cold"])
            ax.set_yticks([0, 1], ["actual warm", "actual cold"])
            ax.set_title(f"{METHOD_LABELS[method]}, +{minutes} min\nIoU {c['iou']:.3f}", fontsize=9)
    fig.suptitle(f"Cold-cloud mask (TIR1 < {COLD_CLOUD_THRESHOLD_K:.0f} K) confusion matrices -- test set, row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def figure_reliability(report: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="perfectly calibrated")
    for minutes, colour in zip(LEAD_MINUTES, ("#c4b5fd", "#8b5cf6", "#5b21b6")):
        rows = [r for r in report["leads"][str(minutes)]["uncertainty"]["reliability"] if r["pixels"]]
        ax.plot([r["meanPredicted"] for r in rows], [r["observedFrequency"] for r in rows], "o-", color=colour, label=f"+{minutes} min")
    ax.set_xlabel("model's predicted probability of cold cloud")
    ax.set_ylabel("how often it actually was cold cloud")
    ax.set_title("Reliability of the model's confidence (test set)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_example(example: dict, store: FrameStore, path) -> None:
    window = insat.region_window("india")
    sl = (slice(window.row0, window.row1), slice(window.col0, window.col1))
    minutes = LEAD_MINUTES[example["lead"]]
    when = store.timestamps[example["anchor"].index]
    panels = [
        (f"last observed ({when:%d %b %H:%M} UTC)", example["last"][sl], "ir"),
        (f"ACTUAL +{minutes} min", example["truth"][sl], "ir"),
        ("ConvLSTM prediction", example["model"][sl], "ir"),
        ("OpenCV optical-flow prediction", example["opticalFlow"][sl], "ir"),
        ("ConvLSTM error (K)", (example["model"] - example["truth"])[sl], "err"),
        ("optical-flow error (K)", (example["opticalFlow"] - example["truth"])[sl], "err"),
        ("persistence error (K)", (example["last"] - example["truth"])[sl], "err"),
        ("ConvLSTM predicted uncertainty (K)", example["scale"][sl], "unc"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for ax, (title, image, kind) in zip(axes.ravel(), panels):
        if kind == "ir":
            shown = ax.imshow(image, cmap="gray_r", vmin=190, vmax=310)
        elif kind == "err":
            shown = ax.imshow(image, cmap="RdBu_r", vmin=-15, vmax=15)
            title += f" -- MAE {np.abs(image).mean():.2f}"
        else:
            shown = ax.imshow(image, cmap="inferno", vmin=0, vmax=8)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(shown, ax=ax, fraction=0.046)
    fig.suptitle("Held-out test example, Indian subcontinent crop, INSAT-3DR TIR1 brightness temperature")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def figure_training(history: dict, path) -> None:
    rows = history["history"]
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [r["epoch"] for r in rows]
    ax.plot(epochs, [r["meanMaeK"] for r in rows], "o-", color="#7c3aed", label="validation MAE (model)")
    ax.axhline(rows[0]["meanPersistenceMaeK"], color="#9ca3af", ls="--", label="validation MAE (persistence)")
    best = int(np.argmin([r["meanMaeK"] for r in rows]))
    ax.axvline(epochs[best], color="#16a34a", ls=":", label=f"checkpoint kept (epoch {epochs[best]})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean absolute error (K), mean of 3 leads")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    store = FrameStore()
    ml_inference.clear_cache()
    loaded = ml_inference.load_model()
    diff = serving_path_check(store)
    assert diff == 0.0, f"serving and training read different brightness temperatures (max diff {diff} K)"

    report, example = evaluate(store, loaded, "test")
    report["servingVsTrainingInputMaxAbsDiffK"] = diff
    config.EVALUATION_PATH.write_text(json.dumps(report, indent=2))

    history = json.loads(config.HISTORY_PATH.read_text())
    figure_mae(report, config.FIGURES_DIR / "test_mae_by_lead.png")
    figure_confusion(report, config.FIGURES_DIR / "test_cold_cloud_confusion.png")
    figure_reliability(report, config.FIGURES_DIR / "test_reliability.png")
    figure_example(example, store, config.FIGURES_DIR / "test_example_india.png")
    figure_training(history, config.FIGURES_DIR / "training_curve.png")

    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    summary = {
        minutes: {
            method: {
                "maeK": round(lead[method]["maeK"], 4), "rmseK": round(lead[method]["rmseK"], 4),
                "coldCloudIou": round(lead[method]["coldCloud"]["iou"], 4),
                "coldCloudPrecision": round(lead[method]["coldCloud"]["precision"], 4),
                "coldCloudRecall": round(lead[method]["coldCloud"]["recall"], 4),
                "coldCloudDice": round(lead[method]["coldCloud"]["dice"], 4),
            } for method in METHODS
        } | {"skill": {k: round(v, 2) if isinstance(v, float) else v for k, v in lead["skill"].items()}, "frames": lead["frames"]}
        for minutes, lead in report["leads"].items()
    }
    metadata = {
        "name": config.MODEL_NAME,
        "version": config.MODEL_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "framework": f"PyTorch {torch.__version__}",
        "parameters": count_parameters(loaded.model),
        "artifactFile": config.CHECKPOINT_PATH.name,
        "artifactSizeBytes": config.CHECKPOINT_PATH.stat().st_size,
        "trainedAt": datetime.fromtimestamp(config.CHECKPOINT_PATH.stat().st_mtime, tz=timezone.utc).isoformat(),
        "task": "Predict INSAT-3DR TIR1 (10.8 um) brightness temperature 30/60/90 minutes ahead from the last "
                f"{N_INPUT_FRAMES} half-hourly TIR1 + WV frames. Self-supervised: the target is the later satellite frame.",
        "notA": "weather forecast, rainfall estimate, storm detector or classifier",
        "input": {"channels": list(INPUT_CHANNELS), "frames": N_INPUT_FRAMES, "units": "K", "cadenceMinutes": 30,
                  "normalisation": checkpoint["normalisation"], "normalisationSource": "training-split frames only"},
        "output": {"leadMinutes": list(LEAD_MINUTES), "mean": "TIR1 brightness temperature (K)",
                   "scale": "Laplace scale b (K) = predicted mean absolute error",
                   "coldCloudProbability": f"P(TIR1 < {COLD_CLOUD_THRESHOLD_K:.0f} K) from the Laplace distribution"},
        "trainingData": {
            "source": "ISRO MOSDAC INSAT-3DR Imager L1C ASIA Mercator sector (3RIMG_L1C_ASIA_MER), 4 km",
            "period": f"{store.timestamps[0]:%Y-%m-%d %H:%M} to {store.timestamps[-1]:%Y-%m-%d %H:%M} UTC",
            "validFrames": len(store.timestamps),
            "sequences": {s: len(store.anchors_for(s)) for s in ("train", "val", "test")},
            "splitRule": "chronological; every input and target frame of a sample lies in one split",
        },
        "training": {
            "seed": config.RANDOM_SEED, "device": history["device"], "optimiser": "AdamW",
            "learningRate": config.LEARNING_RATE, "schedule": "one-cycle", "weightDecay": config.WEIGHT_DECAY,
            "batchSize": config.BATCH_SIZE, "cropPx": config.CROP_SIZE, "itersPerEpoch": config.ITERS_PER_EPOCH,
            "maxEpochs": config.MAX_EPOCHS, "earlyStopPatience": config.EARLY_STOP_PATIENCE,
            "epochsRun": len(history["history"]), "bestEpoch": checkpoint["epoch"],
            "bestValMeanMaeK": history["bestValMeanMaeK"],
            "valPersistenceMeanMaeK": history["untrainedValidation"]["meanPersistenceMaeK"],
            "loss": "L1 on the mean + Laplace NLL on the scale (mean detached)",
            "augmentation": "random crop, horizontal/vertical flip, transpose",
            "imbalanceHandling": f"{int(100 * config.ACTIVITY_SAMPLING_FRACTION)}% of crops sampled in proportion to scene change",
            "wallClockSeconds": history["history"][-1]["elapsedSeconds"],
        },
        "frameSplits": dict(zip(store.index["frameIds"], store.splits)),
        "testMetrics": summary,
        "calibration": {
            minutes: {k: round(v, 3) for k, v in lead["uncertainty"].items() if k != "reliability"}
            for minutes, lead in report["leads"].items()
        },
        "limitations": [
            "Trained and tested on one continuous 34-hour period (6-7 January 2025). The test frames are later in the "
            "same weather situation, not a different day or season; performance in the monsoon is untested.",
            f"Test set is small: {len(store.anchors_for('test'))} consecutive sequences. Consecutive frames are highly "
            "correlated, so treat differences of a few percent as indicative, not proven.",
            "Predictions are smoother than real imagery (the usual behaviour of an error-minimising model).",
            "Predicts cloud-top infrared imagery only. Not a rainfall, lightning, flood or storm forecast, and not "
            "connected to AtmoGuard's risk engine or alerts.",
            "Cold-cloud masks come from a 235 K threshold on the image, not from human-labelled data.",
        ],
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"test sequences: {report['sequences']}  serving-vs-training input diff: {diff} K")
    for minutes, lead in report["leads"].items():
        print(f"+{minutes} min ({lead['frames']} frames)")
        for method in METHODS:
            m, c = lead[method], lead[method]["coldCloud"]
            print(f"   {METHOD_LABELS[method]:22s} MAE {m['maeK']:.3f} K  RMSE {m['rmseK']:.3f} K  "
                  f"cold IoU {c['iou']:.3f}  P {c['precision']:.3f}  R {c['recall']:.3f}  AUC {c['rocAucTemperatureScore']:.4f}")
        print(f"   skill: {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in lead['skill'].items()})}")
        u = lead["uncertainty"]
        print(f"   uncertainty: predicted {u['meanPredictedScaleK']:.2f} K vs actual MAE {u['actualMaeK']:.2f} K; "
              f"50% interval covers {u['coverage50Pct']:.1f}%, 90% covers {u['coverage90Pct']:.1f}%")


if __name__ == "__main__":
    main()
