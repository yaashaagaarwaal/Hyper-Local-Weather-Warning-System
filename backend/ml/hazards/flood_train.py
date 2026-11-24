"""Train the flood-mapping U-Net.

    python -m ml.hazards.flood_train --variant real --seed 0
    python -m ml.hazards.flood_train --all          # every variant x seed in config

Variants:
  real            the 80 official-train Sen1Floods11 chips (India, Pakistan, Sri Lanka)
  real+synthetic  the same, plus the synthetic India training chips
  synthetic       synthetic India chips only (how far does the simulator alone get?)

Whatever the variant, the checkpoint kept is the epoch with the best pooled
water IoU on the REAL validation chips, and the decision threshold is tuned
on those same chips. The real test chips are never read by this script.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from app.services.imagery import flood_data
from app.services.imagery.hazard_common import add_counts, confusion_counts, metrics_from_counts
from app.services.imagery.hazard_models import SmallUNet, count_parameters, segmentation_loss
from ml.hazards import config


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChipSet:
    """Chips held in memory as (radar float16, label int8)."""

    def __init__(self, chips: list[flood_data.FloodChip]):
        self.chips = chips
        self.s1 = [flood_data.read_s1(c).astype(np.float16) for c in chips]
        self.label = [flood_data.read_label(c) for c in chips]

    def __len__(self) -> int:
        return len(self.chips)

    def arrays(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.s1[index].astype(np.float32), self.label[index]


def training_statistics(train: ChipSet) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std of clipped backscatter over the training chips'
    measured pixels."""
    total, total_sq, n = np.zeros(2), np.zeros(2), 0
    for i in range(len(train)):
        s1, _ = train.arrays(i)
        ok = flood_data.data_mask(s1)
        values = np.clip(s1[:, ok], *flood_data.CLIP_DB).astype(np.float64)
        total += values.sum(axis=1)
        total_sq += (values ** 2).sum(axis=1)
        n += values.shape[1]
    mean = total / n
    return mean.astype(np.float32), np.sqrt(total_sq / n - mean ** 2).astype(np.float32)


def augment(x: np.ndarray, y: np.ndarray, v: np.ndarray, rng: np.random.Generator):
    """Geometric only -- backscatter is a physical quantity, so no colour
    jitter. Flips and quarter-turns are safe: water has no preferred
    orientation at chip scale."""
    k = int(rng.integers(4))
    x, y, v = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(v, k)
    if rng.random() < 0.5:
        x, y, v = x[:, :, ::-1], y[:, ::-1], v[:, ::-1]
    return x, y, v


def sample_crop(chips: ChipSet, mean, std, rng: np.random.Generator, crop: int):
    """One random crop; re-drawn (up to 8 times) if it has almost no labelled
    pixels, which would contribute nothing to the loss."""
    for _ in range(8):
        index = int(rng.integers(len(chips)))
        s1, label = chips.arrays(index)
        height, width = label.shape
        r, c = int(rng.integers(height - crop + 1)), int(rng.integers(width - crop + 1))
        s1, label = s1[:, r:r + crop, c:c + crop], label[r:r + crop, c:c + crop]
        valid = flood_data.valid_mask(s1, label)
        if valid.mean() > 0.05:
            break
    x, y, v = augment(flood_data.normalise(s1, mean, std), (label == 1), valid, rng)
    return np.ascontiguousarray(x), np.ascontiguousarray(y, dtype=np.float32), np.ascontiguousarray(v, dtype=np.float32)


@torch.no_grad()
def predict_probability(model, s1: np.ndarray, mean, std, device) -> np.ndarray:
    x = torch.from_numpy(flood_data.normalise(s1, mean, std))[None].to(device)
    return torch.sigmoid(model(x))[0, 0].float().cpu().numpy()


def evaluate(model, chips: ChipSet, mean, std, device, threshold: float = 0.5) -> tuple[dict, list[np.ndarray]]:
    model.eval()
    counts: dict = {}
    probabilities = []
    for i in range(len(chips)):
        s1, label = chips.arrays(i)
        prob = predict_probability(model, s1, mean, std, device)
        probabilities.append(prob)
        add_counts(counts, confusion_counts(prob >= threshold, label == 1, flood_data.valid_mask(s1, label)))
    return metrics_from_counts(counts), probabilities


def tune_threshold(chips: ChipSet, probabilities: list[np.ndarray]) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.arange(0.1, 0.91, 0.05):
        counts: dict = {}
        for i in range(len(chips)):
            s1, label = chips.arrays(i)
            add_counts(counts, confusion_counts(probabilities[i] >= threshold, label == 1, flood_data.valid_mask(s1, label)))
        iou = metrics_from_counts(counts)["iou"] or 0.0
        if iou > best[1]:
            best = (float(round(threshold, 2)), float(iou))
    return best


def run_path(variant: str, seed: int):
    return config.EXPERIMENTS_DIR / f"flood_{variant.replace('+', '_')}_seed{seed}.pt"


def train_one(variant: str, seed: int, epochs: int | None = None) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    epochs = epochs or config.FLOOD_EPOCHS

    real = flood_data.real_catalog()
    real_train = ChipSet([c for c in real if c.split == "train"])
    real_val = ChipSet([c for c in real if c.split == "valid"])
    synthetic_train = None
    if variant != "real":
        synthetic_train = ChipSet([c for c in flood_data.synthetic_catalog() if c.split == "train"])
        if not len(synthetic_train):
            raise SystemExit("No synthetic flood chips found -- run `python -m ml.hazards.synth_flood` first.")
    # Normalisation always comes from the REAL training chips, so every
    # variant (and the served model) shares one definition of the input.
    mean, std = training_statistics(real_train)

    model = SmallUNet(in_channels=2, base=config.FLOOD_BASE, depth=config.FLOOD_DEPTH).to(device)
    n_real = len(real_train) if variant != "synthetic" else 0
    n_syn = len(synthetic_train) if synthetic_train is not None else 0
    chips_per_epoch = max(n_real, 80) if variant != "synthetic" else 80
    iters = chips_per_epoch * config.FLOOD_CROPS_PER_CHIP // config.FLOOD_BATCH
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.FLOOD_LR, weight_decay=config.FLOOD_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=config.FLOOD_LR, total_steps=epochs * iters, pct_start=0.1)
    print(f"[flood {variant} seed {seed}] device={device} params={count_parameters(model):,} "
          f"real_train={n_real} synthetic_train={n_syn} real_val={len(real_val)} iters/epoch={iters}", flush=True)

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = run_path(variant, seed)
    history, best_iou, started = [], -1.0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for _ in range(iters):
            batch = []
            for _ in range(config.FLOOD_BATCH):
                use_synthetic = variant == "synthetic" or (variant == "real+synthetic" and rng.random() < config.FLOOD_SYNTHETIC_MIX)
                batch.append(sample_crop(synthetic_train if use_synthetic else real_train, mean, std, rng, config.FLOOD_CROP))
            x, y, v = (torch.from_numpy(np.stack(a)).to(device) for a in zip(*batch))
            loss = segmentation_loss(model(x)[:, 0], y, v)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += loss.item()
        val, _ = evaluate(model, real_val, mean, std, device)
        iou = val["iou"] or 0.0
        improved = iou > best_iou
        history.append({"epoch": epoch, "trainLoss": running / iters, "realValIou": iou, "realValF1": val["f1"],
                        "elapsedSeconds": round(time.time() - started, 1)})
        if improved:
            best_iou = iou
            torch.save({
                "state_dict": {k: t.cpu() for k, t in model.state_dict().items()},
                "model_config": model.config(),
                "normalisation": {"mean": mean.tolist(), "std": std.tolist()},
                "threshold": 0.5, "epoch": epoch,
                "model_name": config.FLOOD_MODEL_NAME, "model_version": config.FLOOD_MODEL_VERSION,
                "variant": variant, "seed": seed,
            }, path)
        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  epoch {epoch:2d}  loss {running / iters:.4f}  real-val IoU {iou:.4f} {'*' if improved else ''}", flush=True)

    # Decision threshold: tuned on REAL validation only, stored in the checkpoint.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    _, probabilities = evaluate(model, real_val, mean, std, device)
    threshold, tuned_iou = tune_threshold(real_val, probabilities)
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, path)
    summary = {
        "variant": variant, "seed": seed, "device": str(device), "parameters": count_parameters(model),
        "epochs": epochs, "bestEpoch": checkpoint["epoch"], "realValIouAt0.5": best_iou,
        "threshold": threshold, "realValIouAtThreshold": tuned_iou,
        "realTrainChips": n_real, "syntheticTrainChips": n_syn, "trainSeconds": round(time.time() - started, 1),
        "torchVersion": str(torch.__version__), "history": history,
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[flood {variant} seed {seed}] best epoch {checkpoint['epoch']}  real-val IoU {tuned_iou:.4f} "
          f"@ threshold {threshold}  ({summary['trainSeconds']:.0f}s)", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=config.FLOOD_VARIANTS, default="real")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="train every variant x seed listed in ml/hazards/config.py")
    args = parser.parse_args()
    if args.all:
        for variant in config.FLOOD_VARIANTS:
            for seed in config.FLOOD_SEEDS:
                train_one(variant, seed, args.epochs)
    else:
        train_one(args.variant, args.seed, args.epochs)


if __name__ == "__main__":
    main()
