"""Train the lightning-detection U-Net.

    python -m ml.hazards.lightning_prepare          # once
    python -m ml.hazards.lightning_train --variant real --seed 0
    python -m ml.hazards.lightning_train --all

Variants:
  real            GOES-16 infrared + GLM flashes (Americas): train anchors x train blocks
  real+synthetic  the same, plus the synthetic India training samples
  synthetic       synthetic India samples only

Whatever the variant, the checkpoint kept is the epoch with the best average
precision on the REAL validation cells (validation anchors x validation
blocks), and the decision threshold is tuned there too. Real test cells are
never read by this script.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from app.services.imagery import lightning_data as ld
from app.services.imagery.hazard_common import confusion_counts, metrics_from_counts, roc_pr_auc
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


class GoesCache:
    def __init__(self):
        if not config.LIGHTNING_CACHE.exists():
            raise SystemExit("GOES cache missing -- run `python -m ml.hazards.lightning_prepare` first.")
        with np.load(config.LIGHTNING_CACHE) as data:
            self.bt, self.flashes, self.blocks = data["bt"], data["flashes"], data["blocks"]
            self.frame_ids = [str(v) for v in data["frame_ids"]]

    def pair(self, anchor: int) -> np.ndarray:
        return self.bt[[anchor - ld.LOOKBACK_FRAMES, anchor]].astype(np.float32)

    def truth(self, anchor: int) -> np.ndarray:
        return self.flashes[anchor] > 0


class RealSampler:
    """Random crops from training anchors. Half are centred on a cell that
    really had lightning -- otherwise ~99 % of what the network sees is empty."""

    def __init__(self, cache: GoesCache, crop: int, rng: np.random.Generator):
        self.cache, self.crop, self.rng = cache, crop, rng
        self.positives = {a: np.argwhere(cache.truth(a) & (cache.blocks == 0)) for a in ld.TRAIN_ANCHORS}

    def sample(self):
        anchor = int(self.rng.choice(ld.TRAIN_ANCHORS))
        rows, cols = self.cache.bt.shape[1:]
        half = self.crop // 2
        if self.rng.random() < 0.5 and len(self.positives[anchor]):
            cr, cc = self.positives[anchor][int(self.rng.integers(len(self.positives[anchor])))] * 2
            r0 = int(np.clip(cr - half + self.rng.integers(-half // 2, half // 2), 0, rows - self.crop))
            c0 = int(np.clip(cc - half + self.rng.integers(-half // 2, half // 2), 0, cols - self.crop))
        else:
            r0, c0 = int(self.rng.integers(rows - self.crop + 1)), int(self.rng.integers(cols - self.crop + 1))
        r0, c0 = r0 // 2 * 2, c0 // 2 * 2                    # keep 4 km pixels aligned with 8 km cells
        window = (slice(r0, r0 + self.crop), slice(c0, c0 + self.crop))   # crop first: converting whole frames per sample is slow
        pair = np.stack([self.cache.bt[anchor - ld.LOOKBACK_FRAMES][window], self.cache.bt[anchor][window]]).astype(np.float32)
        cells = (slice(r0 // 2, (r0 + self.crop) // 2), slice(c0 // 2, (c0 + self.crop) // 2))
        truth = self.cache.flashes[anchor][cells] > 0
        valid = ld.cell_valid(pair) & (self.cache.blocks[cells] == 0)   # TRAIN blocks only
        return ld.features(pair), truth.astype(np.float32), valid.astype(np.float32)


class SyntheticSet:
    def __init__(self, split: str):
        self.items = []
        for sample in ld.synthetic_catalog():
            if sample.split != split:
                continue
            arrays = ld.read_synthetic(sample)
            for k in range(1, arrays["bt"].shape[0]):
                self.items.append((arrays["bt"][[k - 1, k]], arrays["flash_count"][k] > 0, sample.meta))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, rng: np.random.Generator):
        pair, truth, _ = self.items[int(rng.integers(len(self.items)))]
        return ld.features(pair), truth.astype(np.float32), np.ones(truth.shape, np.float32)


def augment(x, y, v, rng):
    k = int(rng.integers(4))
    x, y, v = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(v, k)
    if rng.random() < 0.5:
        x, y, v = x[:, :, ::-1], y[:, ::-1], v[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(v)


@torch.no_grad()
def predict_probability(model, pair_k: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(ld.features(pair_k))[None].to(device)
    return torch.sigmoid(model(x))[0, 0].float().cpu().numpy()


def real_scores(model, cache: GoesCache, anchors, block: int | None, device) -> tuple[np.ndarray, np.ndarray]:
    """(scores, truth) over the given anchors, restricted to one block split
    (None = every block)."""
    model.eval()
    scores, truths = [], []
    for anchor in anchors:
        pair = cache.pair(anchor)
        member = ld.cell_valid(pair) & ((cache.blocks == block) if block is not None else True)
        scores.append(predict_probability(model, pair, device)[member])
        truths.append(cache.truth(anchor)[member])
    return np.concatenate(scores), np.concatenate(truths)


def best_threshold(scores: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.arange(0.05, 0.96, 0.05):
        csi = metrics_from_counts(confusion_counts(scores >= threshold, truth))["iou"] or 0.0
        if csi > best[1]:
            best = (float(round(threshold, 2)), float(csi))
    return best


def run_path(variant: str, seed: int):
    return config.EXPERIMENTS_DIR / f"lightning_{variant.replace('+', '_')}_seed{seed}.pt"


def train_one(variant: str, seed: int, epochs: int | None = None) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    epochs = epochs or config.LIGHTNING_EPOCHS
    cache = GoesCache()
    real = RealSampler(cache, config.LIGHTNING_CROP, rng)
    synthetic = None
    if variant != "real":
        synthetic = SyntheticSet("train")
        if not len(synthetic):
            raise SystemExit("No synthetic lightning samples found -- run `python -m ml.hazards.synth_lightning` first.")

    model = SmallUNet(in_channels=2, base=config.LIGHTNING_BASE, depth=config.LIGHTNING_DEPTH, out_stride=2).to(device)
    iters = config.LIGHTNING_ITERS_PER_EPOCH
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.LIGHTNING_LR, weight_decay=config.LIGHTNING_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=config.LIGHTNING_LR, total_steps=epochs * iters, pct_start=0.1)
    print(f"[lightning {variant} seed {seed}] device={device} params={count_parameters(model):,} "
          f"synthetic_train_pairs={len(synthetic) if synthetic else 0}", flush=True)

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = run_path(variant, seed)
    history, best_ap, started = [], -1.0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for _ in range(iters):
            batch = []
            for _ in range(config.LIGHTNING_BATCH):
                use_synthetic = variant == "synthetic" or (variant == "real+synthetic" and rng.random() < config.LIGHTNING_SYNTHETIC_MIX)
                batch.append(augment(*(synthetic.sample(rng) if use_synthetic else real.sample()), rng))
            x, y, v = (torch.from_numpy(np.stack(a)).to(device) for a in zip(*batch))
            loss = segmentation_loss(model(x)[:, 0], y, v, pos_weight=config.LIGHTNING_POS_WEIGHT)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += loss.item()
        scores, truth = real_scores(model, cache, ld.VAL_ANCHORS, 1, device)
        ap = roc_pr_auc(scores, truth)["prAuc"] or 0.0
        improved = ap > best_ap
        history.append({"epoch": epoch, "trainLoss": running / iters, "realValPrAuc": ap, "elapsedSeconds": round(time.time() - started, 1)})
        if improved:
            best_ap = ap
            torch.save({
                "state_dict": {k: t.cpu() for k, t in model.state_dict().items()},
                "model_config": model.config(), "threshold": 0.5, "epoch": epoch,
                "input": {"btCentreK": ld.BT_CENTRE_K, "btScaleK": ld.BT_SCALE_K, "trendScaleK": ld.TREND_SCALE_K},
                "model_name": config.LIGHTNING_MODEL_NAME, "model_version": config.LIGHTNING_MODEL_VERSION,
                "variant": variant, "seed": seed,
            }, path)
        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  epoch {epoch:2d}  loss {running / iters:.4f}  real-val PR-AUC {ap:.4f} {'*' if improved else ''}", flush=True)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    scores, truth = real_scores(model, cache, ld.VAL_ANCHORS, 1, device)
    threshold, csi = best_threshold(scores, truth)
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, path)
    summary = {
        "variant": variant, "seed": seed, "device": str(device), "parameters": count_parameters(model),
        "epochs": epochs, "bestEpoch": checkpoint["epoch"], "realValPrAuc": best_ap,
        "threshold": threshold, "realValCsiAtThreshold": csi,
        "syntheticTrainPairs": len(synthetic) if synthetic else 0, "trainSeconds": round(time.time() - started, 1),
        "torchVersion": str(torch.__version__), "history": history,
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[lightning {variant} seed {seed}] best epoch {checkpoint['epoch']}  real-val PR-AUC {best_ap:.4f}  "
          f"CSI {csi:.4f} @ threshold {threshold}  ({summary['trainSeconds']:.0f}s)", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=config.LIGHTNING_VARIANTS, default="real")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        for variant in config.LIGHTNING_VARIANTS:
            for seed in config.LIGHTNING_SEEDS:
                train_one(variant, seed, args.epochs)
    else:
        train_one(args.variant, args.seed, args.epochs)


if __name__ == "__main__":
    main()
