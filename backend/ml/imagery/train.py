"""Train the INSAT-3DR infrared nowcast model.

    python -m ml.imagery.preprocess     # once
    python -m ml.imagery.train
    python -m ml.imagery.evaluate       # test-set metrics + figures + metadata

Early stopping and checkpoint selection use the VALIDATION split only. The
test split is not touched by this script at all.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

from app.services.imagery.nowcast_common import LEAD_MINUTES, LEAD_STEPS
from app.services.imagery.nowcast_model import InsatNowcastNet, count_parameters, nowcast_loss, tiled_forward
from ml.imagery import config
from ml.imagery.dataset import Anchor, FrameStore, TrainSampler


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


def predict_full_frame(model: InsatNowcastNet, store: FrameStore, anchor: Anchor, device: torch.device):
    """Whole-sector (tiled) inference for one sample. Returns
    (mean_K, scale_K), each (L, rows, cols) float32 in kelvin."""
    mean, scale = tiled_forward(model, store.normalised_inputs(anchor), device)
    return mean * store.std[0] + store.mean[0], scale * store.std[0]


def validate(model, store, device) -> dict:
    """Mean absolute error in kelvin on full validation frames, per lead,
    for the model and for persistence (context only)."""
    model.eval()
    abs_model = np.zeros(len(LEAD_STEPS))
    abs_persist = np.zeros(len(LEAD_STEPS))
    n = np.zeros(len(LEAD_STEPS))
    for anchor in store.anchors_for("val"):
        mean, _ = predict_full_frame(model, store, anchor, device)
        last = store.bt("TIR1", anchor.inputs[-1])
        for k, frame in enumerate(anchor.targets):
            if frame < 0:
                continue
            truth = store.bt("TIR1", frame)
            abs_model[k] += np.abs(mean[k] - truth).mean()
            abs_persist[k] += np.abs(last - truth).mean()
            n[k] += 1
    model_mae = abs_model / n
    persist_mae = abs_persist / n
    return {
        "maeK": {str(m): float(v) for m, v in zip(LEAD_MINUTES, model_mae)},
        "persistenceMaeK": {str(m): float(v) for m, v in zip(LEAD_MINUTES, persist_mae)},
        "meanMaeK": float(model_mae.mean()),
        "meanPersistenceMaeK": float(persist_mae.mean()),
    }


def main() -> None:
    seed_everything(config.RANDOM_SEED)
    device = pick_device()
    store = FrameStore()
    sampler = TrainSampler(store)
    model = InsatNowcastNet(in_channels=len(store.mean), base=config.BASE_CHANNELS, n_leads=len(LEAD_STEPS)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=config.LEARNING_RATE, total_steps=config.MAX_EPOCHS * config.ITERS_PER_EPOCH, pct_start=0.1,
    )
    print(f"device={device} params={count_parameters(model):,} train_samples={len(sampler.anchors)} "
          f"val_samples={len(store.anchors_for('val'))}")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("inf"), 0
    started = time.time()
    baseline = validate(model, store, device)  # untrained == persistence, by construction
    print(f"epoch  0  val MAE {baseline['meanMaeK']:.3f} K (persistence {baseline['meanPersistenceMaeK']:.3f} K)")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        sums = np.zeros(3)
        for _ in range(config.ITERS_PER_EPOCH):
            x, y, mask = (torch.from_numpy(a).to(device) for a in sampler.batch())
            mean, scale = model(x)
            loss, l1, nll = nowcast_loss(mean, scale, y, mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimiser.step()
            scheduler.step()
            sums += [loss.item(), l1.item(), nll.item()]
        train_loss, train_l1, train_nll = sums / config.ITERS_PER_EPOCH
        val = validate(model, store, device)
        row = {
            "epoch": epoch, "trainLoss": float(train_loss), "trainL1Normalised": float(train_l1),
            "trainL1K": float(train_l1 * store.std[0]), "trainNll": float(train_nll),
            "lr": float(scheduler.get_last_lr()[0]), "elapsedSeconds": round(time.time() - started, 1), **val,
        }
        history.append(row)
        improved = val["meanMaeK"] < best - 1e-4
        if improved:
            best, stale = val["meanMaeK"], 0
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "model_config": model.config(),
                "normalisation": {"mean": store.mean.tolist(), "std": store.std.tolist()},
                "epoch": epoch, "val": val,
                "model_name": config.MODEL_NAME, "model_version": config.MODEL_VERSION,
            }, config.CHECKPOINT_PATH)
        else:
            stale += 1
        print(f"epoch {epoch:2d}  train L1 {row['trainL1K']:.3f} K  val MAE {val['meanMaeK']:.3f} K "
              f"(persistence {val['meanPersistenceMaeK']:.3f})  {'*' if improved else ' '}  {row['elapsedSeconds']:.0f}s", flush=True)
        config.HISTORY_PATH.write_text(json.dumps({
            "device": str(device), "torchVersion": torch.__version__, "seed": config.RANDOM_SEED,
            "parameters": count_parameters(model), "untrainedValidation": baseline,
            "bestValMeanMaeK": best, "history": history,
        }, indent=2))
        if stale >= config.EARLY_STOP_PATIENCE:
            print(f"early stop: no validation improvement for {stale} epochs")
            break

    print(f"best validation mean MAE {best:.3f} K -> {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
