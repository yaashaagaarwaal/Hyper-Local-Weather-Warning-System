"""Build the lightning training cache from the raw GOES-16 files.

    python -m ml.hazards.lightning_prepare

Reads dataset/raw/goes_glm/ (read-only) through the same functions the API
uses and writes ml/hazards/cache/goes_lightning.npz (gitignored, regenerable):
the 24 infrared frames on the 4 km input grid and, for every anchor frame,
the GLM flash count per 8 km cell within +/-5 minutes.

Also measures, on TRAINING anchors and TRAINING blocks only, how flash
occurrence relates to cloud-top temperature -- the one real-world statistic
the India synthetic generator borrows (see synth_lightning.py).
"""
from __future__ import annotations

import json

import numpy as np

from app.services.imagery import lightning_data as ld
from ml.hazards import config

BT_BINS_K = (180, 200, 205, 210, 215, 220, 225, 230, 235, 240, 250, 260, 340)


def main() -> None:
    frames = ld.abi_frames()
    if len(frames) < ld.LOOKBACK_FRAMES + 1:
        raise SystemExit(f"Need at least {ld.LOOKBACK_FRAMES + 1} GOES ABI frames under {ld.goes_dir()}; found {len(frames)}.")
    bt, flashes, seconds, n_flashes = [], [], [], []
    shape = None
    for frame in frames:
        abi = ld.read_abi(frame.path)
        bt.append(ld.to_input_grid(abi["bt"]).astype(np.float16))
        seconds.append(abi["seconds"])
        if frame.index >= ld.LOOKBACK_FRAMES:
            grid, points = ld.flash_grid_for(abi)
        else:
            grid, points = np.zeros((abi["bt"].shape[0] // 4, abi["bt"].shape[1] // 4), np.uint16), []
        flashes.append(grid)
        n_flashes.append(len(points))
        shape = grid.shape
        print(f"  frame {frame.index:2d} {frame.frame_id}  flashes in window {len(points):6d}  "
              f"cells with a flash {int((grid > 0).sum()):5d}  split {ld.split_of_anchor(frame.index)}", flush=True)
    bt, flashes = np.stack(bt), np.stack(flashes)
    blocks = ld.block_split_map(shape)

    # flash occurrence vs cloud-top temperature, TRAIN anchors x TRAIN blocks only
    hits, totals, per_positive = np.zeros(len(BT_BINS_K) - 1), np.zeros(len(BT_BINS_K) - 1), []
    for index in ld.TRAIN_ANCHORS:
        pair = bt[[index - ld.LOOKBACK_FRAMES, index]].astype(np.float32)
        member = ld.cell_valid(pair) & (blocks == 0)
        coldest = ld.cell_min_bt(pair[1])[member]
        truth = flashes[index][member]
        which = np.digitize(coldest, BT_BINS_K) - 1
        ok = (which >= 0) & (which < len(hits))
        np.add.at(totals, which[ok], 1)
        np.add.at(hits, which[ok], truth[ok] > 0)
        per_positive.append(truth[truth > 0])
    per_positive = np.concatenate(per_positive)
    statistics = {
        "source": "GOES-16 ABI band 13 + GLM, 2 Aug 2024, TRAIN anchors x TRAIN blocks only (Americas, not India)",
        "btBinsK": list(BT_BINS_K),
        "flashProbabilityByBtBin": [float(h / t) if t else None for h, t in zip(hits, totals)],
        "cellsByBtBin": [int(t) for t in totals],
        "meanFlashesPerPositiveCell": float(per_positive.mean()),
        "medianFlashesPerPositiveCell": float(np.median(per_positive)),
        "positiveCellFraction": float(hits.sum() / totals.sum()),
    }
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.LIGHTNING_CACHE, bt=bt, flashes=flashes, seconds=np.asarray(seconds), blocks=blocks,
                        frame_ids=np.asarray([f.frame_id for f in frames]))
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "goes_flash_statistics.json").write_text(json.dumps(statistics, indent=2))
    print(json.dumps(statistics, indent=2))
    print(f"cache -> {config.LIGHTNING_CACHE} ({config.LIGHTNING_CACHE.stat().st_size / 1e6:.1f} MB); "
          f"blocks train/val/test = {[(blocks == k).mean().round(3) for k in range(3)]}")


if __name__ == "__main__":
    main()
