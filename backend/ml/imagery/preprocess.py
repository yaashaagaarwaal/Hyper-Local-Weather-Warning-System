"""Reproducible preprocessing: HDF5 -> a compact training cache.

    python -m ml.imagery.preprocess

For every *valid* frame, copies the raw 10-bit TIR1 and WV counts and that
file's own count->kelvin lookup tables into `ml/imagery/cache/`. Counts are
stored losslessly (uint16) and converted to brightness temperature on the
fly with the per-file table, so the cache is an exact subset of the HDF5
content. The original files are opened read-only and never changed.

Also computes the normalisation statistics -- from TRAINING frames only.
"""
from __future__ import annotations

import json

import numpy as np

from app.services.imagery import insat
from app.services.imagery.nowcast_common import INPUT_CHANNELS
from ml.imagery import config


def build_cache(force: bool = False) -> dict:
    index_path = config.CACHE_DIR / "index.json"
    frames = [f for f in insat.scan_catalog(config.INSAT_DIR, use_cache=False) if f.valid]
    if not frames:
        raise SystemExit(f"No valid INSAT-3DR frames found under {config.INSAT_DIR}")

    if index_path.exists() and not force:
        index = json.loads(index_path.read_text())
        if index["frameIds"] == [f.frame_id for f in frames]:
            return index

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows, cols = insat.EXPECTED_SHAPE
    fill_pixels = {}
    for channel in INPUT_CHANNELS:
        counts = np.lib.format.open_memmap(
            config.CACHE_DIR / f"counts_{channel}.npy", mode="w+", dtype=np.uint16, shape=(len(frames), rows, cols),
        )
        luts = np.zeros((len(frames), 1024), dtype=np.float32)
        n_fill = 0
        for i, frame in enumerate(frames):
            data, lut, fill = insat.read_counts(frame.path, channel)
            counts[i] = data
            luts[i, : lut.size] = lut
            n_fill += int((data == fill).sum()) + int((lut[np.unique(data)] > 900).sum())
        counts.flush()
        del counts
        np.save(config.CACHE_DIR / f"luts_{channel}.npy", luts)
        fill_pixels[channel] = n_fill

    # Normalisation statistics from the training split only.
    train_idx = [i for i, f in enumerate(frames) if config.split_of(f.timestamp) == "train"]
    stats = {}
    for channel in INPUT_CHANNELS:
        counts = np.load(config.CACHE_DIR / f"counts_{channel}.npy", mmap_mode="r")
        luts = np.load(config.CACHE_DIR / f"luts_{channel}.npy")
        total = total_sq = 0.0
        n = 0
        for i in train_idx:
            bt = luts[i][counts[i]].astype(np.float64)
            total += bt.sum()
            total_sq += (bt ** 2).sum()
            n += bt.size
        mean = total / n
        stats[channel] = {"meanK": mean, "stdK": float(np.sqrt(total_sq / n - mean ** 2))}

    index = {
        "frameIds": [f.frame_id for f in frames],
        "timestamps": [f.timestamp.isoformat() for f in frames],
        "sourceFiles": [f.path.name for f in frames],
        "splits": [config.split_of(f.timestamp) for f in frames],
        "channels": list(INPUT_CHANNELS),
        "shape": [rows, cols],
        "fillPixels": fill_pixels,
        "normalisation": stats,
        "normalisationSource": "training-split frames only",
    }
    index_path.write_text(json.dumps(index, indent=2))
    return index


if __name__ == "__main__":
    idx = build_cache(force=True)
    counts = {s: idx["splits"].count(s) for s in ("train", "val", "test", "unused")}
    print(f"cached {len(idx['frameIds'])} frames -> {config.CACHE_DIR}")
    print("frames per split:", counts)
    print("fill pixels:", idx["fillPixels"])
    print("normalisation (train only):", json.dumps(idx["normalisation"], indent=2))
