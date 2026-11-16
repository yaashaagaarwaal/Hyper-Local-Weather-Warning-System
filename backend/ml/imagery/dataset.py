"""Sample construction for the nowcast task: which frames form a sample,
which split it belongs to, and how training crops are drawn."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.services.imagery import insat
from app.services.imagery.nowcast_common import INPUT_CHANNELS, LEAD_STEPS, N_INPUT_FRAMES
from ml.imagery import config


@dataclass(frozen=True)
class Anchor:
    """One sample: inputs are frames `inputs` (oldest first, ending at the
    anchor time); `targets[k]` is the frame index for lead k, or -1 if the
    satellite frame for that time is missing/corrupt/outside the split."""
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    split: str

    @property
    def index(self) -> int:
        return self.inputs[-1]


def build_anchors(timestamps: list[datetime], splits: list[str]) -> list[Anchor]:
    """Leakage rule, enforced here and asserted in the tests: every input
    AND every target frame of a sample must lie in the same split as its
    anchor. Inputs must be strictly consecutive (30 min apart); a missing
    target simply drops that lead time for that sample."""
    position = {ts: i for i, ts in enumerate(timestamps)}
    anchors = []
    for i, ts in enumerate(timestamps):
        split = splits[i]
        if split not in ("train", "val", "test"):
            continue
        inputs = []
        for back in range(N_INPUT_FRAMES - 1, -1, -1):
            j = position.get(ts - back * insat.FRAME_INTERVAL)
            if j is None or splits[j] != split:
                inputs = None
                break
            inputs.append(j)
        if inputs is None:
            continue
        targets = []
        for step in LEAD_STEPS:
            j = position.get(ts + step * insat.FRAME_INTERVAL)
            targets.append(j if j is not None and splits[j] == split else -1)
        if all(t < 0 for t in targets):
            continue
        anchors.append(Anchor(tuple(inputs), tuple(targets), split))
    return anchors


class FrameStore:
    """Memory-mapped view of the preprocessing cache."""

    def __init__(self):
        index_path = config.CACHE_DIR / "index.json"
        if not index_path.exists():
            raise SystemExit("No cache found -- run `python -m ml.imagery.preprocess` first.")
        self.index = json.loads(index_path.read_text())
        self.timestamps = [datetime.fromisoformat(t) for t in self.index["timestamps"]]
        self.splits = self.index["splits"]
        self.counts = {c: np.load(config.CACHE_DIR / f"counts_{c}.npy", mmap_mode="r") for c in INPUT_CHANNELS}
        self.luts = {c: np.load(config.CACHE_DIR / f"luts_{c}.npy") for c in INPUT_CHANNELS}
        norm = self.index["normalisation"]
        self.mean = np.array([norm[c]["meanK"] for c in INPUT_CHANNELS], dtype=np.float32)
        self.std = np.array([norm[c]["stdK"] for c in INPUT_CHANNELS], dtype=np.float32)
        self.rows, self.cols = self.index["shape"]
        self.anchors = build_anchors(self.timestamps, self.splits)

    def anchors_for(self, split: str) -> list[Anchor]:
        return [a for a in self.anchors if a.split == split]

    def bt(self, channel: str, frame: int, r0: int = 0, c0: int = 0, height: int | None = None, width: int | None = None) -> np.ndarray:
        height = height or self.rows
        width = width or self.cols
        return self.luts[channel][frame][self.counts[channel][frame, r0:r0 + height, c0:c0 + width]]

    def normalised_inputs(self, anchor: Anchor, r0=0, c0=0, height=None, width=None) -> np.ndarray:
        """(T, C, H, W) float32, normalised with training-split statistics."""
        out = np.stack([
            np.stack([self.bt(c, f, r0, c0, height, width) for c in INPUT_CHANNELS]) for f in anchor.inputs
        ]).astype(np.float32)
        return (out - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def normalised_targets(self, anchor: Anchor, r0=0, c0=0, height=None, width=None) -> tuple[np.ndarray, np.ndarray]:
        """(L, H, W) normalised TIR1 targets and an (L,) availability flag."""
        height = height or self.rows
        width = width or self.cols
        targets = np.zeros((len(LEAD_STEPS), height, width), dtype=np.float32)
        available = np.zeros(len(LEAD_STEPS), dtype=np.float32)
        for k, frame in enumerate(anchor.targets):
            if frame >= 0:
                targets[k] = (self.bt("TIR1", frame, r0, c0, height, width) - self.mean[0]) / self.std[0]
                available[k] = 1.0
        return targets, available


class TrainSampler:
    """Random training crops, half of them biased toward where the scene is
    changing (see config.ACTIVITY_SAMPLING_FRACTION), with flip/transpose
    augmentation. Seeded, so a run is reproducible."""

    def __init__(self, store: FrameStore, seed: int = config.RANDOM_SEED):
        self.store = store
        self.rng = np.random.default_rng(seed)
        self.anchors = store.anchors_for("train")
        self.activity = [self._activity_map(a) for a in self.anchors]

    def _activity_map(self, anchor: Anchor) -> np.ndarray:
        block = config.ACTIVITY_BLOCK
        diff = np.abs(self.store.bt("TIR1", anchor.inputs[-1]) - self.store.bt("TIR1", anchor.inputs[-2]))
        rows, cols = (diff.shape[0] // block) * block, (diff.shape[1] // block) * block
        coarse = diff[:rows, :cols].reshape(rows // block, block, cols // block, block).mean(axis=(1, 3))
        coarse = coarse.astype(np.float64) + 1e-9
        return coarse / coarse.sum()

    def _corner(self, which: int) -> tuple[int, int]:
        size, store = config.CROP_SIZE, self.store
        if self.rng.random() < config.ACTIVITY_SAMPLING_FRACTION:
            prob = self.activity[which]
            cell = self.rng.choice(prob.size, p=prob.ravel())
            block = config.ACTIVITY_BLOCK
            centre_r = (cell // prob.shape[1]) * block + self.rng.integers(0, block)
            centre_c = (cell % prob.shape[1]) * block + self.rng.integers(0, block)
            r0, c0 = centre_r - size // 2, centre_c - size // 2
        else:
            r0 = self.rng.integers(0, store.rows - size + 1)
            c0 = self.rng.integers(0, store.cols - size + 1)
        return int(np.clip(r0, 0, store.rows - size)), int(np.clip(c0, 0, store.cols - size))

    def batch(self, batch_size: int = config.BATCH_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size, margin = config.CROP_SIZE, config.LOSS_MARGIN
        xs, ys, masks = [], [], []
        for _ in range(batch_size):
            which = int(self.rng.integers(0, len(self.anchors)))
            anchor = self.anchors[which]
            r0, c0 = self._corner(which)
            x = self.store.normalised_inputs(anchor, r0, c0, size, size)
            y, available = self.store.normalised_targets(anchor, r0, c0, size, size)
            # Geometric augmentation only -- brightness temperature is a
            # physical quantity, so no intensity/contrast jitter.
            if self.rng.random() < 0.5:
                x, y = x[..., ::-1], y[..., ::-1]
            if self.rng.random() < 0.5:
                x, y = x[..., ::-1, :], y[..., ::-1, :]
            if self.rng.random() < 0.5:
                x, y = np.swapaxes(x, -1, -2), np.swapaxes(y, -1, -2)
            mask = np.zeros_like(y)
            mask[:, margin:-margin, margin:-margin] = 1.0
            mask *= available[:, None, None]
            xs.append(np.ascontiguousarray(x))
            ys.append(np.ascontiguousarray(y))
            masks.append(mask)
        return np.stack(xs), np.stack(ys), np.stack(masks)
