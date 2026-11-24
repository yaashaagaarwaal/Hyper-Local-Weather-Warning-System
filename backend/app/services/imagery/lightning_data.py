"""Lightning-detection data access (torch-free): real GOES-16 infrared + GLM
flashes, the synthetic India samples, and the input/label definition shared
by training and serving.

THE TASK. Input: two infrared (~10.5 um) cloud-top brightness-temperature
images of the same scene 30 minutes apart, on a ~4 km grid. Output: for each
8 km cell, the probability that at least one lightning flash occurs there
within +/-5 minutes of the later image. Ground truth: flashes detected
optically from orbit by the GOES Geostationary Lightning Mapper (GLM).

Why this shape: a 4 km grid and a 30-minute pair are what INSAT-3DR provides
over India, so a model with this interface can be pointed at Indian imagery
later without redesign -- but nothing here validates it for India (see
docs/flood_lightning_model_report.md). 8 km output cells match the GLM's
own pixel footprint.

Two sources, never mixed without saying so:
- REAL      `dataset/raw/goes_glm/`            GOES-16 ABI band 13 + GLM, the
            AMERICAS, 2 Aug 2024 20-22 UTC.
- SYNTHETIC `dataset/synthetic/india_lightning/` simulated Indian storm
            scenes with simulated flashes (`python -m ml.hazards.synth_lightning`).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.exceptions import AppError
from app.services.imagery.hazard_common import REPO_ROOT, SOURCE_REAL, SOURCE_SYNTHETIC, SYNTHETIC_ROOT

GOES_DIR = REPO_ROOT / "dataset" / "raw" / "goes_glm"
SYNTHETIC_DIR = SYNTHETIC_ROOT / "india_lightning"
GOES_EPOCH = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

INPUT_SPACING_KM = 4            # model input grid
CELL_KM = 8                     # model output / label grid
ABI_TO_INPUT = 2                # ABI CONUS band 13 is 2 km -> average 2x2 to 4 km
LOOKBACK_MINUTES = 30           # spacing of the two input images
ABI_CADENCE_MINUTES = 5
LOOKBACK_FRAMES = LOOKBACK_MINUTES // ABI_CADENCE_MINUTES
LABEL_HALF_WINDOW_S = 300.0     # a flash counts if within +/-5 min of the image

# Fixed, physical input scaling (not fitted): the same numbers apply to any
# infrared window channel, so nothing ties the model to one satellite's statistics.
BT_CENTRE_K, BT_SCALE_K, TREND_SCALE_K = 260.0, 30.0, 10.0
PLAUSIBLE_K = (150.0, 350.0)
MIN_SIDE_PX = 32
MAX_NAN_FRACTION = 0.5
FILL_K = 300.0

# --- leak-free split of the 24 real frames --------------------------------
# Anchors (the later image of each pair) are frames 6..23. Storms persist for
# hours, so a purely temporal split would test on the same storms a few
# minutes later. The domain is therefore ALSO cut into 384 km blocks, each
# assigned to exactly one split; a cell is scored only when both its time and
# its block belong to that split. One anchor is dropped between splits so
# label windows (+/-5 min) never overlap.
TRAIN_ANCHORS = tuple(range(6, 15))
VAL_ANCHORS = (16, 17, 18)
TEST_ANCHORS = (20, 21, 22, 23)
BLOCK_CELLS = 48
BLOCK_SEED = 26077
BLOCK_SHARES = (0.6, 0.2, 0.2)  # train, val, test


class LightningDataError(AppError):
    def __init__(self, detail: str, status_code: int = 404):
        super().__init__(detail, status_code=status_code)


class InvalidLightningInputError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=422)


def goes_dir() -> Path:
    return Path(os.environ.get("GOES_DATA_DIR", GOES_DIR))


def synthetic_dir() -> Path:
    return Path(os.environ.get("LIGHTNING_SYNTHETIC_DIR", SYNTHETIC_DIR))


def _h5py():
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise LightningDataError("The `h5py` package is required to read GOES NetCDF files.", 503) from exc
    return h5py


# --------------------------------------------------------------------------
# GOES fixed-grid projection
# --------------------------------------------------------------------------
def goes_forward(lat, lon, proj: dict):
    """(lat, lon) degrees -> scan angles (x, y) radians."""
    req, rpol = proj["semi_major_axis"], proj["semi_minor_axis"]
    h = proj["perspective_point_height"] + req
    lon0 = np.radians(proj["longitude_of_projection_origin"])
    phi, lam = np.radians(lat), np.radians(lon)
    phic = np.arctan((rpol ** 2 / req ** 2) * np.tan(phi))
    e2 = 1 - rpol ** 2 / req ** 2
    rc = rpol / np.sqrt(1 - e2 * np.cos(phic) ** 2)
    sx = h - rc * np.cos(phic) * np.cos(lam - lon0)
    sy = -rc * np.cos(phic) * np.sin(lam - lon0)
    sz = rc * np.sin(phic)
    return np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), np.arctan(sz / sx)


def goes_inverse(x, y, proj: dict):
    """Scan angles (x, y) radians -> (lat, lon) degrees; NaN off the Earth."""
    req, rpol = proj["semi_major_axis"], proj["semi_minor_axis"]
    h = proj["perspective_point_height"] + req
    lon0 = np.radians(proj["longitude_of_projection_origin"])
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (req ** 2 / rpol ** 2) * np.sin(y) ** 2)
    b = -2 * h * np.cos(x) * np.cos(y)
    c = h ** 2 - req ** 2
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        sx, sy, sz = rs * np.cos(x) * np.cos(y), -rs * np.sin(x), rs * np.cos(x) * np.sin(y)
        lat = np.degrees(np.arctan((req ** 2 / rpol ** 2) * sz / np.sqrt((h - sx) ** 2 + sy ** 2)))
        lon = np.degrees(lon0 - np.arctan(sy / (h - sx)))
    return lat, lon


# --------------------------------------------------------------------------
# Real GOES frames
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AbiFrame:
    index: int
    path: Path
    time: datetime
    frame_id: str


def _abi_start(name: str) -> datetime | None:
    match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", name)
    if not match:
        return None
    year, doy, hour, minute, second = (int(g) for g in match.groups())
    return datetime(year, 1, 1, hour, minute, second, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def abi_frames() -> list[AbiFrame]:
    root = goes_dir()
    return list(_abi_frames(str(root), _stamp(root / "abi")))


def _stamp(path: Path) -> float:
    try:
        return max([path.stat().st_mtime] + [p.stat().st_mtime for p in path.iterdir()])
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _abi_frames(root_str: str, _stamp_value: float) -> tuple[AbiFrame, ...]:
    paths = sorted((Path(root_str) / "abi").rglob("*M6C13_*.nc"))
    frames = []
    for path in paths:
        start = _abi_start(path.name)
        if start is not None:
            frames.append((start, path))
    frames.sort()
    return tuple(AbiFrame(i, path, start, start.strftime("GOES16_%Y%m%dT%H%MZ")) for i, (start, path) in enumerate(frames))


def read_abi(path: Path) -> dict:
    """One ABI band-13 image: brightness temperature (K, NaN where missing)
    on its native 2 km grid, plus what is needed to geolocate it."""
    with _h5py().File(path, "r") as f:
        cmi = f["CMI"]
        raw = cmi[:]
        bt = np.where(raw == cmi.attrs["_FillValue"], np.nan,
                      raw * cmi.attrs["scale_factor"] + cmi.attrs["add_offset"]).astype(np.float32)
        bt[f["DQF"][:] > 1] = np.nan                       # flagged unusable by the provider
        proj = {k: float(np.asarray(f["goes_imager_projection"].attrs[k]).ravel()[0]) for k in
                ("semi_major_axis", "semi_minor_axis", "perspective_point_height", "longitude_of_projection_origin")}
        xs = f["x"][:] * f["x"].attrs["scale_factor"] + f["x"].attrs["add_offset"]
        ys = f["y"][:] * f["y"].attrs["scale_factor"] + f["y"].attrs["add_offset"]
        seconds = float(f["t"][()])
    return {"bt": bt, "x": xs.astype(np.float64), "y": ys.astype(np.float64), "proj": proj,
            "seconds": seconds, "time": GOES_EPOCH + timedelta(seconds=seconds)}


def to_input_grid(bt_2km: np.ndarray) -> np.ndarray:
    """2 km -> 4 km by 2x2 averaging (NaN if any of the four is missing)."""
    rows, cols = bt_2km.shape[0] // ABI_TO_INPUT * ABI_TO_INPUT, bt_2km.shape[1] // ABI_TO_INPUT * ABI_TO_INPUT
    blocks = bt_2km[:rows, :cols].reshape(rows // ABI_TO_INPUT, ABI_TO_INPUT, cols // ABI_TO_INPUT, ABI_TO_INPUT)
    return blocks.mean(axis=(1, 3)).astype(np.float32)


def glm_files_between(start_s: float, end_s: float) -> list[Path]:
    """GLM files (one per 20 s) whose start time falls in [start, end)."""
    out = []
    for path in sorted((goes_dir() / "glm").rglob("*.nc")):
        begin = _abi_start(path.name)
        if begin is None:
            continue
        seconds = (begin - GOES_EPOCH).total_seconds()
        if start_s - 25 <= seconds < end_s + 5:
            out.append(path)
    return out


def read_glm_flashes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat, lon, seconds-since-GOES-epoch) of every flash in one file."""
    with _h5py().File(path, "r") as f:
        offset = f["flash_time_offset_of_first_event"]
        seconds = float(f["product_time"][()]) + offset[:].astype("f8") * offset.attrs["scale_factor"] + offset.attrs["add_offset"]
        return f["flash_lat"][:].astype(np.float64), f["flash_lon"][:].astype(np.float64), seconds


def flash_cells(lat, lon, abi: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map flashes to (row, col) on the 8 km cell grid of an ABI image;
    third value marks flashes that fall inside the image."""
    x, y = goes_forward(lat, lon, abi["proj"])
    dx, dy = abi["x"][1] - abi["x"][0], abi["y"][1] - abi["y"][0]
    col = np.round((x - abi["x"][0]) / dx).astype(np.int64)
    row = np.round((y - abi["y"][0]) / dy).astype(np.int64)
    inside = (col >= 0) & (col < abi["x"].size) & (row >= 0) & (row < abi["y"].size)
    factor = CELL_KM // 2   # ABI pixels per cell
    return row // factor, col // factor, inside


def flash_grid_for(abi: dict) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Flash counts per 8 km cell within +/-5 min of the image, and the
    flashes' (lat, lon)."""
    factor = CELL_KM // 2
    grid = np.zeros((abi["bt"].shape[0] // factor, abi["bt"].shape[1] // factor), dtype=np.uint16)
    points: list[tuple[float, float]] = []
    lo, hi = abi["seconds"] - LABEL_HALF_WINDOW_S, abi["seconds"] + LABEL_HALF_WINDOW_S
    for path in glm_files_between(lo, hi):
        lat, lon, seconds = read_glm_flashes(path)
        row, col, inside = flash_cells(lat, lon, abi)
        keep = inside & (seconds >= lo) & (seconds < hi) & (row < grid.shape[0]) & (col < grid.shape[1])
        np.add.at(grid, (row[keep], col[keep]), 1)
        points.extend(zip(lat[keep].tolist(), lon[keep].tolist()))
    return grid, points


def cell_latlon(abi: dict) -> tuple[np.ndarray, np.ndarray]:
    """Centre (lat, lon) of every 8 km cell."""
    factor = CELL_KM // 2
    rows, cols = abi["bt"].shape[0] // factor, abi["bt"].shape[1] // factor
    xs = abi["x"][: cols * factor].reshape(cols, factor).mean(axis=1)
    ys = abi["y"][: rows * factor].reshape(rows, factor).mean(axis=1)
    return goes_inverse(*np.meshgrid(xs, ys), abi["proj"])


def split_of_anchor(index: int) -> str:
    if index in TRAIN_ANCHORS:
        return "train"
    if index in VAL_ANCHORS:
        return "val"
    if index in TEST_ANCHORS:
        return "test"
    return "unused"


def block_split_map(shape: tuple[int, int]) -> np.ndarray:
    """(rows, cols) int8 on the cell grid: 0 train, 1 val, 2 test block."""
    rows, cols = shape
    n_r, n_c = -(-rows // BLOCK_CELLS), -(-cols // BLOCK_CELLS)
    rng = np.random.default_rng(BLOCK_SEED)
    order = rng.permutation(n_r * n_c)
    assignment = np.zeros(n_r * n_c, dtype=np.int8)
    n_train = int(round(BLOCK_SHARES[0] * order.size))
    n_val = int(round(BLOCK_SHARES[1] * order.size))
    assignment[order[n_train:n_train + n_val]] = 1
    assignment[order[n_train + n_val:]] = 2
    blocks = assignment.reshape(n_r, n_c)
    return np.kron(blocks, np.ones((BLOCK_CELLS, BLOCK_CELLS), dtype=np.int8))[:rows, :cols]


# --------------------------------------------------------------------------
# Synthetic India samples
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    split: str
    path: Path
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def synthetic_catalog() -> list[SyntheticSample]:
    root = synthetic_dir()
    listing = root / "metadata.jsonl"
    try:
        stamp = listing.stat().st_mtime
    except OSError:
        return []
    return list(_synthetic_catalog(str(root), stamp))


@lru_cache(maxsize=4)
def _synthetic_catalog(root_str: str, _stamp_value: float) -> tuple[SyntheticSample, ...]:
    root = Path(root_str)
    samples = []
    for line in (root / "metadata.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        path = root / "samples" / f"{meta['id']}.npz"
        if path.exists():
            samples.append(SyntheticSample(meta["id"], meta["split"], path, meta))
    return tuple(samples)


def read_synthetic(sample: SyntheticSample) -> dict:
    """`bt`: (T, H, W) kelvin, 30 min apart; `flash_count`: (T, H/2, W/2);
    `flashes`: (n, 4) = frame index, lat, lon, minutes from that frame."""
    with np.load(sample.path) as data:
        return {"bt": data["bt"].astype(np.float32), "flash_count": data["flash_count"].astype(np.uint16),
                "flashes": data["flashes"].astype(np.float32)}


# --------------------------------------------------------------------------
# Model input / labels / baseline (identical in training and serving)
# --------------------------------------------------------------------------
def validate_pair(frames_k: np.ndarray) -> None:
    """`frames_k`: (2, H, W) = [image 30 min earlier, image now], kelvin."""
    if not isinstance(frames_k, np.ndarray) or frames_k.ndim != 3 or frames_k.shape[0] != 2:
        raise InvalidLightningInputError("Lightning input must be 2 infrared images (earlier, now) as a (2, rows, cols) array.")
    _, height, width = frames_k.shape
    if min(height, width) < MIN_SIDE_PX:
        raise InvalidLightningInputError(f"Image is too small ({height}x{width}); at least {MIN_SIDE_PX}x{MIN_SIDE_PX} pixels are needed.")
    if not np.issubdtype(frames_k.dtype, np.floating):
        raise InvalidLightningInputError("Lightning input must be floating-point brightness temperature in kelvin.")
    if np.isinf(frames_k).any():
        raise InvalidLightningInputError("Lightning input contains infinite values.")
    missing = float(np.isnan(frames_k).mean())
    if missing > MAX_NAN_FRACTION:
        raise InvalidLightningInputError(f"{100 * missing:.0f}% of the input is missing data; too much to analyse.")
    lo, hi = float(np.nanmin(frames_k)), float(np.nanmax(frames_k))
    if lo < PLAUSIBLE_K[0] or hi > PLAUSIBLE_K[1]:
        raise InvalidLightningInputError(
            f"Values span {lo:.0f}-{hi:.0f}, outside the plausible brightness-temperature range "
            f"{PLAUSIBLE_K[0]:.0f}-{PLAUSIBLE_K[1]:.0f} K. The model only accepts calibrated infrared imagery in kelvin."
        )


def features(frames_k: np.ndarray) -> np.ndarray:
    """(2, H, W) kelvin -> (2, H, W) network input: [how cold the cloud top
    is now, how much it cooled in 30 minutes]. Missing pixels become warm,
    unchanging clear sky."""
    earlier, now = (np.where(np.isfinite(f), f, FILL_K) for f in frames_k)
    return np.stack([(now - BT_CENTRE_K) / BT_SCALE_K, (now - earlier) / TREND_SCALE_K]).astype(np.float32)


def cell_valid(frames_k: np.ndarray) -> np.ndarray:
    """8 km cells whose four 4 km pixels are all present in both images."""
    ok = np.isfinite(frames_k).all(axis=0)
    rows, cols = ok.shape[0] // 2, ok.shape[1] // 2
    return ok[:rows * 2, :cols * 2].reshape(rows, 2, cols, 2).all(axis=(1, 3))


def cell_min_bt(bt_k: np.ndarray) -> np.ndarray:
    """Coldest 4 km pixel in each 8 km cell."""
    rows, cols = bt_k.shape[0] // 2, bt_k.shape[1] // 2
    blocks = np.where(np.isfinite(bt_k), bt_k, np.inf)[:rows * 2, :cols * 2].reshape(rows, 2, cols, 2)
    return blocks.min(axis=(1, 3))


def threshold_baseline(bt_now_k: np.ndarray, threshold_k: float) -> np.ndarray:
    """Classical rule: lightning where the cloud top is colder than a
    threshold (deep convection). No training beyond choosing the number."""
    return cell_min_bt(bt_now_k) < threshold_k
