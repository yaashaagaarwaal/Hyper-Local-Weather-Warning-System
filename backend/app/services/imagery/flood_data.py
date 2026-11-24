"""Flood-mapping data access (torch-free): the real Sen1Floods11 chips, the
synthetic India chips, the preprocessing shared by training and serving, and
the classical threshold baselines.

Two sources, never mixed without saying so:
- REAL      `dataset/raw/sen1floods11/`     Sentinel-1 radar chips with
            human-labelled water masks (Cloud to Street). 512x512 @ 10 m.
- SYNTHETIC `dataset/synthetic/india_flood/` procedural simulations written
            by `python -m ml.hazards.synth_flood`. Same layout, clearly
            tagged; never a satellite observation.

Label convention (the dataset's own): 1 = surface water (flood water AND
permanent water), 0 = not water, -1 = no label (cloud in the optical image
the annotator worked from). Radar pixels can also be NaN (outside the swath).
Both kinds of missing pixel are excluded from the loss and from every metric.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.exceptions import AppError
from app.services.imagery.hazard_common import REPO_ROOT, SOURCE_REAL, SOURCE_SYNTHETIC, SYNTHETIC_ROOT

REAL_DIR = REPO_ROOT / "dataset" / "raw" / "sen1floods11"
SYNTHETIC_DIR = SYNTHETIC_ROOT / "india_flood"

CHANNELS = ("VV", "VH")
CHIP_SIZE = 512
PIXEL_SIZE_M = 10.0
# Sentinel-1 GRD backscatter in decibels. Anything outside this is not
# calibrated sigma0 in dB (e.g. linear power, or an 8-bit picture).
PLAUSIBLE_DB = (-60.0, 30.0)
CLIP_DB = (-35.0, 5.0)
MAX_MISSING_FRACTION = 0.9
MIN_SIDE_PX = 64
MAX_SIDE_PX = 2048

# Acquisition dates of the three South Asian events (Sen1Floods11_Metadata.geojson).
EVENT_INFO = {
    "India": {"country": "India", "s1Date": "2016-08-12", "note": "Monsoon flooding, Brahmaputra valley (Assam)"},
    "Pakistan": {"country": "Pakistan", "s1Date": "2017-06-28", "note": "Indus basin flooding"},
    "Sri-Lanka": {"country": "Sri Lanka", "s1Date": "2017-05-30", "note": "South-west monsoon flooding"},
}


class FloodDataError(AppError):
    def __init__(self, detail: str, status_code: int = 404):
        super().__init__(detail, status_code=status_code)


class InvalidFloodInputError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=422)


@dataclass(frozen=True)
class FloodChip:
    chip_id: str
    source: str              # "real" | "synthetic"
    split: str               # train | valid | test
    region: str              # real: event name; synthetic: Indian state
    path: Path               # real: the S1 GeoTIFF; synthetic: the .npz
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def real_dir() -> Path:
    return Path(os.environ.get("FLOOD_DATA_DIR", REAL_DIR))


def synthetic_dir() -> Path:
    return Path(os.environ.get("FLOOD_SYNTHETIC_DIR", SYNTHETIC_DIR))


# --------------------------------------------------------------------------
# Catalogs
# --------------------------------------------------------------------------
def real_catalog() -> list[FloodChip]:
    """Chips present on disk, with the dataset's OFFICIAL split assignment."""
    root = real_dir()
    return list(_real_catalog(str(root), _dir_stamp(root / "S1Hand")))


def _dir_stamp(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _real_catalog(root_str: str, _stamp: float) -> tuple[FloodChip, ...]:
    root = Path(root_str)
    split_of: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        listing = root / f"flood_{split}_data.csv"
        if not listing.exists():
            continue
        with open(listing) as fh:
            for row in csv.reader(fh):
                if row:
                    split_of[row[0].replace("_S1Hand.tif", "")] = split
    chips = []
    for path in sorted((root / "S1Hand").glob("*_S1Hand.tif")):
        chip_id = path.name.replace("_S1Hand.tif", "")
        if chip_id not in split_of:
            continue  # not in any official split list -> not used anywhere
        event = chip_id.split("_")[0]
        chips.append(FloodChip(chip_id, SOURCE_REAL, split_of[chip_id], event, path, dict(EVENT_INFO.get(event, {}))))
    return tuple(chips)


def synthetic_catalog() -> list[FloodChip]:
    root = synthetic_dir()
    return list(_synthetic_catalog(str(root), _dir_stamp(root / "metadata.jsonl")))


@lru_cache(maxsize=4)
def _synthetic_catalog(root_str: str, _stamp: float) -> tuple[FloodChip, ...]:
    root = Path(root_str)
    listing = root / "metadata.jsonl"
    if not listing.exists():
        return ()
    chips = []
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        path = root / "chips" / f"{meta['id']}.npz"
        if path.exists():
            chips.append(FloodChip(meta["id"], SOURCE_SYNTHETIC, meta["split"], meta["state"], path, meta))
    return tuple(chips)


def get_chip(chip_id: str) -> FloodChip:
    for chip in (*real_catalog(), *synthetic_catalog()):
        if chip.chip_id == chip_id:
            return chip
    raise FloodDataError(f"Unknown flood sample {chip_id!r}.")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _tifffile():
    try:
        import tifffile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise FloodDataError("The `tifffile` package is required to read Sentinel-1 GeoTIFF chips.", 503) from exc
    return tifffile


def _sibling(chip: FloodChip, kind: str) -> Path:
    return chip.path.parents[1] / kind / f"{chip.chip_id}_{kind}.tif"


def read_s1(chip: FloodChip) -> np.ndarray:
    """(2, H, W) float32 backscatter in dB (VV, VH); NaN where missing."""
    if chip.source == SOURCE_SYNTHETIC:
        with np.load(chip.path) as data:
            return data["s1"].astype(np.float32)
    return np.asarray(_tifffile().imread(chip.path), dtype=np.float32)


def read_label(chip: FloodChip) -> np.ndarray:
    """(H, W) int8 in {-1, 0, 1}."""
    if chip.source == SOURCE_SYNTHETIC:
        with np.load(chip.path) as data:
            return data["label"].astype(np.int8)
    return np.asarray(_tifffile().imread(_sibling(chip, "LabelHand"))).astype(np.int8)


def read_permanent_water(chip: FloodChip) -> np.ndarray | None:
    """(H, W) bool: permanent water (real: JRC Global Surface Water; synthetic:
    the simulator's river/sea/pond layer). None if not available."""
    if chip.source == SOURCE_SYNTHETIC:
        with np.load(chip.path) as data:
            return data["permanent"].astype(bool)
    path = _sibling(chip, "JRCWaterHand")
    return (np.asarray(_tifffile().imread(path)) == 1) if path.exists() else None


def read_otsu_reference(chip: FloodChip) -> np.ndarray | None:
    """The dataset's own classical baseline mask (Otsu threshold on the radar
    image), shipped as `S1OtsuLabelHand`. Real chips only."""
    if chip.source != SOURCE_REAL:
        return None
    path = _sibling(chip, "S1OtsuLabelHand")
    return (np.asarray(_tifffile().imread(path)) == 1) if path.exists() else None


def chip_bounds(chip: FloodChip) -> dict | None:
    """Geographic footprint {south, west, north, east} in degrees."""
    if chip.source == SOURCE_SYNTHETIC:
        return chip.meta.get("bounds")
    try:
        with _tifffile().TiffFile(chip.path) as tif:
            geo = tif.geotiff_metadata or {}
            page = tif.pages[0]
            height, width = page.imagelength, page.imagewidth
        scale, tie = geo["ModelPixelScale"], geo["ModelTiepoint"]
        west, north = float(tie[3]), float(tie[4])
        return {"south": north - height * float(scale[1]), "west": west,
                "north": north, "east": west + width * float(scale[0])}
    except Exception:  # noqa: BLE001 -- georeferencing is a nicety, never a reason to fail
        return None


# --------------------------------------------------------------------------
# Preprocessing (identical in training and serving)
# --------------------------------------------------------------------------
def data_mask(s1: np.ndarray) -> np.ndarray:
    """Pixels where both radar bands hold a real measurement."""
    return np.isfinite(s1).all(axis=0)


def valid_mask(s1: np.ndarray, label: np.ndarray) -> np.ndarray:
    return data_mask(s1) & (label >= 0)


def normalise(s1: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Clip to a physical range, z-score with TRAINING-split statistics, and
    put 0 (the training mean) where the radar has no data."""
    clipped = np.clip(s1, *CLIP_DB)
    out = (clipped - mean[:, None, None]) / std[:, None, None]
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def validate_s1(s1: np.ndarray) -> None:
    if not isinstance(s1, np.ndarray) or s1.ndim != 3:
        raise InvalidFloodInputError("Flood input must be a 3-D array (bands, rows, cols).")
    bands, height, width = s1.shape
    if bands != len(CHANNELS):
        raise InvalidFloodInputError(f"Exactly {len(CHANNELS)} radar bands {CHANNELS} are required, got {bands}.")
    if min(height, width) < MIN_SIDE_PX or max(height, width) > MAX_SIDE_PX:
        raise InvalidFloodInputError(f"Image is {height}x{width}; sides must be {MIN_SIDE_PX}-{MAX_SIDE_PX} pixels.")
    if not np.issubdtype(s1.dtype, np.floating):
        raise InvalidFloodInputError("Flood input must be floating-point backscatter in decibels.")
    if np.isinf(s1).any():
        raise InvalidFloodInputError("Flood input contains infinite values.")
    missing = float(1.0 - data_mask(s1).mean())
    if missing > MAX_MISSING_FRACTION:
        raise InvalidFloodInputError(f"{100 * missing:.0f}% of the image has no radar data; too little to map.")
    lo, hi = float(np.nanmin(s1)), float(np.nanmax(s1))
    if lo < PLAUSIBLE_DB[0] or hi > PLAUSIBLE_DB[1]:
        raise InvalidFloodInputError(
            f"Values span {lo:.1f} to {hi:.1f}, outside the plausible Sentinel-1 backscatter range "
            f"{PLAUSIBLE_DB[0]:.0f} to {PLAUSIBLE_DB[1]:.0f} dB. The model only accepts calibrated VV/VH in decibels."
        )
    if hi > 0 and lo >= 0:
        raise InvalidFloodInputError("Values are all non-negative: this looks like linear power or an ordinary picture, not decibels.")


def decode_s1_upload(data: bytes) -> tuple[np.ndarray, dict | None]:
    """Parse an uploaded 2-band Sentinel-1 GeoTIFF (VV, VH in dB). Returns
    (array, bounds-or-None)."""
    import io  # noqa: PLC0415

    tifffile = _tifffile()
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            array = np.asarray(tif.asarray(), dtype=np.float32)
            geo = tif.geotiff_metadata or {}
    except Exception as exc:  # noqa: BLE001
        raise InvalidFloodInputError("File is not a readable TIFF. Upload a 2-band Sentinel-1 GeoTIFF (VV, VH in dB).") from exc
    if array.ndim == 3 and array.shape[-1] == len(CHANNELS) and array.shape[0] != len(CHANNELS):
        array = array.transpose(2, 0, 1)  # (H, W, bands) -> (bands, H, W)
    validate_s1(array)
    bounds = None
    try:
        scale, tie = geo["ModelPixelScale"], geo["ModelTiepoint"]
        model_type = geo.get("GTModelTypeGeoKey")
        if getattr(model_type, "name", None) == "Geographic" or model_type == 2:   # lat/lon grid; projected grids are not converted
            west, north = float(tie[3]), float(tie[4])
            bounds = {"south": north - array.shape[1] * float(scale[1]), "west": west,
                      "north": north, "east": west + array.shape[2] * float(scale[0])}
    except Exception:  # noqa: BLE001
        bounds = None
    return array, bounds


# --------------------------------------------------------------------------
# Classical baseline
# --------------------------------------------------------------------------
def threshold_water(s1: np.ndarray, band: str, threshold_db: float) -> np.ndarray:
    """Open water is a smooth mirror to radar and returns little energy, so
    'darker than a threshold' is the textbook SAR water detector."""
    return np.nan_to_num(s1[CHANNELS.index(band)], nan=np.inf) < threshold_db


def otsu_water(s1: np.ndarray, band: str = "VH") -> tuple[np.ndarray, float]:
    """Per-image Otsu threshold on one band (no training). Returns
    (mask, threshold_dB)."""
    import cv2  # noqa: PLC0415

    values = s1[CHANNELS.index(band)]
    finite = np.isfinite(values)
    lo, hi = CLIP_DB
    scaled = np.zeros(values.shape, dtype=np.uint8)
    scaled[finite] = np.clip((values[finite] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if not finite.any():
        return np.zeros(values.shape, dtype=bool), float("nan")
    level, _ = cv2.threshold(scaled[finite].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_db = lo + float(level) / 255.0 * (hi - lo)
    return finite & (values < threshold_db), threshold_db


def area_km2(pixel_count: int, bounds: dict | None, shape: tuple[int, int]) -> float | None:
    """Ground area of `pixel_count` pixels. Uses the true footprint when the
    chip is georeferenced (a degree of longitude shrinks with latitude)."""
    if bounds is None:
        return None
    height, width = shape
    lat_mid = np.radians((bounds["south"] + bounds["north"]) / 2.0)
    pixel_h_km = (bounds["north"] - bounds["south"]) / height * 110.574
    pixel_w_km = (bounds["east"] - bounds["west"]) / width * 111.320 * float(np.cos(lat_mid))
    return float(pixel_count * pixel_h_km * pixel_w_km)
