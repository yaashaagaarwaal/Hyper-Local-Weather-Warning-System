"""INSAT-3DR Imager L1C (ASIA Mercator sector) HDF5 reader.

The real data this reads lives in `dataset/mosdac_api/MOSDAC_Downloads/`
(ISRO MOSDAC product `3RIMG_L1C_ASIA_MER`, half-hourly, 4 km, 6 channels).
This module is **read-only** with respect to those files -- nothing here
ever opens one for writing -- and is shared by the training code
(`backend/ml/imagery/`) and the serving code (`ml_inference.py`) so that
calibration and geolocation are done exactly one way.

What the files actually contain (verified by inspection, see
`backend/ml/artifacts/imagery/dataset_inspection.md`):
- `IMG_<CH>` : (1, 1616, 1737) uint16 10-bit sensor counts, CH in
  VIS / SWIR / MIR / WV / TIR1 / TIR2.
- `IMG_<CH>_TEMP` : a 1024-entry count -> brightness-temperature (K) lookup
  table. **The table differs from file to file** (per-scan calibration), so
  a count is only meaningful together with its own file's table.
- `X` / `Y` : projection coordinates in metres (Mercator, standard parallel
  17.75 N, origin longitude 77.25 E, WGS84).
- No labels, masks, classes or bounding boxes of any kind.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.core.exceptions import AppError, NotFoundError

BACKEND_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_INSAT_DIR = REPO_ROOT / "dataset" / "mosdac_api" / "MOSDAC_Downloads"

FRAME_INTERVAL = timedelta(minutes=30)
EXPECTED_SHAPE = (1616, 1737)  # rows (north -> south), cols (west -> east)
THERMAL_CHANNELS = ("TIR1", "WV", "TIR2", "MIR")
ALL_CHANNELS = ("VIS", "SWIR", "MIR", "WV", "TIR1", "TIR2")

_FILENAME_RE = re.compile(r"^3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_L1C_ASIA_MER_V\d+R\d+\.h5$")

# Mercator parameters, as recorded in every file's `Projection_Information`
# attributes (identical across the dataset; verified in the inspection
# report by reproducing the stored corner coordinates to < 1 mm).
_SEMI_MAJOR = 6378137.0
_SEMI_MINOR = 6356752.3142
_ECC = math.sqrt(1 - (_SEMI_MINOR / _SEMI_MAJOR) ** 2)
_STD_PARALLEL = math.radians(17.75)
_LON_ORIGIN = 77.25
_K0 = math.cos(_STD_PARALLEL) / math.sqrt(1 - (_ECC * math.sin(_STD_PARALLEL)) ** 2)
_X0 = -3473242.733735  # centre of the first (west-most) column
_Y0 = 5401854.420193  # centre of the first (north-most) row
_DX = 4001.4317208928987
_DY = 4000.463224575855

# The sector's own stated geographic extent.
SECTOR_BOUNDS = {"south": -10.0, "west": 44.5, "north": 45.5, "east": 110.0}

# Named sub-regions offered by the API. Bounds are (south, west, north, east).
REGIONS: dict[str, dict] = {
    "north_india": {"label": "North India (Delhi-NCR centred)", "bounds": (20.0, 66.0, 37.0, 90.0)},
    "india": {"label": "Indian subcontinent", "bounds": (5.0, 64.0, 38.0, 100.0)},
    "full_sector": {"label": "Full ASIA Mercator sector", "bounds": (-10.0, 44.5, 45.5, 110.0)},
}
# The project's own area of interest (same box as dataset/scripts/fetch_gibs_imagery.py).
DELHI_NCR_BOUNDS = (26.8, 76.5, 29.2, 78.3)


class InsatDataUnavailableError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=503)


class InsatFrameError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=422)


def insat_dir() -> Path:
    override = os.environ.get("INSAT_DATA_DIR")
    return Path(override) if override else DEFAULT_INSAT_DIR


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415 -- optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise InsatDataUnavailableError(
            "h5py is not installed, so INSAT-3DR HDF5 files cannot be read."
        ) from exc
    return h5py


# --------------------------------------------------------------------------
# Geolocation
# --------------------------------------------------------------------------
def latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
    """Forward ellipsoidal Mercator, in the product's own projection."""
    phi = math.radians(lat)
    x = _SEMI_MAJOR * _K0 * math.radians(lon - _LON_ORIGIN)
    con = ((1 - _ECC * math.sin(phi)) / (1 + _ECC * math.sin(phi))) ** (_ECC / 2)
    y = _SEMI_MAJOR * _K0 * math.log(math.tan(math.pi / 4 + phi / 2) * con)
    return x, y


def latlon_to_pixel(lat: float, lon: float) -> tuple[float, float]:
    """(row, col) as floats; row 0 is the northern edge, col 0 the western."""
    x, y = latlon_to_xy(lat, lon)
    return (_Y0 - y) / _DY, (x - _X0) / _DX


def pixel_to_latlon(row: float, col: float) -> tuple[float, float]:
    x = _X0 + col * _DX
    y = _Y0 - row * _DY
    lon = _LON_ORIGIN + math.degrees(x / (_SEMI_MAJOR * _K0))
    ts = math.exp(-y / (_SEMI_MAJOR * _K0))
    phi = math.pi / 2 - 2 * math.atan(ts)
    for _ in range(8):  # standard fixed-point iteration for the ellipsoid
        con = _ECC * math.sin(phi)
        phi = math.pi / 2 - 2 * math.atan(ts * ((1 - con) / (1 + con)) ** (_ECC / 2))
    return math.degrees(phi), lon


@dataclass(frozen=True)
class PixelWindow:
    row0: int
    row1: int  # exclusive
    col0: int
    col1: int  # exclusive

    @property
    def shape(self) -> tuple[int, int]:
        return self.row1 - self.row0, self.col1 - self.col0

    def bounds(self) -> dict:
        north, west = pixel_to_latlon(self.row0, self.col0)
        south, east = pixel_to_latlon(self.row1 - 1, self.col1 - 1)
        return {"south": round(south, 3), "west": round(west, 3), "north": round(north, 3), "east": round(east, 3)}


FULL_WINDOW = PixelWindow(0, EXPECTED_SHAPE[0], 0, EXPECTED_SHAPE[1])


def bounds_to_window(bounds: tuple[float, float, float, float]) -> PixelWindow:
    south, west, north, east = bounds
    r_north, c_west = latlon_to_pixel(north, west)
    r_south, c_east = latlon_to_pixel(south, east)
    row0 = max(0, int(round(r_north)))
    row1 = min(EXPECTED_SHAPE[0], int(round(r_south)) + 1)
    col0 = max(0, int(round(c_west)))
    col1 = min(EXPECTED_SHAPE[1], int(round(c_east)) + 1)
    if row1 - row0 < 2 or col1 - col0 < 2:
        raise InsatFrameError(f"Bounds {bounds} fall outside the INSAT-3DR ASIA sector.")
    return PixelWindow(row0, row1, col0, col1)


def region_window(region: str) -> PixelWindow:
    if region not in REGIONS:
        raise InsatFrameError(f"Unknown region {region!r}. Valid: {sorted(REGIONS)}")
    return bounds_to_window(REGIONS[region]["bounds"])


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameInfo:
    frame_id: str  # e.g. "20250107T1445Z"
    timestamp: datetime
    path: Path
    valid: bool
    error: str | None = None


def parse_timestamp(filename: str) -> datetime | None:
    match = _FILENAME_RE.match(filename)
    if not match:
        return None
    try:
        naive = datetime.strptime(match.group(1) + match.group(2), "%d%b%Y%H%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc)


def frame_id_for(timestamp: datetime) -> str:
    return timestamp.strftime("%Y%m%dT%H%MZ")


def validate_file(path: Path, channels: tuple[str, ...] = ("TIR1", "WV")) -> str | None:
    """Return None if the file is a readable L1C frame, else a short reason.

    Truncated downloads are the real failure mode in this dataset (4 of the
    63 `.h5` files end mid-stream), and HDF5 reports those at open time.
    """
    h5py = _require_h5py()
    try:
        with h5py.File(path, "r") as handle:
            for channel in channels:
                for key in (f"IMG_{channel}", f"IMG_{channel}_TEMP"):
                    if key not in handle:
                        return f"missing dataset {key}"
                shape = handle[f"IMG_{channel}"].shape
                if tuple(shape[-2:]) != EXPECTED_SHAPE:
                    return f"unexpected grid {tuple(shape)} for IMG_{channel}"
                # Touch the last chunk: a file truncated after its superblock
                # still opens, but fails here.
                handle[f"IMG_{channel}"][0, -1, -1]
    except Exception as exc:  # noqa: BLE001 -- any read failure means "unusable"
        return f"{type(exc).__name__}: {str(exc)[:120]}"
    return None


_CATALOG_CACHE: dict[tuple[str, tuple], list[FrameInfo]] = {}


def scan_catalog(directory: Path | None = None, *, use_cache: bool = True) -> list[FrameInfo]:
    """Every `3RIMG_*_L1C_ASIA_MER_*.h5` under `directory`, time-ordered,
    each marked valid/corrupt. `.part` (incomplete download) files are not
    `.h5` files and are simply never listed."""
    directory = directory or insat_dir()
    if not directory.exists():
        return []
    files = sorted(p for p in directory.rglob("*.h5") if parse_timestamp(p.name))
    signature = tuple((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in files)
    key = (str(directory), signature)
    if use_cache and key in _CATALOG_CACHE:
        return _CATALOG_CACHE[key]

    frames = []
    for path in files:
        timestamp = parse_timestamp(path.name)
        error = validate_file(path)
        frames.append(FrameInfo(frame_id_for(timestamp), timestamp, path, error is None, error))
    frames.sort(key=lambda f: f.timestamp)
    _CATALOG_CACHE.clear()
    _CATALOG_CACHE[key] = frames
    return frames


def valid_frames_by_time(directory: Path | None = None) -> dict[datetime, FrameInfo]:
    return {f.timestamp: f for f in scan_catalog(directory) if f.valid}


def get_frame(frame_id: str, directory: Path | None = None) -> FrameInfo:
    for frame in scan_catalog(directory):
        if frame.frame_id == frame_id:
            return frame
    raise NotFoundError(f"Unknown INSAT-3DR frame id: {frame_id!r}")


def input_sequence(anchor: FrameInfo, n_inputs: int, directory: Path | None = None) -> list[FrameInfo]:
    """The `n_inputs` strictly consecutive (30-minute spaced) valid frames
    ending at `anchor`. Raises if any is missing/corrupt -- a gap is never
    papered over by substituting a different time."""
    by_time = valid_frames_by_time(directory)
    if not anchor.valid:
        raise InsatFrameError(f"Frame {anchor.frame_id} is corrupt and cannot be used: {anchor.error}")
    sequence = []
    for back in range(n_inputs - 1, -1, -1):
        when = anchor.timestamp - back * FRAME_INTERVAL
        frame = by_time.get(when)
        if frame is None:
            raise InsatFrameError(
                f"Frame {anchor.frame_id} cannot be used as a nowcast anchor: the required input frame at "
                f"{when:%Y-%m-%d %H:%M} UTC is missing or corrupt ({n_inputs} consecutive half-hourly frames are needed)."
            )
        sequence.append(frame)
    return sequence


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_counts(path: Path, channel: str, window: PixelWindow = FULL_WINDOW) -> tuple[np.ndarray, np.ndarray, int]:
    """Raw 10-bit counts for `window`, plus this file's own count->K lookup
    table and fill value."""
    if channel not in ALL_CHANNELS:
        raise InsatFrameError(f"Unknown channel {channel!r}. Valid: {ALL_CHANNELS}")
    h5py = _require_h5py()
    try:
        with h5py.File(path, "r") as handle:
            dataset = handle[f"IMG_{channel}"]
            counts = dataset[0, window.row0:window.row1, window.col0:window.col1]
            fill = int(dataset.attrs["_FillValue"][0])
            lut_key = f"IMG_{channel}_TEMP"
            lut = handle[lut_key][:] if lut_key in handle else np.empty(0, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise InsatFrameError(f"Could not read {channel} from {path.name}: {type(exc).__name__}: {exc}") from exc
    return counts, lut, fill


def counts_to_bt(counts: np.ndarray, lut: np.ndarray, fill: int) -> np.ndarray:
    """Brightness temperature in kelvin; fill pixels become NaN."""
    if lut.size == 0:
        raise InsatFrameError("This channel has no brightness-temperature lookup table.")
    bt = lut[np.clip(counts, 0, lut.size - 1)].astype(np.float32)
    bt[counts == fill] = np.nan
    bt[bt > 900] = np.nan  # the LUT's own 999.0 fill value
    return bt


def read_bt(path: Path, channel: str = "TIR1", window: PixelWindow = FULL_WINDOW) -> np.ndarray:
    counts, lut, fill = read_counts(path, channel, window)
    return counts_to_bt(counts, lut, fill)
