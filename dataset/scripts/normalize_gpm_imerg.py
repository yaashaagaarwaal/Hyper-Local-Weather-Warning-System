"""Normalizes a raw GPM IMERG HDF5 granule (from `download_gpm_imerg.py`)
into a clean CSV matching AtmoGuard's existing weather-column conventions,
and maps each grid point to the nearest AtmoGuard `Location`.

NOT YET RUN against a real file: no GPM sample has been downloaded (no
NASA Earthdata account configured on this machine as of writing) -- this
script is written and ready, per the instruction to prepare reusable
preprocessing without fabricating output.

Normalization applied (never invents a value for anything missing/masked):
  - IMERG's `Grid/time` is seconds since 1970-01-01 UTC -> converted to an
    explicit UTC ISO8601 string, never assumed to already be local time.
  - `Grid/precipitation` units are already mm/hr (IMERG's native unit,
    confirmed in its product documentation) -- renamed to
    `rainfall_mm_hour` for consistency with AtmoGuard's own column name,
    value otherwise unchanged.
  - IMERG's fill value (-9999.9, per its own metadata) is converted to
    `None`/NaN in the output, never treated as a real reading of -9999.9mm.
  - Coordinates are IMERG's own lat/lon grid (0.1 degree cells, WGS84) --
    passed through unchanged, just subset to the Delhi-NCR bbox.

Grid mapping reuses the same haversine + MAX_USABLE_RADIUS_KM convention as
`normalize_era5.py` (duplicated, not imported -- dataset/ scripts run
standalone outside the backend venv).

Run as `python dataset/scripts/normalize_gpm_imerg.py <path-to-hdf5-file>`
from the repo root, once a real sample exists.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse the exact same helpers as normalize_era5.py rather than redefine
# them differently -- both scripts live in dataset/scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_era5 import MAX_USABLE_RADIUS_KM, haversine_km, nearest_known_location  # noqa: E402

FILL_VALUE = -9999.9
# Delhi-NCR + Agra, matching download_era5.py / download_gpm_imerg.py.
BBOX_WSEN = (76.7, 26.7, 78.2, 29.1)


def normalize(h5_path: Path, out_csv: Path) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        raise SystemExit("h5py/numpy are required -- run: pip install h5py numpy") from e

    with h5py.File(h5_path, "r") as f:
        grid = f["Grid"]
        time_s = grid["time"][0]
        timestamp = datetime.fromtimestamp(float(time_s), tz=timezone.utc).isoformat()

        lats = grid["lat"][:]
        lons = grid["lon"][:]
        precip = grid["precipitation"][0]  # shape: (lon, lat) per IMERG's own convention

        w, s, e, n = BBOX_WSEN
        lat_idx = np.where((lats >= s) & (lats <= n))[0]
        lon_idx = np.where((lons >= w) & (lons <= e))[0]

        rows = []
        for li in lon_idx:
            for ai in lat_idx:
                lon, lat = float(lons[li]), float(lats[ai])
                value = float(precip[li, ai])
                rainfall = None if abs(value - FILL_VALUE) < 1e-3 else value
                nearest_name, distance_km = nearest_known_location(lat, lon)
                rows.append({
                    "timestamp_utc": timestamp,
                    "latitude": lat,
                    "longitude": lon,
                    "rainfall_mm_hour": rainfall,
                    "nearest_atmoguard_location": nearest_name,
                    "distance_to_nearest_location_km": distance_km,
                    "source": "GPM IMERG V07 (NASA GES DISC)",
                })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python dataset/scripts/normalize_gpm_imerg.py <path-to-hdf5-file>")
    src = Path(sys.argv[1])
    dest = Path(__file__).resolve().parent.parent / "processed" / "gpm_imerg" / f"{src.stem}_normalized.csv"
    normalize(src, dest)
