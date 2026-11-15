"""Normalizes a raw ERA5 NetCDF sample (from `download_era5.py`) into a
clean CSV matching AtmoGuard's existing weather-column conventions, and
maps each grid point to the nearest AtmoGuard `Location`/grid cell.

NOT YET RUN against a real file: no ERA5 sample has been downloaded (no CDS
account configured on this machine as of writing) -- this script is written
and ready, per the instruction to prepare reusable preprocessing without
fabricating output. Running it requires `pip install xarray netCDF4` (not
installed yet -- no point installing before there's a real file to read).

Normalization applied, matching backend/ml/config.py's BASE_WEATHER_COLUMNS
naming and units (never invents a value for anything missing):
  - timestamps -> UTC ISO8601 (ERA5's `time` dimension is already UTC, but
    made explicit rather than assumed)
  - 2m_temperature (Kelvin) -> temperature_c (Celsius): x - 273.15
  - 2m_dewpoint_temperature (Kelvin) -> dew_point_c: x - 273.15
  - total_precipitation (metres, ERA5's own hourly-accumulated convention)
    -> rainfall_mm_hour: x * 1000
  - 10m_u/v_component_of_wind (m/s each) -> wind_speed_kmh = hypot(u,v)*3.6,
    wind_direction_deg = (270 - atan2(v,u)*180/pi) % 360
  - surface_pressure / mean_sea_level_pressure (Pa) -> *_hpa: x / 100
  - total_cloud_cover (fraction 0-1) -> cloud_cover_pct: x * 100
  - volumetric_soil_water_layer_1/2/3 (m3/m3) -> soil_moisture_0_7cm /
    7_28cm / 28_100cm: unchanged (already the same unit AtmoGuard uses)
  - convective_available_potential_energy (J/kg) -> cape_j_kg: unchanged
  - any NaN/masked cell is kept as NaN in the output -- never filled.

Grid mapping reuses the same haversine distance + MAX_USABLE_RADIUS_KM
"too far to be meaningful" rule already established in
backend/app/services/grid_mapping_service.py (duplicated here, not
imported, since dataset/ scripts run standalone outside the backend venv).

Run as `python dataset/scripts/normalize_era5.py <path-to-nc-file>` from
the repo root, once a real sample exists.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

MAX_USABLE_RADIUS_KM = 45.0  # matches app/services/risk_engine/thresholds.py

# Minimal copy of frontend/src/data/locations.js's Delhi-NCR points, used
# only to label which AtmoGuard location a grid point is nearest to -- not
# a second source of truth; re-derive from the real locations table if this
# drifts.
KNOWN_LOCATIONS = [
    ("loc-ndls", "New Delhi", 28.6315, 77.2167),
    ("loc-gurugram", "Gurugram", 28.4595, 77.0266),
    ("loc-noida", "Noida", 28.5355, 77.3910),
    ("loc-ghaziabad", "Ghaziabad", 28.6692, 77.4538),
    ("loc-faridabad", "Faridabad", 28.4089, 77.3178),
    ("loc-agra", "Agra", 27.1767, 78.0081),
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_known_location(lat: float, lon: float) -> tuple[str | None, float | None]:
    if not KNOWN_LOCATIONS:
        return None, None
    best = min(KNOWN_LOCATIONS, key=lambda loc: haversine_km(lat, lon, loc[2], loc[3]))
    dist = haversine_km(lat, lon, best[2], best[3])
    if dist > MAX_USABLE_RADIUS_KM:
        return None, dist
    return best[1], dist


def kelvin_to_celsius(k: float) -> float:
    return k - 273.15


def wind_speed_direction(u: float, v: float) -> tuple[float, float]:
    speed_kmh = math.hypot(u, v) * 3.6
    direction_deg = (270 - math.degrees(math.atan2(v, u))) % 360
    return speed_kmh, direction_deg


def normalize(nc_path: Path, out_csv: Path) -> None:
    try:
        import xarray as xr
    except ImportError as e:
        raise SystemExit(
            "xarray is not installed -- run: pip install xarray netCDF4 "
            "(only needed once a real ERA5 sample exists to process)"
        ) from e

    ds = xr.open_dataset(nc_path)
    rows = []
    for t in ds["valid_time"].values if "valid_time" in ds else ds["time"].values:
        for lat in ds["latitude"].values:
            for lon in ds["longitude"].values:
                point = ds.sel(latitude=lat, longitude=lon, method="nearest")
                point = point.sel(valid_time=t) if "valid_time" in ds else point.sel(time=t)

                def val(name):
                    v = point[name].values if name in point else None
                    return None if v is None or (hasattr(v, "item") and str(v.item()) == "nan") else float(v)

                u, v = val("u10"), val("v10")
                wind_speed, wind_dir = (None, None) if u is None or v is None else wind_speed_direction(u, v)
                nearest_name, distance_km = nearest_known_location(float(lat), float(lon))

                temp_k, dew_k, precip_m = val("t2m"), val("d2m"), val("tp")
                rows.append({
                    "timestamp_utc": str(t),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "temperature_c": None if temp_k is None else kelvin_to_celsius(temp_k),
                    "dew_point_c": None if dew_k is None else kelvin_to_celsius(dew_k),
                    "rainfall_mm_hour": None if precip_m is None else precip_m * 1000,
                    "wind_speed_kmh": wind_speed,
                    "wind_direction_deg": wind_dir,
                    "surface_pressure_hpa": (val("sp") or 0) / 100 if val("sp") is not None else None,
                    "pressure_hpa": (val("msl") or 0) / 100 if val("msl") is not None else None,
                    "cloud_cover_pct": (val("tcc") or 0) * 100 if val("tcc") is not None else None,
                    "soil_moisture_0_7cm": val("swvl1"),
                    "soil_moisture_7_28cm": val("swvl2"),
                    "soil_moisture_28_100cm": val("swvl3"),
                    "cape_j_kg": val("cape"),
                    "nearest_atmoguard_location": nearest_name,
                    "distance_to_nearest_location_km": distance_km,
                    "source": "ERA5 (Copernicus Climate Data Store)",
                })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python dataset/scripts/normalize_era5.py <path-to-nc-file>")
    src = Path(sys.argv[1])
    dest = Path(__file__).resolve().parent.parent / "processed" / "era5" / f"{src.stem}_normalized.csv"
    normalize(src, dest)
