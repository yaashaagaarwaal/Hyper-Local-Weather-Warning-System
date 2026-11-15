"""Phase 4 Stage 1: downloads ERA5 hourly single-level reanalysis data for
Delhi-NCR from the Copernicus Climate Data Store (CDS).

REQUIRES a free CDS account and API key -- this script cannot run without
one, and nobody but the account holder should create that account or hold
its key:
  1. Register at https://cds.climate.copernicus.eu (free, no institution
     needed).
  2. Copy the API key shown on your CDS account page into `~/.cdsapirc`:
         url: https://cds.climate.copernicus.eu/api
         key: <your-personal-access-token>
  3. Accept the "Licence to use Copernicus Products" on the CDS website for
     the `reanalysis-era5-single-levels` dataset (one-time, via the web UI's
     download form) -- the API refuses requests until this is done.
  4. `pip install cdsapi` inside the backend venv (or a separate one for
     dataset/ scripts).

Variables chosen to match AtmoGuard's existing feature set
(backend/ml/config.py BASE_WEATHER_COLUMNS) so a future model could use
ERA5 history as a second, longer-range source alongside the current
Open-Meteo-derived CSV -- never blended silently; see the project's own
"never claim a feature works without real data" rule.

Bounding box: approximately Delhi NCR + Agra, matching
frontend/src/data/locations.js's existing demo locations.

Run as `python dataset/scripts/download_era5.py --sample` for a small
1-week sample first (per Stage 1's "verify before expanding" instruction);
drop `--sample` for a full-period request once the sample is verified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "raw" / "era5"

# North, West, South, East -- covers Delhi/NCR (Gurugram, Noida, Ghaziabad,
# Faridabad) and Agra, matching frontend/src/data/locations.js.
DELHI_NCR_BBOX = [29.1, 76.7, 26.7, 78.2]

# Mapped to backend/ml/config.py's BASE_WEATHER_COLUMNS equivalents:
#   2m_temperature            -> temperature_c
#   2m_dewpoint_temperature   -> dew_point_c
#   total_precipitation       -> rainfall_mm_hour (needs unit conversion, see processing notes)
#   10m_u/v_component_of_wind -> wind_speed_kmh / wind_direction_deg (needs vector->speed/dir conversion)
#   surface_pressure          -> surface_pressure_hpa
#   mean_sea_level_pressure   -> pressure_hpa
#   total_cloud_cover         -> cloud_cover_pct
#   volumetric_soil_water_layer_1/2/3 -> soil_moisture_0_7cm / 7_28cm / 28_100cm
#   convective_available_potential_energy -> cape_j_kg
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "convective_available_potential_energy",
]


def build_request(sample: bool) -> dict:
    if sample:
        years, months, days = ["2024"], ["07"], [f"{d:02d}" for d in range(1, 8)]
    else:
        years = [str(y) for y in range(2022, 2026)]
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]

    return {
        "product_type": "reanalysis",
        "variable": ERA5_VARIABLES,
        "year": years,
        "month": months,
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": DELHI_NCR_BBOX,
        "format": "netcdf",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="download a small 1-week sample only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        raise SystemExit(
            "No ~/.cdsapirc found. This script cannot create a CDS account or hold its "
            "key for you -- register at https://cds.climate.copernicus.eu (free), then "
            "put your API key in ~/.cdsapirc as described in this file's module docstring, "
            "and re-run."
        )

    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit("cdsapi is not installed -- run: pip install cdsapi") from e

    request = build_request(args.sample)
    out_name = "era5_delhi_ncr_sample.nc" if args.sample else "era5_delhi_ncr_full.nc"
    out_path = OUT_DIR / out_name

    print(f"Requesting ERA5 single-levels data (sample={args.sample}) -> {out_path}", flush=True)
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))

    manifest = {
        "dataset": "reanalysis-era5-single-levels",
        "variables": ERA5_VARIABLES,
        "bbox_north_west_south_east": DELHI_NCR_BBOX,
        "sample": args.sample,
        "request": request,
        "output_file": out_name,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "Copernicus Climate Data Store (https://cds.climate.copernicus.eu)",
        "license": "Licence to use Copernicus Products -- free, attribution to "
        "Copernicus Climate Change Service (and ECMWF where specified) required.",
    }
    (OUT_DIR / f"{out_path.stem}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Manifest written to {out_path.stem}_manifest.json", flush=True)


if __name__ == "__main__":
    main()
