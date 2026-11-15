"""Phase 4 Stage 1: downloads NASA GPM IMERG V07 half-hourly precipitation
granules for Delhi-NCR via NASA Earthdata.

REQUIRES a free NASA Earthdata Login -- this script cannot create that
account or hold its credentials for you:
  1. Register at https://urs.earthdata.nasa.gov (free).
  2. `pip install earthaccess` (NASA's own official client -- handles auth,
     search, and spatial/temporal subsetting against GES DISC).
  3. Either run this script interactively once (earthaccess.login() will
     prompt for username/password and offer to save them to ~/.netrc), or
     create ~/.netrc yourself with:
         machine urs.earthdata.nasa.gov login <your-username> password <your-password>
     Do not paste credentials into chat or commit ~/.netrc anywhere.

Bounding box matches download_era5.py's DELHI_NCR_BBOX (same region) so the
two sources can later be compared/aligned without re-deriving the area.

Run as `python dataset/scripts/download_gpm_imerg.py --sample` for a small
1-week sample first; drop `--sample` for the full period once verified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "raw" / "gpm_imerg"

# West, South, East, North -- earthaccess's bounding_box() takes this order
# (note: different order from CDS's area=[N,W,S,E] in download_era5.py).
DELHI_NCR_BBOX_WSEN = (76.7, 26.7, 78.2, 29.1)

SHORT_NAME = "GPM_3IMERGHH"  # IMERG Final Precipitation L3 Half Hourly 0.1deg
VERSION = "07"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="download a small 1-week sample only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    netrc_path = Path.home() / ".netrc"
    has_earthdata_entry = netrc_path.exists() and "urs.earthdata.nasa.gov" in netrc_path.read_text()
    if not has_earthdata_entry:
        raise SystemExit(
            "No NASA Earthdata credentials found in ~/.netrc. This script cannot create "
            "that account or hold its password for you -- register (free) at "
            "https://urs.earthdata.nasa.gov, then either run earthaccess.login() "
            "interactively once to save credentials, or add a 'machine urs.earthdata.nasa.gov' "
            "entry to ~/.netrc yourself, and re-run."
        )

    try:
        import earthaccess
    except ImportError as e:
        raise SystemExit("earthaccess is not installed -- run: pip install earthaccess") from e

    earthaccess.login(strategy="netrc")

    if args.sample:
        temporal = ("2024-07-01", "2024-07-07")
    else:
        temporal = ("2022-01-01", "2025-12-31")

    print(f"Searching {SHORT_NAME} v{VERSION} for {temporal} over Delhi-NCR...", flush=True)
    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        bounding_box=DELHI_NCR_BBOX_WSEN,
        temporal=temporal,
    )
    print(f"Found {len(results)} granules.", flush=True)
    if not results:
        raise SystemExit("No granules found -- check the short_name/version are still current on GES DISC.")

    downloaded = earthaccess.download(results, str(OUT_DIR))

    manifest = {
        "short_name": SHORT_NAME,
        "version": VERSION,
        "bounding_box_w_s_e_n": DELHI_NCR_BBOX_WSEN,
        "temporal_range": temporal,
        "sample": args.sample,
        "granule_count": len(results),
        "downloaded_files": [str(Path(f).name) for f in downloaded],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "NASA GES DISC via earthaccess (https://disc.gsfc.nasa.gov)",
        "license": "NASA data policy -- free for public use; cite the GPM data product "
        "reference on https://gpm.nasa.gov/data/policy. Not independently confirmed as "
        "formal public domain -- see dataset/scripts/README.md.",
    }
    (OUT_DIR / f"imerg_{'sample' if args.sample else 'full'}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"Downloaded {len(downloaded)} files to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
