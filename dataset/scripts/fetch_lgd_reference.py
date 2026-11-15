"""Fetches India's authoritative LGD (Local Government Directory) district
and state reference tables from a community-maintained mirror of LGD's own
exports (ramseraph.github.io/opendata/lgd -- no API exists on
lgdirectory.gov.in itself, confirmed during Phase 4 research).

This is reference/lookup data (district_code, district_name, state_code,
state_name, census codes) used ONLY to build the geoBoundaries<->IFI
crosswalk -- it is not a boundary geometry source and not a second
acquisition of the flood inventory or geoBoundaries themselves.

Run as `python dataset/scripts/fetch_lgd_reference.py` from the repo root.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "boundaries" / "lgd"

FILES = {
    "districts.parquet": "https://ramseraph.github.io/opendata/lgd/data/districts.8edd1bf95e5bfeec.parquet",
    "states.parquet": "https://ramseraph.github.io/opendata/lgd/data/states.3a760944dab3b3ff.parquet",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = OUT_DIR / name
        print(f"Downloading {url} -> {dest}", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size} bytes", flush=True)

    import pandas as pd

    districts = pd.read_parquet(OUT_DIR / "districts.parquet")
    states = pd.read_parquet(OUT_DIR / "states.parquet")

    source_doc = f"""# LGD (Local Government Directory) reference tables

- **Provider:** community mirror of LGD's own exports, https://ramseraph.github.io/opendata/lgd
  (lgdirectory.gov.in itself has no public API -- confirmed during Phase 4 research)
- **Underlying authority:** Government of India, Ministry of Panchayati Raj --
  Local Government Directory (https://lgdirectory.gov.in)
- **Downloaded:** {datetime.now(timezone.utc).isoformat()}
- **Downloaded by:** `dataset/scripts/fetch_lgd_reference.py`
- **Districts:** {len(districts)} rows, columns: {list(districts.columns)}
- **States:** {len(states)} rows, columns: {list(states.columns)}

## Use

Reference/lookup only -- joins geoBoundaries' `shapeName` and the India Flood
Inventory's district names to an authoritative (district_code, state_code,
state_name) triple. See `dataset/processed/district_crosswalk/` for the
actual crosswalk built from this table.

## Known limitation

This mirror is a third-party convenience copy, not the government portal
itself -- treat `district_code`/`state_code` as authoritative LGD identifiers,
but re-verify against lgdirectory.gov.in directly for anything safety-critical.
"""
    (OUT_DIR / "SOURCE.md").write_text(source_doc)
    print(f"Wrote {OUT_DIR / 'SOURCE.md'}", flush=True)


if __name__ == "__main__":
    main()
