"""Phase 4: downloads India's ADM2 (district) boundary GeoJSON from
geoBoundaries -- no account or API key needed, this is a fully open dataset.

Fetches the current metadata from geoBoundaries' own API (rather than
hardcoding a URL) so the exact license, source, and download link used are
always the ones geoBoundaries is serving right now, then downloads that
exact GeoJSON file.

License: Open Data Commons Open Database License 1.0 (ODbL) -- confirmed
directly from the API response, not assumed. Source: Pathways Data Pvt.
Ltd., derived from lgdirectory.gov.in (India's own Local Government
Directory) -- see dataset/boundaries/SOURCE.md after running this.

Run as `python dataset/scripts/download_geoboundaries_india_adm2.py` from
the repo root.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
OUT_DIR = Path(__file__).resolve().parent.parent / "boundaries"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(API_URL) as resp:
        meta = json.loads(resp.read())

    gj_url = meta["gjDownloadURL"]
    out_path = OUT_DIR / "geoBoundaries-IND-ADM2.geojson"

    print(f"Downloading {gj_url} -> {out_path}", flush=True)
    urllib.request.urlretrieve(gj_url, out_path)
    size = out_path.stat().st_size
    print(f"Downloaded {size} bytes", flush=True)

    with open(out_path) as f:
        gj = json.load(f)
    feature_count = len(gj.get("features", []))

    source_doc = f"""# India ADM2 (District) boundaries -- source record

- **Provider:** geoBoundaries (https://www.geoboundaries.org)
- **API metadata used:** {API_URL}
- **Boundary type:** {meta['boundaryType']} ({meta['boundaryCanonical']})
- **Underlying source:** {meta['boundarySource']}
- **License:** {meta['boundaryLicense']} -- license detail source: {meta['licenseSource']}
- **Source data update date:** {meta['sourceDataUpdateDate']}
- **geoBoundaries build date:** {meta['buildDate']}
- **Admin unit count (per geoBoundaries metadata):** {meta['admUnitCount']}
- **Actual feature count in downloaded file:** {feature_count}
- **File:** `{out_path.name}` ({size} bytes)
- **Downloaded:** {datetime.now(timezone.utc).isoformat()}
- **Downloaded by:** `dataset/scripts/download_geoboundaries_india_adm2.py`

## Attribution (per the ODbL's attribution requirement)

Boundary data from geoBoundaries (https://www.geoboundaries.org), licensed
under the Open Data Commons Open Database License (ODbL) 1.0. Underlying
source: {meta['boundarySource']}.

## Known limitation

`admUnitCount` ({meta['admUnitCount']}) is lower than India's current
district count (India has undergone further district splits since this
layer's {meta['boundaryYearRepresented']} vintage) -- some newer districts
will not have their own polygon here and will only match at an older
parent district's boundary.
"""
    (OUT_DIR / "SOURCE.md").write_text(source_doc)
    print(f"Wrote {OUT_DIR / 'SOURCE.md'}", flush=True)
    print(f"Feature count: {feature_count}", flush=True)


if __name__ == "__main__":
    main()
