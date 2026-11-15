"""Phase 4 Stage 1: downloads the India Flood Inventory-Impacts (IFI-Impacts)
v4 dataset from its official Zenodo record (DOI 10.5281/zenodo.4742142,
record id 16994648) -- no account or API key needed, this is a fully open
Zenodo record.

License: CC-BY-NC-4.0 (non-commercial use only) -- see
dataset/raw/india_flood_inventory/SOURCE.md after running this.

Every file's md5 checksum (from Zenodo's own API, not computed after the
fact) is verified after download; a mismatch raises rather than silently
keeping a corrupted file.

Run as `python dataset/scripts/download_india_flood_inventory.py` from the
repo root.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RECORD_API_URL = "https://zenodo.org/api/records/16994648"
RECORD_LANDING_URL = "https://zenodo.org/records/16994648"
OUT_DIR = Path(__file__).resolve().parent.parent / "raw" / "india_flood_inventory"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(RECORD_API_URL) as resp:
        record = json.loads(resp.read())

    license_id = record.get("metadata", {}).get("license", {}).get("id", "unknown")
    pub_date = record.get("metadata", {}).get("publication_date", "unknown")
    title = record.get("metadata", {}).get("title", "unknown")

    downloaded = []
    for entry in record["files"]:
        filename = entry["key"]
        url = entry["links"]["self"]
        expected_md5 = entry["checksum"].removeprefix("md5:")
        dest = OUT_DIR / filename

        print(f"Downloading {filename} ({entry['size']} bytes)...", flush=True)
        urllib.request.urlretrieve(url, dest)

        actual_md5 = _md5(dest)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"checksum mismatch for {filename}: expected {expected_md5}, got {actual_md5}"
            )
        print(f"  OK -- md5 verified ({actual_md5})", flush=True)
        downloaded.append({"filename": filename, "size": entry["size"], "md5": actual_md5})

    source_doc = f"""# India Flood Inventory-Impacts (IFI-Impacts) v4 -- source record

- **Title:** {title}
- **Official record:** {RECORD_LANDING_URL}
- **License:** {license_id} -- **non-commercial use only**, attribution required.
- **Publication date (Zenodo):** {pub_date}
- **Downloaded:** {datetime.now(timezone.utc).isoformat()}
- **Downloaded by:** `dataset/scripts/download_india_flood_inventory.py`

## Files (checksum-verified at download time)

{chr(10).join(f"- `{d['filename']}` -- {d['size']} bytes, md5 `{d['md5']}`" for d in downloaded)}

## Known limitation (per the project's own dataset, carried over honestly)

Primary source is IMD's "Disastrous Weather Events" database (~89%), with
the remainder from EM-DAT and the Dartmouth Flood Observatory -- these are
event *records* (date, district, severity, area, casualties), not a
continuous flood/no-flood time series and not raw hydrological
measurements. Does not include flood discharge or catchment variables.
"""
    (OUT_DIR / "SOURCE.md").write_text(source_doc)
    print(f"\nWrote {OUT_DIR / 'SOURCE.md'}", flush=True)
    print(f"All {len(downloaded)} files downloaded and checksum-verified into {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
