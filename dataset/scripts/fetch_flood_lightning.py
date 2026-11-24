"""Acquire small, real, login-free flood and lightning imagery samples.

    python dataset/scripts/fetch_flood_lightning.py flood
    python dataset/scripts/fetch_flood_lightning.py lightning --date 2024-07-10 --hours 20-21

FLOOD  -- Sen1Floods11 hand-labelled chips (Cloud to Street; public Google
          Cloud bucket `sen1floods11`, no authentication). Only the South Asian
          events (India, Pakistan, Sri-Lanka) are fetched, not the 14 GB set:
          Sentinel-1 (VV/VH radar) + Sentinel-2 (13-band optical) 512x512 chips
          with a human-labelled water mask, plus the JRC permanent-water and
          S1-Otsu reference masks and the official train/valid/test splits.
LIGHTNING -- NOAA GOES-R Geostationary Lightning Mapper (GLM) L2 flash files
          and the matching ABI L2 CMIP band-13 (10.3 um IR) image, from NOAA's
          public AWS bucket (no authentication). GOES sees the Americas, NOT
          India -- no India lightning data is open without an Earthdata login
          (NASA LIS redirects to a login and was not touched).

Everything lands in dataset/raw/<name>/ (gitignored except SOURCE.md). Files
are verified against the sizes/MD5 the servers publish; existing verified
files are skipped, so re-running resumes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "dataset" / "raw"
GCS_API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
GCS_MEDIA = "https://storage.googleapis.com/sen1floods11/"
FLOOD_EVENTS = ("India", "Pakistan", "Sri-Lanka")
FLOOD_SUBDIRS = ("S1Hand", "S2Hand", "LabelHand", "JRCWaterHand", "S1OtsuLabelHand")
GOES_BUCKET = "https://noaa-goes16.s3.amazonaws.com"


def http_get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dataset-fetch/1.0"}), timeout=timeout) as r:
        return r.read()


def download(url: str, dest: Path, size: int | None, md5_b64: str | None = None, md5_hex: str | None = None) -> str:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = http_get(url, timeout=300)
    if size is not None and len(data) != size:
        raise IOError(f"{dest.name}: got {len(data)} bytes, expected {size}")
    digest = hashlib.md5(data)  # noqa: S324 -- integrity check against the server's published hash
    if md5_b64 and base64.b64encode(digest.digest()).decode() != md5_b64:
        raise IOError(f"{dest.name}: MD5 mismatch")
    if md5_hex and digest.hexdigest() != md5_hex:
        raise IOError(f"{dest.name}: MD5 mismatch")
    dest.write_bytes(data)
    return "ok"


def run_pool(jobs, label: str) -> None:
    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in [pool.submit(*job) for job in jobs]:
            try:
                skipped += fut.result() == "skip"
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAILED: {exc}", file=sys.stderr)
    print(f"{label}: {done} ok ({skipped} already present), {failed} failed")
    if failed:
        sys.exit(1)


# --------------------------------------------------------------------------
def gcs_list(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        url = f"{GCS_API}?prefix={urllib.parse.quote(prefix)}&maxResults=1000" + (f"&pageToken={token}" if token else "")
        page = json.loads(http_get(url))
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return items


def fetch_flood() -> None:
    out = RAW / "sen1floods11"
    jobs, index = [], []
    for sub in FLOOD_SUBDIRS:
        for item in gcs_list(f"v1.1/data/flood_events/HandLabeled/{sub}/"):
            name = item["name"].rsplit("/", 1)[-1]
            if name.split("_")[0] not in FLOOD_EVENTS:
                continue
            dest = out / sub / name
            jobs.append((download, GCS_MEDIA + item["name"], dest, int(item["size"]), item["md5Hash"]))
            index.append({"file": f"{sub}/{name}", "bytes": int(item["size"]), "md5_b64": item["md5Hash"]})
    for extra in ("v1.1/Sen1Floods11_Metadata.geojson",
                  *(f"v1.1/splits/flood_handlabeled/flood_{s}_data.csv" for s in ("train", "valid", "test", "bolivia"))):
        for item in gcs_list(extra):
            if item["name"] == extra:
                jobs.append((download, GCS_MEDIA + extra, out / extra.rsplit("/", 1)[-1], int(item["size"]), item["md5Hash"]))
    total = sum(e["bytes"] for e in index)
    print(f"flood: {len(index)} chips-files, {total / 1e6:.0f} MB ({', '.join(FLOOD_EVENTS)})")
    run_pool(jobs, "flood")
    (out / "manifest.json").write_text(json.dumps({"source": GCS_MEDIA, "events": FLOOD_EVENTS, "files": index}, indent=1))


# --------------------------------------------------------------------------
def s3_list(prefix: str) -> list[dict]:
    items, token = [], None
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        url = f"{GOES_BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        root = ET.fromstring(http_get(url))
        for c in root.findall("s:Contents", ns):
            items.append({"key": c.find("s:Key", ns).text, "size": int(c.find("s:Size", ns).text), "etag": c.find("s:ETag", ns).text.strip('"')})
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not token:
            return items


def fetch_lightning(date: str, hours: range) -> None:
    from datetime import date as _date
    day = _date.fromisoformat(date)
    doy = day.timetuple().tm_yday
    out = RAW / "goes_glm"
    jobs, files = [], []
    for hour in hours:
        for product, label in (("GLM-L2-LCFA", "glm"), ("ABI-L2-CMIPC", "abi")):
            for item in s3_list(f"{product}/{day.year}/{doy:03d}/{hour:02d}/"):
                name = item["key"].rsplit("/", 1)[-1]
                if label == "abi" and "M6C13_" not in name:  # band 13 = 10.3 um clean IR window
                    continue
                md5 = item["etag"] if "-" not in item["etag"] else None  # multipart ETags are not an MD5
                jobs.append((download, f"{GOES_BUCKET}/{item['key']}", out / label / f"{day:%Y%m%d}" / name, item["size"], None, md5))
                files.append({"file": f"{label}/{day:%Y%m%d}/{name}", "bytes": item["size"]})
    print(f"lightning: {len(files)} files, {sum(f['bytes'] for f in files) / 1e6:.0f} MB (GOES-16, {date}, hours {hours.start}-{hours.stop - 1} UTC)")
    run_pool(jobs, "lightning")
    manifest = out / "manifest.json"
    prior = json.loads(manifest.read_text())["files"] if manifest.exists() else []
    merged = {f["file"]: f for f in prior + files}
    manifest.write_text(json.dumps({"source": GOES_BUCKET, "files": list(merged.values())}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("flood")
    li = sub.add_parser("lightning")
    li.add_argument("--date", required=True, help="YYYY-MM-DD (GOES-16 / GOES-East coverage)")
    li.add_argument("--hours", default="20-21", help="UTC hour range, e.g. 20-21")
    args = parser.parse_args()
    if args.cmd == "flood":
        fetch_flood()
    else:
        lo, hi = (int(x) for x in args.hours.split("-"))
        fetch_lightning(args.date, range(lo, hi + 1))
