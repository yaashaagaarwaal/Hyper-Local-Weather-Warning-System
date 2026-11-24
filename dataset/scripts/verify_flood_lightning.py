"""Open and sanity-check every downloaded flood / lightning file, then write
`dataset/processed/flood_lightning/{report.json,REPORT.md}` and quick-look
figures. Read-only on dataset/raw/.

    backend/.venv/bin/python dataset/scripts/verify_flood_lightning.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FLOOD = REPO / "dataset" / "raw" / "sen1floods11"
GLM = REPO / "dataset" / "raw" / "goes_glm"
OUT = REPO / "dataset" / "processed" / "flood_lightning"
GOES_EPOCH = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
def verify_flood() -> dict:
    split_of = {}
    for split in ("train", "valid", "test"):
        with open(FLOOD / f"flood_{split}_data.csv") as fh:
            for s1, label in csv.reader(fh):
                split_of[s1.replace("_S1Hand.tif", "")] = split
    chips = sorted(p.name.replace("_S1Hand.tif", "") for p in (FLOOD / "S1Hand").glob("*_S1Hand.tif"))
    rows, problems = [], []
    for chip in chips:
        try:
            s1 = tifffile.imread(FLOOD / "S1Hand" / f"{chip}_S1Hand.tif")
            s2 = tifffile.imread(FLOOD / "S2Hand" / f"{chip}_S2Hand.tif")
            lab = tifffile.imread(FLOOD / "LabelHand" / f"{chip}_LabelHand.tif")
            jrc = tifffile.imread(FLOOD / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{chip}: unreadable ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        valid = lab >= 0
        row = {
            "chip": chip, "event": chip.split("_")[0], "split": split_of.get(chip, "none"),
            "s1Shape": list(s1.shape), "s2Shape": list(s2.shape), "labelShape": list(lab.shape),
            "labelValues": sorted(int(v) for v in np.unique(lab)),
            "validPct": float(100 * valid.mean()),
            "floodWaterPct": float(100 * (lab == 1).sum() / max(valid.sum(), 1)),
            "s1NanPct": float(100 * np.isnan(s1).mean()),
            "s1VvDbRange": [float(np.nanmin(s1[0])), float(np.nanmax(s1[0]))],
            "s2MaxReflectance": int(s2.max()),
            "jrcPermanentWaterPct": float(100 * (jrc == 1).mean()),
        }
        if s1.shape != (2, 512, 512) or s2.shape != (13, 512, 512) or lab.shape != (512, 512):
            problems.append(f"{chip}: unexpected shapes {s1.shape} {s2.shape} {lab.shape}")
        if not set(row["labelValues"]) <= {-1, 0, 1}:
            problems.append(f"{chip}: label values {row['labelValues']}")
        rows.append(row)

    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event"]].append(r)
    summary = {
        "chips": len(rows), "problems": problems,
        "perEvent": {e: {"chips": len(v), "meanFloodWaterPct": round(float(np.mean([x["floodWaterPct"] for x in v])), 2),
                         "chipsWithAnyFlood": sum(x["floodWaterPct"] > 0 for x in v)} for e, v in sorted(by_event.items())},
        "perSplit": dict(Counter(r["split"] for r in rows)),
        "labelledPixelClassBalance": {
            "floodWaterPct": round(float(np.average([r["floodWaterPct"] for r in rows], weights=[r["validPct"] for r in rows])), 2),
        },
        "chipsWithNanInS1": sum(r["s1NanPct"] > 0 for r in rows),
        "chipsWithInvalidLabelPixels": sum(r["validPct"] < 100 for r in rows),
        "chipsPerSplitOutsideThisSubset": "official splits list all 11 events; only India/Pakistan/Sri-Lanka were downloaded",
        "rows": rows,
    }

    # quick-look: the India chip with the most flood water
    pick = max((r for r in rows if r["event"] == "India"), key=lambda r: r["floodWaterPct"])
    s1 = tifffile.imread(FLOOD / "S1Hand" / f"{pick['chip']}_S1Hand.tif")
    s2 = tifffile.imread(FLOOD / "S2Hand" / f"{pick['chip']}_S2Hand.tif")
    lab = tifffile.imread(FLOOD / "LabelHand" / f"{pick['chip']}_LabelHand.tif")
    rgb = np.clip(s2[[3, 2, 1]].transpose(1, 2, 0) / 3000.0, 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    ax[0].imshow(rgb); ax[0].set_title(f"Sentinel-2 true colour -- {pick['chip']}")
    ax[1].imshow(s1[0], cmap="gray", vmin=-25, vmax=0); ax[1].set_title("Sentinel-1 VV radar (dB)")
    ax[2].imshow(np.ma.masked_less(lab, 0), cmap="Blues", vmin=0, vmax=1); ax[2].set_title(f"Human-labelled water ({pick['floodWaterPct']:.0f}% of valid pixels)")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "flood_quicklook_india.png", dpi=70); plt.close(fig)
    return summary


# --------------------------------------------------------------------------
def goes_xy(lat, lon, proj):
    req, rpol = proj["semi_major_axis"], proj["semi_minor_axis"]
    h = proj["perspective_point_height"] + req
    lon0 = np.radians(proj["longitude_of_projection_origin"])
    phi, lam = np.radians(lat), np.radians(lon)
    phic = np.arctan((rpol ** 2 / req ** 2) * np.tan(phi))
    e2 = 1 - rpol ** 2 / req ** 2
    rc = rpol / np.sqrt(1 - e2 * np.cos(phic) ** 2)
    sx = h - rc * np.cos(phic) * np.cos(lam - lon0)
    sy = -rc * np.cos(phic) * np.sin(lam - lon0)
    sz = rc * np.sin(phic)
    return np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), np.arctan(sz / sx)


def read_abi(path):
    with h5py.File(path, "r") as f:
        cmi = f["CMI"]
        raw = cmi[:]
        bt = np.where(raw == cmi.attrs["_FillValue"], np.nan, raw * cmi.attrs["scale_factor"] + cmi.attrs["add_offset"])
        proj = {k: float(np.asarray(f["goes_imager_projection"].attrs[k]).ravel()[0]) for k in
                ("semi_major_axis", "semi_minor_axis", "perspective_point_height", "longitude_of_projection_origin")}
        xs = f["x"][:] * f["x"].attrs["scale_factor"] + f["x"].attrs["add_offset"]
        ys = f["y"][:] * f["y"].attrs["scale_factor"] + f["y"].attrs["add_offset"]
        t = GOES_EPOCH.timestamp() + float(f["t"][()])
        dqf_bad = float((f["DQF"][:] > 1).mean())
    return bt.astype(np.float32), xs, ys, proj, datetime.fromtimestamp(t, timezone.utc), dqf_bad


def verify_lightning() -> dict:
    glm_files = sorted((GLM / "glm").rglob("*.nc"))
    abi_files = sorted((GLM / "abi").rglob("*.nc"))
    flashes, per_file, problems = [], [], []
    for path in glm_files:
        try:
            with h5py.File(path, "r") as f:
                lat, lon, e = f["flash_lat"][:], f["flash_lon"][:], f["flash_energy"][:]
                off = f["flash_time_offset_of_first_event"]  # seconds relative to this file's product_time
                tflash = GOES_EPOCH.timestamp() + float(f["product_time"][()]) + off[:].astype("f8") * off.attrs["scale_factor"] + off.attrs["add_offset"]
                dqf = f["flash_quality_flag"][:]
            per_file.append(len(lat))
            flashes.append(np.column_stack([lat, lon, e.astype("f8"), tflash, dqf]))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path.name}: {type(exc).__name__}: {str(exc)[:80]}")
    allf = np.concatenate(flashes)
    lat, lon = allf[:, 0], allf[:, 1]

    abi_meta = []
    for path in abi_files:
        try:
            bt, xs, ys, proj, when, bad = read_abi(path)
            abi_meta.append({"file": path.name, "time": when.isoformat(), "shape": list(bt.shape),
                             "btMinK": float(np.nanmin(bt)), "btMaxK": float(np.nanmax(bt)),
                             "nanPct": float(100 * np.isnan(bt).mean()), "badQualityPct": round(100 * bad, 3)})
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path.name}: {type(exc).__name__}: {str(exc)[:80]}")

    # quick-look: the ABI IR frame nearest the busiest 5-minute window, with the flashes inside it
    bins = Counter(((allf[:, 3] // 300) * 300).astype(int))
    busiest = max(bins, key=bins.get)
    frame_path = min(abi_files, key=lambda p: abs(read_abi(p)[4].timestamp() - (busiest + 150)))
    bt, xs, ys, proj, when, _ = read_abi(frame_path)
    sel = (allf[:, 3] >= when.timestamp() - 150) & (allf[:, 3] < when.timestamp() + 150)
    fx, fy = goes_xy(allf[sel, 0], allf[sel, 1], proj)
    px = np.interp(fx, xs, np.arange(xs.size))
    py = np.interp(-fy, -ys, np.arange(ys.size))  # y decreases with row index
    inside = (fx >= xs.min()) & (fx <= xs.max()) & (fy >= ys.min()) & (fy <= ys.max())
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.imshow(bt, cmap="gray_r", vmin=190, vmax=310)
    ax.scatter(px[inside], py[inside], s=6, c="#ff2d95", alpha=0.8, label=f"GLM flashes, 5 min window ({int(inside.sum())})")
    ax.set_title(f"GOES-16 ABI band 13 (10.3 um) + GLM lightning flashes, {when:%Y-%m-%d %H:%M} UTC (CONUS, real data)")
    ax.legend(loc="lower left"); ax.axis("off")
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "lightning_quicklook_conus.png", dpi=70); plt.close(fig)
    cold = None
    if inside.any():  # does lightning sit under cold cloud tops? (a sanity check that ABI and GLM align)
        vals = bt[np.clip(py[inside].astype(int), 0, bt.shape[0] - 1), np.clip(px[inside].astype(int), 0, bt.shape[1] - 1)]
        cold = {"flashesChecked": int(inside.sum()), "medianBtUnderFlashK": float(np.nanmedian(vals)),
                "medianBtWholeSceneK": float(np.nanmedian(bt)), "pctFlashesUnder250K": float(100 * np.nanmean(vals < 250))}
    return {
        "glmFiles": len(glm_files), "abiFiles": len(abi_files), "problems": problems,
        "flashes": int(len(allf)), "flashesPerFile": {"min": int(min(per_file)), "median": float(np.median(per_file)), "max": int(max(per_file))},
        "filesWithZeroFlashes": int(sum(n == 0 for n in per_file)),
        "latRange": [float(lat.min()), float(lat.max())], "lonRange": [float(lon.min()), float(lon.max())],
        "timeRangeUtc": [datetime.fromtimestamp(allf[:, 3].min(), timezone.utc).isoformat(), datetime.fromtimestamp(allf[:, 3].max(), timezone.utc).isoformat()],
        "flashEnergyJ": {"median": float(np.median(allf[:, 2])), "max": float(allf[:, 2].max())},
        "qualityFlagCounts": {int(k): int(v) for k, v in Counter(allf[:, 4].astype(int)).items()},
        "abiFrames": abi_meta[:1] + abi_meta[-1:], "abiBtOverall": [min(m["btMinK"] for m in abi_meta), max(m["btMaxK"] for m in abi_meta)],
        "alignmentCheck": cold,
        "coversIndia": bool(((lon > 60) & (lon < 100)).any()),
    }


if __name__ == "__main__":
    report = {"flood": verify_flood(), "lightning": verify_lightning()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    f, l = report["flood"], report["lightning"]
    print(json.dumps({"flood": {k: v for k, v in f.items() if k != "rows"}, "lightning": l}, indent=1))
