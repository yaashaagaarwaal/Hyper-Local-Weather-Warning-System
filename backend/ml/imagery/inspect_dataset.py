"""Dataset inspection for the INSAT-3DR HDF5 files -- run BEFORE choosing a
model, and the reason the model is what it is.

    python -m ml.imagery.inspect_dataset

Writes `ml/artifacts/imagery/dataset_inspection.{json,md}` and quick-look
figures. Opens every file read-only.

(The only Python file shipped next to the data, `dataset/mosdac_api/mdapi.py`,
is ISRO MOSDAC's bulk-download client -- it logs in and downloads, it does
not read or plot HDF5 -- so the visualisation here is this project's own.)
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import timedelta

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.services.imagery import insat  # noqa: E402
from app.services.imagery.nowcast_common import COLD_CLOUD_THRESHOLD_K  # noqa: E402
from ml.imagery import config  # noqa: E402
from ml.imagery.dataset import build_anchors  # noqa: E402

LABEL_HINTS = ("label", "mask", "class", "bbox", "box", "annotation", "target", "truth", "category")


def describe_structure(path) -> dict:
    datasets, label_like = [], []
    with h5py.File(path, "r") as handle:
        root_attrs = {k: (v.decode() if isinstance(v, bytes) else np.asarray(v).tolist()) for k, v in handle.attrs.items()}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append({
                    "name": name, "shape": list(obj.shape), "dtype": str(obj.dtype),
                    "compression": obj.compression,
                    "longName": obj.attrs["long_name"].decode() if "long_name" in obj.attrs else None,
                    "units": obj.attrs["units"].decode() if "units" in obj.attrs else None,
                })
                if any(h in name.lower() for h in LABEL_HINTS):
                    label_like.append(name)

        handle.visititems(visit)
        groups = [k for k in handle if isinstance(handle[k], h5py.Group)]
    return {"rootAttributes": root_attrs, "datasets": datasets, "groups": groups, "labelLikeDatasets": label_like}


def per_file_stats(frame: insat.FrameInfo) -> dict:
    out = {"sizeBytes": frame.path.stat().st_size}
    with h5py.File(frame.path, "r") as handle:
        out["sunElevationDeg"] = float(handle.attrs["Sun_Elevation(Degrees)"][0])
        out["acquisitionStart"] = handle.attrs["Acquisition_Start_Time"].decode()
        out["acquisitionEnd"] = handle.attrs["Acquisition_End_Time"].decode()
        for channel in insat.ALL_CHANNELS:
            data = handle[f"IMG_{channel}"][0]
            fill = int(handle[f"IMG_{channel}"].attrs["_FillValue"][0])
            out[channel] = {
                "countMin": int(data.min()), "countMax": int(data.max()),
                "fillFraction": float((data == fill).mean()),
                "fullyFilledRows": int(((data == fill).mean(axis=1) > 0.5).sum()),
            }
            if channel == "TIR1":
                out["tir1Md5"] = hashlib.md5(data.tobytes()).hexdigest()  # noqa: S324 -- duplicate detection only
                lut = handle["IMG_TIR1_TEMP"][:]
                out["tir1LutMd5"] = hashlib.md5(lut.tobytes()).hexdigest()  # noqa: S324
                bt = lut[data]
                out["tir1"] = {
                    "btMinK": float(bt.min()), "btMaxK": float(bt.max()), "btMeanK": float(bt.mean()),
                    "nonFinite": int((~np.isfinite(bt)).sum()),
                    "coldCloudFraction": float((bt < COLD_CLOUD_THRESHOLD_K).mean()),
                }
    return out


def contiguous_runs(timestamps) -> list[list]:
    runs, current = [], [timestamps[0]]
    for prev, cur in zip(timestamps, timestamps[1:]):
        if cur - prev == insat.FRAME_INTERVAL:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)
    return runs


def quicklook(frame: insat.FrameInfo, out_path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    with h5py.File(frame.path, "r") as handle:
        for ax, channel in zip(axes.ravel(), insat.ALL_CHANNELS):
            data = handle[f"IMG_{channel}"][0]
            if channel in insat.THERMAL_CHANNELS:
                image, cmap, label = handle[f"IMG_{channel}_TEMP"][:][data], "gray_r", "brightness temperature (K)"
            elif channel == "VIS":
                image, cmap, label = handle["IMG_VIS_ALBEDO"][:][data], "gray", "albedo (%)"
            else:
                image, cmap, label = handle["IMG_SWIR_RADIANCE"][:][data], "gray", "radiance"
            shown = ax.imshow(image, cmap=cmap, extent=(44.5, 110.0, 0, 1), aspect="auto")
            ax.set_title(f"{channel} -- {handle[f'IMG_{channel}'].attrs['central_wavelength'][0]:.2f} um")
            ax.set_yticks([])
            ax.set_xlabel("longitude (deg E)")
            fig.colorbar(shown, ax=ax, fraction=0.046, label=label)
    fig.suptitle(f"INSAT-3DR Imager L1C, {frame.timestamp:%d %b %Y %H:%M} UTC -- all six channels (real data, as stored)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=70)
    plt.close(fig)


def timeline_figure(frames, expected, out_path) -> None:
    fig, ax = plt.subplots(figsize=(15, 2.6))
    colours = {"train": "#2563eb", "val": "#d97706", "test": "#16a34a"}
    by_time = {f.timestamp: f for f in frames}
    for when in expected:
        frame = by_time.get(when)
        if frame is None:
            ax.scatter(when, 0, marker="x", color="#9ca3af", s=40)
        elif not frame.valid:
            ax.scatter(when, 0, marker="X", color="#dc2626", s=70)
        else:
            ax.scatter(when, 0, marker="s", color=colours[config.split_of(when)], s=60)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=f"valid ({s})") for s, c in colours.items()]
    handles += [plt.Line2D([], [], marker="X", ls="", color="#dc2626", label="corrupt (truncated file)"),
                plt.Line2D([], [], marker="x", ls="", color="#9ca3af", label="absent / .part")]
    ax.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.55), frameon=False)
    ax.set_yticks([])
    ax.set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frames = insat.scan_catalog(config.INSAT_DIR, use_cache=False)
    valid = [f for f in frames if f.valid]
    corrupt = [f for f in frames if not f.valid]
    part_files = sorted(p.name for p in config.INSAT_DIR.rglob("*.part"))
    other_py = sorted(str(p.relative_to(config.INSAT_DIR.parent)) for p in config.INSAT_DIR.parent.glob("*.py"))

    structure = describe_structure(valid[0].path)
    stats = {f.frame_id: per_file_stats(f) for f in valid}

    timestamps = [f.timestamp for f in valid]
    expected = []
    when = min(f.timestamp for f in frames)
    while when <= max(f.timestamp for f in frames):
        expected.append(when)
        when += timedelta(minutes=30)
    runs = contiguous_runs(timestamps)
    splits = [config.split_of(t) for t in timestamps]
    anchors = build_anchors(timestamps, splits)
    duplicates = [h for h, n in Counter(s["tir1Md5"] for s in stats.values()).items() if n > 1]
    cold = np.array([s["tir1"]["coldCloudFraction"] for s in stats.values()])

    report = {
        "product": "INSAT-3DR Imager L1C, ASIA Mercator sector (3RIMG_L1C_ASIA_MER), ISRO MOSDAC",
        "directory": str(config.INSAT_DIR.relative_to(config.BACKEND_DIR.parent)),
        "companionPythonFiles": other_py,
        "companionScriptNote": "mdapi.py is MOSDAC's download client (needs account credentials in a config.json that "
                               "is not present); it contains no HDF5 reading or plotting code and was not run.",
        "files": {
            "h5Total": len(frames), "valid": len(valid), "corrupt": len(corrupt), "partialDownloads": len(part_files),
            "corruptFiles": [{"file": f.path.name, "sizeBytes": f.path.stat().st_size, "error": f.error} for f in corrupt],
            "partFiles": part_files,
        },
        "structure": structure,
        "grid": {"rows": insat.EXPECTED_SHAPE[0], "cols": insat.EXPECTED_SHAPE[1], "resolutionKm": 4,
                 "projection": "Mercator (std parallel 17.75 N, origin 77.25 E, WGS84)", "bounds": insat.SECTOR_BOUNDS},
        "channels": list(insat.ALL_CHANNELS),
        "labels": {
            "present": False, "labelLikeDatasets": structure["labelLikeDatasets"],
            "conclusion": "No labels, masks, classes or boxes exist. Supervised classification, segmentation and "
                          "detection are not possible. Class distribution is therefore undefined.",
        },
        "time": {
            "first": timestamps[0].isoformat(), "last": timestamps[-1].isoformat(), "cadenceMinutes": 30,
            "expectedSlots": len(expected), "validSlots": len(valid),
            "missingOrCorruptSlots": [t.isoformat() for t in expected if t not in set(timestamps)],
            "contiguousRuns": [{"start": r[0].isoformat(), "end": r[-1].isoformat(), "frames": len(r)} for r in runs],
        },
        "quality": {
            "duplicateTir1Frames": len(duplicates),
            "fillPixelsTir1": sum(s["TIR1"]["fillFraction"] > 0 for s in stats.values()),
            "fillPixelsWv": sum(s["WV"]["fillFraction"] > 0 for s in stats.values()),
            "nonFiniteTir1Pixels": int(sum(s["tir1"]["nonFinite"] for s in stats.values())),
            "framesWithDroppedScanLines": sum(s["TIR1"]["fullyFilledRows"] > 0 for s in stats.values()),
            "distinctTir1CalibrationTables": len({s["tir1LutMd5"] for s in stats.values()}),
        },
        "derivedColdCloud": {
            "thresholdK": COLD_CLOUD_THRESHOLD_K,
            "meanFraction": float(cold.mean()), "minFraction": float(cold.min()), "maxFraction": float(cold.max()),
            "note": "Not a label in the data -- a physical threshold on TIR1, used only to score predictions as a mask. "
                    "Strongly imbalanced: about 1 pixel in 20.",
        },
        "taskSupport": {
            "classification": False, "segmentation": False, "detection": False,
            "sequentialPrediction": True,
            "reason": "Half-hourly frames of the same scene with exact timestamps: the later frame is ground truth "
                      "for a prediction made from earlier frames.",
        },
        "samples": {
            split: {
                "frames": splits.count(split),
                "sequences": sum(a.split == split for a in anchors),
                "targetsPerLead": [sum(a.split == split and a.targets[k] >= 0 for a in anchors) for k in range(3)],
            } for split in ("train", "val", "test")
        },
        "perFrame": stats,
    }
    (config.ARTIFACTS_DIR / "dataset_inspection.json").write_text(json.dumps(report, indent=2))

    daytime = max(valid, key=lambda f: stats[f.frame_id]["sunElevationDeg"])
    quicklook(daytime, config.FIGURES_DIR / "quicklook_all_channels.png")
    timeline_figure(frames, expected, config.FIGURES_DIR / "frame_timeline.png")

    lines = [
        "# INSAT-3DR HDF5 dataset -- inspection report",
        "",
        "Generated by `python -m ml.imagery.inspect_dataset`. Every number below was read from the files.",
        "",
        f"- **Product**: {report['product']}",
        f"- **Location**: `{report['directory']}`",
        f"- **Files**: {len(frames)} `.h5` -- **{len(valid)} readable, {len(corrupt)} corrupt** (truncated downloads), "
        f"plus {len(part_files)} `.part` stubs (incomplete downloads, ignored).",
        f"- **Grid**: {insat.EXPECTED_SHAPE[0]} x {insat.EXPECTED_SHAPE[1]} pixels at 4 km, Mercator, "
        "44.5-110 E, 10 S-45.5 N.",
        "- **Channels** (uint16, 10-bit counts): VIS 0.65 um, SWIR 1.63 um, MIR 3.9 um, WV 6.9 um, TIR1 10.8 um, TIR2 12.0 um. "
        "Each thermal channel has its own count->kelvin lookup table **per file** "
        f"({report['quality']['distinctTir1CalibrationTables']} distinct TIR1 tables across {len(valid)} files).",
        f"- **Time span**: {timestamps[0]:%d %b %Y %H:%M} -> {timestamps[-1]:%d %b %Y %H:%M} UTC, every 30 min; "
        f"{len(valid)} of {len(expected)} slots usable.",
        "- **Labels / masks / classes / boxes**: **none.** Only imagery, calibration tables, viewing geometry and coordinates.",
        "",
        "## What the data supports",
        "",
        "| Task | Supported? | Why |",
        "|---|---|---|",
        "| Image classification (CNN/ResNet/EfficientNet) | No | no class labels |",
        "| Segmentation (U-Net) | No | no masks |",
        "| Object detection (YOLO) | No | no bounding boxes |",
        "| **Sequential prediction (ConvLSTM)** | **Yes** | genuine half-hourly sequence; the next frame is the ground truth |",
        "",
        "## Integrity",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Corrupt files | {len(corrupt)}: " + ", ".join(f"{f.timestamp:%d %b %H:%M}" for f in corrupt) + " (all truncated mid-download) |",
        f"| Duplicate frames (TIR1 md5) | {len(duplicates)} |",
        f"| Fill / missing pixels in TIR1, WV | {report['quality']['fillPixelsTir1']} / {report['quality']['fillPixelsWv']} frames affected |",
        f"| Dropped scan lines | {report['quality']['framesWithDroppedScanLines']} frames |",
        f"| Non-finite TIR1 temperatures | {report['quality']['nonFiniteTir1Pixels']} pixels |",
        f"| Time gaps | {len(runs)} unbroken runs: " + ", ".join(str(len(r)) for r in runs) + " frames |",
        "",
        "## Imbalance",
        "",
        f"Cold cloud tops (TIR1 < {COLD_CLOUD_THRESHOLD_K:.0f} K) cover {100 * cold.mean():.1f} % of pixels on average "
        f"({100 * cold.min():.1f}-{100 * cold.max():.1f} %). Most of the scene barely changes in 30 minutes.",
        "",
        "## Usable samples (3 consecutive input frames -> +30/+60/+90 min targets, chronological split)",
        "",
        "| Split | Frames | Sequences | Targets at +30 / +60 / +90 min |",
        "|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        s = report["samples"][split]
        lines.append(f"| {split} | {s['frames']} | {s['sequences']} | {' / '.join(map(str, s['targetsPerLead']))} |")
    lines += [
        "",
        "Figures: `figures/quicklook_all_channels.png`, `figures/frame_timeline.png`.",
        "",
    ]
    (config.ARTIFACTS_DIR / "dataset_inspection.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
