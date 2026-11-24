"""Flood-mapping feature: sample catalog, prediction, rendering.

One request = one radar chip (a real Sen1Floods11 chip, a synthetic India
chip, or an uploaded Sentinel-1 GeoTIFF). Returned side by side and
separately labelled:
- the TRAINED U-Net's water-probability map and mask, when the artifact is
  usable ("trained-ml");
- the CLASSICAL backscatter threshold, always ("classical");
- the human label and error map, when the chip has one.

It maps water in one radar image. It is not a flood forecast and is not
connected to the risk engine or alerts.
"""
from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from app.services.imagery import flood_data, hazard_inference, nowcast_client, storage
from app.services.imagery.hazard_common import (
    SOURCE_REAL, SOURCE_SYNTHETIC, SYNTHETIC_NOTICE, confusion_counts, metrics_from_counts, rounded,
)
from app.services.imagery.ml_inference import ModelUnavailableError

ENGINE_ML = "trained-ml"
ENGINE_CLASSICAL = "classical-fallback"
_ML_TINT = (255, 64, 200)        # BGR violet -- trained-model output (same convention as the nowcast panel)
_CLASSICAL_TINT = (230, 200, 0)  # BGR cyan  -- classical output
_LABEL_OUTLINE = (0, 230, 255)   # BGR yellow -- the human label
DEFAULT_VH_THRESHOLD_DB = -20.0

TASK_NOTE = (
    "Maps surface water (flood water and permanent water alike) in one Sentinel-1 radar image. It does not "
    "forecast floods or estimate depth, and it is not connected to the risk engine or alerts."
)
CLASSICAL_NOTE = (
    "Classical method: pixels darker than one VH backscatter threshold are called water (smooth water reflects "
    "the radar beam away). No training beyond choosing the threshold, no confidence estimate."
)
UPLOAD_NOTE = "Uploaded image: there is no label to check the result against."


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _grey(band_db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((np.nan_to_num(band_db, nan=lo) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    image[~np.isfinite(band_db)] = (60, 30, 30)       # no radar data
    return image


def _tint(base: np.ndarray, alpha: np.ndarray, colour) -> np.ndarray:
    alpha = np.clip(alpha, 0, 1)[..., None].astype(np.float32)
    return (base.astype(np.float32) * (1 - alpha) + np.array(colour, np.float32) * alpha).astype(np.uint8)


def _outline(image: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, colour, 1, cv2.LINE_AA)
    return image


def render_error(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Blue = water found, red = false alarm, orange = missed water,
    light grey = correctly dry, dark grey = not labelled."""
    image = np.full((*pred.shape, 3), 225, np.uint8)
    image[pred & truth] = (200, 120, 30)
    image[pred & ~truth] = (50, 50, 220)
    image[~pred & truth] = (0, 150, 255)
    image[~valid] = (90, 90, 90)
    return image


def render_probability(prob: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.clip(prob * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def render_mask_only(mask: np.ndarray, colour) -> np.ndarray:
    """Transparent PNG (BGRA) of a mask, for laying over a map."""
    out = np.zeros((*mask.shape, 4), np.uint8)
    out[mask] = (*colour, 170)
    return out


def _save(image: np.ndarray, out_dir, name: str) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    return storage.runtime_media_url(storage.save_processed_image(buf.tobytes(), out_dir, name))


# --------------------------------------------------------------------------
# catalog / status
# --------------------------------------------------------------------------
def _split_note(source: str, split: str, trained_on: str | None) -> str:
    if source == SOURCE_SYNTHETIC:
        seen = trained_on in ("real+synthetic", "synthetic") and split == "train"
        return (SYNTHETIC_NOTICE + (" This chip was in the model's training mix." if seen
                                    else " The served model did not train on this chip."))
    if split == "train":
        return "This chip was used to TRAIN the model, so its accuracy here is optimistic and proves nothing."
    if split == "valid":
        return "This chip was used to choose the training stopping point and threshold (validation), not to fit weights."
    if split == "test":
        return "Held-out test chip: never seen in training or model selection. This is an honest check."
    return "Not part of any dataset split."


def list_samples() -> dict:
    real = [{
        "sampleId": c.chip_id, "source": SOURCE_REAL, "split": c.split, "event": c.region,
        "country": c.meta.get("country", c.region), "acquired": c.meta.get("s1Date"), "label": True,
        "title": f"{c.meta.get('country', c.region)} {c.chip_id.split('_')[-1]}",
    } for c in flood_data.real_catalog()]
    synthetic = [{
        "sampleId": c.chip_id, "source": SOURCE_SYNTHETIC, "split": c.split, "state": c.meta["state"],
        "region": c.meta["region"], "place": c.meta["place"], "scenario": c.meta["scenario"],
        "month": c.meta["monthName"], "floodLabel": c.meta["floodLabel"], "label": True,
        "title": f"{c.meta['place']} -- {c.meta['scenario']}",
    } for c in flood_data.synthetic_catalog()]
    return {
        "real": real, "synthetic": synthetic,
        "realSource": "Sen1Floods11 (Cloud to Street): Sentinel-1 radar chips with human-labelled water masks -- "
                      "India (Assam, Aug 2016), Pakistan (Jun 2017), Sri Lanka (May 2017)",
        "syntheticSource": "AtmoGuard India flood simulator (ml.hazards.synth_flood). " + SYNTHETIC_NOTICE,
        "realAvailable": bool(real), "syntheticAvailable": bool(synthetic),
    }


def status() -> dict:
    out = nowcast_client.hazard_status("flood")
    out["dataset"] = {"realChips": len(flood_data.real_catalog()), "syntheticChips": len(flood_data.synthetic_catalog())}
    return out


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------
def _area(pixels: int, bounds: dict | None, shape) -> float | None:
    area = flood_data.area_km2(pixels, bounds, shape)
    return round(area, 4) if area is not None else None


def _summarise(mask: np.ndarray, data: np.ndarray, permanent: np.ndarray | None, bounds, shape) -> dict:
    measured = int(data.sum())
    water = int((mask & data).sum())
    out = {"waterPixels": water, "waterAreaPct": round(100.0 * water / max(measured, 1), 2),
           "waterAreaKm2": _area(water, bounds, shape)}
    if permanent is not None:
        new = int((mask & data & ~permanent).sum())
        out.update({"floodPixels": new, "floodAreaPct": round(100.0 * new / max(measured, 1), 2),
                    "floodAreaKm2": _area(new, bounds, shape)})
    return out


def _run(s1: np.ndarray, label: np.ndarray | None, permanent: np.ndarray | None, bounds: dict | None,
         sample: dict, reference_otsu: np.ndarray | None = None) -> dict:
    flood_data.validate_s1(s1)
    data = flood_data.data_mask(s1)
    shape = s1.shape[1:]
    metadata = hazard_inference.load_metadata("flood")

    # --- trained model (optional) ---
    probability, model_info = None, {"available": False}
    try:
        reply = nowcast_client.hazard_predict("flood", s1)
        probability = reply["probability"]
        model_info = {"available": True, **reply["metadata"], "device": reply["device"], "inferenceMs": reply["inference_ms"],
                      "threshold": reply["threshold"]}
    except ModelUnavailableError as exc:
        model_info = {"available": False, "reason": exc.reason}

    # --- classical baseline (always) ---
    tuned = (metadata.get("baselineThresholdsDb") or {}).get("VH")
    threshold_db = float(tuned) if tuned is not None else DEFAULT_VH_THRESHOLD_DB
    classical_mask = flood_data.threshold_water(s1, "VH", threshold_db) & data

    result_id, out_dir = storage.new_processed_dir("flood")
    truth = (label == 1) if label is not None else None
    valid = flood_data.valid_mask(s1, label) if label is not None else None
    vh = _grey(s1[1], -32.0, -7.0)

    def overlay(mask: np.ndarray, alpha: np.ndarray, colour) -> np.ndarray:
        image = _tint(vh, np.where(mask, alpha, 0.0), colour)
        return _outline(image, truth & valid, _LABEL_OUTLINE) if truth is not None else image

    images = {
        "radarVvUrl": _save(_grey(s1[0], -25.0, 0.0), out_dir, "radar_vv"),
        "radarVhUrl": _save(vh, out_dir, "radar_vh"),
    }
    classical = {
        "method": "VH backscatter threshold", "thresholdDb": threshold_db,
        "thresholdSource": "tuned on the real training chips" if tuned is not None else "default (no trained-model metadata found)",
        "overlayUrl": _save(overlay(classical_mask, np.full(shape, 0.55), _CLASSICAL_TINT), out_dir, "classical_overlay"),
        **_summarise(classical_mask, data, permanent, bounds, shape),
    }
    ml = None
    if probability is not None:
        ml_mask = (probability >= model_info["threshold"]) & data
        flagged = probability[ml_mask]
        ml = {
            "overlayUrl": _save(overlay(ml_mask, 0.3 + 0.45 * probability, _ML_TINT), out_dir, "ml_overlay"),
            "probabilityUrl": _save(render_probability(np.where(data, probability, 0.0)), out_dir, "ml_probability"),
            "maskUrl": _save(render_mask_only(ml_mask, _ML_TINT), out_dir, "ml_mask"),
            "threshold": model_info["threshold"],
            **_summarise(ml_mask, data, permanent, bounds, shape),
            # Mean predicted probability over the pixels it calls water.
            "meanWaterConfidencePct": round(float(100 * flagged.mean()), 1) if flagged.size else None,
            # Share of pixels the model is unsure about (probability 0.3-0.7).
            "uncertainAreaPct": round(float(100 * ((probability > 0.3) & (probability < 0.7) & data).mean()), 2),
            "maxProbability": round(float(probability[data].max()), 4) if data.any() else None,
        }

    verification = None
    if truth is not None:
        images["labelUrl"] = _save(_outline(_tint(vh, np.where(truth & valid, 0.5, 0.0), (255, 160, 60)), ~valid, (90, 90, 90)), out_dir, "label")
        verification = {"labelledPct": round(float(100 * valid.mean()), 1),
                        "labelWaterPct": round(float(100 * (truth & valid).sum() / max(valid.sum(), 1)), 2),
                        "classical": rounded(metrics_from_counts(confusion_counts(classical_mask, truth, valid)))}
        classical["errorUrl"] = _save(render_error(classical_mask, truth, valid), out_dir, "classical_error")
        if ml is not None:
            verification["ml"] = rounded(metrics_from_counts(confusion_counts(ml_mask, truth, valid)))
            ml["errorUrl"] = _save(render_error(ml_mask, truth, valid), out_dir, "ml_error")
        if reference_otsu is not None:
            verification["datasetOtsuReference"] = rounded(metrics_from_counts(confusion_counts(reference_otsu, truth, valid)))

    centre = None
    if bounds:
        centre = {"latitude": round((bounds["south"] + bounds["north"]) / 2, 5), "longitude": round((bounds["west"] + bounds["east"]) / 2, 5)}
    result = {
        "resultId": result_id,
        "engine": ENGINE_ML if ml is not None else ENGINE_CLASSICAL,
        "model": model_info, "sample": sample,
        "geo": {"bounds": bounds, "centre": centre, "pixelSizeM": flood_data.PIXEL_SIZE_M,
                "widthPx": int(shape[1]), "heightPx": int(shape[0]),
                "footprintKm2": _area(int(shape[0] * shape[1]), bounds, shape)},
        "radar": {"missingPct": round(float(100 * (1 - data.mean())), 2)},
        "permanentWater": None if permanent is None else {
            "source": sample.get("permanentWaterSource"), "areaKm2": _area(int((permanent & data).sum()), bounds, shape)},
        "images": images, "ml": ml, "classical": classical, "verification": verification,
        "notes": {"task": TASK_NOTE, "classical": CLASSICAL_NOTE,
                  "area": "Flooded area = water found outside the permanent-water layer. Without that layer only total water is reported."},
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    storage.save_result(out_dir, result)
    return result


def predict_sample(sample_id: str) -> dict:
    chip = flood_data.get_chip(sample_id)
    trained_on = hazard_inference.load_metadata("flood").get("trainedOn")
    synthetic = chip.source == SOURCE_SYNTHETIC
    sample = {
        "sampleId": chip.chip_id, "source": chip.source, "split": chip.split,
        "splitNote": _split_note(chip.source, chip.split, trained_on),
        "sourceLabel": ("SYNTHETIC simulation (AtmoGuard India flood simulator)" if synthetic
                        else "REAL Sentinel-1 observation (Sen1Floods11, human-labelled)"),
        "permanentWaterSource": "simulator river/sea/pond layer" if synthetic else "JRC Global Surface Water",
    }
    if synthetic:
        sample.update({k: chip.meta[k] for k in ("state", "region", "place", "scenario", "monthName", "season", "floodKind", "floodLabel")})
        sample["notice"] = SYNTHETIC_NOTICE
    else:
        sample.update({"event": chip.region, "country": chip.meta.get("country"), "acquired": chip.meta.get("s1Date"),
                       "eventNote": chip.meta.get("note")})
    return _run(flood_data.read_s1(chip), flood_data.read_label(chip), flood_data.read_permanent_water(chip),
                flood_data.chip_bounds(chip), sample, flood_data.read_otsu_reference(chip))


def predict_upload(data: bytes, filename: str) -> dict:
    if len(data) > storage.MAX_IMAGE_BYTES:
        raise flood_data.InvalidFloodInputError(f"File is larger than {storage.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    s1, bounds = flood_data.decode_s1_upload(data)
    sample = {"sampleId": None, "source": "upload", "split": None, "splitNote": UPLOAD_NOTE,
              "sourceLabel": f"Uploaded Sentinel-1 GeoTIFF ({filename})", "filename": filename}
    return _run(s1, None, None, bounds, sample)
