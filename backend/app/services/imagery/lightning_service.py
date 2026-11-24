"""Lightning-detection feature: sample catalog, prediction, rendering.

Three kinds of input, each labelled for what it is:
- REAL GOES-16 infrared over the AMERICAS, checked against real GLM flashes;
- SYNTHETIC India storm scenes, checked against SIMULATED flashes;
- REAL INSAT-3DR infrared over INDIA, with NO lightning ground truth at all
  (the model's output there is unverified and shown as experimental).

Returned side by side: the trained U-Net's flash-probability map
("trained-ml") and the classical cold-cloud threshold ("classical").

It infers lightning from cloud-top temperature. It does not detect flashes,
it is not validated for India, and it is not connected to the risk engine
or alerts.
"""
from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from app.services.imagery import hazard_inference, insat, nowcast_client, storage
from app.services.imagery import lightning_data as ld
from app.services.imagery.hazard_common import (
    SOURCE_REAL, SOURCE_SYNTHETIC, SYNTHETIC_NOTICE, confusion_counts, metrics_from_counts, rounded,
)
from app.services.imagery.ml_inference import ModelUnavailableError

ENGINE_ML = "trained-ml"
ENGINE_CLASSICAL = "classical-fallback"
SOURCE_INSAT = "real-insat"
INSAT_PREFIX = "INSAT:"
_ML_TINT = (255, 64, 200)        # BGR violet
_CLASSICAL_TINT = (230, 200, 0)  # BGR cyan
_TRUTH_COLOUR = (0, 230, 255)    # BGR yellow
DEFAULT_THRESHOLD_K = 215.0
MAX_LOCATIONS = 150
_IR_RANGE_K = (190.0, 310.0)

TASK_NOTE = (
    "Estimates where lightning is likely from two infrared cloud-top images 30 minutes apart. It does not detect "
    "flashes (no lightning sensor is involved), it is not validated for India, and it is not connected to the "
    "risk engine or alerts."
)
CLASSICAL_NOTE = (
    "Classical method: lightning is assumed wherever the cloud top is colder than one temperature threshold "
    "(deep convection). No training beyond choosing the threshold, no probability."
)
INSAT_NOTICE = (
    "REAL INSAT-3DR infrared over India, but there is NO lightning ground truth for it. The model was trained on "
    "the Americas and has never been checked over India; treat this output as an unverified experiment. Cold "
    "ground (snow, high winter terrain) can be mistaken for storm cloud."
)
GOES_NOTICE = "REAL GOES-16 observation over the Americas (not India), checked against real GLM lightning flashes."


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def render_ir(bt: np.ndarray) -> np.ndarray:
    lo, hi = _IR_RANGE_K
    grey = 255 - np.clip((np.nan_to_num(bt, nan=hi) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def _upsample(cells: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(cells.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _tint(base: np.ndarray, alpha: np.ndarray, colour) -> np.ndarray:
    alpha = np.clip(alpha, 0, 1)[..., None].astype(np.float32)
    return (base.astype(np.float32) * (1 - alpha) + np.array(colour, np.float32) * alpha).astype(np.uint8)


def _outline(image: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, colour, 1, cv2.LINE_AA)
    return image


def render_overlay(bt: np.ndarray, strength: np.ndarray, flagged: np.ndarray, colour, truth: np.ndarray | None) -> np.ndarray:
    shape = bt.shape
    alpha = np.where(_upsample(flagged, shape) > 0, 0.3 + 0.5 * _upsample(strength, shape), 0.0)
    image = _tint(render_ir(bt), alpha, colour)
    if truth is not None:
        image = _outline(image, _upsample(truth, shape) > 0, _TRUTH_COLOUR)
    return image


def render_probability(prob: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.applyColorMap(np.clip(_upsample(prob, shape) * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def render_transparent(prob: np.ndarray, flagged: np.ndarray, colour, scale: int = 8) -> np.ndarray:
    """BGRA probability overlay for a map (transparent where not flagged)."""
    out = np.zeros((*prob.shape, 4), np.uint8)
    out[flagged] = (*colour, 0)
    out[..., 3] = np.where(flagged, 90 + 150 * prob, 0).astype(np.uint8)
    return cv2.resize(out, (prob.shape[1] * scale, prob.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def _fit(image: np.ndarray, max_dim: int = 1250) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_dim / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def _save(image: np.ndarray, out_dir, name: str) -> str:
    ok, buf = cv2.imencode(".png", _fit(image))
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    return storage.runtime_media_url(storage.save_processed_image(buf.tobytes(), out_dir, name))


# --------------------------------------------------------------------------
# catalog / status
# --------------------------------------------------------------------------
def _insat_anchors() -> list:
    try:
        by_time = insat.valid_frames_by_time()
    except Exception:  # noqa: BLE001 -- INSAT files are optional
        return []
    return [frame for time, frame in sorted(by_time.items()) if (time - insat.FRAME_INTERVAL) in by_time]


def list_samples() -> dict:
    frames = ld.abi_frames()
    real = [{
        "sampleId": f.frame_id, "source": SOURCE_REAL, "split": ld.split_of_anchor(f.index),
        "timestamp": f.time.isoformat(), "title": f"GOES-16 {f.time:%d %b %Y %H:%M} UTC", "label": True,
    } for f in frames if f.index >= ld.LOOKBACK_FRAMES]
    synthetic = [{
        "sampleId": s.sample_id, "source": SOURCE_SYNTHETIC, "split": s.split, "state": s.meta["state"],
        "region": s.meta["region"], "place": s.meta["place"], "scenario": s.meta["scenario"], "month": s.meta["monthName"],
        "lightningLabel": s.meta["lightningLabel"], "title": f"{s.meta['place']} -- {s.meta['scenario']}", "label": True,
    } for s in ld.synthetic_catalog()]
    india = [{
        "sampleId": f"{INSAT_PREFIX}{f.frame_id}", "source": SOURCE_INSAT, "split": None,
        "timestamp": f.timestamp.isoformat(), "title": f"INSAT-3DR {f.timestamp:%d %b %Y %H:%M} UTC", "label": False,
    } for f in _insat_anchors()]
    return {
        "real": real, "synthetic": synthetic, "insat": india,
        "realSource": "NOAA GOES-16: ABI band 13 infrared + GLM lightning flashes, CONUS sector, 2 Aug 2024 -- the Americas, not India",
        "syntheticSource": "AtmoGuard India thunderstorm simulator (ml.hazards.synth_lightning). " + SYNTHETIC_NOTICE,
        "insatSource": "ISRO MOSDAC INSAT-3DR TIR1 infrared (Jan 2025). " + INSAT_NOTICE,
        "insatRegions": [{"key": key, "label": spec["label"]} for key, spec in insat.REGIONS.items() if key != "full_sector"],
    }


def status() -> dict:
    out = nowcast_client.hazard_status("lightning")
    out["dataset"] = {"realGoesFrames": len(ld.abi_frames()), "syntheticScenes": len(ld.synthetic_catalog()),
                      "insatFramesIndia": len(_insat_anchors())}
    out["validatedForIndia"] = False
    return out


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------
def _split_note(source: str, split: str | None, trained_on: str | None) -> str:
    if source == SOURCE_INSAT:
        return INSAT_NOTICE
    if source == SOURCE_SYNTHETIC:
        seen = trained_on in ("real+synthetic", "synthetic") and split == "train"
        return SYNTHETIC_NOTICE + (" This scene was in the model's training mix." if seen else " The served model did not train on this scene.")
    if split == "train":
        return "This time was used to TRAIN the model (in 60 % of the area), so scores here are optimistic."
    if split == "val":
        return "This time was used for model selection (validation) in 20 % of the area."
    if split == "test":
        return ("Held-out test time. Only the held-out test blocks (20 % of the area) are an honest check; "
                "the rest of the image lies in blocks the model trained on at earlier times.")
    return "This time sits in the gap left between splits; it was not used for training or scoring."


def _locations(prob: np.ndarray, flagged: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> list[dict]:
    rows, cols = np.nonzero(flagged)
    order = np.argsort(prob[rows, cols])[::-1][:MAX_LOCATIONS]
    return [{"latitude": round(float(lat[rows[i], cols[i]]), 3), "longitude": round(float(lon[rows[i], cols[i]]), 3),
             "probability": round(float(prob[rows[i], cols[i]]), 3)} for i in order
            if np.isfinite(lat[rows[i], cols[i]]) and np.isfinite(lon[rows[i], cols[i]])]


def _regular_latlon(bounds: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    lat = bounds["north"] - (np.arange(rows) + 0.5) / rows * (bounds["north"] - bounds["south"])
    lon = bounds["west"] + (np.arange(cols) + 0.5) / cols * (bounds["east"] - bounds["west"])
    return np.repeat(lat[:, None], cols, axis=1), np.repeat(lon[None, :], rows, axis=0)


def _run(pair: np.ndarray, truth: np.ndarray | None, lat: np.ndarray, lon: np.ndarray, sample: dict,
         score_masks: dict[str, np.ndarray] | None = None, map_bounds: dict | None = None, truth_points: list | None = None) -> dict:
    ld.validate_pair(pair)
    # Whole 8 km cells only: drop a trailing odd row/column of 4 km pixels.
    pair = pair[:, : pair.shape[1] // 2 * 2, : pair.shape[2] // 2 * 2]
    valid = ld.cell_valid(pair)
    metadata = hazard_inference.load_metadata("lightning")

    probability, model_info = None, {"available": False}
    try:
        reply = nowcast_client.hazard_predict("lightning", pair)
        probability = reply["probability"]
        model_info = {"available": True, **reply["metadata"], "device": reply["device"], "inferenceMs": reply["inference_ms"],
                      "threshold": reply["threshold"]}
    except ModelUnavailableError as exc:
        model_info = {"available": False, "reason": exc.reason}

    tuned = metadata.get("baselineThresholdK")
    threshold_k = float(tuned) if tuned is not None else DEFAULT_THRESHOLD_K
    coldest = ld.cell_min_bt(pair[1])
    classical_mask = (coldest < threshold_k) & valid
    cells = int(valid.sum())
    cell_area = ld.CELL_KM ** 2

    result_id, out_dir = storage.new_processed_dir("lightning")
    bt_now = pair[1]
    images = {
        "infraredUrl": _save(render_ir(bt_now), out_dir, "infrared_now"),
        "infraredEarlierUrl": _save(render_ir(pair[0]), out_dir, "infrared_earlier"),
    }
    classical = {
        "method": "cold cloud-top threshold", "thresholdK": threshold_k,
        "thresholdSource": "tuned on the real GOES training cells" if tuned is not None else "default (no trained-model metadata found)",
        "overlayUrl": _save(render_overlay(bt_now, classical_mask.astype(np.float32), classical_mask, _CLASSICAL_TINT, truth), out_dir, "classical_overlay"),
        "flaggedCells": int(classical_mask.sum()), "flaggedAreaKm2": int(classical_mask.sum() * cell_area),
        "flaggedAreaPct": round(100.0 * classical_mask.sum() / max(cells, 1), 3),
    }
    ml = None
    if probability is not None:
        flagged = (probability >= model_info["threshold"]) & valid
        ml = {
            "overlayUrl": _save(render_overlay(bt_now, probability, flagged, _ML_TINT, truth), out_dir, "ml_overlay"),
            "probabilityUrl": _save(render_probability(np.where(valid, probability, 0.0), bt_now.shape), out_dir, "ml_probability"),
            "threshold": round(float(model_info["threshold"]), 4),
            "flaggedCells": int(flagged.sum()), "flaggedAreaKm2": int(flagged.sum() * cell_area),
            "flaggedAreaPct": round(100.0 * flagged.sum() / max(cells, 1), 3),
            "maxProbability": round(float(probability[valid].max()), 4) if cells else None,
            "meanFlaggedConfidencePct": round(float(100 * probability[flagged].mean()), 1) if flagged.any() else None,
            "lightningDetected": bool(flagged.any()),
            "locations": _locations(probability, flagged, lat, lon),
            "locationsNote": f"Centres of the up-to-{MAX_LOCATIONS} highest-probability 8 km cells -- areas where lightning is "
                             "likely, not positions of individual flashes.",
        }
        if map_bounds is not None:
            ml["mapOverlayUrl"] = _save(render_transparent(probability, flagged, _ML_TINT), out_dir, "ml_map_overlay")

    verification = None
    if truth is not None:
        truth = truth & valid
        images["truthUrl"] = _save(render_overlay(bt_now, truth.astype(np.float32), truth, _TRUTH_COLOUR, None), out_dir, "truth")
        verification = {"truthCells": int(truth.sum()), "scopes": {}}
        for name, member in (score_masks or {"allCells": valid}).items():
            member = member & valid
            scope = {"cells": int(member.sum()), "truthCells": int((truth & member).sum()),
                     "classical": rounded(metrics_from_counts(confusion_counts(classical_mask, truth, member)))}
            if ml is not None:
                scope["ml"] = rounded(metrics_from_counts(confusion_counts(flagged, truth, member)))
            verification["scopes"][name] = scope

    finite = np.isfinite(lat) & np.isfinite(lon)
    result = {
        "resultId": result_id, "engine": ENGINE_ML if ml is not None else ENGINE_CLASSICAL,
        "model": model_info, "sample": sample, "validatedForIndia": False,
        "geo": {"bounds": map_bounds or {"south": round(float(lat[finite].min()), 3), "north": round(float(lat[finite].max()), 3),
                                         "west": round(float(lon[finite].min()), 3), "east": round(float(lon[finite].max()), 3)},
                "mapOverlaySupported": map_bounds is not None,
                "imageGridKm": ld.INPUT_SPACING_KM, "cellKm": ld.CELL_KM,
                "widthCells": int(valid.shape[1]), "heightCells": int(valid.shape[0])},
        "cloud": {"minBtK": round(float(np.nanmin(bt_now)), 1),
                  "coldCloudPct": round(float(100 * (np.nan_to_num(bt_now, nan=999.0) < 235.0).mean()), 2)},
        "images": images, "ml": ml, "classical": classical, "verification": verification,
        "truthFlashes": None if truth_points is None else {
            "count": len(truth_points),
            "points": [{"latitude": round(float(a), 3), "longitude": round(float(b), 3)} for a, b in truth_points[:MAX_LOCATIONS * 4]]},
        "notes": {"task": TASK_NOTE, "classical": CLASSICAL_NOTE},
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    storage.save_result(out_dir, result)
    return result


def predict_sample(sample_id: str, region: str = "north_india") -> dict:
    trained_on = hazard_inference.load_metadata("lightning").get("trainedOn")

    # --- real INSAT-3DR over India: no ground truth ---
    if sample_id.startswith(INSAT_PREFIX):
        anchor = insat.get_frame(sample_id[len(INSAT_PREFIX):])
        earlier, now = insat.input_sequence(anchor, 2)
        window = insat.region_window(region)
        pair = np.stack([insat.read_bt(earlier.path, "TIR1", window), insat.read_bt(now.path, "TIR1", window)]).astype(np.float32)
        rows, cols = pair.shape[1] // 2, pair.shape[2] // 2
        lat = np.empty((rows, cols))
        lon = np.empty((rows, cols))
        for r in range(rows):
            for c in (0, cols - 1):
                lat[r, c], lon[r, c] = insat.pixel_to_latlon(window.row0 + 2 * r + 0.5, window.col0 + 2 * c + 0.5)
            lat[r, :] = lat[r, 0]                                  # Mercator: latitude depends on the row only,
            lon[r, :] = np.linspace(lon[r, 0], lon[r, -1], cols)   # longitude is linear in the column
        sample = {"sampleId": sample_id, "source": SOURCE_INSAT, "split": None, "splitNote": INSAT_NOTICE, "notice": INSAT_NOTICE,
                  "sourceLabel": "REAL INSAT-3DR infrared over India -- NO lightning ground truth",
                  "timestamp": anchor.timestamp.isoformat(), "region": region, "regionLabel": insat.REGIONS[region]["label"],
                  "experimental": True}
        # INSAT's grid is Mercator, like a web map, so the overlay can be laid straight onto one.
        return _run(pair, None, lat, lon, sample, map_bounds=window.bounds())

    # --- synthetic India scene ---
    for item in ld.synthetic_catalog():
        if item.sample_id == sample_id:
            arrays = ld.read_synthetic(item)
            pair = arrays["bt"][[-2, -1]]
            truth = arrays["flash_count"][-1] > 0
            bounds = item.meta["bounds"]
            lat, lon = _regular_latlon(bounds, truth.shape)
            last = arrays["flashes"][arrays["flashes"][:, 0] == arrays["bt"].shape[0] - 1]
            sample = {"sampleId": sample_id, "source": SOURCE_SYNTHETIC, "split": item.split,
                      "splitNote": _split_note(SOURCE_SYNTHETIC, item.split, trained_on), "notice": SYNTHETIC_NOTICE,
                      "sourceLabel": "SYNTHETIC simulation (AtmoGuard India thunderstorm simulator) -- simulated infrared, simulated lightning",
                      **{k: item.meta[k] for k in ("state", "region", "place", "scenario", "monthName", "season", "stormKind",
                                                   "lightningLabel", "foundation", "usesRealInsatBackground")}}
            return _run(pair, truth, lat, lon, sample, map_bounds=bounds, truth_points=[(row[1], row[2]) for row in last])

    # --- real GOES-16 over the Americas ---
    frames = {f.frame_id: f for f in ld.abi_frames()}
    frame = frames.get(sample_id)
    if frame is None:
        raise ld.LightningDataError(f"Unknown lightning sample {sample_id!r}.")
    if frame.index < ld.LOOKBACK_FRAMES:
        raise ld.LightningDataError(f"{sample_id} has no image {ld.LOOKBACK_MINUTES} minutes earlier to pair it with.", 422)
    now = ld.read_abi(frame.path)
    earlier = ld.read_abi(ld.abi_frames()[frame.index - ld.LOOKBACK_FRAMES].path)
    pair = np.stack([ld.to_input_grid(earlier["bt"]), ld.to_input_grid(now["bt"])])
    grid, points = ld.flash_grid_for(now)
    lat, lon = ld.cell_latlon(now)
    split = ld.split_of_anchor(frame.index)
    blocks = ld.block_split_map(grid.shape)
    scopes = {"allCells": np.ones(grid.shape, bool)}
    if split == "test":
        scopes = {"heldOutTestBlocks": blocks == 2, "allCells": np.ones(grid.shape, bool)}
    sample = {"sampleId": sample_id, "source": SOURCE_REAL, "split": split, "splitNote": _split_note(SOURCE_REAL, split, trained_on),
              "notice": GOES_NOTICE, "sourceLabel": "REAL GOES-16 infrared + GLM lightning -- the Americas, not India",
              "timestamp": frame.time.isoformat()}
    return _run(pair, grid > 0, lat, lon, sample, score_masks=scopes, truth_points=points)
