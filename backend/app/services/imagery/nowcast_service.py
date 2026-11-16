"""Orchestration + rendering for the INSAT-3DR nowcast feature.

One request = one anchor time + one region. Produces, side by side and
separately labelled:
- the TRAINED MODEL's prediction (with per-pixel uncertainty and cold-cloud
  probability), when a usable model artifact exists;
- the CLASSICAL OpenCV optical-flow extrapolation, always;
- the ACTUAL later satellite frame and error maps, whenever that later frame
  really exists in the dataset -- so every prediction shown can be checked.

Nothing here is a weather forecast: it extrapolates infrared cloud-top
imagery up to 90 minutes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from app.services.imagery import insat, ml_inference, nowcast_client, storage
from app.services.imagery.nowcast_common import (
    COLD_CLOUD_THRESHOLD_K,
    INPUT_CHANNELS,
    LEAD_MINUTES,
    LEAD_STEPS,
    N_INPUT_FRAMES,
    cold_cloud_metrics,
    laplace_cdf,
    optical_flow_forecast,
    regression_metrics,
)

CONTEXT_HALO_PX = 64      # extra real data read around a region so its edges see inflow
MAX_RENDER_DIM = 1100
_IR_RANGE_K = (190.0, 310.0)
_ERROR_RANGE_K = 15.0
_UNCERTAINTY_RANGE_K = 8.0
_ML_TINT = (255, 64, 200)       # BGR violet -- trained-model output
_CLASSICAL_TINT = (230, 200, 0)  # BGR cyan  -- classical output
_TRUTH_OUTLINE = (0, 230, 255)  # BGR yellow -- what the satellite actually saw

ENGINE_ML = "trained-ml"
ENGINE_CLASSICAL = "classical-fallback"

TASK_NOTE = (
    "Short-range extrapolation of INSAT-3DR infrared (10.8 um) cloud-top brightness temperature. "
    "It predicts what the satellite image will look like, not rainfall, storms or any weather hazard, "
    "and is not connected to the risk engine or alerts."
)
CLASSICAL_NOTE = (
    "Classical OpenCV method: Farneback optical flow between the last two frames, then the last frame "
    "is moved further along that motion. No training, no uncertainty estimate."
)
COLD_CLOUD_NOTE = (
    f"Cold-cloud mask = brightness temperature below {COLD_CLOUD_THRESHOLD_K:.0f} K (the classic GOES "
    "Precipitation Index cloud-top threshold). A cloud-top temperature criterion, not a rain or storm detection."
)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _fit(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = MAX_RENDER_DIM / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def _delhi_box(image: np.ndarray, window: insat.PixelWindow) -> np.ndarray:
    south, west, north, east = insat.DELHI_NCR_BOUNDS
    r0, c0 = insat.latlon_to_pixel(north, west)
    r1, c1 = insat.latlon_to_pixel(south, east)
    p0 = (int(round(c0 - window.col0)), int(round(r0 - window.row0)))
    p1 = (int(round(c1 - window.col0)), int(round(r1 - window.row0)))
    height, width = image.shape[:2]
    if 0 <= p0[0] < width and 0 <= p1[0] < width and 0 <= p0[1] < height and 0 <= p1[1] < height:
        cv2.rectangle(image, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_ir(bt: np.ndarray) -> np.ndarray:
    """Conventional infrared rendering: cold cloud tops bright, warm surface dark."""
    lo, hi = _IR_RANGE_K
    grey = 255 - np.clip((np.nan_to_num(bt, nan=hi) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def _tint(base: np.ndarray, alpha: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    alpha = np.clip(alpha, 0, 1)[..., None].astype(np.float32)
    return (base.astype(np.float32) * (1 - alpha) + np.array(colour, dtype=np.float32) * alpha).astype(np.uint8)


def _outline(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, colour, 1, cv2.LINE_AA)
    return image


def render_cold_overlay(bt: np.ndarray, strength: np.ndarray, colour, truth_mask: np.ndarray | None) -> np.ndarray:
    """`strength` in [0, 1]: a probability for the model, a 0/1 mask for the
    classical method. Shaded where >= 0.5; the actual cold cloud, if known,
    is outlined on top."""
    image = _tint(render_ir(bt), np.where(strength >= 0.5, 0.25 + 0.5 * strength, 0.0), colour)
    if truth_mask is not None:
        image = _outline(image, truth_mask, _TRUTH_OUTLINE)
    return image


def render_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Blue = predicted too cold, red = predicted too warm, white = correct."""
    err = np.clip(np.nan_to_num(pred - truth) / _ERROR_RANGE_K, -1, 1).astype(np.float32)
    white = np.full((*err.shape, 3), 255.0, dtype=np.float32)
    warm = np.array([40, 40, 220], dtype=np.float32)   # BGR red
    cold = np.array([220, 90, 30], dtype=np.float32)   # BGR blue
    mag = np.abs(err)[..., None]
    return (white * (1 - mag) + np.where(err[..., None] > 0, warm, cold) * mag).astype(np.uint8)


def render_uncertainty(scale_k: np.ndarray) -> np.ndarray:
    level = np.clip(scale_k / _UNCERTAINTY_RANGE_K * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(level, cv2.COLORMAP_INFERNO)


def _save(image: np.ndarray, out_dir, name: str, window: insat.PixelWindow) -> str:
    ok, buf = cv2.imencode(".png", _fit(_delhi_box(image, window)))
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    return storage.runtime_media_url(storage.save_processed_image(buf.tobytes(), out_dir, name))


# --------------------------------------------------------------------------
# Catalog / status helpers
# --------------------------------------------------------------------------
def _frame_splits() -> dict[str, str]:
    return ml_inference.load_metadata().get("frameSplits", {})


def _split_summary(frame_ids: list[str], splits: dict[str, str]) -> tuple[str, str]:
    kinds = {splits.get(fid, "unseen") for fid in frame_ids}
    if "train" in kinds:
        return "train", ("These frames were used to TRAIN the model, so its accuracy here is optimistic and is not "
                         "evidence that it works.")
    if "val" in kinds:
        return "val", "These frames were used to choose the training stopping point (validation), not to fit the weights."
    if kinds == {"test"}:
        return "test", "Held-out test frames: never seen in training or model selection. This is an honest check."
    return "unseen", "Frames outside the original dataset split: never seen by the model."


def list_frames() -> dict:
    frames = insat.scan_catalog()
    if not frames:
        return {
            "available": False,
            "reason": f"No INSAT-3DR L1C files found under {insat.insat_dir().name}/.",
            "frames": [], "regions": [],
        }
    by_time = {f.timestamp: f for f in frames if f.valid}
    splits = _frame_splits()
    out = []
    for frame in frames:
        can_anchor = frame.valid and all(
            (frame.timestamp - back * insat.FRAME_INTERVAL) in by_time for back in range(N_INPUT_FRAMES)
        )
        verifiable = [
            minutes for step, minutes in zip(LEAD_STEPS, LEAD_MINUTES)
            if (frame.timestamp + step * insat.FRAME_INTERVAL) in by_time
        ]
        involved = [insat.frame_id_for(frame.timestamp - back * insat.FRAME_INTERVAL) for back in range(N_INPUT_FRAMES)]
        involved += [insat.frame_id_for(frame.timestamp + m // 30 * insat.FRAME_INTERVAL) for m in verifiable]
        out.append({
            "frameId": frame.frame_id,
            "timestamp": frame.timestamp.isoformat(),
            "valid": frame.valid,
            "error": frame.error,
            "canAnchorNowcast": can_anchor,
            "verifiableLeadMinutes": verifiable if can_anchor else [],
            "split": _split_summary(involved, splits)[0] if can_anchor else splits.get(frame.frame_id, "unseen"),
        })
    return {
        "available": True,
        "source": "ISRO MOSDAC -- INSAT-3DR Imager L1C, ASIA Mercator sector, 4 km, half-hourly",
        "frames": out,
        "validCount": sum(f.valid for f in frames),
        "corruptCount": sum(not f.valid for f in frames),
        "regions": [
            {"key": key, "label": spec["label"], "bounds": insat.region_window(key).bounds(),
             "widthPx": insat.region_window(key).shape[1], "heightPx": insat.region_window(key).shape[0]}
            for key, spec in insat.REGIONS.items()
        ],
    }


def frame_preview(frame_id: str, region: str, channel: str) -> dict:
    frame = insat.get_frame(frame_id)
    if not frame.valid:
        raise insat.InsatFrameError(f"Frame {frame_id} is corrupt: {frame.error}")
    if channel not in insat.THERMAL_CHANNELS:
        raise insat.InsatFrameError(f"Preview supports thermal channels {insat.THERMAL_CHANNELS}, got {channel!r}.")
    window = insat.region_window(region)
    bt = insat.read_bt(frame.path, channel, window)
    out_dir = storage.PROCESSED_DIR / "insat_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    if channel == "WV":  # water vapour is much colder overall; stretch to its own range
        lo, hi = 200.0, 265.0
        grey = 255 - np.clip((bt - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    else:
        image = render_ir(bt)
    url = _save(image, out_dir, f"{frame_id}_{region}_{channel}", window)
    return {
        "frameId": frame_id, "timestamp": frame.timestamp.isoformat(), "region": region, "channel": channel,
        "url": url, "minK": round(float(np.nanmin(bt)), 1), "maxK": round(float(np.nanmax(bt)), 1),
        "meanK": round(float(np.nanmean(bt)), 1),
        "coldCloudPct": round(float(100 * (bt < COLD_CLOUD_THRESHOLD_K).mean()), 2) if channel == "TIR1" else None,
        "bounds": window.bounds(),
    }


# --------------------------------------------------------------------------
# The nowcast itself
# --------------------------------------------------------------------------
def _expand(window: insat.PixelWindow, halo: int) -> insat.PixelWindow:
    rows, cols = insat.EXPECTED_SHAPE
    return insat.PixelWindow(
        max(0, window.row0 - halo), min(rows, window.row1 + halo),
        max(0, window.col0 - halo), min(cols, window.col1 + halo),
    )


def _round(metrics: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def _verify(pred: np.ndarray, truth: np.ndarray) -> dict:
    cold = cold_cloud_metrics(pred, truth)
    return {
        **_round(regression_metrics(pred, truth)),
        "coldCloud": _round({k: cold[k] for k in ("iou", "dice", "precision", "recall")}),
    }


def run_nowcast(anchor_frame_id: str, region: str) -> dict:
    anchor = insat.get_frame(anchor_frame_id)
    sequence = insat.input_sequence(anchor, N_INPUT_FRAMES)
    window = insat.region_window(region)
    context = _expand(window, CONTEXT_HALO_PX)
    inner = (slice(window.row0 - context.row0, window.row1 - context.row0),
             slice(window.col0 - context.col0, window.col1 - context.col0))

    frames_k = np.stack([
        np.stack([insat.read_bt(frame.path, channel, context) for channel in INPUT_CHANNELS]) for frame in sequence
    ]).astype(np.float32)
    ml_inference.validate_frames(frames_k)

    # --- trained model (optional) ---
    prediction = None
    model_info: dict = {"available": False}
    try:
        prediction, served_by = nowcast_client.predict(frames_k)
        model_info = {
            "available": True,
            "name": served_by.get("name"),
            "version": served_by.get("version"),
            "architecture": served_by.get("architecture"),
            "device": prediction.device,
            "inferenceMs": prediction.inference_ms,
        }
    except ml_inference.ModelUnavailableError as exc:
        model_info = {"available": False, "reason": exc.reason}

    # --- classical baseline (always) ---
    classical = optical_flow_forecast(frames_k[-2, 0], frames_k[-1, 0])
    last = frames_k[-1, 0][inner]

    by_time = insat.valid_frames_by_time()
    result_id, out_dir = storage.new_processed_dir("nowcast")

    inputs_out = [
        {"frameId": frame.frame_id, "timestamp": frame.timestamp.isoformat(),
         "url": _save(render_ir(frames_k[i, 0][inner]), out_dir, f"input_{i}", window)}
        for i, frame in enumerate(sequence)
    ]

    involved = [f.frame_id for f in sequence]
    leads_out = []
    for k, (step, minutes) in enumerate(zip(LEAD_STEPS, LEAD_MINUTES)):
        valid_time = anchor.timestamp + step * insat.FRAME_INTERVAL
        truth_frame = by_time.get(valid_time)
        truth = insat.read_bt(truth_frame.path, "TIR1", window) if truth_frame else None
        truth_mask = (truth < COLD_CLOUD_THRESHOLD_K) if truth is not None else None
        if truth_frame:
            involved.append(truth_frame.frame_id)

        classical_bt = classical[k][inner]
        lead: dict = {
            "leadMinutes": minutes,
            "validTime": valid_time.isoformat(),
            "classical": {
                "predictionUrl": _save(render_ir(classical_bt), out_dir, f"classical_{minutes}", window),
                "coldCloudUrl": _save(
                    render_cold_overlay(classical_bt, (classical_bt < COLD_CLOUD_THRESHOLD_K).astype(np.float32),
                                        _CLASSICAL_TINT, truth_mask),
                    out_dir, f"classical_cold_{minutes}", window),
                "coldCloudAreaPct": round(float(100 * (classical_bt < COLD_CLOUD_THRESHOLD_K).mean()), 2),
            },
            "ml": None,
            "actual": None,
            "verification": None,
        }

        if prediction is not None:
            mean_k, scale_k = prediction.mean_k[k][inner], prediction.scale_k[k][inner]
            prob = laplace_cdf(COLD_CLOUD_THRESHOLD_K, mean_k, scale_k)
            flagged = prob >= 0.5
            lead["ml"] = {
                "predictionUrl": _save(render_ir(mean_k), out_dir, f"ml_{minutes}", window),
                "coldCloudUrl": _save(render_cold_overlay(mean_k, prob, _ML_TINT, truth_mask), out_dir, f"ml_cold_{minutes}", window),
                "uncertaintyUrl": _save(render_uncertainty(scale_k), out_dir, f"ml_uncertainty_{minutes}", window),
                "confidence": {
                    # The model's own estimate of its mean absolute error here.
                    "expectedErrorK": round(float(scale_k.mean()), 3),
                    "coldCloudAreaPct": round(float(100 * flagged.mean()), 2),
                    # Mean predicted probability over the pixels it flags as cold cloud.
                    "coldCloudConfidencePct": round(float(100 * prob[flagged].mean()), 1) if flagged.any() else None,
                },
            }

        if truth is not None:
            lead["actual"] = {
                "frameId": truth_frame.frame_id,
                "url": _save(render_ir(truth), out_dir, f"actual_{minutes}", window),
                "coldCloudAreaPct": round(float(100 * truth_mask.mean()), 2),
            }
            verification = {"classical": _verify(classical_bt, truth), "persistence": _verify(last, truth)}
            lead["classical"]["errorUrl"] = _save(render_error(classical_bt, truth), out_dir, f"classical_error_{minutes}", window)
            if prediction is not None:
                verification["ml"] = _verify(mean_k, truth)
                # Was the model's stated uncertainty honest for this image?
                verification["ml"]["within90PctInterval"] = round(
                    float((np.abs(mean_k - truth) <= scale_k * np.log(10.0)).mean()), 4)
                lead["ml"]["errorUrl"] = _save(render_error(mean_k, truth), out_dir, f"ml_error_{minutes}", window)
            lead["verification"] = verification
        leads_out.append(lead)

    split, split_note = _split_summary(involved, _frame_splits())
    result = {
        "resultId": result_id,
        "engine": ENGINE_ML if prediction is not None else ENGINE_CLASSICAL,
        "model": model_info,
        "anchorFrameId": anchor.frame_id,
        "anchorTime": anchor.timestamp.isoformat(),
        "region": region,
        "regionLabel": insat.REGIONS[region]["label"],
        "bounds": window.bounds(),
        "grid": {"widthPx": window.shape[1], "heightPx": window.shape[0], "resolutionKm": 4},
        "dataSplit": split,
        "dataSplitNote": split_note,
        "inputs": inputs_out,
        "leads": leads_out,
        "coldCloudThresholdK": COLD_CLOUD_THRESHOLD_K,
        "notes": {"task": TASK_NOTE, "classical": CLASSICAL_NOTE, "coldCloud": COLD_CLOUD_NOTE},
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    storage.save_result(out_dir, result)
    return result
