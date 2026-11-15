"""Core OpenCV operations for the imagery/video processing module.

Every function here is pure (array/bytes in, array/bytes/dict out) -- no
file I/O except `probe_video`/`extract_frames`, which must stream a video
file (OpenCV's VideoCapture has no in-memory byte-buffer API). This file
intentionally contains no FastAPI/storage concerns; see
app/services/imagery/storage.py for file handling and
app/api/v1/endpoints/imagery.py for the HTTP layer.

This is a classical image-processing baseline (OpenCV only) -- there is no
trained cloud/storm/segmentation model behind any of this. See
app/services/imagery/detection.py's module docstring for why, and
dataset/imagery/metadata/SOURCES.md / PROJECT_HANDOFF.md for what data
exists and doesn't.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import AppError


class ImageDecodeError(AppError):
    def __init__(self, detail: str = "Could not decode this file as an image."):
        super().__init__(detail, status_code=422)


class VideoDecodeError(AppError):
    def __init__(self, detail: str = "Could not decode this file as a video."):
        super().__init__(detail, status_code=422)


MAX_DIM = 1280  # resize cap for processing/display -- real originals can be larger


def decode_image(data: bytes) -> np.ndarray:
    """Decodes by actually inspecting the bytes (cv2.imdecode), never by
    trusting a file extension or Content-Type header.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError()
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ImageDecodeError("Failed to encode processed output as PNG.")
    return buf.tobytes()


def basic_info(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {"width": int(w), "height": int(h), "channels": int(channels)}


def resize_max_dim(image: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def brightness_stats(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "meanBrightness": round(float(gray.mean()), 2),
        "stdBrightness": round(float(gray.std()), 2),
    }


def denoise(image: np.ndarray) -> np.ndarray:
    """Non-local-means denoising -- real OpenCV algorithm (cv2.fastNlMeansDenoisingColored),
    moderate strength so texture isn't destroyed on a 1-2 MP frame.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast-Limited Adaptive Histogram Equalization) on the L
    channel of LAB -- improves local contrast (e.g. faint cloud edges in haze)
    without blowing out color like global histogram equalization would.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def _align_shapes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two independently-sourced frames are rarely pixel-identical in size
    -- resize the second to the first's dimensions so every pairwise op
    below has a well-defined per-pixel correspondence.
    """
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return a, b


def compare_images(image_a: np.ndarray, image_b: np.ndarray) -> dict:
    """Plain per-pixel absolute difference -- not a learned "did this scene
    change" model, just arithmetic. `changedPixelPct` uses a fixed abs-diff
    threshold (30/255 per channel, averaged) so small compression/encoding
    noise doesn't count as "changed".
    """
    image_a, image_b = _align_shapes(image_a, image_b)
    diff = cv2.absdiff(image_a, image_b)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    changed_mask = diff_gray > 30
    return {
        "heatmap": heatmap,
        "meanAbsDiff": round(float(diff_gray.mean()), 3),
        "changedPixelPct": round(float(changed_mask.mean() * 100), 2),
    }


_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _compass_label(deg: float) -> str:
    idx = int(((deg % 360) / 22.5) + 0.5) % 16
    return _COMPASS[idx]


def optical_flow(image_a: np.ndarray, image_b: np.ndarray) -> dict:
    """Dense Farneback optical flow (cv2.calcOpticalFlowFarneback) between
    two genuinely time-ordered frames of the SAME scene. This is classical
    motion estimation, not a storm-prediction model -- it describes how
    pixels/cloud-texture moved between these two frames, nothing about the
    future. See detection.py docstring for the "no storm prediction from
    optical flow" boundary this project draws.
    """
    image_a, image_b = _align_shapes(image_a, image_b)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    hsv = np.zeros((*gray_a.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue range is 0-179
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mean_magnitude = float(magnitude.mean())
    # magnitude-weighted circular mean, so near-static background pixels
    # don't drown out the signal from the few genuinely moving regions.
    # `angle` here is cv2.cartToPolar's convention: 0deg = +x (right), and
    # it increases towards +y (DOWN in image/array coordinates, since row
    # index grows downward) -- i.e. 0=right, 90=down, 180=left, 270=up.
    # Compass bearings are 0=up/N, 90=right/E, 180=down/S, 270=left/W, so
    # converting one to the other is a +90 deg rotation, not a straight copy.
    weights = magnitude.flatten()
    if weights.sum() > 1e-6:
        angles_rad = np.deg2rad(angle.flatten())
        mean_sin = float(np.average(np.sin(angles_rad), weights=weights))
        mean_cos = float(np.average(np.cos(angles_rad), weights=weights))
        image_angle_deg = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360)
    else:
        image_angle_deg = 0.0
    dominant_bearing_deg = (image_angle_deg + 90) % 360

    return {
        "visualization": visualization,
        "meanMagnitudePx": round(mean_magnitude, 4),
        "dominantDirectionDeg": round(dominant_bearing_deg, 1),
        "dominantDirectionLabel": _compass_label(dominant_bearing_deg),
        "note": (
            "Classical dense optical flow between two frames -- describes "
            "observed pixel motion only, not a storm track or forecast. "
            "Direction is frame-relative (N/up = top of frame, E/right = "
            "right edge, etc.) and only matches true geographic compass "
            "directions for a north-up overhead image, such as the NASA "
            "GIBS EPSG:4326 satellite/precipitation samples in this project's "
            "own dataset -- not guaranteed for an arbitrary uploaded photo/video."
        ),
    }


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoDecodeError()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoDecodeError("Video has no readable frames.")
    return {
        "frameCount": frame_count,
        "fps": round(float(fps), 3) if fps else None,
        "durationSeconds": round(frame_count / fps, 2) if fps else None,
        "width": width,
        "height": height,
    }


def extract_frames(path: Path, max_frames: int = 12) -> list[tuple[int, float, np.ndarray]]:
    """Evenly-sampled frames across the whole video (not just the first N),
    so a short demo clip's motion is actually represented end to end.
    """
    info = probe_video(path)
    frame_count = info["frameCount"]
    fps = info["fps"] or 0.0
    n = max(1, min(max_frames, frame_count))
    indices = sorted(set(int(round(i * (frame_count - 1) / max(n - 1, 1))) for i in range(n)))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoDecodeError()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        timestamp = idx / fps if fps else 0.0
        frames.append((idx, round(timestamp, 3), frame))
    cap.release()
    return frames


def write_video(frames: list[np.ndarray], out_path: Path, fps: float = 2.0) -> dict:
    """Builds a real video file from an ordered list of frames with
    cv2.VideoWriter. Caller is responsible for only passing genuinely
    time-ordered frames -- this function just encodes whatever order it's
    given.

    Codec: 'avc1' (H.264), not 'mp4v' (MPEG-4 Part 2) -- verified by
    actually loading both in a real browser <video> tag: mp4v produced a
    file OpenCV itself could read back (so a naive probe looked fine) but
    Chrome refused to decode it (readyState stayed 0, networkState
    NETWORK_NO_SOURCE); avc1 played correctly (readyState 4,
    HAVE_ENOUGH_DATA, correct dimensions/duration). No ffmpeg binary exists
    on this machine to transcode after the fact, so the encoder must
    produce a browser-playable file directly.
    """
    if not frames:
        raise AppError("No frames to encode.", status_code=400)
    h, w = frames[0].shape[:2]
    resized = [f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise AppError("Video encoder failed to open (avc1/H.264 unavailable).", status_code=500)
    for frame in resized:
        writer.write(frame)
    writer.release()

    return {"frameCount": len(resized), "fps": fps, "width": w, "height": h, "path": str(out_path)}
