"""Classical (non-ML) detection/analysis helpers for satellite & precipitation
imagery: cloud detection, "storm-region" candidate clustering, rainfall
colorization, and segmentation.

Why no trained model: a CNN/U-Net/ConvLSTM cloud or storm detector needs
pixel-labelled ground truth (cloud/no-cloud masks, or confirmed storm-cell
boxes) over enough imagery to generalize. This project has neither -- the
18-image NASA GIBS sample in dataset/imagery/ is for demonstrating the
*pipeline*, not for training (18 images is also far too few regardless).
Per the task's explicit instruction, this module is a documented,
functional image-processing BASELINE using only classical OpenCV
techniques (thresholding, connected components, k-means, optical flow in
processing.py) -- every function below says so again at the point of use,
and nothing here claims to be a trained weather/cloud/storm model.

"Storm-region identification" in particular only flags "unusually large,
unusually bright/dense contiguous cloud cluster" by pixel statistics -- it
is NOT meteorological storm detection (no CAPE/reflectivity/rotation
signature is available in a browse PNG) and callers must not present it as
such. Likewise `processing.optical_flow`'s output is observed pixel motion
between two frames, never a predicted storm track.
"""
from __future__ import annotations

import cv2
import numpy as np

CLOUD_DETECTION_NOTE = (
    "Heuristic brightness threshold (Otsu's method) on grayscale intensity -- "
    "flags bright/white regions consistent with cloud tops in a true-color or "
    "IR-style browse image. Not a trained cloud-mask classifier."
)
STORM_REGION_NOTE = (
    "Candidate dense-cloud-cluster regions ranked by pixel area only "
    "(connected components over the cloud mask). This is NOT a storm "
    "detector -- no convective/reflectivity signal is available in a "
    "browse image. Treat as 'a large bright cluster is here', nothing more."
)


def detect_clouds(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_value, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    overlay = image.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)

    return {
        "mask": mask,
        "overlay": overlay,
        "otsuThreshold": round(float(threshold_value), 2),
        "cloudCoveragePct": round(float((mask > 0).mean() * 100), 2),
        "note": CLOUD_DETECTION_NOTE,
    }


def identify_storm_candidate_regions(image: np.ndarray, cloud_mask: np.ndarray, top_n: int = 5) -> dict:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cloud_mask, connectivity=8)
    image_area = cloud_mask.shape[0] * cloud_mask.shape[1]
    min_area = image_area * 0.003  # ignore specks under ~0.3% of the frame

    candidates = []
    for label_id in range(1, num_labels):  # 0 is background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = (
            stats[label_id, cv2.CC_STAT_LEFT], stats[label_id, cv2.CC_STAT_TOP],
            stats[label_id, cv2.CC_STAT_WIDTH], stats[label_id, cv2.CC_STAT_HEIGHT],
        )
        cx, cy = centroids[label_id]
        candidates.append({
            "boundingBox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "centroid": {"x": round(float(cx), 1), "y": round(float(cy), 1)},
            "areaPx": int(area),
            "areaPct": round(float(area / image_area * 100), 2),
        })
    candidates.sort(key=lambda c: c["areaPx"], reverse=True)
    candidates = candidates[:top_n]

    overlay = image.copy()
    for rank, c in enumerate(candidates, start=1):
        b = c["boundingBox"]
        cv2.rectangle(overlay, (b["x"], b["y"]), (b["x"] + b["width"], b["y"] + b["height"]), (0, 0, 255), 2)
        cv2.putText(overlay, f"#{rank}", (b["x"], max(b["y"] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return {"overlay": overlay, "regions": candidates, "note": STORM_REGION_NOTE}


def colorize_intensity(image: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Applies a perceptual colormap to a single-channel intensity image
    (e.g. a grayscale "how much" field) -- a visualization aid, not an
    analysis. If given an already-3-channel browse image (e.g. the IMERG
    samples in this dataset, which GIBS already colorizes server-side),
    converts to grayscale intensity first so the operation is always
    well-defined, rather than silently no-op'ing on color input.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.applyColorMap(gray, colormap)


def segment_otsu(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return {
        "mask": mask,
        "overlay": overlay,
        "foregroundPct": round(float((mask > 0).mean() * 100), 2),
        "note": "Binary Otsu threshold segmentation -- a classical two-class baseline, not semantic segmentation.",
    }


def segment_kmeans(image: np.ndarray, k: int = 3) -> dict:
    """k-means color clustering as a classical, unsupervised segmentation
    baseline -- clusters pixels by color only (no spatial/semantic model),
    explicitly not comparable to a trained segmentation network.
    """
    small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) if max(image.shape[:2]) > 640 else image
    pixels = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    segmented_small = centers[labels.flatten()].reshape(small.shape)
    segmented = cv2.resize(segmented_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) \
        if segmented_small.shape[:2] != image.shape[:2] else segmented_small

    cluster_pct = {
        int(i): round(float((labels == i).mean() * 100), 2) for i in range(k)
    }
    return {
        "overlay": segmented,
        "clusterCount": k,
        "clusterPct": cluster_pct,
        "note": f"k-means (k={k}) color clustering -- classical unsupervised baseline, not a trained segmentation model.",
    }
