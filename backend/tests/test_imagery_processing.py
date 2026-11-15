"""Unit tests for the pure OpenCV operations in
app/services/imagery/processing.py and detection.py -- synthetic arrays
only, so these are deterministic and don't depend on the live GIBS sample
set (see test_imagery_api.py for tests against the real downloaded data).
"""
import cv2
import numpy as np
import pytest

from app.services.imagery import detection, processing


def _solid(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_decode_image_round_trip():
    original = _solid((10, 20, 30))
    data = processing.encode_png(original)
    decoded = processing.decode_image(data)
    assert decoded.shape == original.shape
    assert np.array_equal(decoded, original)


def test_decode_image_rejects_garbage_bytes():
    with pytest.raises(processing.ImageDecodeError):
        processing.decode_image(b"this is not an image")


def test_resize_max_dim_only_downscales():
    small = _solid((1, 2, 3), size=32)
    assert processing.resize_max_dim(small, max_dim=1280) is small  # untouched, already small

    large = _solid((1, 2, 3), size=2000)
    resized = processing.resize_max_dim(large, max_dim=1280)
    assert max(resized.shape[:2]) == 1280


def test_denoise_and_contrast_preserve_shape():
    img = np.random.default_rng(0).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    assert processing.denoise(img).shape == img.shape
    assert processing.enhance_contrast(img).shape == img.shape


def test_compare_images_identical_frames_have_near_zero_diff():
    img = np.random.default_rng(1).integers(0, 255, (50, 50, 3), dtype=np.uint8)
    result = processing.compare_images(img, img.copy())
    assert result["meanAbsDiff"] == 0.0
    assert result["changedPixelPct"] == 0.0


def test_compare_images_detects_a_real_change():
    before = _solid((0, 0, 0), size=50)
    after = _solid((0, 0, 0), size=50)
    after[10:40, 10:40] = (255, 255, 255)  # a real, large, unambiguous change
    result = processing.compare_images(before, after)
    assert result["meanAbsDiff"] > 50
    assert 30 < result["changedPixelPct"] < 50  # the 30x30 square is 36% of the 50x50 frame
    assert result["heatmap"].shape == before.shape


def test_optical_flow_detects_rightward_translation():
    # Farneback needs real spatial structure to match against -- pure random
    # noise hits the aperture problem and gives unreliable flow, so build a
    # frame with actual gradients/blobs (closer to real cloud texture) rather
    # than np.random noise.
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(base, (30, 50), 15, (200, 200, 200), -1)
    cv2.circle(base, (60, 30), 10, (150, 150, 150), -1)
    base = cv2.GaussianBlur(base, (9, 9), 0)
    # Shift the whole frame 8px to the right -- every pixel's true motion is
    # unambiguous (dx=+8, dy=0), so Farneback should recover ~rightward flow.
    shifted = np.zeros_like(base)
    shifted[:, 8:] = base[:, :-8]
    shifted[:, :8] = base[:, :8]

    result = processing.optical_flow(base, shifted)
    assert result["meanMagnitudePx"] > 0
    assert result["visualization"].shape == base.shape
    # Rightward motion -> angle near 0 deg -> compass label E/ENE/ESE
    assert result["dominantDirectionLabel"] in {"E", "ENE", "ESE", "NE", "SE"}


def test_probe_and_write_video_round_trip(tmp_path):
    frames = [_solid((i * 20, 0, 0), size=40) for i in range(5)]
    out_path = tmp_path / "test.mp4"
    info = processing.write_video(frames, out_path, fps=4.0)
    assert info["frameCount"] == 5
    assert out_path.exists() and out_path.stat().st_size > 0

    probed = processing.probe_video(out_path)
    assert probed["frameCount"] == 5
    assert probed["fps"] == 4.0


def test_write_video_rejects_empty_frame_list(tmp_path):
    with pytest.raises(Exception):
        processing.write_video([], tmp_path / "empty.mp4")


def test_detect_clouds_coverage_bounds():
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    result = detection.detect_clouds(black)
    assert 0 <= result["cloudCoveragePct"] <= 100
    assert result["mask"].shape == (40, 40)


def test_identify_storm_candidate_regions_on_synthetic_bright_blob():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:60, 20:60] = 255  # a large, unambiguous bright square
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    result = detection.identify_storm_candidate_regions(img, mask)
    assert len(result["regions"]) == 1
    region = result["regions"][0]
    assert region["areaPx"] == 1600  # 40x40
    assert result["overlay"].shape == img.shape


def test_segment_otsu_and_kmeans_preserve_shape():
    img = np.random.default_rng(3).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    otsu = detection.segment_otsu(img)
    assert otsu["overlay"].shape == img.shape

    kmeans = detection.segment_kmeans(img, k=3)
    assert kmeans["overlay"].shape == img.shape
    assert kmeans["clusterCount"] == 3
    assert abs(sum(kmeans["clusterPct"].values()) - 100.0) < 1.0
