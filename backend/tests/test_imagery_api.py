"""API-level tests for the imagery module, against the real, small NASA
GIBS sample set in dataset/imagery/ (see dataset/imagery/metadata/SOURCES.md)
plus small synthetic uploads generated in-test. No DB involved -- this
module is pure filesystem + OpenCV.
"""
import io

import cv2
import numpy as np
import pytest

from app.services.imagery import storage


def _png_bytes(color=(10, 20, 30), size=48) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = color
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _mp4_bytes(frame_count=6, size=48) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 4.0, (size, size))
        for i in range(frame_count):
            frame = np.full((size, size, 3), (i * 30) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path.read_bytes()


def test_list_samples_returns_the_real_curated_dataset(client):
    resp = client.get("/api/v1/imagery/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) == 18
    assert all(s["id"].startswith("sample:") for s in samples)
    assert all(s["url"].startswith("/media/imagery-dataset/") for s in samples)
    categories = {s["category"] for s in samples}
    assert categories == {"satellite_truecolor", "precipitation_rate_sequence_frame"}


def test_serves_a_real_sample_image_as_static_file(client):
    samples = client.get("/api/v1/imagery/samples").json()
    sample = samples[0]
    resp = client.get(sample["url"])
    assert resp.status_code == 200
    decoded = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None  # genuinely a decodable image, not an error page


def test_upload_image_round_trip(client):
    resp = client.post("/api/v1/imagery/upload", files={"file": ("test.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 48 and body["height"] == 48
    assert body["originalFilename"] == "test.png"

    media = client.get(body["url"])
    assert media.status_code == 200


def test_upload_image_rejects_garbage_bytes(client):
    resp = client.post("/api/v1/imagery/upload", files={"file": ("test.png", b"not a real image", "image/png")})
    assert resp.status_code == 422


def test_upload_image_rejects_disallowed_extension_directly():
    with pytest.raises(storage.UnsupportedFileTypeError):
        storage.save_upload(_png_bytes(), "malicious.exe", kind="image")


def test_upload_image_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(storage, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(storage.UploadTooLargeError):
        storage.save_upload(_png_bytes(), "test.png", kind="image")


def test_process_sample_image_runs_all_operations(client):
    sample_id = "sample:truecolor_modis-terra_2026-11-14"
    resp = client.post(
        f"/api/v1/imagery/process/{sample_id}",
        params=[("operations", op) for op in ["denoise", "contrast", "cloud_detection", "storm_regions", "segment_otsu", "segment_kmeans"]],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {o["operation"] for o in body["outputs"]} == {
        "denoise", "contrast", "cloud_detection", "storm_regions", "segment_otsu", "segment_kmeans",
    }
    assert "cloudCoveragePct" in body["stats"]
    assert "stormCandidateRegions" in body["stats"]

    # every output must actually be a fetchable, decodable image
    for output in body["outputs"]:
        media = client.get(output["url"])
        assert media.status_code == 200
        decoded = cv2.imdecode(np.frombuffer(media.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None

    # and the result must be retrievable again by id
    result_resp = client.get(f"/api/v1/imagery/results/{body['resultId']}")
    assert result_resp.status_code == 200
    assert result_resp.json()["resultId"] == body["resultId"]


def test_process_rejects_unknown_operation(client):
    resp = client.post(
        "/api/v1/imagery/process/sample:truecolor_modis-terra_2026-11-14",
        params={"operations": "not_a_real_operation"},
    )
    assert resp.status_code == 422


def test_process_unknown_image_id_is_404(client):
    resp = client.post("/api/v1/imagery/process/does-not-exist")
    assert resp.status_code == 404


def test_compare_two_real_precipitation_frames(client):
    resp = client.post(
        "/api/v1/imagery/compare",
        params={
            "beforeId": "sample:precip_imerg30_00_20260805T070000Z",
            "afterId": "sample:precip_imerg30_09_20260805T113000Z",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["meanAbsDiff"] >= 0
    assert 0 <= body["changedPixelPct"] <= 100


def test_motion_analysis_on_real_precipitation_sequence(client):
    resp = client.post(
        "/api/v1/imagery/motion",
        params={
            "beforeId": "sample:precip_imerg30_00_20260805T070000Z",
            "afterId": "sample:precip_imerg30_09_20260805T113000Z",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "note" in body and "not a storm track" in body["note"]
    assert body["dominantDirectionLabel"] in {
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    }


def test_video_upload_probe_and_frame_extraction(client):
    upload = client.post("/api/v1/imagery/video/upload", files={"file": ("clip.mp4", _mp4_bytes(frame_count=6), "video/mp4")})
    assert upload.status_code == 200
    video = upload.json()
    assert video["frameCount"] == 6
    assert video["fps"] == 4.0

    frames_resp = client.post(f"/api/v1/imagery/video/{video['assetId']}/frames", params={"maxFrames": 3})
    assert frames_resp.status_code == 200
    frames = frames_resp.json()
    assert frames["frameCount"] == 3
    assert len(frames["frames"]) == 3

    # each extracted frame is a first-class image id usable by /process
    first_frame_id = frames["frames"][0]["imageId"]
    process_resp = client.post(f"/api/v1/imagery/process/{first_frame_id}", params={"operations": "contrast"})
    assert process_resp.status_code == 200


def test_video_upload_rejects_non_video_bytes(client):
    resp = client.post("/api/v1/imagery/video/upload", files={"file": ("clip.mp4", b"not a real video", "video/mp4")})
    assert resp.status_code == 422


def test_generate_video_from_real_precipitation_sequence(client):
    samples = client.get("/api/v1/imagery/samples").json()
    precip_ids = [s["id"] for s in sorted(samples, key=lambda s: s.get("sequence_index", -1)) if s["category"] == "precipitation_rate_sequence_frame"]
    assert len(precip_ids) == 10

    resp = client.post("/api/v1/imagery/video/generate", json={"imageIds": precip_ids, "fps": 2.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["frameCount"] == 10
    assert body["sourceImageIds"] == precip_ids

    media = client.get(body["url"])
    assert media.status_code == 200
    assert len(media.content) > 0


def test_generate_video_requires_at_least_two_frames(client):
    resp = client.post("/api/v1/imagery/video/generate", json={"imageIds": ["sample:truecolor_modis-terra_2026-11-14"]})
    assert resp.status_code == 422
