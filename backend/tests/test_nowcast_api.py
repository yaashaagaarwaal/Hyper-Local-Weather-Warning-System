"""API integration tests for the trained-model endpoints
(`/imagery/ml/*`, `/imagery/insat/*`), against the real INSAT-3DR files and
the real trained artifact, through the real inference worker process.
Skipped where data/artifact/torch are not available. No DB.

This file must never `import torch` (not even via importorskip): the whole
point of the worker is that the API process stays torch-free.
"""
import importlib.util
import os
import signal
import sys
import time

import pytest

pytest.importorskip("h5py")

from app.services.imagery import insat, ml_inference, nowcast_client  # noqa: E402

REAL_DATA = insat.DEFAULT_INSAT_DIR.exists() and any(insat.DEFAULT_INSAT_DIR.rglob("*.h5"))
ARTIFACT = (
    (ml_inference.MODEL_DIR / ml_inference.CHECKPOINT_NAME).exists()
    and importlib.util.find_spec("torch") is not None
)
needs_data = pytest.mark.skipif(not REAL_DATA, reason="INSAT-3DR MOSDAC files not present")
needs_model = pytest.mark.skipif(not (REAL_DATA and ARTIFACT), reason="needs real data and the trained artifact")

TEST_ANCHOR = "20250107T1915Z"   # first held-out test anchor; +30/+60/+90 all verifiable
TRAIN_ANCHOR = "20250106T1815Z"
FIRST_FRAME = "20250106T1345Z"   # nothing before it
CORRUPT_FRAME = "20250107T0745Z"


@pytest.fixture(autouse=True)
def _cpu_and_clean_cache(monkeypatch):
    monkeypatch.setenv("NOWCAST_DEVICE", "cpu")
    monkeypatch.delenv("NOWCAST_MODEL_DIR", raising=False)
    monkeypatch.delenv("INSAT_DATA_DIR", raising=False)
    yield


def _assert_png(client, url):
    resp = client.get(url)
    assert resp.status_code == 200, url
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


# --- status -----------------------------------------------------------------
@needs_model
def test_status_describes_the_loaded_model(client):
    body = client.get("/api/v1/imagery/ml/status").json()
    assert body["available"] is True
    assert body["name"] == "insat_nowcast" and body["version"]
    assert body["device"] == "cpu"
    assert body["leadMinutes"] == [30, 60, 90]
    assert body["inputChannels"] == ["TIR1", "WV"]
    assert body["testMetrics"]["30"]["model"]["maeK"] > 0


def test_status_is_ok_and_honest_when_the_artifact_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    resp = client.get("/api/v1/imagery/ml/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False and body["reason"]
    assert "testMetrics" not in body  # no metrics claimed for a model that is not there


# --- frame catalog / preview ----------------------------------------------------
@needs_data
def test_frame_catalog_lists_valid_corrupt_and_split(client):
    body = client.get("/api/v1/imagery/insat/frames").json()
    assert body["available"] is True
    assert body["validCount"] == 59 and body["corruptCount"] == 4
    frames = {f["frameId"]: f for f in body["frames"]}
    assert frames[CORRUPT_FRAME]["valid"] is False and frames[CORRUPT_FRAME]["canAnchorNowcast"] is False
    assert frames[FIRST_FRAME]["canAnchorNowcast"] is False
    assert frames[TEST_ANCHOR]["canAnchorNowcast"] is True
    assert frames[TEST_ANCHOR]["verifiableLeadMinutes"] == [30, 60, 90]
    assert {r["key"] for r in body["regions"]} == {"north_india", "india", "full_sector"}
    if ARTIFACT:
        assert frames[TEST_ANCHOR]["split"] == "test"
        assert frames[TRAIN_ANCHOR]["split"] == "train"


def test_frame_catalog_is_unavailable_not_an_error_without_data(client, tmp_path, monkeypatch):
    monkeypatch.setenv("INSAT_DATA_DIR", str(tmp_path))
    resp = client.get("/api/v1/imagery/insat/frames")
    assert resp.status_code == 200
    assert resp.json()["available"] is False and resp.json()["frames"] == []


@needs_data
def test_frame_preview_renders_a_real_png(client):
    resp = client.get(f"/api/v1/imagery/insat/frames/{TEST_ANCHOR}/preview", params={"region": "north_india", "channel": "TIR1"})
    assert resp.status_code == 200
    body = resp.json()
    assert 170 < body["minK"] < body["maxK"] < 340
    assert 0 <= body["coldCloudPct"] <= 100
    _assert_png(client, body["url"])


@needs_data
def test_frame_preview_error_paths(client):
    base = "/api/v1/imagery/insat/frames"
    assert client.get(f"{base}/19990101T0000Z/preview").status_code == 404
    assert client.get(f"{base}/{CORRUPT_FRAME}/preview").status_code == 422
    assert client.get(f"{base}/{TEST_ANCHOR}/preview", params={"region": "atlantis"}).status_code == 422
    assert client.get(f"{base}/{TEST_ANCHOR}/preview", params={"channel": "VIS"}).status_code == 422


# --- nowcast ------------------------------------------------------------------
@needs_model
def test_nowcast_returns_model_classical_and_verification(client):
    resp = client.post("/api/v1/imagery/ml/nowcast", json={"anchorFrameId": TEST_ANCHOR, "region": "north_india"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["engine"] == "trained-ml"
    assert body["model"]["available"] is True and body["model"]["version"]
    assert body["model"]["device"] == "cpu" and body["model"]["inferenceMs"] > 0
    assert body["dataSplit"] == "test"
    assert len(body["inputs"]) == 3
    assert [lead["leadMinutes"] for lead in body["leads"]] == [30, 60, 90]

    for lead in body["leads"]:
        ml, classical, actual, verification = lead["ml"], lead["classical"], lead["actual"], lead["verification"]
        # trained-model output: image, overlay, uncertainty, confidence
        for key in ("predictionUrl", "coldCloudUrl", "uncertaintyUrl", "errorUrl"):
            _assert_png(client, ml[key])
        assert ml["confidence"]["expectedErrorK"] > 0
        assert 0 <= ml["confidence"]["coldCloudAreaPct"] <= 100
        # classical output is present and separate
        for key in ("predictionUrl", "coldCloudUrl", "errorUrl"):
            _assert_png(client, classical[key])
        assert "confidence" not in classical
        # the real later frame, and metrics for all three methods
        _assert_png(client, actual["url"])
        assert set(verification) == {"ml", "classical", "persistence"}
        for method in verification.values():
            assert method["maeK"] > 0 and method["rmseK"] >= method["maeK"]
            assert set(method["coldCloud"]) == {"iou", "dice", "precision", "recall"}
        assert 0 <= verification["ml"]["within90PctInterval"] <= 1

    # all rendered images share the region's pixel grid
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(client.get(body["leads"][0]["ml"]["predictionUrl"]).content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (body["grid"]["heightPx"], body["grid"]["widthPx"])

    # stored and retrievable through the existing results endpoint
    stored = client.get(f"/api/v1/imagery/results/{body['resultId']}")
    assert stored.status_code == 200 and stored.json()["anchorFrameId"] == TEST_ANCHOR


@needs_model
def test_nowcast_flags_training_data_as_not_an_honest_check(client):
    body = client.post("/api/v1/imagery/ml/nowcast", json={"anchorFrameId": TRAIN_ANCHOR, "region": "north_india"}).json()
    assert body["dataSplit"] == "train"
    assert "TRAIN" in body["dataSplitNote"]


@needs_data
def test_nowcast_falls_back_to_classical_when_the_model_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    resp = client.post("/api/v1/imagery/ml/nowcast", json={"anchorFrameId": TEST_ANCHOR, "region": "north_india"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "classical-fallback"
    assert body["model"]["available"] is False and body["model"]["reason"]
    for lead in body["leads"]:
        assert lead["ml"] is None  # nothing is passed off as a model prediction
        _assert_png(client, lead["classical"]["predictionUrl"])
        assert set(lead["verification"]) == {"classical", "persistence"}


@needs_data
def test_nowcast_falls_back_when_the_model_file_is_corrupt(client, tmp_path, monkeypatch):
    (tmp_path / ml_inference.CHECKPOINT_NAME).write_bytes(b"garbage" * 1000)
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    body = client.post("/api/v1/imagery/ml/nowcast", json={"anchorFrameId": TEST_ANCHOR, "region": "north_india"}).json()
    assert body["engine"] == "classical-fallback"
    assert "corrupt" in body["model"]["reason"]


@needs_data
def test_nowcast_error_paths(client):
    url = "/api/v1/imagery/ml/nowcast"
    assert client.post(url, json={"anchorFrameId": "19990101T0000Z"}).status_code == 404
    assert client.post(url, json={"anchorFrameId": TEST_ANCHOR, "region": "atlantis"}).status_code == 422
    assert client.post(url, json={"region": "north_india"}).status_code == 422  # missing anchor
    corrupt = client.post(url, json={"anchorFrameId": CORRUPT_FRAME})
    assert corrupt.status_code == 422 and "corrupt" in corrupt.json()["error"]
    gap = client.post(url, json={"anchorFrameId": FIRST_FRAME})
    assert gap.status_code == 422 and "consecutive" in gap.json()["error"]


@needs_data
def test_existing_classical_endpoints_still_work_alongside(client):
    """The upgrade must not change the classical pipeline's API."""
    samples = client.get("/api/v1/imagery/samples")
    assert samples.status_code == 200 and len(samples.json()) > 0
    sample_id = samples.json()[0]["id"]
    processed = client.post(f"/api/v1/imagery/process/{sample_id}", params={"operations": ["cloud_detection"]})
    assert processed.status_code == 200
    assert processed.json()["outputs"][0]["operation"] == "cloud_detection"


# --- process isolation --------------------------------------------------------
@needs_model
def test_api_process_never_imports_torch(client):
    """Regression test for a real crash: PyTorch + XGBoost in one process
    segfaults on macOS. Inference must happen in the worker, not here."""
    assert client.get("/api/v1/imagery/ml/status").json()["available"] is True
    resp = client.post("/api/v1/imagery/ml/nowcast", json={"anchorFrameId": TEST_ANCHOR, "region": "north_india"})
    assert resp.json()["engine"] == "trained-ml"
    assert "torch" not in sys.modules
    assert nowcast_client.worker_pid() not in (None, os.getpid())


@needs_model
def test_model_stays_loaded_between_requests_and_worker_recovers_from_a_crash(client):
    assert client.get("/api/v1/imagery/ml/status").json()["available"] is True
    first_pid = nowcast_client.worker_pid()
    assert client.get("/api/v1/imagery/ml/status").json()["available"] is True
    assert nowcast_client.worker_pid() == first_pid  # same worker => model cached, not reloaded

    os.kill(first_pid, signal.SIGKILL)
    time.sleep(0.3)
    body = client.get("/api/v1/imagery/ml/status").json()
    assert body["available"] is True  # a fresh worker was started transparently
    assert nowcast_client.worker_pid() != first_pid


@needs_data
def test_unstartable_worker_degrades_to_classical(client, monkeypatch):
    nowcast_client.shutdown()

    def refuse():
        raise OSError("no such interpreter")

    monkeypatch.setattr(nowcast_client, "_start", refuse)
    status = client.get("/api/v1/imagery/ml/status").json()
    assert status["available"] is False and "worker" in status["reason"]
    body = client.post("/api/v1/imagery/ml/nowcast", json={"anchorFrameId": TEST_ANCHOR, "region": "north_india"}).json()
    assert body["engine"] == "classical-fallback"
    assert all(lead["ml"] is None for lead in body["leads"])
