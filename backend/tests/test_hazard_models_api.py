"""API integration tests for the flood-mapping and lightning-detection
models (`/hazard-models/*`), against the real datasets and the real trained
artifacts, through the real inference worker process. Skipped where data /
artifacts / torch are not available. No DB.

This file must never `import torch`: the API process stays torch-free.
"""
import importlib.util
import io
import sys

import numpy as np
import pytest

from app.services.imagery import flood_data, hazard_inference
from app.services.imagery import lightning_data as ld
from app.services.imagery.hazard_common import HAZARD_MODEL_DIR

TORCH = importlib.util.find_spec("torch") is not None
FLOOD_DATA = bool(flood_data.real_catalog())
FLOOD_SYN = bool(flood_data.synthetic_catalog())
LTG_SYN = bool(ld.synthetic_catalog())
GOES = len(ld.abi_frames()) >= ld.LOOKBACK_FRAMES + 1
FLOOD_MODEL = TORCH and (HAZARD_MODEL_DIR / hazard_inference.SPECS["flood"]["checkpoint"]).exists()
LTG_MODEL = TORCH and (HAZARD_MODEL_DIR / hazard_inference.SPECS["lightning"]["checkpoint"]).exists()
needs_flood = pytest.mark.skipif(not (FLOOD_DATA and FLOOD_MODEL), reason="needs Sen1Floods11 chips and the trained flood artifact")
needs_flood_syn = pytest.mark.skipif(not (FLOOD_SYN and FLOOD_MODEL), reason="needs synthetic flood chips and the trained artifact")
needs_ltg_syn = pytest.mark.skipif(not (LTG_SYN and LTG_MODEL), reason="needs synthetic lightning scenes and the trained artifact")
needs_goes = pytest.mark.skipif(not (GOES and LTG_MODEL), reason="needs GOES-16 files and the trained lightning artifact")

API = "/api/v1/hazard-models"


@pytest.fixture(autouse=True)
def _cpu_and_default_dirs(monkeypatch):
    monkeypatch.setenv("NOWCAST_DEVICE", "cpu")
    for key in ("HAZARD_MODEL_DIR", "FLOOD_DATA_DIR", "FLOOD_SYNTHETIC_DIR", "GOES_DATA_DIR", "LIGHTNING_SYNTHETIC_DIR"):
        monkeypatch.delenv(key, raising=False)
    yield


def _png(client, url):
    resp = client.get(url)
    assert resp.status_code == 200, url
    assert resp.headers["content-type"] == "image/png" and resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def _flood_id(split, event="India"):
    return next(c.chip_id for c in flood_data.real_catalog() if c.split == split and c.region == event)


# --- status / metrics ---------------------------------------------------------
@needs_flood
def test_status_describes_both_models_and_keeps_real_and_synthetic_scores_apart(client):
    body = client.get(f"{API}/status").json()
    flood = body["flood"]
    assert flood["available"] is True and flood["name"] == "flood_unet" and flood["device"] == "cpu"
    assert 0 < flood["threshold"] < 1 and flood["parameters"] > 100_000 and flood["isolatedWorker"] is True
    assert 0 < flood["testMetrics"]["real"]["iou"] < 1 and flood["testMetrics"]["real"]["chips"] == 29
    assert "simulator" in flood["syntheticMetrics"]["note"]              # synthetic scores are a separate, labelled block
    assert flood["limitations"]
    if LTG_MODEL:
        lightning = body["lightning"]
        assert lightning["available"] is True and lightning["validatedForIndia"] is False
        assert any("NOT validated for India" in line for line in lightning["limitations"])
        assert client.get(f"{API}/lightning/status").json()["name"] == "lightning_unet"


def test_unknown_model_and_unknown_figure_are_404(client):
    assert client.get(f"{API}/tornado/status").status_code == 404
    assert client.get(f"{API}/tornado/metrics").status_code == 404
    assert client.get(f"{API}/flood/figures/secrets.png").status_code == 404
    assert client.get(f"{API}/flood/figures/..%2F..%2Fflood_unet_v1.pt").status_code == 404


@needs_flood
def test_metrics_report_separates_real_from_synthetic_and_serves_its_figures(client):
    body = client.get(f"{API}/flood/metrics").json()
    assert body["available"] is True
    assert "test never consulted" in body["promotion"]["rule"]
    assert {"real", "real+synthetic", "synthetic"} >= set(body["variantComparison"])
    assert "realTest" in body["promoted"] and "syntheticTest" in body["promoted"]
    assert body["baselines"]["tunedOn"] == "real training chips only"
    for figure in body["figures"]:
        _png(client, figure["url"])


def test_metrics_without_a_report_say_so(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    body = client.get(f"{API}/lightning/metrics").json()
    assert body["available"] is False and "evaluation" in body["reason"]


# --- flood ----------------------------------------------------------------------
@needs_flood
def test_flood_catalog_lists_real_chips_with_their_official_splits(client):
    body = client.get(f"{API}/flood/samples").json()
    assert len(body["real"]) == 138 and all(s["source"] == "real" for s in body["real"])
    assert {s["split"] for s in body["real"]} == {"train", "valid", "test"}
    assert all(s["source"] == "synthetic" for s in body["synthetic"])
    assert "not a satellite observation" in body["syntheticSource"]


@needs_flood
def test_flood_prediction_on_a_real_held_out_india_chip(client):
    body = client.post(f"{API}/flood/predict", json={"sampleId": _flood_id("test")}).json()
    assert body["engine"] == "trained-ml" and body["model"]["available"] is True and body["model"]["device"] == "cpu"
    assert body["sample"]["source"] == "real" and body["sample"]["split"] == "test" and "Held-out" in body["sample"]["splitNote"]
    geo = body["geo"]
    assert 6 < geo["centre"]["latitude"] < 37 and 68 < geo["centre"]["longitude"] < 98
    assert geo["footprintKm2"] == pytest.approx(23.5, rel=0.2)
    ml, classical, check = body["ml"], body["classical"], body["verification"]
    assert 0 <= ml["waterAreaPct"] <= 100 and ml["waterAreaKm2"] <= geo["footprintKm2"]
    assert ml["floodAreaKm2"] <= ml["waterAreaKm2"]                       # flood = water outside permanent water
    assert 0 < ml["threshold"] < 1 and (ml["meanWaterConfidencePct"] is None or ml["meanWaterConfidencePct"] >= 100 * ml["threshold"] - 1e-6)
    assert classical["thresholdSource"] == "tuned on the real training chips"
    for key in ("ml", "classical", "datasetOtsuReference"):
        assert set(check[key]) >= {"iou", "f1", "precision", "recall", "tp", "fp", "fn", "tn"}
    assert check["ml"]["tp"] + check["ml"]["fp"] + check["ml"]["fn"] + check["ml"]["tn"] == check["ml"]["pixels"] <= 512 * 512
    for url in (ml["overlayUrl"], ml["probabilityUrl"], ml["maskUrl"], ml["errorUrl"], classical["overlayUrl"], classical["errorUrl"],
                body["images"]["radarVvUrl"], body["images"]["radarVhUrl"], body["images"]["labelUrl"]):
        _png(client, url)
    assert client.get(f"/api/v1/imagery/results/{body['resultId']}").json()["resultId"] == body["resultId"]


@needs_flood
def test_flood_prediction_is_deterministic_and_training_chips_are_flagged(client):
    sample = _flood_id("test")
    a = client.post(f"{API}/flood/predict", json={"sampleId": sample}).json()
    b = client.post(f"{API}/flood/predict", json={"sampleId": sample}).json()
    assert a["ml"]["waterPixels"] == b["ml"]["waterPixels"] and a["verification"] == b["verification"]
    train = client.post(f"{API}/flood/predict", json={"sampleId": _flood_id("train")}).json()
    assert train["sample"]["split"] == "train" and "TRAIN" in train["sample"]["splitNote"]


@needs_flood_syn
def test_flood_prediction_on_a_synthetic_chip_is_labelled_synthetic_everywhere(client):
    chip = next(c for c in flood_data.synthetic_catalog() if c.split == "test" and c.meta["floodLabel"] == "flood")
    body = client.post(f"{API}/flood/predict", json={"sampleId": chip.chip_id}).json()
    sample = body["sample"]
    assert sample["source"] == "synthetic" and sample["sourceLabel"].startswith("SYNTHETIC")
    assert "SYNTHETIC SIMULATION" in sample["notice"] and "SYNTHETIC SIMULATION" in sample["splitNote"]
    assert sample["state"] == chip.meta["state"] and sample["scenario"] == chip.meta["scenario"]
    assert body["geo"]["bounds"] == chip.meta["bounds"] and body["verification"]["ml"]["iou"] is not None


@needs_flood
def test_flood_upload_of_a_real_radar_geotiff_and_of_junk(client):
    chip = flood_data.get_chip(_flood_id("test"))
    upload = client.post(f"{API}/flood/predict/upload", files={"file": (chip.path.name, chip.path.read_bytes(), "image/tiff")})
    assert upload.status_code == 200
    body = upload.json()
    assert body["engine"] == "trained-ml" and body["sample"]["source"] == "upload" and body["verification"] is None
    assert body["geo"]["bounds"]["south"] == pytest.approx(flood_data.chip_bounds(chip)["south"])      # georeference survives
    assert body["ml"]["waterAreaKm2"] is not None and "floodAreaKm2" not in body["ml"]                 # no permanent-water layer

    for name, data in (("junk.tif", b"this is not a tiff"), ("empty.tif", b"")):
        resp = client.post(f"{API}/flood/predict/upload", files={"file": (name, data, "image/tiff")})
        assert resp.status_code == 422 and "TIFF" in resp.json()["error"]
    tifffile = pytest.importorskip("tifffile")
    for array, fragment in ((np.zeros((64, 64, 3), np.uint8), "2 radar bands"),
                            (np.random.default_rng(0).uniform(0, 1, (2, 128, 128)).astype(np.float32), "linear power")):
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, array)
        resp = client.post(f"{API}/flood/predict/upload", files={"file": ("x.tif", buffer.getvalue(), "image/tiff")})
        assert resp.status_code == 422 and fragment in resp.json()["error"]


def test_flood_unknown_sample_is_a_404_and_a_missing_id_is_a_422(client):
    assert client.post(f"{API}/flood/predict", json={"sampleId": "India_0"}).status_code == 404
    assert client.post(f"{API}/flood/predict", json={}).status_code == 422


# --- flood: the model must be optional -------------------------------------------
@pytest.mark.skipif(not FLOOD_DATA, reason="needs Sen1Floods11 chips")
def test_flood_falls_back_to_the_threshold_when_the_artifact_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    status = client.get(f"{API}/flood/status").json()
    assert status["available"] is False and ("not found" in status["reason"] or "PyTorch" in status["reason"])
    body = client.post(f"{API}/flood/predict", json={"sampleId": _flood_id("test")}).json()
    assert body["engine"] == "classical-fallback" and body["ml"] is None and body["model"]["available"] is False
    assert body["classical"]["waterAreaPct"] >= 0 and "ml" not in body["verification"]
    assert body["classical"]["thresholdSource"].startswith("default")      # no metadata either: says so
    _png(client, body["classical"]["overlayUrl"])


@pytest.mark.skipif(not (FLOOD_DATA and TORCH), reason="needs Sen1Floods11 chips and torch")
def test_flood_falls_back_when_the_artifact_is_corrupt(client, tmp_path, monkeypatch):
    (tmp_path / hazard_inference.SPECS["flood"]["checkpoint"]).write_bytes(b"\x00not a checkpoint\x00" * 50)
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    body = client.post(f"{API}/flood/predict", json={"sampleId": _flood_id("test")}).json()
    assert body["engine"] == "classical-fallback" and "corrupt" in body["model"]["reason"]


# --- lightning --------------------------------------------------------------------
@needs_ltg_syn
def test_lightning_catalog_keeps_the_three_kinds_of_input_apart(client):
    body = client.get(f"{API}/lightning/samples").json()
    assert all(s["source"] == "synthetic" for s in body["synthetic"]) and len(body["synthetic"]) == 408
    assert all(s["source"] == "real" for s in body["real"]) and all(s["source"] == "real-insat" and s["label"] is False for s in body["insat"])
    assert "not India" in body["realSource"] and "NO lightning ground truth" in body["insatSource"]
    if GOES:
        assert len(body["real"]) == 18 and {s["split"] for s in body["real"]} == {"train", "val", "test", "unused"}


@needs_ltg_syn
def test_lightning_prediction_on_a_synthetic_india_scene(client):
    sample = next(s for s in ld.synthetic_catalog() if s.split == "test" and s.meta["totalFlashes"] > 50)
    body = client.post(f"{API}/lightning/predict", json={"sampleId": sample.sample_id}).json()
    assert body["engine"] == "trained-ml" and body["validatedForIndia"] is False
    assert body["sample"]["source"] == "synthetic" and "simulated lightning" in body["sample"]["sourceLabel"]
    assert body["geo"]["bounds"] == sample.meta["bounds"] and body["geo"]["mapOverlaySupported"] is True
    assert body["geo"]["widthCells"] == 64 and body["geo"]["cellKm"] == 8
    ml = body["ml"]
    assert 0 < ml["threshold"] < 1 and 0 <= ml["maxProbability"] <= 1 and ml["flaggedAreaKm2"] == ml["flaggedCells"] * 64
    bounds = sample.meta["bounds"]
    for point in ml["locations"]:
        assert bounds["south"] < point["latitude"] < bounds["north"] and bounds["west"] < point["longitude"] < bounds["east"]
        assert point["probability"] >= ml["threshold"] - 1e-6
    scope = body["verification"]["scopes"]["allCells"]
    assert scope["truthCells"] == body["verification"]["truthCells"] > 0 and set(scope) >= {"ml", "classical"}
    assert body["truthFlashes"]["count"] > 0
    for url in (ml["overlayUrl"], ml["probabilityUrl"], ml["mapOverlayUrl"], body["classical"]["overlayUrl"],
                body["images"]["infraredUrl"], body["images"]["truthUrl"]):
        _png(client, url)


@needs_ltg_syn
def test_lightning_free_synthetic_scene_has_nothing_to_score(client):
    sample = next(s for s in ld.synthetic_catalog() if s.meta["stormKind"] == "clear")
    body = client.post(f"{API}/lightning/predict", json={"sampleId": sample.sample_id}).json()
    assert body["verification"]["truthCells"] == 0 and body["truthFlashes"]["count"] == 0
    assert body["verification"]["scopes"]["allCells"]["classical"]["recall"] is None      # undefined, not invented


@needs_goes
def test_lightning_prediction_on_real_goes_scores_the_held_out_blocks_separately(client):
    frame = ld.abi_frames()[ld.TEST_ANCHORS[0]]
    body = client.post(f"{API}/lightning/predict", json={"sampleId": frame.frame_id}).json()
    assert body["sample"]["source"] == "real" and body["sample"]["split"] == "test" and "not India" in body["sample"]["sourceLabel"]
    assert body["geo"]["widthCells"] == 625 and body["geo"]["heightCells"] == 375 and body["geo"]["mapOverlaySupported"] is False
    assert body["geo"]["bounds"]["east"] < -40                                       # the Americas
    scopes = body["verification"]["scopes"]
    assert list(scopes) == ["heldOutTestBlocks", "allCells"]
    assert 0 < scopes["heldOutTestBlocks"]["cells"] < scopes["allCells"]["cells"]
    assert body["truthFlashes"]["count"] > 1000 and len(body["ml"]["locations"]) <= 150
    train = client.post(f"{API}/lightning/predict", json={"sampleId": ld.abi_frames()[ld.TRAIN_ANCHORS[0]].frame_id}).json()
    assert list(train["verification"]["scopes"]) == ["allCells"] and "TRAIN" in train["sample"]["splitNote"]


@needs_goes
def test_goes_frame_without_an_earlier_image_is_rejected(client):
    first = ld.abi_frames()[0].frame_id
    resp = client.post(f"{API}/lightning/predict", json={"sampleId": first})
    assert resp.status_code == 422 and "30 minutes earlier" in resp.json()["error"]


@pytest.mark.skipif(not LTG_MODEL, reason="needs the trained lightning artifact")
def test_lightning_on_real_insat_is_marked_unverified_with_no_scores(client):
    insat = client.get(f"{API}/lightning/samples").json()["insat"]
    if not insat:
        pytest.skip("INSAT-3DR files not present")
    body = client.post(f"{API}/lightning/predict", json={"sampleId": insat[-1]["sampleId"], "region": "north_india"}).json()
    assert body["sample"]["source"] == "real-insat" and body["sample"]["experimental"] is True
    assert body["verification"] is None and body["truthFlashes"] is None and "truthUrl" not in body["images"]
    assert "NO lightning ground truth" in body["sample"]["notice"] and body["validatedForIndia"] is False
    assert 60 < body["geo"]["bounds"]["west"] < body["geo"]["bounds"]["east"] < 100
    bad = client.post(f"{API}/lightning/predict", json={"sampleId": insat[-1]["sampleId"], "region": "mars"})
    assert bad.status_code in (404, 422)


def test_lightning_unknown_sample_is_a_404(client):
    assert client.post(f"{API}/lightning/predict", json={"sampleId": "GOES16_19990101T0000Z"}).status_code == 404


@pytest.mark.skipif(not LTG_SYN, reason="needs synthetic lightning scenes")
def test_lightning_falls_back_to_the_temperature_threshold_without_a_model(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    sample = next(s for s in ld.synthetic_catalog() if s.split == "test" and s.meta["totalFlashes"] > 50)
    body = client.post(f"{API}/lightning/predict", json={"sampleId": sample.sample_id}).json()
    assert body["engine"] == "classical-fallback" and body["ml"] is None
    assert body["classical"]["thresholdK"] == 215.0 and body["classical"]["thresholdSource"].startswith("default")
    assert "ml" not in body["verification"]["scopes"]["allCells"]


def test_the_api_process_still_has_not_imported_torch():
    assert "torch" not in sys.modules
