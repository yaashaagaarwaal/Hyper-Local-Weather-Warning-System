"""Network + inference-engine tests for the flood-mapping and
lightning-detection models (hazard_models.py, hazard_inference.py).

Imports PyTorch, so it lives in `tests_torch/` and runs in its own process
(see tests/test_nowcast_isolation.py). CPU only. No database, no conftest.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.services.imagery import flood_data, hazard_inference  # noqa: E402
from app.services.imagery import lightning_data as ld  # noqa: E402
from app.services.imagery.hazard_common import HAZARD_MODEL_DIR  # noqa: E402
from app.services.imagery.hazard_models import SmallUNet, count_parameters, segmentation_loss  # noqa: E402
from app.services.imagery.ml_inference import ModelUnavailableError  # noqa: E402

FLOOD_ARTIFACT = HAZARD_MODEL_DIR / hazard_inference.SPECS["flood"]["checkpoint"]
LIGHTNING_ARTIFACT = HAZARD_MODEL_DIR / hazard_inference.SPECS["lightning"]["checkpoint"]
needs_flood = pytest.mark.skipif(not FLOOD_ARTIFACT.exists(), reason="trained flood artifact not present")
needs_lightning = pytest.mark.skipif(not LIGHTNING_ARTIFACT.exists(), reason="trained lightning artifact not present")


@pytest.fixture(autouse=True)
def _cpu_and_clean_cache(monkeypatch):
    monkeypatch.setenv("NOWCAST_DEVICE", "cpu")
    monkeypatch.delenv("HAZARD_MODEL_DIR", raising=False)
    hazard_inference.clear_cache()
    yield
    hazard_inference.clear_cache()


def radar(height=128, width=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s1 = np.stack([rng.normal(-9, 2, (height, width)), rng.normal(-16, 2, (height, width))]).astype(np.float32)
    s1[:, 30:80, 30:90] -= 11.0
    return s1


def infrared(height=128, width=128) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    storm = np.exp(-((yy - 60) ** 2 + (xx - 60) ** 2) / (2 * 14 ** 2))
    return np.stack([295 - 60 * storm, 295 - 95 * storm]).astype(np.float32)    # top cooled sharply in 30 min


def save_checkpoint(directory, key, **overrides):
    spec = hazard_inference.SPECS[key]
    model = SmallUNet(in_channels=overrides.pop("in_channels", 2), base=8, depth=2, out_stride=overrides.pop("out_stride", spec["out_stride"]))
    checkpoint = {"state_dict": model.state_dict(), "model_config": model.config(), "threshold": 0.5,
                  "normalisation": {"mean": [-9.0, -16.0], "std": [3.0, 4.0]}, "model_name": key, "model_version": "test"}
    checkpoint.update(overrides)
    torch.save(checkpoint, directory / spec["checkpoint"])


# --- architecture ---------------------------------------------------------------
@pytest.mark.parametrize("shape", [(64, 64), (96, 128), (70, 90), (257, 131)])
def test_full_resolution_output_matches_any_input_size(shape):
    model = SmallUNet(in_channels=2, base=8, depth=3).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 2, *shape))
    assert out.shape == (2, 1, *shape) and out.dtype == torch.float32 and bool(torch.isfinite(out).all())


@pytest.mark.parametrize("shape, expected", [((64, 64), (32, 32)), ((128, 96), (64, 48)), ((129, 63), (65, 32))])
def test_half_resolution_output_for_the_lightning_grid(shape, expected):
    model = SmallUNet(in_channels=2, base=8, depth=3, out_stride=2).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 2, *shape)).shape == (1, 1, *expected)


def test_prediction_for_one_image_does_not_depend_on_its_batch_mates():
    """GroupNorm, not BatchNorm: serving one image must equal training-time behaviour."""
    model = SmallUNet(in_channels=2, base=8, depth=3).eval()
    x = torch.randn(3, 2, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x)[1], model(x[1:2])[0], atol=1e-5)


def test_config_round_trips_and_bad_arguments_are_rejected():
    model = SmallUNet(in_channels=2, base=8, depth=3, out_stride=2, dropout=0.0)
    clone = SmallUNet(**model.config())
    clone.load_state_dict(model.state_dict())
    assert count_parameters(clone) == count_parameters(model) > 0
    with pytest.raises(ValueError):
        SmallUNet(in_channels=2, out_stride=4)
    with pytest.raises(ValueError):
        SmallUNet(in_channels=2, depth=1)


def test_loss_ignores_invalid_pixels_and_trains():
    torch.manual_seed(0)
    logits = torch.randn(2, 32, 32, requires_grad=True)
    target = (torch.rand(2, 32, 32) > 0.8).float()
    valid = torch.ones(2, 32, 32)
    valid[:, :8] = 0
    base = segmentation_loss(logits, target, valid)
    poisoned = logits.detach().clone()
    poisoned[:, :8] = 50.0                                   # garbage where there is no label
    assert torch.isclose(segmentation_loss(poisoned, target, valid), base, atol=1e-5)
    base.backward()
    assert torch.all(logits.grad[:, :8] == 0) and bool(torch.isfinite(logits.grad).all())
    assert torch.isfinite(segmentation_loss(logits, target, torch.zeros(2, 32, 32)))         # nothing labelled: no NaN
    assert segmentation_loss(logits, target, valid, pos_weight=4.0) > base                   # rare class up-weighted

    model = SmallUNet(in_channels=2, base=8, depth=2, dropout=0.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.from_numpy(flood_data.normalise(radar(64, 64), np.array([-9, -16], np.float32), np.array([3, 4], np.float32)))[None]
    y = torch.zeros(1, 64, 64)
    y[:, 30:64, 30:64] = 1
    losses = []
    for _ in range(25):
        loss = segmentation_loss(model(x)[:, 0], y, torch.ones(1, 64, 64))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7


# --- engine: loading ---------------------------------------------------------------
@pytest.mark.parametrize("key", ["flood", "lightning"])
def test_missing_artifact_is_reported_not_raised(key, tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    status = hazard_inference.model_status(key)
    assert status["available"] is False and "not found" in status["reason"] and status["classicalFallback"]
    with pytest.raises(ModelUnavailableError):
        hazard_inference.load_model(key)


def test_unknown_model_key():
    assert hazard_inference.model_status("tornado")["available"] is False
    with pytest.raises(ModelUnavailableError):
        hazard_inference.load_model("tornado")


@pytest.mark.parametrize("payload", [b"", b"not a checkpoint", b"\x80\x04" + b"\x00" * 200])
def test_corrupt_artifact_is_unavailable(payload, tmp_path, monkeypatch):
    (tmp_path / hazard_inference.SPECS["flood"]["checkpoint"]).write_bytes(payload)
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    with pytest.raises(ModelUnavailableError) as exc:
        hazard_inference.load_model("flood")
    assert "corrupt" in exc.value.reason


def test_artifacts_from_the_wrong_model_or_with_bad_numbers_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    for key, overrides, fragment in (
        ("flood", {"out_stride": 2}, "different input/output layout"),             # a lightning net saved as the flood model
        ("lightning", {"out_stride": 1}, "different input/output layout"),
        ("flood", {"in_channels": 5}, "different input/output layout"),
        ("flood", {"threshold": 1.5}, "invalid decision threshold"),
        ("flood", {"normalisation": {"mean": [0.0, 0.0], "std": [1.0, 0.0]}}, "invalid normalisation"),
        ("lightning", {"calibration": {"a": -1.0, "b": 0.0}}, "calibration"),
        ("flood", {"model_config": {"nonsense": True}}, "expected architecture"),
    ):
        hazard_inference.clear_cache()
        save_checkpoint(tmp_path, key, **overrides)
        with pytest.raises(ModelUnavailableError) as exc:
            hazard_inference.load_model(key)
        assert fragment in exc.value.reason, exc.value.reason


def test_model_is_cached_and_a_replaced_artifact_is_picked_up(tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    save_checkpoint(tmp_path, "flood")
    first = hazard_inference.load_model("flood")
    assert hazard_inference.load_model("flood") is first
    save_checkpoint(tmp_path, "flood", threshold=0.3, model_version="newer", padding=list(range(50)))
    second = hazard_inference.load_model("flood")
    assert second is not first and second.checkpoint["threshold"] == 0.3
    (tmp_path / hazard_inference.SPECS["flood"]["metadata"]).write_text("{not json")
    hazard_inference.clear_cache()
    assert hazard_inference.model_status("flood")["available"] is True                 # broken metadata never blocks serving


# --- engine: prediction ---------------------------------------------------------------
def test_flood_prediction_shape_range_and_handling_of_missing_radar(tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    save_checkpoint(tmp_path, "flood")
    s1 = radar(100, 140)
    s1[:, :20, :20] = np.nan
    result = hazard_inference.predict("flood", s1)
    assert result.probability.shape == (100, 140) and result.probability.dtype == np.float32
    assert np.isfinite(result.probability).all() and 0 <= result.probability.min() and result.probability.max() <= 1
    assert result.threshold == 0.5 and result.device == "cpu" and result.inference_ms > 0
    again = hazard_inference.predict("flood", s1)
    assert np.array_equal(result.probability, again.probability)                        # deterministic in eval mode


def test_lightning_prediction_is_on_the_half_grid_and_calibration_is_monotonic(tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))
    save_checkpoint(tmp_path, "lightning")
    raw = hazard_inference.predict("lightning", infrared())
    assert raw.probability.shape == (64, 64) and raw.threshold == 0.5
    hazard_inference.clear_cache()
    path = tmp_path / hazard_inference.SPECS["lightning"]["checkpoint"]
    checkpoint = torch.load(path, weights_only=True)             # same weights, calibration added
    checkpoint["calibration"] = {"a": 0.5, "b": -1.5}
    torch.save(checkpoint, path)
    calibrated = hazard_inference.predict("lightning", infrared())
    assert calibrated.threshold == pytest.approx(1 / (1 + np.exp(1.5)), abs=1e-6)          # sigmoid(0.5*logit(0.5) - 1.5)
    # same cells are flagged before and after calibration: only the stated confidence changes
    assert np.array_equal(raw.probability >= raw.threshold, calibrated.probability >= calibrated.threshold)
    order = np.argsort(raw.probability.ravel())
    assert np.all(np.diff(calibrated.probability.ravel()[order]) >= -1e-7)


@pytest.mark.parametrize("key, bad", [
    ("flood", np.zeros((3, 96, 96), np.float32) - 10),
    ("flood", np.full((2, 96, 96), 200.0, np.float32)),
    ("flood", np.zeros((2, 8, 8), np.float32) - 10),
    ("lightning", np.zeros((2, 64, 64), np.float32) + 25.0),
    ("lightning", np.zeros((4, 64, 64), np.float32) + 280.0),
])
def test_invalid_input_is_a_422_even_when_no_model_exists(key, bad, tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_MODEL_DIR", str(tmp_path))       # empty: validation must come first
    with pytest.raises((flood_data.InvalidFloodInputError, ld.InvalidLightningInputError)) as exc:
        hazard_inference.predict(key, bad)
    assert exc.value.status_code == 422


# --- the real trained artifacts -------------------------------------------------------
@needs_flood
def test_trained_flood_model_loads_and_finds_obvious_water():
    status = hazard_inference.model_status("flood")
    assert status["available"] is True and status["name"] == "flood_unet" and status["device"] == "cpu"
    assert status["artifactSizeBytes"] == FLOOD_ARTIFACT.stat().st_size and status["testMetrics"]["real"]["iou"] > 0
    s1 = np.stack([np.full((256, 256), -8.0), np.full((256, 256), -14.0)]).astype(np.float32)
    s1 += np.random.default_rng(0).normal(0, 1.5, s1.shape).astype(np.float32)
    s1[0, 64:192, 64:192] -= 14.0
    s1[1, 64:192, 64:192] -= 15.0                               # a large, very dark (calm-water) square
    result = hazard_inference.predict("flood", s1)
    mask = result.probability >= result.threshold
    assert mask[80:176, 80:176].mean() > 0.9 and mask[:40, :40].mean() < 0.1


@needs_flood
def test_flood_metadata_records_the_official_split_of_every_chip():
    metadata = json.loads((HAZARD_MODEL_DIR / hazard_inference.SPECS["flood"]["metadata"]).read_text())
    official = {c.chip_id: c.split for c in flood_data.real_catalog()}
    if official:
        assert metadata["chipSplits"] == official
    assert metadata["trainedOn"] in ("real", "real+synthetic", "synthetic") and 0 < metadata["threshold"] < 1
    assert "real" in metadata["testMetrics"] and "simulator" in metadata["syntheticMetrics"]["note"]


@needs_lightning
def test_trained_lightning_model_prefers_a_cooling_storm_to_clear_sky():
    status = hazard_inference.model_status("lightning")
    assert status["available"] is True and status["name"] == "lightning_unet"
    assert status["probabilityCalibration"].startswith("Platt")
    storm = hazard_inference.predict("lightning", infrared())
    clear = hazard_inference.predict("lightning", np.full((2, 128, 128), 296.0, np.float32))
    assert storm.probability.shape == (64, 64)
    assert storm.probability[26:34, 26:34].mean() > 10 * clear.probability.mean()
    assert clear.probability.max() < storm.threshold                      # clear sky: nothing flagged
