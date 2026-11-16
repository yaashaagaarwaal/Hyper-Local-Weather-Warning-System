"""Model, inference-engine and baseline tests for the INSAT-3DR nowcast
(app/services/imagery/nowcast_model.py, ml_inference.py, nowcast_common.py).

These import PyTorch, so they live OUTSIDE `tests/` and run in their own
process: PyTorch and XGBoost cannot share a process on macOS (see
app/services/imagery/nowcast_worker.py), and the main suite exercises the
XGBoost models. `tests/test_nowcast_isolation.py` runs this directory as a
subprocess, so a plain `pytest` still covers it; to run it directly:

    pytest tests_torch -q

Runs on CPU for determinism. Tests that need the trained artifact are
skipped if it has not been produced yet. No database, no conftest.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.services.imagery import ml_inference  # noqa: E402
from app.services.imagery.nowcast_common import (  # noqa: E402
    COLD_CLOUD_THRESHOLD_K,
    LEAD_STEPS,
    N_INPUT_FRAMES,
    cold_cloud_metrics,
    laplace_cdf,
    mask_metrics_from_counts,
    optical_flow_forecast,
    regression_metrics,
)
from app.services.imagery.nowcast_model import InsatNowcastNet, nowcast_loss, tiled_forward  # noqa: E402

ARTIFACT = ml_inference.MODEL_DIR / ml_inference.CHECKPOINT_NAME
needs_artifact = pytest.mark.skipif(not ARTIFACT.exists(), reason="trained model artifact not present")


@pytest.fixture(autouse=True)
def _cpu_and_clean_cache(monkeypatch):
    monkeypatch.setenv("NOWCAST_DEVICE", "cpu")
    monkeypatch.delenv("NOWCAST_MODEL_DIR", raising=False)
    ml_inference.clear_cache()
    yield
    ml_inference.clear_cache()


def plausible_frames(height=64, width=96, seed=0) -> np.ndarray:
    """(T, C, H, W) kelvin: smooth blobs, TIR1 warmer than WV, like real data."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(height, width)).astype(np.float32)
    import cv2

    base = cv2.GaussianBlur(base, (0, 0), 6)
    base = (base - base.min()) / (base.max() - base.min())
    frames = []
    for t in range(N_INPUT_FRAMES):
        shifted = np.roll(base, 2 * t, axis=1)
        frames.append(np.stack([300.0 - 90.0 * shifted, 255.0 - 40.0 * shifted]))
    return np.stack(frames).astype(np.float32)


# --- architecture -------------------------------------------------------------
def test_forward_shapes_types_and_positive_uncertainty():
    model = InsatNowcastNet().eval()
    x = torch.randn(2, N_INPUT_FRAMES, 2, 32, 48)
    with torch.no_grad():
        mean, scale = model(x)
    assert mean.shape == scale.shape == (2, len(LEAD_STEPS), 32, 48)
    assert mean.dtype == scale.dtype == torch.float32
    assert bool((scale > 0).all())


def test_untrained_network_is_exactly_persistence():
    """Zero-initialised head => predicted change is zero => every lead
    equals the last observed TIR1 frame. Training starts from the baseline."""
    model = InsatNowcastNet().eval()
    x = torch.randn(1, N_INPUT_FRAMES, 2, 32, 32)
    with torch.no_grad():
        mean, _ = model(x)
    for lead in range(len(LEAD_STEPS)):
        torch.testing.assert_close(mean[0, lead], x[0, -1, 0])


@pytest.mark.parametrize("shape", [(1, 3, 2, 30, 32), (1, 3, 5, 32, 32), (3, 2, 32, 32)])
def test_forward_rejects_bad_tensor_layouts(shape):
    with pytest.raises(ValueError):
        InsatNowcastNet()(torch.zeros(shape))


def test_tiled_inference_matches_single_pass():
    torch.manual_seed(0)
    model = InsatNowcastNet(base=8).eval()
    for parameter in model.head.parameters():  # make the output non-trivial
        torch.nn.init.normal_(parameter, std=0.05)
    x = np.random.default_rng(0).normal(size=(N_INPUT_FRAMES, 2, 200, 240)).astype(np.float32)
    with torch.no_grad():
        whole_mean, whole_scale = model(torch.from_numpy(x)[None])
    tiled_mean, tiled_scale = tiled_forward(model, x, torch.device("cpu"), tile=128, halo=48)
    np.testing.assert_allclose(tiled_mean, whole_mean[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(tiled_scale, whole_scale[0].numpy(), atol=1e-5)


def test_tiled_inference_handles_sizes_not_divisible_by_four():
    model = InsatNowcastNet(base=8).eval()
    x = np.zeros((N_INPUT_FRAMES, 2, 70, 93), dtype=np.float32)
    mean, scale = tiled_forward(model, x, torch.device("cpu"))
    assert mean.shape == scale.shape == (len(LEAD_STEPS), 70, 93)


def test_loss_ignores_masked_leads_and_is_finite():
    mean = torch.zeros(1, 3, 8, 8, requires_grad=True)
    scale = torch.ones(1, 3, 8, 8)
    target = torch.zeros(1, 3, 8, 8)
    target[:, 2] = 1000.0  # a "missing" lead filled with rubbish
    mask = torch.ones(1, 3, 8, 8)
    mask[:, 2] = 0.0
    loss, l1, _ = nowcast_loss(mean, scale, target, mask)
    assert torch.isfinite(loss)
    assert float(l1) == 0.0  # the rubbish lead contributed nothing
    all_masked, _, _ = nowcast_loss(mean, scale, target, torch.zeros_like(mask))
    assert torch.isfinite(all_masked)


# --- input validation ("invalid images") -----------------------------------------
@pytest.mark.parametrize("bad, message", [
    (np.zeros((64, 64), dtype=np.float32), "4-D"),
    (np.full((2, 2, 64, 64), 280.0, dtype=np.float32), "consecutive frames"),
    (np.full((3, 1, 64, 64), 280.0, dtype=np.float32), "channels"),
    (np.full((3, 2, 16, 64), 280.0, dtype=np.float32), "too small"),
    (np.full((3, 2, 64, 64), 128, dtype=np.uint8), "floating-point"),
    (np.full((3, 2, 64, 64), 128.0, dtype=np.float32), "plausible"),        # an 8-bit picture, not kelvin
    (np.full((3, 2, 64, 64), 700.0, dtype=np.float32), "plausible"),        # raw counts, not kelvin
])
def test_invalid_inputs_are_rejected_with_a_clear_reason(bad, message):
    with pytest.raises(ml_inference.InvalidNowcastInputError, match=message):
        ml_inference.validate_frames(bad)


def test_mostly_missing_and_infinite_inputs_are_rejected():
    frames = plausible_frames()
    holes = frames.copy()
    holes[:, :, :40] = np.nan
    with pytest.raises(ml_inference.InvalidNowcastInputError, match="missing data"):
        ml_inference.validate_frames(holes)
    infinite = frames.copy()
    infinite[0, 0, 0, 0] = np.inf
    with pytest.raises(ml_inference.InvalidNowcastInputError, match="infinite"):
        ml_inference.validate_frames(infinite)


def test_a_few_missing_pixels_are_tolerated():
    frames = plausible_frames()
    frames[0, 0, 5, 5] = np.nan
    ml_inference.validate_frames(frames)
    normalised = ml_inference.normalise(frames, np.array([273.0, 245.0], dtype=np.float32), np.array([20.0, 10.0], dtype=np.float32))
    assert np.isfinite(normalised).all()
    assert normalised.dtype == np.float32


# --- model artifact: loading, caching, failure modes --------------------------------
@needs_artifact
def test_trained_model_loads_and_is_cached():
    first = ml_inference.load_model()
    assert ml_inference.load_model() is first  # cached, not reloaded
    assert str(first.device) == "cpu"
    assert first.mean.shape == first.std.shape == (2,)
    assert first.size_bytes == ARTIFACT.stat().st_size
    assert ml_inference.load_model(force_reload=True) is not first


@needs_artifact
def test_status_reports_version_metrics_and_device():
    status = ml_inference.model_status()
    assert status["available"] is True
    assert status["name"] == "insat_nowcast"
    assert status["version"]
    assert status["device"] == "cpu"
    assert status["parameters"] > 0
    assert status["artifactSizeBytes"] == ARTIFACT.stat().st_size
    assert set(status["testMetrics"]) == {"30", "60", "90"}
    for lead in status["testMetrics"].values():
        assert {"model", "opticalFlow", "persistence", "skill"} <= set(lead)
    assert status["limitations"]


@needs_artifact
def test_checkpoint_and_metadata_agree_and_frames_are_split_disjointly():
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    metadata = json.loads((ml_inference.MODEL_DIR / ml_inference.METADATA_NAME).read_text())
    assert metadata["input"]["normalisation"] == checkpoint["normalisation"]
    assert metadata["version"] == checkpoint["model_version"]
    assert metadata["artifactSizeBytes"] == ARTIFACT.stat().st_size
    assert set(metadata["frameSplits"].values()) == {"train", "val", "test"}


def test_missing_artifact_is_unavailable_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    with pytest.raises(ml_inference.ModelUnavailableError, match="not found"):
        ml_inference.load_model()
    status = ml_inference.model_status()
    assert status["available"] is False
    assert "not found" in status["reason"]
    assert status["classicalFallback"]


def test_corrupt_artifact_is_unavailable_not_a_crash(tmp_path, monkeypatch):
    (tmp_path / ml_inference.CHECKPOINT_NAME).write_bytes(b"\x00not a torch checkpoint" * 500)
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    with pytest.raises(ml_inference.ModelUnavailableError, match="corrupt"):
        ml_inference.load_model()
    assert ml_inference.model_status()["available"] is False


@needs_artifact
def test_truncated_artifact_is_unavailable(tmp_path, monkeypatch):
    data = ARTIFACT.read_bytes()
    (tmp_path / ml_inference.CHECKPOINT_NAME).write_bytes(data[: len(data) // 3])
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    with pytest.raises(ml_inference.ModelUnavailableError):
        ml_inference.load_model()


def test_artifact_from_a_different_architecture_is_refused(tmp_path, monkeypatch):
    other = InsatNowcastNet(base=8)
    torch.save({
        "state_dict": other.state_dict(),
        "model_config": {"in_channels": 2, "base": 16, "n_leads": 3, "min_scale": 1e-3},  # does not match the weights
        "normalisation": {"mean": [273.0, 245.0], "std": [20.0, 10.0]},
    }, tmp_path / ml_inference.CHECKPOINT_NAME)
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    with pytest.raises(ml_inference.ModelUnavailableError, match="architecture"):
        ml_inference.load_model()


def test_artifact_with_bad_normalisation_is_refused(tmp_path, monkeypatch):
    model = InsatNowcastNet(base=8)
    torch.save({
        "state_dict": model.state_dict(), "model_config": model.config(),
        "normalisation": {"mean": [273.0, 245.0], "std": [20.0, 0.0]},
    }, tmp_path / ml_inference.CHECKPOINT_NAME)
    monkeypatch.setenv("NOWCAST_MODEL_DIR", str(tmp_path))
    with pytest.raises(ml_inference.ModelUnavailableError, match="normalisation"):
        ml_inference.load_model()


def test_device_selection_honours_override_and_falls_back_to_cpu(monkeypatch):
    assert str(ml_inference.select_device(torch)) == "cpu"  # forced by the fixture
    monkeypatch.delenv("NOWCAST_DEVICE")
    assert str(ml_inference.select_device(torch)) in {"cpu", "mps", "cuda"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(ml_inference.select_device(torch)) == "cpu"


# --- inference ---------------------------------------------------------------
@needs_artifact
@pytest.mark.parametrize("height, width", [(64, 96), (70, 93)])
def test_prediction_output_dimensions_types_and_ranges(height, width):
    prediction = ml_inference.predict(plausible_frames(height, width))
    assert prediction.mean_k.shape == prediction.scale_k.shape == (len(LEAD_STEPS), height, width)
    assert prediction.mean_k.dtype == prediction.scale_k.dtype == np.float32
    assert np.isfinite(prediction.mean_k).all()
    assert (prediction.scale_k > 0).all()
    assert 150.0 < prediction.mean_k.min() and prediction.mean_k.max() < 350.0  # still a temperature
    assert prediction.device == "cpu"
    assert prediction.inference_ms > 0


@needs_artifact
def test_prediction_is_deterministic():
    frames = plausible_frames()
    a, b = ml_inference.predict(frames), ml_inference.predict(frames)
    np.testing.assert_array_equal(a.mean_k, b.mean_k)
    np.testing.assert_array_equal(a.scale_k, b.scale_k)


@needs_artifact
def test_uncertainty_grows_with_lead_time():
    prediction = ml_inference.predict(plausible_frames(96, 96))
    means = [float(prediction.scale_k[k].mean()) for k in range(len(LEAD_STEPS))]
    assert means[0] < means[1] < means[2]


@needs_artifact
def test_predict_rejects_invalid_images_before_running_the_model():
    with pytest.raises(ml_inference.InvalidNowcastInputError):
        ml_inference.predict(np.full((3, 2, 64, 64), 128.0, dtype=np.float32))


# --- classical baseline + metrics ----------------------------------------------
def test_optical_flow_baseline_tracks_a_moving_pattern_better_than_persistence():
    import cv2

    yy, xx = np.mgrid[0:128, 0:160].astype(np.float32)

    def scene(shift):
        blob = np.exp(-(((xx - 50 - shift) ** 2) + ((yy - 64) ** 2)) / (2 * 14.0 ** 2))
        return (295.0 - 80.0 * cv2.GaussianBlur(blob, (0, 0), 2)).astype(np.float32)

    prev, last, truth = scene(0), scene(4), scene(8)
    forecast = optical_flow_forecast(prev, last, lead_steps=(1,))[0]
    assert forecast.shape == last.shape and forecast.dtype == np.float32
    assert np.abs(forecast - truth).mean() < np.abs(last - truth).mean()


def test_regression_and_mask_metrics_on_known_values():
    truth = np.full((4, 4), 250.0, dtype=np.float32)
    truth[:2] = 220.0  # 8 cold pixels
    pred = truth.copy()
    pred[0] = 240.0    # 4 cold pixels missed
    reg = regression_metrics(pred, truth)
    assert reg["maeK"] == pytest.approx(4 * 20.0 / 16)
    assert reg["pixels"] == 16
    cold = cold_cloud_metrics(pred, truth, COLD_CLOUD_THRESHOLD_K)
    assert (cold["tp"], cold["fp"], cold["fn"], cold["tn"]) == (4, 0, 4, 8)
    assert cold["precision"] == 1.0
    assert cold["recall"] == 0.5
    assert cold["iou"] == 0.5
    assert cold["dice"] == pytest.approx(2 / 3)


def test_undefined_mask_metrics_are_none_not_a_made_up_number():
    empty = mask_metrics_from_counts({"tp": 0, "fp": 0, "fn": 0, "tn": 10})
    assert empty["iou"] is None and empty["precision"] is None and empty["recall"] is None
    assert empty["accuracy"] == 1.0


def test_laplace_probability_is_a_valid_monotonic_probability():
    mean = np.array([200.0, 235.0, 270.0], dtype=np.float32)
    prob = laplace_cdf(235.0, mean, np.full(3, 5.0, dtype=np.float32))
    assert prob[0] > 0.99 and prob[1] == pytest.approx(0.5) and prob[2] < 0.01
    assert ((prob >= 0) & (prob <= 1)).all()
