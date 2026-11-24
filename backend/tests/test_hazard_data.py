"""Flood / lightning data layer (torch-free): real-file readers, the
preprocessing shared by training and serving, baselines and metrics.
Tests that need the downloaded datasets are skipped when they are absent.
No DB.
"""
import io

import numpy as np
import pytest

from app.services.imagery import flood_data, hazard_common
from app.services.imagery import lightning_data as ld

REAL_FLOOD = (flood_data.REAL_DIR / "S1Hand").exists() and any((flood_data.REAL_DIR / "S1Hand").glob("*.tif"))
REAL_GOES = (ld.GOES_DIR / "abi").exists() and len(ld.abi_frames()) >= ld.LOOKBACK_FRAMES + 1
needs_flood = pytest.mark.skipif(not REAL_FLOOD, reason="Sen1Floods11 chips not present")
needs_goes = pytest.mark.skipif(not REAL_GOES, reason="GOES-16 files not present")
PROJ = {"semi_major_axis": 6378137.0, "semi_minor_axis": 6356752.31414, "perspective_point_height": 35786023.0,
        "longitude_of_projection_origin": -75.0}


def radar(height=96, width=96, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s1 = np.stack([rng.normal(-9, 2, (height, width)), rng.normal(-16, 2, (height, width))]).astype(np.float32)
    s1[:, 20:50, 20:60] -= 10.0          # a dark "water" patch
    return s1


# --- metrics ----------------------------------------------------------------
def test_confusion_counts_respect_the_valid_mask():
    pred = np.array([[1, 1], [0, 0]], bool)
    truth = np.array([[1, 0], [1, 0]], bool)
    assert hazard_common.confusion_counts(pred, truth) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    valid = np.array([[1, 0], [1, 1]], bool)
    assert hazard_common.confusion_counts(pred, truth, valid) == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}


def test_metrics_are_none_not_invented_when_undefined():
    empty = hazard_common.metrics_from_counts({"tp": 0, "fp": 0, "fn": 0, "tn": 10})
    assert empty["iou"] is None and empty["precision"] is None and empty["recall"] is None
    assert empty["accuracy"] == 1.0
    m = hazard_common.metrics_from_counts({"tp": 6, "fp": 2, "fn": 2, "tn": 10})
    assert m["iou"] == pytest.approx(0.6) and m["f1"] == pytest.approx(0.75) and m["falseAlarmRatio"] == pytest.approx(0.25)


def test_roc_pr_auc_perfect_random_and_degenerate():
    truth = np.array([0] * 500 + [1] * 500, bool)
    perfect = hazard_common.roc_pr_auc(truth.astype(float), truth)
    assert perfect["rocAuc"] == pytest.approx(1.0, abs=1e-3) and perfect["prAuc"] == pytest.approx(1.0, abs=1e-3)
    rng = np.random.default_rng(0)
    noise = hazard_common.roc_pr_auc(rng.random(20000), rng.random(20000) < 0.5)
    assert noise["rocAuc"] == pytest.approx(0.5, abs=0.02)
    assert hazard_common.roc_pr_auc(np.zeros(5), np.zeros(5, bool)) == {"rocAuc": None, "prAuc": None}


def test_platt_scaling_is_monotonic_so_rankings_never_change():
    p = np.linspace(0.001, 0.999, 200)
    q = hazard_common.apply_platt(p, 0.46, -1.57)
    assert np.all(np.diff(q) > 0) and q.min() > 0 and q.max() < 1
    assert hazard_common.apply_platt(0.0, 1.0, 0.0) > 0 and hazard_common.apply_platt(1.0, 1.0, 0.0) < 1   # no inf/nan at the ends


# --- flood: preprocessing / validation / baselines ---------------------------
def test_normalise_clips_and_fills_missing_radar_with_the_training_mean():
    s1 = radar()
    s1[:, :5, :5] = np.nan
    s1[0, 10, 10] = -80.0
    mean, std = np.array([-9.0, -16.0], np.float32), np.array([3.0, 4.0], np.float32)
    x = flood_data.normalise(s1, mean, std)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert np.all(x[:, :5, :5] == 0.0)
    assert x[0, 10, 10] == pytest.approx((flood_data.CLIP_DB[0] + 9.0) / 3.0)


def test_valid_mask_excludes_unlabelled_and_missing_pixels():
    s1 = radar()
    s1[1, 0, 0] = np.nan
    label = np.zeros(s1.shape[1:], np.int8)
    label[1, 1] = -1
    valid = flood_data.valid_mask(s1, label)
    assert not valid[0, 0] and not valid[1, 1] and valid[2, 2]


@pytest.mark.parametrize("bad, message", [
    (np.zeros((96, 96), np.float32), "3-D"),
    (np.zeros((3, 96, 96), np.float32) - 10, "2 radar bands"),
    (np.zeros((2, 16, 16), np.float32) - 10, "sides must be"),
    ((np.zeros((2, 96, 96)) - 10).astype(np.int16), "floating-point"),
    (np.full((2, 96, 96), np.nan, np.float32), "no radar data"),
    (np.full((2, 96, 96), 120.0, np.float32), "plausible"),
    (np.random.default_rng(0).uniform(0, 1, (2, 96, 96)).astype(np.float32), "linear power"),
])
def test_validate_s1_rejects_what_the_model_was_not_trained_on(bad, message):
    with pytest.raises(flood_data.InvalidFloodInputError) as exc:
        flood_data.validate_s1(bad)
    assert message in exc.value.message and exc.value.status_code == 422


def test_validate_s1_rejects_infinite_values():
    s1 = radar()
    s1[0, 0, 0] = np.inf
    with pytest.raises(flood_data.InvalidFloodInputError):
        flood_data.validate_s1(s1)


def test_threshold_and_otsu_find_the_dark_patch_and_never_flag_missing_pixels():
    s1 = radar()
    s1[:, 80:, 80:] = np.nan
    mask = flood_data.threshold_water(s1, "VH", -22.0)
    assert mask[20:50, 20:60].mean() > 0.9 and mask[60:75, 60:75].mean() < 0.05 and not mask[80:, 80:].any()
    otsu, threshold = flood_data.otsu_water(s1, "VH")
    assert -30 < threshold < -15 and otsu[20:50, 20:60].mean() > 0.9 and not otsu[80:, 80:].any()


def test_area_uses_the_real_footprint_and_is_none_without_georeferencing():
    bounds = {"south": 26.0, "north": 26.0 + 5.12 / 110.574, "west": 93.0, "east": 93.0 + 5.12 / (111.320 * np.cos(np.radians(26.023)))}
    assert flood_data.area_km2(512 * 512, bounds, (512, 512)) == pytest.approx(26.2144, rel=1e-3)   # 5.12 km x 5.12 km
    assert flood_data.area_km2(100, None, (512, 512)) is None


def test_upload_decoding_accepts_a_radar_geotiff_and_rejects_everything_else():
    tifffile = pytest.importorskip("tifffile")
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, radar(128, 128))
    array, bounds = flood_data.decode_s1_upload(buffer.getvalue())
    assert array.shape == (2, 128, 128) and bounds is None
    for junk in (b"not an image", b"\x89PNG\r\n\x1a\n" + b"0" * 64, b""):
        with pytest.raises(flood_data.InvalidFloodInputError):
            flood_data.decode_s1_upload(junk)
    photo = io.BytesIO()
    tifffile.imwrite(photo, np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(flood_data.InvalidFloodInputError):
        flood_data.decode_s1_upload(photo.getvalue())


# --- flood: the real dataset --------------------------------------------------
@needs_flood
def test_real_catalog_uses_the_official_split_lists_with_no_chip_in_two_splits():
    chips = flood_data.real_catalog()
    assert len(chips) == 138 and len({c.chip_id for c in chips}) == 138
    counts = {s: sum(c.split == s for c in chips) for s in ("train", "valid", "test")}
    assert counts == {"train": 80, "valid": 29, "test": 29}
    assert {c.region for c in chips} == {"India", "Pakistan", "Sri-Lanka"}
    assert all(c.source == "real" for c in chips)


@needs_flood
def test_real_chip_reads_have_the_documented_shapes_values_and_georeference():
    chip = next(c for c in flood_data.real_catalog() if c.region == "India" and c.split == "test")
    s1, label = flood_data.read_s1(chip), flood_data.read_label(chip)
    assert s1.shape == (2, 512, 512) and s1.dtype == np.float32
    assert label.shape == (512, 512) and set(np.unique(label)) <= {-1, 0, 1}
    assert flood_data.read_permanent_water(chip).dtype == bool and flood_data.read_otsu_reference(chip).shape == (512, 512)
    bounds = flood_data.chip_bounds(chip)
    assert 6 < bounds["south"] < bounds["north"] < 37 and 68 < bounds["west"] < bounds["east"] < 98      # inside India
    assert flood_data.area_km2(512 * 512, bounds, (512, 512)) == pytest.approx(26.2, rel=0.15)             # ~5.1 km square


@needs_flood
def test_reading_never_modifies_the_original_files():
    chip = flood_data.real_catalog()[0]
    before = (chip.path.stat().st_size, chip.path.stat().st_mtime_ns)
    flood_data.read_s1(chip), flood_data.read_label(chip), flood_data.chip_bounds(chip)
    assert (chip.path.stat().st_size, chip.path.stat().st_mtime_ns) == before


def test_unknown_flood_sample_is_a_404():
    with pytest.raises(flood_data.FloodDataError) as exc:
        flood_data.get_chip("no-such-chip")
    assert exc.value.status_code == 404


# --- lightning: projection, grids, splits -------------------------------------
def test_goes_projection_round_trips():
    lat, lon = np.array([25.0, 40.0, 10.0]), np.array([-80.0, -100.0, -60.0])
    x, y = ld.goes_forward(lat, lon, PROJ)
    lat2, lon2 = ld.goes_inverse(x, y, PROJ)
    assert np.allclose(lat, lat2, atol=1e-6) and np.allclose(lon, lon2, atol=1e-6)
    off_earth = ld.goes_inverse(np.array([0.2]), np.array([0.2]), PROJ)
    assert np.isnan(off_earth[0]).all()


def test_to_input_grid_averages_2x2_and_propagates_missing():
    bt = np.arange(16, dtype=np.float32).reshape(4, 4)
    bt[0, 0] = np.nan
    out = ld.to_input_grid(bt)
    assert out.shape == (2, 2) and np.isnan(out[0, 0]) and out[1, 1] == pytest.approx(12.5)


def test_block_split_is_deterministic_disjoint_and_close_to_60_20_20():
    a, b = ld.block_split_map((375, 625)), ld.block_split_map((375, 625))
    assert np.array_equal(a, b) and set(np.unique(a)) == {0, 1, 2}
    shares = [(a == k).mean() for k in range(3)]
    assert shares[0] == pytest.approx(0.6, abs=0.06) and shares[1] == pytest.approx(0.2, abs=0.06)
    block = a[:ld.BLOCK_CELLS, :ld.BLOCK_CELLS]
    assert (block == block[0, 0]).all()                      # a block is never split between two sets


def test_time_splits_do_not_overlap_and_leave_a_gap_between_label_windows():
    sets = [set(ld.TRAIN_ANCHORS), set(ld.VAL_ANCHORS), set(ld.TEST_ANCHORS)]
    assert not (sets[0] & sets[1]) and not (sets[1] & sets[2]) and not (sets[0] & sets[2])
    # label windows are +/-5 min and frames are 5 min apart: adjacent anchors would share flashes
    assert min(ld.VAL_ANCHORS) - max(ld.TRAIN_ANCHORS) >= 2 and min(ld.TEST_ANCHORS) - max(ld.VAL_ANCHORS) >= 2
    assert min(ld.TRAIN_ANCHORS) >= ld.LOOKBACK_FRAMES


def test_features_cell_helpers_and_baseline():
    pair = np.full((2, 64, 64), 290.0, np.float32)
    pair[1, 10:20, 10:20] = 205.0
    pair[1, 0, 0] = np.nan
    x = ld.features(pair)
    assert x.shape == (2, 64, 64) and x.dtype == np.float32 and np.isfinite(x).all()
    assert x[0, 15, 15] == pytest.approx((205.0 - ld.BT_CENTRE_K) / ld.BT_SCALE_K)
    assert x[1, 15, 15] == pytest.approx((205.0 - 290.0) / ld.TREND_SCALE_K)       # cooled 85 K in 30 min
    valid = ld.cell_valid(pair)
    assert valid.shape == (32, 32) and not valid[0, 0] and valid[1, 1]
    flagged = ld.threshold_baseline(pair[1], 211.0)
    assert flagged[5:10, 5:10].all() and flagged.sum() == 25


@pytest.mark.parametrize("bad, message", [
    (np.zeros((64, 64), np.float32) + 280, "2 infrared images"),
    (np.zeros((3, 64, 64), np.float32) + 280, "2 infrared images"),
    (np.zeros((2, 8, 8), np.float32) + 280, "too small"),
    ((np.zeros((2, 64, 64)) + 280).astype(np.int32), "floating-point"),
    (np.full((2, 64, 64), np.nan, np.float32), "missing data"),
    (np.zeros((2, 64, 64), np.float32) + 20.0, "plausible"),       # Celsius, or an 8-bit picture
])
def test_validate_pair_rejects_what_the_model_was_not_trained_on(bad, message):
    with pytest.raises(ld.InvalidLightningInputError) as exc:
        ld.validate_pair(bad)
    assert message in exc.value.message and exc.value.status_code == 422


# --- lightning: the real GOES files -------------------------------------------
@needs_goes
def test_goes_catalog_is_time_ordered_five_minutes_apart():
    frames = ld.abi_frames()
    assert len(frames) == 24 and [f.index for f in frames] == list(range(24))
    gaps = {round((b.time - a.time).total_seconds()) for a, b in zip(frames, frames[1:])}
    assert gaps == {300}


@needs_goes
def test_real_flashes_land_under_cold_cloud_tops():
    """Independent check that GLM flashes and the infrared grid are aligned:
    cells with a flash must be far colder than cells without."""
    frame = ld.abi_frames()[ld.TEST_ANCHORS[0]]
    abi = ld.read_abi(frame.path)
    grid, points = ld.flash_grid_for(abi)
    assert abi["bt"].shape == (1500, 2500) and grid.shape == (375, 625) and int(grid.sum()) == len(points) > 1000
    coldest = ld.cell_min_bt(ld.to_input_grid(abi["bt"]))
    finite = np.isfinite(coldest)
    assert np.median(coldest[(grid > 0) & finite]) < np.median(coldest[(grid == 0) & finite]) - 40
    lat, lon = ld.cell_latlon(abi)
    assert np.nanmin(lon) < -110 and np.nanmax(lon) < -40          # the Americas, nowhere near India (68-98 E)


@needs_goes
def test_training_cache_equals_what_the_api_reads_from_the_raw_files():
    cache_path = hazard_common.BACKEND_ROOT / "ml" / "hazards" / "cache" / "goes_lightning.npz"
    if not cache_path.exists():
        pytest.skip("training cache not built")
    index = ld.TEST_ANCHORS[-1]
    abi = ld.read_abi(ld.abi_frames()[index].path)
    with np.load(cache_path) as cache:
        assert np.array_equal(cache["bt"][index], ld.to_input_grid(abi["bt"]).astype(np.float16), equal_nan=True)
        assert np.array_equal(cache["flashes"][index], ld.flash_grid_for(abi)[0])
        assert np.array_equal(cache["blocks"], ld.block_split_map(cache["flashes"][index].shape))
