"""HDF5 loading + preprocessing tests for the INSAT-3DR reader
(app/services/imagery/insat.py) and the nowcast sample builder
(ml/imagery/dataset.py).

Synthetic HDF5 files (written to a tmp dir, same layout as the real
product) cover the failure modes deterministically; the real-dataset tests
at the bottom run only when the MOSDAC files are present.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from app.services.imagery import insat  # noqa: E402
from app.services.imagery.nowcast_common import N_INPUT_FRAMES  # noqa: E402

ROWS, COLS = insat.EXPECTED_SHAPE


def _name(when: datetime) -> str:
    return f"3RIMG_{when:%d%b%Y}_{when:%H%M}_L1C_ASIA_MER_V01R00.h5".upper().replace(".H5", ".h5")


def write_frame(directory, when: datetime, *, value: int = 600, shape=(ROWS, COLS), channels=("TIR1", "WV"), fill_block=False):
    """A structurally faithful stand-in for one L1C file: 10-bit counts plus
    a 1024-entry count->kelvin table that decreases with count, like the
    real one."""
    path = directory / _name(when)
    with h5py.File(path, "w") as handle:
        for channel in channels:
            counts = np.full((1, *shape), value, dtype=np.uint16)
            counts[0, :8, :8] = np.arange(64, dtype=np.uint16).reshape(8, 8) + 400
            if fill_block:
                counts[0, 100:110, 100:110] = 1023
            dataset = handle.create_dataset(f"IMG_{channel}", data=counts, compression="gzip")
            dataset.attrs["_FillValue"] = np.array([1023], dtype=np.uint16)
            handle.create_dataset(f"IMG_{channel}_TEMP", data=np.linspace(340.0, 180.0, 1024, dtype=np.float32))
    return path


T0 = datetime(2025, 1, 7, 12, 15, tzinfo=timezone.utc)
HALF_HOUR = timedelta(minutes=30)


# --- filenames / timestamps -------------------------------------------------
def test_parse_timestamp_from_real_filename_pattern():
    assert insat.parse_timestamp("3RIMG_07JAN2025_1445_L1C_ASIA_MER_V01R00.h5") == datetime(2025, 1, 7, 14, 45, tzinfo=timezone.utc)


@pytest.mark.parametrize("name", [
    "3RIMG_07JAN2025_1445_L1C_ASIA_MER_V01R00.h5.part",  # incomplete download
    "3RIMG_07XXX2025_1445_L1C_ASIA_MER_V01R00.h5",       # not a month
    "3RIMG_07JAN2025_2575_L1C_ASIA_MER_V01R00.h5",       # not a time
    "something_else.h5",
])
def test_parse_timestamp_rejects_non_frames(name):
    assert insat.parse_timestamp(name) is None


# --- geolocation -----------------------------------------------------------
def test_sector_corners_map_to_grid_corners():
    """The product states its own corner coordinates; the projection maths
    must reproduce them (to well under a pixel)."""
    assert insat.latlon_to_pixel(45.5, 44.5) == pytest.approx((0.0, 0.0), abs=1e-3)
    assert insat.latlon_to_pixel(-10.0, 110.0) == pytest.approx((ROWS - 1, COLS - 1), abs=1e-3)


def test_pixel_latlon_round_trip():
    for lat, lon in [(28.6, 77.2), (-5.0, 50.0), (40.0, 105.0)]:
        row, col = insat.latlon_to_pixel(lat, lon)
        assert insat.pixel_to_latlon(row, col) == pytest.approx((lat, lon), abs=1e-6)


def test_region_windows_are_inside_the_grid_and_contain_delhi():
    delhi_row, delhi_col = insat.latlon_to_pixel(28.6, 77.2)
    for key in insat.REGIONS:
        window = insat.region_window(key)
        assert 0 <= window.row0 < window.row1 <= ROWS
        assert 0 <= window.col0 < window.col1 <= COLS
        assert window.row0 <= delhi_row < window.row1 and window.col0 <= delhi_col < window.col1
    assert insat.region_window("full_sector").shape == (ROWS, COLS)


def test_unknown_region_and_out_of_sector_bounds_are_rejected():
    with pytest.raises(insat.InsatFrameError):
        insat.region_window("mars")
    with pytest.raises(insat.InsatFrameError):
        insat.bounds_to_window((60.0, 120.0, 70.0, 130.0))


# --- reading / calibration -------------------------------------------------
def test_read_bt_applies_this_files_own_lookup_table(tmp_path):
    path = write_frame(tmp_path, T0)
    window = insat.PixelWindow(0, 16, 0, 20)
    counts, lut, fill = insat.read_counts(path, "TIR1", window)
    bt = insat.read_bt(path, "TIR1", window)
    assert counts.shape == bt.shape == (16, 20)
    assert bt.dtype == np.float32
    assert fill == 1023
    np.testing.assert_array_equal(bt, lut[counts])
    assert 180.0 <= bt.min() and bt.max() <= 340.0


def test_fill_pixels_become_nan_not_a_temperature(tmp_path):
    path = write_frame(tmp_path, T0, fill_block=True)
    bt = insat.read_bt(path, "TIR1", insat.PixelWindow(96, 116, 96, 116))
    assert np.isnan(bt[4:14, 4:14]).all()
    assert np.isfinite(bt[:4]).all()


def test_read_rejects_unknown_channel(tmp_path):
    path = write_frame(tmp_path, T0)
    with pytest.raises(insat.InsatFrameError):
        insat.read_counts(path, "XRAY")


def test_original_file_is_not_modified_by_reading(tmp_path):
    path = write_frame(tmp_path, T0)
    before = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
    insat.validate_file(path)
    insat.read_bt(path, "TIR1")
    insat.read_bt(path, "WV", insat.PixelWindow(0, 8, 0, 8))
    assert (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes()) == before


# --- corruption handling -----------------------------------------------------
def test_truncated_file_is_reported_corrupt(tmp_path):
    path = write_frame(tmp_path, T0)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    assert insat.validate_file(path) is not None


def test_garbage_file_is_reported_corrupt(tmp_path):
    path = tmp_path / _name(T0)
    path.write_bytes(b"this is not an HDF5 file" * 100)
    assert insat.validate_file(path) is not None


def test_missing_channel_and_wrong_grid_are_reported(tmp_path):
    assert "missing dataset" in insat.validate_file(write_frame(tmp_path, T0, channels=("TIR1",)))
    assert "unexpected grid" in insat.validate_file(write_frame(tmp_path, T0 + HALF_HOUR, shape=(64, 64)))


# --- catalog ---------------------------------------------------------------
def test_catalog_orders_frames_flags_corrupt_and_ignores_other_files(tmp_path):
    write_frame(tmp_path, T0 + HALF_HOUR)
    write_frame(tmp_path, T0)
    bad = write_frame(tmp_path, T0 + 2 * HALF_HOUR)
    bad.write_bytes(bad.read_bytes()[:2000])
    (tmp_path / (_name(T0 + 3 * HALF_HOUR) + ".part")).write_bytes(b"x" * 100)
    (tmp_path / "notes.h5").write_bytes(b"x")

    frames = insat.scan_catalog(tmp_path, use_cache=False)
    assert [f.timestamp for f in frames] == [T0, T0 + HALF_HOUR, T0 + 2 * HALF_HOUR]
    assert [f.valid for f in frames] == [True, True, False]
    assert frames[2].error
    assert frames[0].frame_id == "20250107T1215Z"


def test_catalog_of_missing_directory_is_empty(tmp_path):
    assert insat.scan_catalog(tmp_path / "nope", use_cache=False) == []


def test_input_sequence_requires_strictly_consecutive_frames(tmp_path):
    for step in (0, 1, 2, 4):  # 3 is missing
        write_frame(tmp_path, T0 + step * HALF_HOUR)
    frames = insat.scan_catalog(tmp_path, use_cache=False)
    by_id = {f.frame_id: f for f in frames}

    sequence = insat.input_sequence(by_id["20250107T1315Z"], N_INPUT_FRAMES, tmp_path)
    assert [f.timestamp for f in sequence] == [T0, T0 + HALF_HOUR, T0 + 2 * HALF_HOUR]

    with pytest.raises(insat.InsatFrameError, match="missing or corrupt"):
        insat.input_sequence(by_id["20250107T1415Z"], N_INPUT_FRAMES, tmp_path)  # needs the absent 13:45
    with pytest.raises(insat.InsatFrameError):
        insat.input_sequence(by_id["20250107T1215Z"], N_INPUT_FRAMES, tmp_path)  # nothing before it


# --- sample construction / leakage -------------------------------------------
def test_no_frame_is_shared_between_train_val_and_test():
    """The leakage guarantee: build samples from the REAL frame timeline and
    the REAL split boundaries, and check that every frame a sample touches
    (inputs and targets) belongs to that sample's own split."""
    from ml.imagery import config
    from ml.imagery.dataset import build_anchors

    start = datetime(2025, 1, 6, 13, 45, tzinfo=timezone.utc)
    # every half-hour slot, minus a few gaps like the real data has
    timestamps = [start + i * HALF_HOUR for i in range(69) if i not in (3, 18, 24, 27, 30, 31, 35, 36, 37, 39)]
    splits = [config.split_of(t) for t in timestamps]
    anchors = build_anchors(timestamps, splits)

    used = {"train": set(), "val": set(), "test": set()}
    for anchor in anchors:
        frames = list(anchor.inputs) + [t for t in anchor.targets if t >= 0]
        assert {splits[f] for f in frames} == {anchor.split}
        # inputs are strictly consecutive half-hours
        times = [timestamps[f] for f in anchor.inputs]
        assert all(b - a == HALF_HOUR for a, b in zip(times, times[1:]))
        used[anchor.split].update(frames)

    assert all(used.values()), "each split must contain at least one sample"
    assert not (used["train"] & used["val"])
    assert not (used["train"] & used["test"])
    assert not (used["val"] & used["test"])
    # chronological: everything in train precedes validation precedes test
    assert max(timestamps[f] for f in used["train"]) < min(timestamps[f] for f in used["val"])
    assert max(timestamps[f] for f in used["val"]) < min(timestamps[f] for f in used["test"])


def test_missing_target_drops_only_that_lead_time():
    from ml.imagery.dataset import build_anchors

    timestamps = [T0 + i * HALF_HOUR for i in (0, 1, 2, 4, 5)]  # slot 3 missing
    anchors = build_anchors(timestamps, ["train"] * len(timestamps))
    first = next(a for a in anchors if a.index == 2)
    assert first.targets == (-1, 3, 4)  # +30 min absent, +60/+90 present


# --- the real dataset (skipped when the MOSDAC files are not on disk) ----------
REAL_AVAILABLE = insat.DEFAULT_INSAT_DIR.exists() and any(insat.DEFAULT_INSAT_DIR.rglob("*.h5"))
needs_real = pytest.mark.skipif(not REAL_AVAILABLE, reason="INSAT-3DR MOSDAC files not present")


@needs_real
def test_real_dataset_catalog_matches_the_inspection_report():
    frames = insat.scan_catalog(insat.DEFAULT_INSAT_DIR)
    assert len(frames) == 63
    assert sum(f.valid for f in frames) == 59
    corrupt = sorted(f.frame_id for f in frames if not f.valid)
    assert corrupt == ["20250107T0715Z", "20250107T0745Z", "20250107T0815Z", "20250107T0915Z"]
    assert len({f.timestamp for f in frames}) == len(frames)  # no duplicate times


@needs_real
def test_real_frame_reads_as_plausible_brightness_temperature():
    frame = insat.get_frame("20250107T1915Z", insat.DEFAULT_INSAT_DIR)
    window = insat.region_window("north_india")
    for channel in ("TIR1", "WV"):
        bt = insat.read_bt(frame.path, channel, window)
        assert bt.shape == window.shape
        assert np.isfinite(bt).all()
        assert 170.0 < bt.min() and bt.max() < 340.0


@needs_real
def test_real_files_carry_per_file_calibration_tables():
    a = insat.get_frame("20250107T1915Z", insat.DEFAULT_INSAT_DIR)
    b = insat.get_frame("20250106T1345Z", insat.DEFAULT_INSAT_DIR)
    tiny = insat.PixelWindow(0, 2, 0, 2)
    _, lut_a, _ = insat.read_counts(a.path, "TIR1", tiny)
    _, lut_b, _ = insat.read_counts(b.path, "TIR1", tiny)
    assert lut_a.shape == lut_b.shape == (1024,)
    assert not np.array_equal(lut_a, lut_b), "calibration must be applied per file, not once globally"
