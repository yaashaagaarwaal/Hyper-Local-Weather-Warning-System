"""India synthetic flood / lightning generators (ml/hazards/synth_*.py):
determinism, image-mask alignment, metadata, Indian geography, and that the
output is always tagged synthetic. Torch-free. No DB.
"""
import json

import numpy as np
import pytest

from app.services.imagery import flood_data
from app.services.imagery import lightning_data as ld
from ml.hazards import synth_flood, synth_lightning
from ml.hazards.india_geo import FLOOD_SITES, LIGHTNING_SITES, season_of

INDIA_BOX = (6.5, 37.5, 68.0, 97.5)   # south, north, west, east
REQUIRED_STATES = {
    "Uttar Pradesh", "Delhi", "Uttarakhand", "Himachal Pradesh", "Punjab", "Rajasthan", "Gujarat", "Maharashtra",
    "Bihar", "West Bengal", "Assam", "Odisha", "Karnataka", "Kerala", "Tamil Nadu", "Andhra Pradesh", "Telangana",
    "Madhya Pradesh", "Chhattisgarh", "Meghalaya", "Sikkim", "Arunachal Pradesh", "Nagaland",
}
# Stand-in for the fitted values so these tests need no downloaded data.
FLOOD_CALIBRATION = {"realIndiaTrain": {"chips": 0}, "scale": {"water": [1.0, 1.0], "land": [1.0, 1.0]},
                     "shift": {"water": [0.0, 0.0], "land": [0.0, 0.0]}}
LIGHTNING_CALIBRATION = {"flashRateMultiplier": 0.1}


# --- geography ----------------------------------------------------------------
@pytest.mark.parametrize("sites", [FLOOD_SITES, LIGHTNING_SITES])
def test_every_site_is_inside_india_and_all_requested_states_are_covered(sites):
    south, north, west, east = INDIA_BOX
    for site in sites:
        assert south <= site["lat"] <= north and west <= site["lon"] <= east, site["place"]
        assert all(1 <= m <= 12 for m in site["months"])
    assert REQUIRED_STATES <= {s["state"] for s in sites}
    assert {s["region"] for s in sites} == {"North", "West", "East", "South", "Central", "North-East"}


def test_requested_scenarios_are_all_present():
    flood = " | ".join(s["scenario"] for s in FLOOD_SITES)
    for phrase in ("Ganga and Yamuna", "Brahmaputra", "Bihar and eastern Uttar Pradesh", "Urban waterlogging",
                   "Coastal flooding", "Agricultural flooding", "Himalayan flash flood"):
        assert phrase in flood
    assert {s["state"] for s in FLOOD_SITES if s["kind"] == "urban"} >= {"Delhi", "Maharashtra", "Tamil Nadu", "Karnataka"}
    assert {s["state"] for s in FLOOD_SITES if s["kind"] == "coastal"} == {"Odisha", "Kerala", "West Bengal"}
    lightning = " | ".join(s["scenario"] for s in LIGHTNING_SITES)
    for phrase in ("Pre-monsoon thunderstorm", "Monsoon thunderstorm", "Severe convective storm", "Bay of Bengal",
                   "Arabian Sea", "Himalayan thunderstorm", "Western disturbance"):
        assert phrase in lightning
    assert season_of(5) == "pre-monsoon" and season_of(7) == "south-west monsoon" and season_of(1) == "winter"


# --- flood --------------------------------------------------------------------
@pytest.mark.parametrize("site_index", [0, 4, 5, 15, 17, 7])   # meandering, urban, flash, braided, coastal, agricultural
def test_flood_chip_is_deterministic_and_its_mask_is_aligned_with_the_scene(site_index):
    variant = (site_index + 1) % synth_flood.VARIANTS_PER_SITE          # a flooded variant
    a = synth_flood.build_chip(123, site_index, variant)
    b = synth_flood.build_chip(123, site_index, variant)
    assert np.array_equal(a["s1"], b["s1"], equal_nan=True) and np.array_equal(a["label"], b["label"])
    assert not np.array_equal(a["label"], synth_flood.build_chip(124, site_index, variant)["label"])    # the seed matters
    assert a["s1"].shape == (2, 512, 512) and a["label"].shape == (512, 512) and a["label"].dtype == bool
    # the mask is DERIVED from the land cover the radar image was rendered from
    assert np.array_equal(a["label"], np.isin(a["landcover"], synth_flood.WATER_CLASSES))
    assert (a["permanent"] <= a["label"]).all()                         # permanent water is a subset of water
    flood = a["label"] & ~a["permanent"]
    assert flood.any() and not a["dry"]
    # water really is darker than land in the simulated radar (that is what a model has to learn)
    ok = ~a["missing"]
    assert np.nanmean(a["s1"][1][a["label"] & ok]) < np.nanmean(a["s1"][1][~a["label"] & ok]) - 1.0
    assert np.isfinite(a["s1"][:, ok]).all()


def test_dry_variant_has_permanent_water_only():
    site_index = 0
    chip = synth_flood.build_chip(123, site_index, site_index % synth_flood.VARIANTS_PER_SITE)
    assert chip["dry"] and chip["severity"] == 0.0
    assert np.array_equal(chip["label"], chip["permanent"])


def test_flood_generate_writes_readable_tagged_samples_with_full_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(synth_flood, "FLOOD_SITES", FLOOD_SITES[:2])
    summary = synth_flood.generate(tmp_path, seed=7, per_site=2, quicklooks=False, calibration=FLOOD_CALIBRATION)
    assert summary["chips"] == 4 and summary["synthetic"] is True
    records = [json.loads(line) for line in (tmp_path / "metadata.jsonl").read_text().splitlines()]
    south, north, west, east = INDIA_BOX
    for meta in records:
        assert meta["synthetic"] is True and "not a satellite observation" in meta["dataKind"]
        for key in ("state", "region", "place", "latitude", "longitude", "bounds", "month", "season", "scenario",
                    "floodLabel", "split", "floodFraction", "floodedAreaKm2", "seed"):
            assert key in meta, key
        assert south <= meta["latitude"] <= north and west <= meta["longitude"] <= east
        assert meta["floodLabel"] in ("flood", "no-flood") and meta["split"] in ("train", "valid", "test")
        with np.load(tmp_path / meta["image"]["file"]) as data:
            assert data["s1"].shape == (2, 512, 512) and data["label"].shape == (512, 512)
            assert set(np.unique(data["label"])) <= {0, 1}
            assert data["label"].mean() == pytest.approx(meta["waterFraction"], abs=1e-3)
    # the served catalog reads exactly this layout and keeps the synthetic tag
    monkeypatch.setenv("FLOOD_SYNTHETIC_DIR", str(tmp_path))
    chips = flood_data.synthetic_catalog()
    assert len(chips) == 4 and all(c.source == "synthetic" for c in chips)
    assert flood_data.read_s1(chips[0]).shape == (2, 512, 512) and flood_data.chip_bounds(chips[0])["south"] > 6


def test_flood_calibration_only_shifts_and_scales_within_each_class():
    chip = synth_flood.build_chip(1, 0, 1)
    calibration = {"scale": {"water": [1.2, 1.1], "land": [0.9, 1.0]}, "shift": {"water": [-1.0, 2.0], "land": [0.5, 0.0]}}
    out = synth_flood.apply_calibration(chip["s1"], chip["label"], calibration)
    water = chip["label"] & ~chip["missing"]
    assert np.allclose(out[0][water], chip["s1"][0][water] * 1.2 - 1.0, atol=1e-4)
    assert np.allclose(out[1][~chip["label"] & ~chip["missing"]], chip["s1"][1][~chip["label"] & ~chip["missing"]], atol=1e-4)


def test_on_disk_flood_dataset_if_present():
    chips = flood_data.synthetic_catalog()
    if not chips:
        pytest.skip("synthetic flood dataset not generated")
    assert len(chips) == 240 and {c.split for c in chips} == {"train", "valid", "test"}
    assert REQUIRED_STATES <= {c.meta["state"] for c in chips}
    assert all(c.meta["synthetic"] is True for c in chips)
    assert {"flood", "no-flood"} == {c.meta["floodLabel"] for c in chips}


# --- lightning ----------------------------------------------------------------
@pytest.mark.parametrize("site_index", [0, 5, 6, 3, 16, 30])   # squall, isolated, mcs, orographic, coastal, wd
def test_lightning_scene_is_deterministic_physical_and_its_flashes_follow_the_storm(site_index):
    a = synth_lightning.build_scene(123, site_index, 1, use_insat=False)
    b = synth_lightning.build_scene(123, site_index, 1, use_insat=False)
    assert np.array_equal(a["bt"], b["bt"])
    assert a["bt"].shape == (4, 128, 128) and np.isfinite(a["bt"]).all()
    assert 180 < a["bt"].min() and a["bt"].max() < 335 and a["positive"] and len(a["cells"]) >= 1
    meta, arrays = synth_lightning.finish_sample(a, 123, site_index, 1.0)
    assert arrays["flash_count"].shape == (4, 64, 64) and arrays["flashes"].shape[1] == 4
    # flash_count is the gridded form of the flash list (image-label alignment)
    assert int(arrays["flash_count"].sum()) <= len(arrays["flashes"])           # flashes outside the scene are dropped
    if meta["totalFlashes"]:
        counts, bt = arrays["flash_count"], arrays["bt"].astype(np.float32)
        coldest = bt.reshape(4, 64, 2, 64, 2).min(axis=(2, 4))
        assert coldest[counts > 0].mean() < coldest[counts == 0].mean() - 15     # lightning sits under cold tops
        b0 = meta["bounds"]
        inside = arrays["flashes"][(arrays["flashes"][:, 1] > b0["south"]) & (arrays["flashes"][:, 1] < b0["north"])]
        assert len(inside) >= int(arrays["flash_count"].sum()) * 0.9


def test_cold_cloud_is_not_the_same_thing_as_lightning():
    """A decaying anvil stays cold after its lightning has stopped -- the
    reason a temperature threshold over-warns."""
    cell = synth_lightning.Cell(0, 0, 0, 0, birth=-165, life=220, top_k=200, radius=80, core=8, flash_peak=100, shear=0)
    assert cell.phase(0) > 0.72 and cell.flash_rate(0) == 0.0
    bt = synth_lightning.render_cell(np.random.default_rng(0), np.full((128, 128), 300.0, np.float32), cell, 0)
    assert bt.min() < 225 and (bt < 235).sum() > 200          # a large cold anvil, and no lightning
    assert synth_lightning.Cell(0, 0, 0, 0, -60, 200, 200, 80, 8, 100, 0).flash_rate(0) > 50      # growing cell: active


@pytest.mark.parametrize("variant", [8, 9, 10, 11])   # clear, stratiform, cirrus, fog (in some order)
def test_negative_scenes_never_contain_lightning(variant):
    scene = synth_lightning.build_scene(123, 2, variant, use_insat=False)
    assert not scene["positive"] and scene["negativeKind"] in ("clear", "stratiform", "cirrus", "fog") and scene["cells"] == []
    meta, arrays = synth_lightning.finish_sample(scene, 123, 2, 5.0)
    assert meta["totalFlashes"] == 0 and meta["lightningLabel"] == "no-lightning" and arrays["flash_count"].sum() == 0


def test_lightning_generate_writes_readable_tagged_samples_with_full_metadata(tmp_path, monkeypatch):
    summary = synth_lightning.generate(tmp_path, seed=7, quicklooks=False, calibration=LIGHTNING_CALIBRATION, sites=[0], use_insat=False)
    assert summary["samples"] == 12 and summary["synthetic"] is True
    records = [json.loads(line) for line in (tmp_path / "metadata.jsonl").read_text().splitlines()]
    assert {r["lightningLabel"] for r in records} == {"lightning", "no-lightning"}
    for meta in records:
        assert meta["synthetic"] is True and "not a satellite observation" in meta["dataKind"]
        for key in ("state", "region", "place", "latitude", "longitude", "bounds", "month", "season", "scenario",
                    "lightningLabel", "split", "totalFlashes", "flashesPerFrame", "foundation", "seed"):
            assert key in meta, key
        assert meta["bounds"]["south"] < meta["latitude"] < meta["bounds"]["north"]
    monkeypatch.setenv("LIGHTNING_SYNTHETIC_DIR", str(tmp_path))
    samples = ld.synthetic_catalog()
    arrays = ld.read_synthetic(samples[0])
    assert len(samples) == 12 and arrays["bt"].shape == (4, 128, 128) and arrays["flash_count"].shape == (4, 64, 64)
    ld.validate_pair(arrays["bt"][[-2, -1]])            # exactly what the model is given


def test_on_disk_lightning_dataset_if_present():
    samples = ld.synthetic_catalog()
    if not samples:
        pytest.skip("synthetic lightning dataset not generated")
    assert len(samples) == 408 and {s.split for s in samples} == {"train", "valid", "test"}
    assert REQUIRED_STATES <= {s.meta["state"] for s in samples}
    assert all(s.meta["synthetic"] is True for s in samples)
    labels = [s.meta["lightningLabel"] for s in samples]
    assert labels.count("no-lightning") >= 100 and labels.count("lightning") >= 200      # positives AND negatives
    real_background = [s for s in samples if s.meta["usesRealInsatBackground"]]
    assert all(s.meta["synthetic"] is True and "simulated" in s.meta["foundation"] for s in real_background)
