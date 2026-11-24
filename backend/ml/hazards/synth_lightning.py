"""Generate the India-specific SYNTHETIC lightning / thunderstorm dataset.

    python -m ml.hazards.synth_lightning            # 408 samples -> dataset/synthetic/india_lightning/

EVERY OUTPUT OF THIS SCRIPT IS A SIMULATION. No sample is a satellite
observation and no flash is a detected lightning stroke. No lightning
ground truth for India exists in this project.

Each sample is a 512 x 512 km scene centred on a real Indian place
(ml/hazards/india_geo.py): four infrared cloud-top brightness-temperature
images 30 minutes apart on a 4 km grid (INSAT-3DR's cadence and resolution),
plus simulated lightning flashes for each image.

How a scene is built:
1. Clear-sky background from the month, latitude, coast and terrain of the
   site (hot pre-monsoon plains, cool winter north, cold high Himalaya, an
   even-temperature sea on the correct side for each coast).
2. Convective cells with a life cycle -- towering growth (rapid cooling of
   the top), a mature stage with a cold core and spreading anvil, then decay
   into a warm, thinning, lightning-free anvil. How many cells, how they are
   arranged and how they move follows the Indian scenario: a fast south-east
   moving Kalbaishakhi squall line, scattered monsoon cells, one large
   monsoon cluster with a lightning-free stratiform shield, cells anchored
   on the Himalayan front or along a coast, a winter western-disturbance
   cloud band with a few weak embedded cells.
3. Lightning follows the cell's life cycle, not just its temperature: it
   peaks while the top is still growing and stops in decay. That makes
   "cold cloud" and "lightning" differ the way they do in real data.
4. Lightning-free scenes (clear sky, stratiform monsoon deck, cirrus and
   dead anvils, winter fog) are generated at the same sites as negatives.
5. REAL-DATA ANCHOR: one global multiplier on flash rate is tuned so that,
   among cells colder than 215 K, the share containing a flash equals the
   share measured in the real GOES-16/GLM TRAINING cells
   (ml/artifacts/hazards/goes_flash_statistics.json). That statistic is from
   the Americas -- borrowed because no Indian one is available, not because
   it is known to hold over India.
6. Where the INSAT-3DR files are present, half of the western-disturbance
   scenes use four consecutive REAL INSAT-3DR infrared frames (January 2025)
   as the cloud background, with simulated cells and simulated flashes added.
   They are still tagged synthetic: the lightning in them is invented.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ml.hazards import config
from ml.hazards.india_geo import LIGHTNING_NEGATIVE_KINDS, LIGHTNING_SITES, MONTH_NAMES, season_of

GENERATOR_VERSION = "1.0.0"
N = 128
PX_KM = 4.0
CELL_KM = 8.0
FRAME_MINUTES = (-90, -60, -30, 0)
DEFAULT_SEED = 26077
POSITIVES_PER_SITE = 8
NEGATIVES_PER_SITE = 4
COLD_K = 215.0
TROPOPAUSE_K = 188.0
# afternoon clear-sky surface brightness temperature of the northern plains, by month
PLAINS_BT_K = (291, 295, 303, 310, 314, 309, 302, 301, 302, 303, 298, 293)


def noise(rng: np.random.Generator, sigma: float) -> np.ndarray:
    field = ndi.gaussian_filter(rng.standard_normal((N, N)).astype(np.float32), sigma, mode="reflect")
    return (field - field.mean()) / (field.std() + 1e-6)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def grid_km():
    """x east, y north (km) of every 4 km pixel; row 0 is the northern edge."""
    axis = (np.arange(N) - (N - 1) / 2) * PX_KM
    return np.meshgrid(axis, -axis)


X_KM, Y_KM = grid_km()


@dataclass
class Cell:
    x: float; y: float          # position (km) at the last frame (t = 0)
    vx: float; vy: float        # km per minute
    birth: float; life: float   # minutes relative to the last frame; lifetime
    top_k: float                # coldest core temperature reached
    radius: float               # maximum anvil radius, km
    core: float                 # core width (sigma), km
    flash_peak: float           # flashes per 10 min at peak, before the global multiplier
    shear: float                # direction (rad) the anvil is blown towards

    def phase(self, minute: float) -> float:
        return (minute - self.birth) / self.life

    def centre(self, minute: float) -> tuple[float, float]:
        return self.x + self.vx * minute, self.y + self.vy * minute

    def flash_rate(self, minute: float) -> float:
        """Lightning leads the anvil: strongest at the end of the growth
        stage, none in a young tower or a decaying anvil."""
        p = self.phase(minute)
        if p < 0.08 or p > 0.72:
            return 0.0
        return self.flash_peak * float(np.exp(-((p - 0.32) / 0.16) ** 2) + 0.35 * (0.3 < p < 0.62))


# --------------------------------------------------------------------------
# scene pieces
# --------------------------------------------------------------------------
def background(rng, site: dict, month: int, minute_index: int, base: dict) -> np.ndarray:
    field = base["surface"] - 0.6 * minute_index * base["cooling"] + 0.25 * noise(rng, 2.0)
    return field.astype(np.float32)


def make_base(rng, site: dict, month: int) -> dict:
    winter = {12: 1.0, 1: 1.0, 2: 1.0, 11: 0.5, 3: 0.5}.get(month, 0.0)
    surface = PLAINS_BT_K[month - 1] + winter * (26.0 - site["lat"]) * 0.6 + rng.uniform(-5, 2) \
        + 2.0 * noise(rng, 12) + 0.8 * noise(rng, 3)
    kind = site["kind"]
    if kind in ("orographic", "wd") or site["state"] in ("Uttarakhand", "Himachal Pradesh", "Sikkim", "Arunachal Pradesh"):
        # high ground to the north / north-east is much colder (snow and altitude)
        himalaya = site["state"] in ("Uttarakhand", "Himachal Pradesh", "Sikkim", "Arunachal Pradesh", "Punjab", "Delhi", "Uttar Pradesh")
        direction = np.radians(rng.uniform(60, 120) if himalaya else rng.uniform(0, 360))
        along = (X_KM * np.cos(direction) + Y_KM * np.sin(direction)) / 256.0
        start = rng.uniform(-0.1, 0.3) if site["kind"] == "orographic" else rng.uniform(0.45, 0.8)
        height = smoothstep((along - start) / 0.5 + 0.12 * noise(rng, 6))
        surface = surface - height * (rng.uniform(26, 44) if himalaya else rng.uniform(7, 13))
        front = {"direction": float(direction), "start": float(start)}
    else:
        front = None
    sea = None
    if site.get("coast"):
        bearing = np.radians(rng.uniform(-35, 35) + (0 if site["coast"] == "east" else 180))   # direction of the sea
        offshore = X_KM * np.cos(bearing) + Y_KM * np.sin(bearing) - rng.uniform(-40, 60) + 18 * noise(rng, 10)
        sea = offshore > 0
        surface = np.where(sea, rng.uniform(297, 301) + 0.4 * noise(rng, 8), surface)
        front = {"coast": True, "bearing": float(bearing)}
    cumulus = (noise(rng, 1.6) > 1.1) & (sea is None or ~sea)
    surface = surface - cumulus * rng.uniform(2, 7)      # fair-weather cumulus
    return {"surface": surface.astype(np.float32), "cooling": rng.uniform(0.3, 1.2), "front": front, "sea": sea}


def render_cell(rng, bt: np.ndarray, cell: Cell, minute: float) -> np.ndarray:
    p = cell.phase(minute)
    if p <= 0 or p >= 1:
        return bt
    cx, cy = cell.centre(minute)
    start_k = 268.0
    growth = smoothstep(p / 0.3) if p < 0.3 else 1.0 - 0.5 * smoothstep((p - 0.6) / 0.4)
    core_k = start_k - (start_k - cell.top_k) * growth
    anvil_k = core_k + 6.0 + 16.0 * p
    radius = max(cell.radius * smoothstep((p - 0.12) / 0.5), cell.core * 1.6)
    opacity = 1.0 if p < 0.7 else 1.0 - 0.7 * smoothstep((p - 0.7) / 0.3)
    # anvil: an ellipse stretched and displaced downwind of the core
    ax, ay = np.cos(cell.shear), np.sin(cell.shear)
    dx, dy = X_KM - (cx + 0.3 * radius * ax), Y_KM - (cy + 0.3 * radius * ay)
    along, across = dx * ax + dy * ay, -dx * ay + dy * ax
    reach = np.hypot(along / 1.5, across) / radius * (1.0 + 0.18 * noise(rng, 5))
    weight = opacity / (1.0 + np.exp((reach - 1.0) * 7.0))
    core = np.exp(-((X_KM - cx) ** 2 + (Y_KM - cy) ** 2) / (2 * (cell.core * (0.6 + 0.8 * min(p / 0.3, 1.0))) ** 2))
    cloud_k = anvil_k + (core_k - anvil_k) * core + 1.1 * noise(rng, 1.5)
    return np.minimum(bt, bt * (1 - weight) + cloud_k * weight).astype(np.float32)


def render_layer(rng, bt: np.ndarray, cover: float, top_k: tuple[float, float], opacity: tuple[float, float], scale: float,
                 drift: float = 0.0, texture: np.ndarray | None = None) -> np.ndarray:
    """A lightning-free cloud sheet (stratiform deck, cirrus, fog)."""
    pattern = texture if texture is not None else noise(rng, scale)
    if drift:
        pattern = ndi.shift(pattern, (0, drift), mode="reflect", order=1)
    weight = smoothstep((pattern - np.quantile(pattern, 1 - cover)) * 2.0 + 0.5) * rng.uniform(*opacity)
    cloud_k = rng.uniform(*top_k) + 2.5 * noise(rng, scale / 2)
    return np.minimum(bt, bt * (1 - weight) + cloud_k * weight).astype(np.float32)


def make_cells(rng, site: dict, base: dict) -> list[Cell]:
    kind, cape, month = site["kind"], site["cape"], None

    def top() -> float:
        return float(max(TROPOPAUSE_K, 232.0 - 40.0 * cape * rng.uniform(0.75, 1.05)))

    def cell(x, y, vx, vy, radius, life=None, birth=None, strength=1.0) -> Cell:
        t = top()
        life = life or rng.uniform(100, 220)
        birth = birth if birth is not None else rng.uniform(-life * 0.95, -5)
        peak = (6 + 220 * ((240 - t) / 50) ** 2) * float(rng.lognormal(0, 0.45)) * strength
        return Cell(float(x), float(y), vx, vy, float(birth), float(life), t, float(radius), float(rng.uniform(5, 11)),
                    float(peak), float(np.arctan2(vy, vx) + rng.normal(0, 0.4)) if (vx or vy) else float(rng.uniform(0, 6.28)))

    def wind(speed_ms: tuple[float, float], towards_deg: tuple[float, float]):
        speed = rng.uniform(*speed_ms) * 0.06                        # m/s -> km/min
        bearing = np.radians(rng.uniform(*towards_deg))              # math angle: 0 = east, 90 = north
        return speed * np.cos(bearing), speed * np.sin(bearing)

    cells: list[Cell] = []
    if kind == "isolated":
        vx, vy = wind((3, 8), (0, 360))
        for _ in range(int(rng.integers(3, 9))):
            cells.append(cell(rng.uniform(-220, 220), rng.uniform(-220, 220), vx, vy, rng.uniform(20, 60)))
    elif kind == "squall":                                           # line moving south-east
        vx, vy = wind((11, 18), (-70, -20))
        normal = np.arctan2(vy, vx) + np.pi / 2
        ox, oy, length = rng.uniform(-80, 80), rng.uniform(-80, 80), rng.uniform(160, 340)
        birth0 = rng.uniform(-150, -60)
        for s in np.linspace(-0.5, 0.5, int(rng.integers(5, 10))):
            cells.append(cell(ox + s * length * np.cos(normal) + rng.normal(0, 12), oy + s * length * np.sin(normal) + rng.normal(0, 12),
                              vx, vy, rng.uniform(40, 80), life=rng.uniform(180, 300), birth=birth0 + rng.normal(0, 25)))
    elif kind == "mcs":                                              # one long-lived cluster
        vx, vy = wind((4, 9), (150, 250))
        ox, oy = rng.uniform(-120, 120), rng.uniform(-120, 120)
        birth0 = rng.uniform(-330, -120)
        for _ in range(int(rng.integers(3, 6))):
            cells.append(cell(ox + rng.normal(0, 35), oy + rng.normal(0, 35), vx, vy, rng.uniform(110, 180),
                              life=rng.uniform(360, 520), birth=birth0 + rng.normal(0, 40)))
        for _ in range(int(rng.integers(1, 4))):                    # fresh cells on the flank
            cells.append(cell(ox + rng.normal(0, 90), oy + rng.normal(0, 90), vx, vy, rng.uniform(25, 55)))
    elif kind == "orographic":                                       # anchored on the mountain front
        front = base["front"]
        for _ in range(int(rng.integers(4, 9))):
            s, t = rng.uniform(-230, 230), (front["start"] - 0.05) * 256 + rng.normal(0, 22)
            d = front["direction"]
            cells.append(cell(t * np.cos(d) - s * np.sin(d), t * np.sin(d) + s * np.cos(d),
                              *wind((0, 2.5), (0, 360)), rng.uniform(20, 50)))
    elif kind == "coastal":
        bearing = base["front"]["bearing"]
        vx, vy = wind((2, 7), (np.degrees(bearing) + 150, np.degrees(bearing) + 210))   # drifting onshore
        for _ in range(int(rng.integers(3, 8))):
            s, t = rng.uniform(-230, 230), rng.normal(0, 40)
            cells.append(cell(t * np.cos(bearing) - s * np.sin(bearing), t * np.sin(bearing) + s * np.cos(bearing),
                              vx, vy, rng.uniform(25, 65)))
    elif kind == "wd":                                               # weak cells embedded in the cloud band
        vx, vy = wind((10, 16), (-10, 30))
        for _ in range(int(rng.integers(1, 4))):
            cells.append(cell(rng.uniform(-200, 200), rng.uniform(-200, 200), vx, vy, rng.uniform(20, 45), strength=0.35))
    return cells


def simulate_flashes(rng, cells: list[Cell], multiplier: float) -> list[list[tuple[float, float, float]]]:
    """Per frame: list of (x km, y km, minutes from that frame)."""
    frames = []
    for minute in FRAME_MINUTES:
        flashes = []
        for cell in cells:
            count = int(rng.poisson(cell.flash_rate(minute) * multiplier))
            if not count:
                continue
            cx, cy = cell.centre(minute)
            in_anvil = rng.random(count) < 0.06
            spread = np.where(in_anvil, cell.radius * 0.45, cell.core * 1.2)
            x = cx + spread * rng.standard_normal(count) + 0.3 * cell.core * np.cos(cell.shear)
            y = cy + spread * rng.standard_normal(count) + 0.3 * cell.core * np.sin(cell.shear)
            flashes.extend(zip(x.tolist(), y.tolist(), rng.uniform(-5, 5, count).tolist()))
        frames.append(flashes)
    return frames


def flash_grid(flashes: list[tuple[float, float, float]]) -> np.ndarray:
    grid = np.zeros((N // 2, N // 2), np.uint16)
    if flashes:
        x, y = np.array([f[0] for f in flashes]), np.array([f[1] for f in flashes])
        half = N * PX_KM / 2
        col, row = np.floor((x + half) / CELL_KM).astype(int), np.floor((half - y) / CELL_KM).astype(int)
        ok = (col >= 0) & (col < N // 2) & (row >= 0) & (row < N // 2)
        np.add.at(grid, (row[ok], col[ok]), 1)
    return grid


def insat_background(rng, site: dict):
    """Four consecutive REAL INSAT-3DR TIR1 frames around the site, or None."""
    try:
        from app.services.imagery import insat  # noqa: PLC0415

        by_time = insat.valid_frames_by_time()
        starts = [t for t in sorted(by_time) if all((t + k * insat.FRAME_INTERVAL) in by_time for k in range(4))]
        if not starts:
            return None
        start = starts[int(rng.integers(len(starts)))]
        row, col = insat.latlon_to_pixel(site["lat"], site["lon"])
        r0, c0 = int(round(row)) - N // 2, int(round(col)) - N // 2
        if r0 < 0 or c0 < 0 or r0 + N > insat.EXPECTED_SHAPE[0] or c0 + N > insat.EXPECTED_SHAPE[1]:
            return None
        window = insat.PixelWindow(r0, r0 + N, c0, c0 + N)
        frames = [by_time[start + k * insat.FRAME_INTERVAL] for k in range(4)]
        bt = np.stack([insat.read_bt(f.path, "TIR1", window) for f in frames]).astype(np.float32)
        if not np.isfinite(bt).all():
            return None
        return bt, [f.frame_id for f in frames]
    except Exception:  # noqa: BLE001 -- the real background is optional
        return None


def sample_seed(seed: int, site_index: int, variant: int) -> int:
    return int(seed) * 1_000_003 + site_index * 1009 + variant + 500_000


def build_scene(seed: int, site_index: int, variant: int, use_insat: bool = True) -> dict:
    """Everything except the flashes (which depend on the global multiplier)."""
    site = LIGHTNING_SITES[site_index]
    rng = np.random.default_rng(sample_seed(seed, site_index, variant))
    positive = variant < POSITIVES_PER_SITE
    month = int(rng.choice(site["months"]))
    base = make_base(rng, site, month)
    foundation = "fully procedural simulation"
    negative_kind = None
    cells: list[Cell] = []
    real = insat_background(rng, site) if (use_insat and positive and site["kind"] == "wd" and variant % 2 == 0) else None
    if real is not None:
        frames, frame_ids = real
        frames = list(frames)
        foundation = (f"REAL INSAT-3DR TIR1 infrared frames {frame_ids[0]}..{frame_ids[-1]} (ISRO MOSDAC) as the cloud "
                      "background; convective cells and ALL lightning are simulated")
        month = 1
    else:
        frames = [background(rng, site, month, k, base) for k in range(len(FRAME_MINUTES))]
    if positive:
        cells = make_cells(rng, site, base)
        if site["kind"] == "wd" and real is None:
            texture = noise(rng, 22)
            cover, top_k = rng.uniform(0.35, 0.75), (236.0, 256.0)
            frames = [render_layer(rng, f, cover, top_k, (0.85, 1.0), 22, drift=(k - 3) * 6.0, texture=texture) for k, f in enumerate(frames)]
        if site["kind"] == "mcs":                                   # lightning-free stratiform shield
            texture = noise(rng, 26)
            frames = [render_layer(rng, f, rng.uniform(0.2, 0.4), (214.0, 232.0), (0.9, 1.0), 26, drift=(k - 3) * 2.0, texture=texture)
                      for k, f in enumerate(frames)]
    else:
        negative_kind = LIGHTNING_NEGATIVE_KINDS[(variant - POSITIVES_PER_SITE + site_index) % len(LIGHTNING_NEGATIVE_KINDS)]
        texture = noise(rng, 20)
        if negative_kind == "stratiform":
            cover, top_k = rng.uniform(0.4, 0.9), (rng.uniform(226, 246), rng.uniform(246, 260))
            frames = [render_layer(rng, f, cover, top_k, (0.9, 1.0), 20, drift=(k - 3) * 3.0, texture=texture) for k, f in enumerate(frames)]
        elif negative_kind == "cirrus":                             # thin cirrus plus a dead, cold anvil remnant
            frames = [render_layer(rng, f, rng.uniform(0.2, 0.5), (212.0, 230.0), (0.3, 0.7), 9, drift=(k - 3) * 5.0,
                                   texture=noise(np.random.default_rng(sample_seed(seed, site_index, variant) + 7), 9))
                      for k, f in enumerate(frames)]
            ghost = Cell(rng.uniform(-150, 150), rng.uniform(-150, 150), 0.2, 0.1, -rng.uniform(170, 230), 240.0,
                         rng.uniform(198, 212), rng.uniform(60, 120), 9.0, 0.0, rng.uniform(0, 6.28))
            cells_ghost = [ghost]
            frames = [render_cell(rng, f, ghost, m) for f, m in zip(frames, FRAME_MINUTES)]
            del cells_ghost
        elif negative_kind == "fog":                                # low stratus / fog: almost invisible in infrared
            frames = [render_layer(rng, f, rng.uniform(0.3, 0.8), (float(np.median(f)) - 4.0, float(np.median(f)) - 0.5),
                                   (0.9, 1.0), 20, texture=texture) for f in frames]
    for cell in cells:
        frames = [render_cell(rng, f, cell, m) for f, m in zip(frames, FRAME_MINUTES)]
    bt = np.clip(np.stack(frames), TROPOPAUSE_K - 4, 330.0).astype(np.float32)
    return {"site": site, "variant": variant, "positive": positive, "month": month, "bt": bt, "cells": cells,
            "foundation": foundation, "negativeKind": negative_kind, "usesRealInsat": real is not None, "rng": rng}


# --------------------------------------------------------------------------
# calibration of the flash-rate multiplier against real GOES/GLM statistics
# --------------------------------------------------------------------------
def goes_cold_cell_flash_share() -> tuple[float, dict]:
    path = config.ARTIFACTS_DIR / "goes_flash_statistics.json"
    if not path.exists():
        raise SystemExit("goes_flash_statistics.json missing -- run `python -m ml.hazards.lightning_prepare` first.")
    stats = json.loads(path.read_text())
    bins, probs, counts = stats["btBinsK"], stats["flashProbabilityByBtBin"], stats["cellsByBtBin"]
    hits = sum(p * c for lo, p, c in zip(bins[:-1], probs, counts) if lo < COLD_K and p is not None)
    total = sum(c for lo, p, c in zip(bins[:-1], probs, counts) if lo < COLD_K and p is not None)
    return hits / total, stats


def cold_cell_flash_share(scenes: list[dict], multiplier: float, seed: int) -> float:
    hits = total = 0
    for i, scene in enumerate(scenes):
        rng = np.random.default_rng(seed + i)
        for k, flashes in enumerate(simulate_flashes(rng, scene["cells"], multiplier)):
            coldest = scene["bt"][k].reshape(N // 2, 2, N // 2, 2).min(axis=(1, 3))
            cold = coldest < COLD_K
            hits += int((flash_grid(flashes)[cold] > 0).sum())
            total += int(cold.sum())
    return hits / max(total, 1)


def fit_multiplier(seed: int) -> dict:
    target, stats = goes_cold_cell_flash_share()
    scenes = [build_scene(seed, s, v, use_insat=False) for s in range(len(LIGHTNING_SITES)) for v in (1, 5)]
    lo, hi = 0.01, 20.0
    for _ in range(16):
        mid = float(np.sqrt(lo * hi))
        if cold_cell_flash_share(scenes, mid, seed) < target:
            lo = mid
        else:
            hi = mid
    multiplier = float(np.sqrt(lo * hi))
    return {"statistic": f"share of 8 km cells colder than {COLD_K:.0f} K that contain a flash within +/-5 min",
            "goesTrainValue": target, "syntheticValue": cold_cell_flash_share(scenes, multiplier, seed),
            "flashRateMultiplier": multiplier, "pilotScenes": len(scenes), "goesSource": stats["source"]}


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def split_of(site_index: int, variant: int) -> str:
    slot = (variant + site_index) % 6
    return "valid" if slot == 4 else "test" if slot == 5 else "train"


def finish_sample(scene: dict, seed: int, site_index: int, multiplier: float) -> tuple[dict, dict]:
    site, bt = scene["site"], scene["bt"]
    per_frame = simulate_flashes(scene["rng"], scene["cells"], multiplier)
    counts = np.stack([flash_grid(f) for f in per_frame])
    lat0, lon0 = site["lat"], site["lon"]
    km_per_deg_lon = 111.320 * np.cos(np.radians(lat0))
    rows = [(k, lat0 + y / 110.574, lon0 + x / km_per_deg_lon, dt) for k, flashes in enumerate(per_frame) for x, y, dt in flashes]
    half = N * PX_KM / 2
    sample_id = f"SYNLTG_{site['state'].replace(' ', '')}_{site_index:02d}{scene['variant']:02d}"
    month = scene["month"]
    total = int(counts.sum())
    meta = {
        "id": sample_id, "synthetic": True, "dataKind": "synthetic simulation -- not a satellite observation, not detected lightning",
        "generator": f"ml.hazards.synth_lightning v{GENERATOR_VERSION}", "seed": sample_seed(seed, site_index, scene["variant"]),
        "state": site["state"], "region": site["region"], "place": site["place"],
        "latitude": lat0, "longitude": lon0,
        "bounds": {"south": round(lat0 - half / 110.574, 4), "west": round(lon0 - half / km_per_deg_lon, 4),
                   "north": round(lat0 + half / 110.574, 4), "east": round(lon0 + half / km_per_deg_lon, 4)},
        "month": month, "monthName": MONTH_NAMES[month - 1], "season": season_of(month),
        "scenario": site["scenario"] if scene["positive"] else f"No thunderstorm ({scene['negativeKind']})",
        "stormKind": site["kind"] if scene["positive"] else scene["negativeKind"],
        "lightningLabel": "lightning" if total > 0 else "no-lightning",
        "flashesPerFrame": [int(c.sum()) for c in counts], "totalFlashes": total,
        "lightningCellsLastFrame": int((counts[-1] > 0).sum()),
        "peakFlashDensityPer8kmCellPer10Min": int(counts.max()),
        "stormCells": len(scene["cells"]),
        "minBtK": round(float(bt.min()), 1), "coldCloudFractionLastFrame": round(float((bt[-1] < 235).mean()), 4),
        "frameMinutes": list(FRAME_MINUTES), "imageGridKm": PX_KM, "labelGridKm": CELL_KM,
        "split": split_of(site_index, scene["variant"]),
        "image": {"file": f"samples/{sample_id}.npz", "arrays": {
            "bt": "float16 (4, 128, 128): cloud-top brightness temperature in kelvin, frames 30 min apart, row 0 = north",
            "flash_count": "uint16 (4, 64, 64): simulated flashes per 8 km cell within +/-5 min of each frame (aligned with bt)",
            "flashes": "float32 (n, 4): frame index, latitude, longitude, minutes from that frame",
        }},
        "foundation": scene["foundation"], "usesRealInsatBackground": scene["usesRealInsat"],
    }
    arrays = {"bt": bt.astype(np.float16), "flash_count": counts, "flashes": np.asarray(rows, np.float32).reshape(-1, 4)}
    return meta, arrays


def quicklook(rows: list[tuple[dict, dict]], path: Path, title: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.5 * len(rows)), squeeze=False)
    for ax, (meta, arrays) in zip(axes, rows):
        b = meta["bounds"]
        extent = (b["west"], b["east"], b["south"], b["north"])
        for k in range(4):
            ax[k].imshow(arrays["bt"][k].astype(np.float32), cmap="gray_r", vmin=190, vmax=310, extent=extent, aspect="auto")
            flashes = arrays["flashes"][arrays["flashes"][:, 0] == k]
            ax[k].scatter(flashes[:, 2], flashes[:, 1], s=1.5, c="#ffd400", linewidths=0)
            ax[k].set_title(f"t{FRAME_MINUTES[k]:+d} min  ({meta['flashesPerFrame'][k]} sim. flashes)", fontsize=8)
            ax[k].tick_params(labelsize=6)
        ax[0].set_ylabel(f"{meta['place']}, {meta['state']}\n{meta['scenario']} ({meta['monthName']})", fontsize=7)
    fig.suptitle(f"{title} -- SYNTHETIC SIMULATION (simulated infrared, simulated lightning in yellow)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def generate(out_dir: Path, seed: int = DEFAULT_SEED, quicklooks: bool = True, calibration: dict | None = None,
             sites: list[int] | None = None, use_insat: bool = True) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    calibration = calibration or fit_multiplier(seed)
    records, examples = [], {}
    for site_index in (sites if sites is not None else range(len(LIGHTNING_SITES))):
        site = LIGHTNING_SITES[site_index]
        for variant in range(POSITIVES_PER_SITE + NEGATIVES_PER_SITE):
            scene = build_scene(seed, site_index, variant, use_insat=use_insat)
            meta, arrays = finish_sample(scene, seed, site_index, calibration["flashRateMultiplier"])
            np.savez_compressed(out_dir / "samples" / f"{meta['id']}.npz", **arrays)
            records.append(meta)
            key = meta["stormKind"]
            examples.setdefault(key, [])
            if len(examples[key]) < 3 and all(m["state"] != meta["state"] for m, _ in examples[key]):
                examples[key].append((meta, arrays))
        print(f"  {site_index + 1:2d}/{len(LIGHTNING_SITES)}  {site['state']:<18} {site['place']:<22} {site['scenario']}", flush=True)

    (out_dir / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    if quicklooks:
        (out_dir / "quicklooks").mkdir(exist_ok=True)
        for kind, rows in examples.items():
            quicklook(rows, out_dir / "quicklooks" / f"{kind}.png", f"India synthetic lightning: {kind}")

    def count(key):
        out: dict = {}
        for r in records:
            out[r[key]] = out.get(r[key], 0) + 1
        return dict(sorted(out.items()))

    summary = {
        "dataset": "AtmoGuard India synthetic lightning / thunderstorm scenes", "synthetic": True,
        "generator": f"ml.hazards.synth_lightning v{GENERATOR_VERSION}", "seed": seed,
        "generatedAt": datetime.now(timezone.utc).isoformat(), "samples": len(records),
        "bySplit": count("split"), "byRegion": count("region"), "byState": count("state"),
        "byScenario": count("scenario"), "byStormKind": count("stormKind"), "byLabel": count("lightningLabel"),
        "samplesWithRealInsatBackground": sum(r["usesRealInsatBackground"] for r in records),
        "totalSimulatedFlashes": sum(r["totalFlashes"] for r in records),
        "calibration": calibration,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=config.LIGHTNING_SYNTHETIC_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-quicklooks", action="store_true")
    parser.add_argument("--no-insat", action="store_true", help="never use real INSAT-3DR frames as a background")
    args = parser.parse_args()
    summary = generate(args.out, args.seed, not args.no_quicklooks, use_insat=not args.no_insat)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("byState",)}, indent=2))


if __name__ == "__main__":
    main()
