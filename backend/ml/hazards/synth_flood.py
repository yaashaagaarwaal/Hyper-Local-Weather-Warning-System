"""Generate the India-specific SYNTHETIC flood dataset.

    python -m ml.hazards.synth_flood                 # 240 chips -> dataset/synthetic/india_flood/
    python -m ml.hazards.synth_flood --per-site 2 --out /tmp/x   # small run

EVERY OUTPUT OF THIS SCRIPT IS A SIMULATION. Nothing here is a satellite
observation, and no chip reconstructs a real flood event.

What is simulated: a 512x512 Sentinel-1-like radar chip (VV, VH backscatter
in dB, 10 m pixels, same layout as Sen1Floods11) with a pixel-aligned water
mask, at 40 real Indian sites (ml/hazards/india_geo.py). For each chip:

1. Terrain and water are built first -- a river style that suits the site
   (meandering Ganga/Yamuna-type, braided Brahmaputra/Kosi-type with sand
   bars, a narrow Himalayan valley, a coast with tidal creeks and embankments,
   an urban street grid, or bunded paddy) and a small elevation model.
2. The flood is derived from that terrain, not painted on: water rises to a
   stage and fills what is below it AND connected to the river/sea (roads on
   embankments and village mounds stay dry; levee breaches let water through).
   Urban waterlogging fills local low spots; agricultural flooding fills whole
   bunded fields.
3. Land cover follows Indian patterns: small field parcels (25-60 m in
   Bihar/Bengal, 90-180 m in Punjab), season-dependent crops, village
   clusters with tree cover on higher ground, roads, bridges, sand bars,
   mangrove and aquaculture ponds on the east coast.
4. Radar backscatter is assigned per land-cover class from typical C-band
   values, including the classic confusions (dry sand, radar shadow and
   smooth roads are dark like water; flooded vegetation and flooded streets
   are bright), then multiplicative speckle and the sensor noise floor are
   applied.
5. REAL-DATA CALIBRATION: one global shift/scale per (water|land, band) makes
   the pooled synthetic statistics equal those of the REAL Sen1Floods11 India
   TRAINING chips (no validation/test chip is read), so the simulator's
   brightness is anchored to real Indian radar imagery.

The mask is exact because the scene is built from it (`label`: 1 = water,
0 = not water, same convention as the real dataset; `permanent` separates
river/sea/ponds from flood water).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from ml.hazards import config
from ml.hazards.india_geo import FLOOD_SITES, MONTH_NAMES, season_of

GENERATOR_VERSION = "1.0.0"
N = 512
PX_M = 10.0
DEFAULT_SEED = 26077
VARIANTS_PER_SITE = 6     # 4 train, 1 valid, 1 test; one of the six is a dry (no-flood) scene
SPECKLE_LOOKS = 4.4       # equivalent number of looks of Sentinel-1 IW GRD
NOISE_FLOOR_DB = (-31.0, -33.0)  # VV, VH residual noise after the product's thermal-noise removal

# land-cover codes
(PADDY, FALLOW, TALL_CROP, TREES, BUILT, ROAD, SAND, ROCK, SHADOW, BRIDGE,
 PERM_WATER, FLOOD_OPEN, FLOOD_VEG, FLOOD_URBAN) = range(14)
LANDCOVER_NAMES = ["paddy/crop", "fallow or bare soil", "tall crop", "trees", "built-up", "road", "sand bar",
                   "rock/steep slope", "radar shadow", "bridge", "permanent water", "flood water (open)",
                   "flooded vegetation", "flooded street"]
WATER_CLASSES = (PERM_WATER, FLOOD_OPEN, FLOOD_VEG, FLOOD_URBAN)
# typical C-band sigma0 (VV dB, VH dB, within-class texture dB)
BACKSCATTER = {
    PADDY: (-10.5, -17.0, 1.0), FALLOW: (-11.5, -19.5, 1.2), TALL_CROP: (-8.0, -13.5, 1.0), TREES: (-7.5, -13.0, 0.8),
    BUILT: (-4.0, -11.5, 3.0), ROAD: (-13.5, -21.0, 1.0), SAND: (-15.5, -23.5, 1.2), ROCK: (-8.0, -15.0, 2.0),
    SHADOW: (-21.0, -28.0, 1.0), BRIDGE: (-2.0, -10.0, 2.0), PERM_WATER: (-21.0, -29.5, 1.2),
    FLOOD_OPEN: (-19.5, -28.0, 1.5), FLOOD_VEG: (-8.5, -17.5, 1.8), FLOOD_URBAN: (-11.0, -19.0, 3.0),
}
ROUGH_WATER = (-12.5, -21.0)   # wind-roughened water is much brighter than calm water


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def noise(rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Smooth random field, zero mean, unit variance."""
    field = ndi.gaussian_filter(rng.standard_normal((N, N)).astype(np.float32), sigma, mode="reflect")
    return (field - field.mean()) / (field.std() + 1e-6)


def unit_hash(ids: np.ndarray, salt: float) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per integer id."""
    return np.modf(np.abs(np.sin(ids.astype(np.float64) * 12.9898 + salt) * 43758.5453))[0].astype(np.float32)


def meander(rng, half_width_px: float, amplitude=(30, 110), wavelength=(300, 700), centre_shift=0.25, angle=None):
    """A sinuous channel across the chip. Returns (mask, distance-to-centreline px, angle)."""
    angle = rng.uniform(0, np.pi) if angle is None else angle
    t = np.linspace(-1.3 * N, 1.3 * N, 700)
    amp, lam = rng.uniform(*amplitude), rng.uniform(*wavelength)
    off = (amp * np.sin(2 * np.pi * t / lam + rng.uniform(0, 6.28))
           + 0.45 * amp * np.sin(2 * np.pi * t / (lam * 2.3) + rng.uniform(0, 6.28)) + rng.uniform(-centre_shift, centre_shift) * N)
    x = N / 2 + t * np.cos(angle) - off * np.sin(angle)
    y = N / 2 + t * np.sin(angle) + off * np.cos(angle)
    line = np.zeros((N, N), np.uint8)
    cv2.polylines(line, [np.stack([x, y], axis=1).astype(np.int32)], False, 1, 1)
    dist = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
    half = half_width_px * (1 + 0.22 * noise(rng, 45))
    return dist < np.maximum(half, 0.8), dist, angle


def field_parcels(rng, size_m: tuple[float, float]) -> np.ndarray:
    """Integer parcel id per pixel: a warped, staggered grid of small fields."""
    alpha = rng.uniform(0, np.pi)
    pw, ph = rng.uniform(*size_m) / PX_M, rng.uniform(*size_m) / PX_M * rng.uniform(1.0, 2.2)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    xx = xx + 5 * noise(rng, 35)
    yy = yy + 5 * noise(rng, 35)
    u = np.floor((xx * np.cos(alpha) + yy * np.sin(alpha)) / pw)
    v = np.floor((-xx * np.sin(alpha) + yy * np.cos(alpha)) / ph + unit_hash(u, 1.7) * 0.9)
    return (u * 7919 + v).astype(np.int64)


def crop_mix(site: dict, month: int) -> tuple[float, float, float]:
    """(paddy/crop, tall crop, trees) shares; the rest is fallow/bare."""
    monsoon = month in (6, 7, 8, 9)
    paddy, tall, trees = (0.60, 0.14, 0.10) if monsoon else (0.38, 0.14, 0.10)
    state = site["state"]
    if state in ("Punjab",):
        paddy, tall, trees = (0.78, 0.08, 0.04) if monsoon else (0.55, 0.08, 0.04)
    elif state in ("Rajasthan", "Gujarat", "Madhya Pradesh", "Maharashtra", "Telangana"):
        paddy, tall, trees = (0.45, 0.18, 0.07) if monsoon else (0.25, 0.15, 0.07)
    elif state in ("Kerala", "Andhra Pradesh", "Tamil Nadu"):
        paddy, tall, trees = 0.45, 0.12, 0.28      # coconut / banana gardens between paddies
    elif state in ("Assam", "West Bengal", "Bihar", "Odisha"):
        paddy, tall, trees = (0.66, 0.10, 0.12) if monsoon else (0.40, 0.10, 0.12)
    return paddy, tall, trees


def rural_landcover(rng, site: dict, month: int, dem: np.ndarray, blocked: np.ndarray):
    """Fields + villages + roads. Returns (landcover, dem with mounds/embankments, road mask)."""
    pid = field_parcels(rng, site.get("parcel_m", (40, 80)))
    r = unit_hash(pid, rng.uniform(0, 100))
    paddy, tall, trees = crop_mix(site, month)
    lc = np.full((N, N), FALLOW, np.uint8)
    lc[r < paddy] = PADDY
    lc[(r >= paddy) & (r < paddy + tall)] = TALL_CROP
    lc[(r >= paddy + tall) & (r < paddy + tall + trees)] = TREES
    dem = dem.copy()

    # villages: built-up clusters on locally higher ground, ringed by trees
    centres = []
    for _ in range(int(rng.integers(site.get("villages", (2, 4))[0], site.get("villages", (2, 4))[1] + 1))):
        candidates = rng.integers(30, N - 30, size=(24, 2))
        ok = [c for c in candidates if not blocked[c[0], c[1]]]
        if not ok:
            continue
        cy, cx = max(ok, key=lambda c: dem[c[0], c[1]])
        radius = rng.uniform(8, 24)
        yy, xx = np.mgrid[0:N, 0:N]
        shape = np.hypot(yy - cy, xx - cx) / radius + 0.35 * noise(rng, 6)
        core, ring = (shape < 1.0) & ~blocked, (shape < 1.5) & ~blocked
        lc[ring & (rng.random((N, N)) < 0.6)] = TREES
        lc[core & (rng.random((N, N)) < 0.72)] = BUILT
        dem[ring] += 1.2 * np.clip(1.5 - shape[ring], 0, 1)   # settlement mound
        centres.append((int(cx), int(cy)))

    # roads on low embankments, linking villages and chip edges
    road = np.zeros((N, N), np.uint8)
    nodes = centres + [(int(rng.integers(0, N)), int(rng.choice([0, N - 1]))) for _ in range(2)] \
                    + [(int(rng.choice([0, N - 1])), int(rng.integers(0, N))) for _ in range(1)]
    rng.shuffle(nodes)
    for a, b in zip(nodes[:-1], nodes[1:]):
        mid = ((a[0] + b[0]) // 2 + int(rng.normal(0, 18)), (a[1] + b[1]) // 2 + int(rng.normal(0, 18)))
        cv2.polylines(road, [np.array([a, mid, b], np.int32)], False, 1, 1)
    road = road.astype(bool)
    dem[ndi.binary_dilation(road, iterations=1)] += rng.uniform(0.8, 1.8)
    return lc, dem, road


def connected_to(candidate: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Pixels of `candidate` that are connected to `source` (water can only
    reach what it can flow to)."""
    labels, _ = ndi.label(candidate | source)
    keep = np.unique(labels[source])
    return np.isin(labels, keep[keep > 0]) & candidate


def split_flood_water(rng, lc: np.ndarray, flood: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Shallow water under crops/trees leaves vegetation standing above it
    (bright, double-bounce) -- still water, but it does not look like it."""
    emergent = flood & np.isin(lc, (PADDY, TALL_CROP, TREES)) & (depth < rng.uniform(0.25, 0.7))
    lc = lc.copy()
    lc[flood] = FLOOD_OPEN
    lc[emergent] = FLOOD_VEG
    return lc


# --------------------------------------------------------------------------
# scene builders: each returns (landcover, permanent_water_mask, details)
# --------------------------------------------------------------------------
def scene_meandering(rng, site, month, severity):
    half = rng.uniform(*site.get("river_m", (150, 400))) / PX_M / 2
    river, dist, _ = meander(rng, half)
    rise = rng.uniform(1.0, 4.0)                       # metres of rise 1 km from the bank
    dem = rise * np.sqrt(np.maximum(dist - half, 0) * PX_M / 1000.0) + 0.6 * noise(rng, 28) + 0.15 * noise(rng, 6)
    if rng.random() < 0.6:                             # abandoned channel: a low swale that floods first
        old, _, _ = meander(rng, rng.uniform(4, 10), amplitude=(60, 140))
        dem[ndi.binary_dilation(old, iterations=3)] -= rng.uniform(0.8, 1.8)
    perm = river.copy()
    if rng.random() < 0.3:                             # oxbow lake / village pond
        cy, cx = rng.integers(60, N - 60, 2)
        yy, xx = np.mgrid[0:N, 0:N]
        perm |= (np.hypot((yy - cy) / rng.uniform(6, 16), (xx - cx) / rng.uniform(10, 30)) < 1) & ~river
    embanked = rng.random() < 0.4
    if embanked:                                       # levee with breaches
        levee = (np.abs(dist - half - 6) < 1.5) & (noise(rng, 30) > -1.1)
        dem[levee] += rng.uniform(1.5, 3.0)
    lc, dem, road = rural_landcover(rng, site, month, dem, ndi.binary_dilation(perm, iterations=4))
    stage = severity * rng.uniform(1.2, 4.2)
    flood = connected_to((dem < stage) & ~perm, perm) if severity > 0 else np.zeros((N, N), bool)
    lc[road] = ROAD
    lc = split_flood_water(rng, lc, flood, stage - dem)
    lc[perm] = PERM_WATER
    lc[road & ndi.binary_dilation(river, iterations=1) & river] = BRIDGE
    return lc, perm, {"riverWidthM": round(2 * half * PX_M), "stageM": round(float(stage), 2), "embanked": bool(embanked)}


def scene_braided(rng, site, month, severity):
    belt, belt_dist, angle = meander(rng, rng.uniform(150, 260), amplitude=(20, 60), wavelength=(500, 900), centre_shift=0.15)
    channels = np.zeros((N, N), bool)
    for _ in range(int(rng.integers(3, 6))):
        ch, _, _ = meander(rng, rng.uniform(6, 26), amplitude=(25, 70), wavelength=(180, 420), centre_shift=0.3, angle=angle + rng.normal(0, 0.08))
        channels |= ch
    perm = channels & belt
    bars = belt & ~perm
    dem = np.where(belt, 0.7 + 0.5 * noise(rng, 10), 2.2 + 2.5 * np.sqrt(np.maximum(belt_dist - 200, 0) * PX_M / 1000.0)
                   + 0.6 * noise(rng, 28)).astype(np.float32)
    lc, dem, road = rural_landcover(rng, site, month, dem, belt)
    vegetated = bars & (noise(rng, 14) > 0.5)          # older, stable islands carry grass and crops
    lc[bars] = SAND
    lc[vegetated] = TALL_CROP
    stage = severity * rng.uniform(1.0, 4.0)
    flood = connected_to((dem < stage) & ~perm, perm) if severity > 0 else np.zeros((N, N), bool)
    lc[road & ~belt] = ROAD
    lc = split_flood_water(rng, lc, flood, stage - dem)
    lc[perm] = PERM_WATER
    return lc, perm, {"beltWidthM": int(belt.sum(axis=0).mean() * PX_M), "stageM": round(float(stage), 2)}


def scene_flash(rng, site, month, severity):
    half = rng.uniform(1.2, 3.0)
    stream, dist, _ = meander(rng, half, amplitude=(15, 50), wavelength=(250, 500), centre_shift=0.2)
    slope = rng.uniform(0.9, 2.6)                      # metres of climb per pixel away from the stream
    dem = dist * slope + 55 * noise(rng, 30) + 12 * noise(rng, 8)
    dem -= dem[stream].mean() if stream.any() else 0
    look = rng.choice([-1.0, 1.0])                     # radar looks from the east or the west
    facing = np.gradient(dem, axis=1) * look
    lc = np.full((N, N), TREES, np.uint8)
    lc[(noise(rng, 12) > 0.9) | (dem > np.quantile(dem, 0.9))] = ROCK
    lc[facing < -np.quantile(np.abs(facing), 0.9)] = SHADOW      # slopes hidden from the radar: dark, NOT water
    floor = dist < rng.uniform(12, 30) * (1 + 0.4 * noise(rng, 30))
    terraces = floor & (unit_hash(field_parcels(rng, (25, 50)), 3.3) < 0.7)
    lc[floor] = FALLOW
    lc[terraces] = PADDY
    lc[floor & (noise(rng, 5) > 1.3)] = BUILT                   # roadside settlements on the valley floor
    lc[(np.abs(dist - rng.uniform(7, 11)) < 0.8) & floor] = ROAD
    perm = stream
    reach = half + severity * rng.uniform(5, 16) * (1 + 0.5 * noise(rng, 25))
    flood = (dist < reach) & ~perm if severity > 0 else np.zeros((N, N), bool)
    lc[flood] = FLOOD_OPEN
    lc[perm] = PERM_WATER
    return lc, perm, {"valleySlopeMPerPx": round(float(slope), 2), "floodReachM": round(float(np.mean(reach)) * PX_M)}


def scene_coastal(rng, site, month, severity):
    backwater = "Kuttanad" in site["place"]
    alpha = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    inland = (xx - N / 2) * np.cos(alpha) + (yy - N / 2) * np.sin(alpha) + rng.uniform(60, 170) + 35 * noise(rng, 45)
    sea = inland < 0
    creeks = np.zeros((N, N), bool)
    for _ in range(int(rng.integers(2, 5)) if not backwater else int(rng.integers(4, 8))):
        ch, _, _ = meander(rng, rng.uniform(2, 7), amplitude=(30, 90), wavelength=(200, 450), centre_shift=0.4)
        creeks |= ch
    perm = sea | creeks
    dem = np.maximum(inland, 0) * rng.uniform(0.004, 0.012) + 0.45 * noise(rng, 22) + 0.12 * noise(rng, 5)
    if backwater:
        dem -= 0.8                                    # reclaimed paddy lying below water level
    dyke = (ndi.binary_dilation(perm, iterations=4) & ~ndi.binary_dilation(perm, iterations=2)) & (noise(rng, 26) > -0.9)
    dem[dyke] += rng.uniform(1.8, 3.0)                # embankment with breaches
    lc, dem, road = rural_landcover(rng, site, month, dem, ndi.binary_dilation(perm, iterations=4))
    near = (inland > 0) & (inland < rng.uniform(40, 110))
    if not backwater:
        lc[near & (noise(rng, 12) > 0.1)] = TREES     # mangrove fringe
    ponds = near & (unit_hash(field_parcels(rng, (60, 130)), 9.1) < (0.18 if not backwater else 0.05)) & ~dyke
    perm |= ponds                                     # aquaculture ponds
    stage = severity * rng.uniform(1.0, 3.2)
    flood = connected_to((dem < stage) & ~perm, perm) if severity > 0 else np.zeros((N, N), bool)
    lc[road] = ROAD
    lc = split_flood_water(rng, lc, flood, stage - dem)
    lc[perm] = PERM_WATER
    return lc, perm, {"surgeM": round(float(stage), 2), "seaFraction": round(float(sea.mean()), 3)}


def scene_agricultural(rng, site, month, severity):
    half = rng.uniform(*site.get("river_m", (60, 150))) / PX_M / 2
    canal, dist, _ = meander(rng, half, amplitude=(15, 60))
    perm = canal.copy()
    dem = 1.5 * np.sqrt(np.maximum(dist - half, 0) * PX_M / 1000.0) + 0.5 * noise(rng, 30) + 0.1 * noise(rng, 6)
    lc, dem, road = rural_landcover(rng, site, month, dem, ndi.binary_dilation(perm, iterations=3))
    # bunded fields fill and hold water as whole parcels
    pid = field_parcels(rng, site.get("parcel_m", (40, 80)))
    ids, inverse = np.unique(pid, return_inverse=True)
    parcel_height = (np.bincount(inverse.ravel(), dem.ravel()) / np.bincount(inverse.ravel()))[inverse].reshape(N, N)
    stage = severity * rng.uniform(0.6, 2.2)
    wet = (parcel_height < stage) & (unit_hash(pid, 5.5) < 0.85) & np.isin(lc, (PADDY, FALLOW, TALL_CROP))
    flood = (wet | connected_to((dem < stage * 0.7) & ~perm, perm)) & ~perm if severity > 0 else np.zeros((N, N), bool)
    flood &= ~np.isin(lc, (BUILT,))
    lc[road] = ROAD
    lc = split_flood_water(rng, lc, flood & ~road, stage - parcel_height)
    lc[perm] = PERM_WATER
    return lc, perm, {"stageM": round(float(stage), 2), "canalWidthM": round(2 * half * PX_M)}


def scene_urban(rng, site, month, severity):
    alpha = rng.uniform(0, np.pi / 2)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    u = xx * np.cos(alpha) + yy * np.sin(alpha) + 4 * noise(rng, 40)
    v = -xx * np.sin(alpha) + yy * np.cos(alpha) + 4 * noise(rng, 40)
    bu, bv = rng.uniform(7, 15), rng.uniform(7, 15)   # block size in pixels (70-150 m)
    street = (np.mod(u, bu) < 1.2) | (np.mod(v, bv) < 1.2) | (np.mod(u, bu * 5) < 2.5) | (np.mod(v, bv * 5) < 2.5)
    block = (np.floor(u / bu) * 7919 + np.floor(v / bv)).astype(np.int64)
    r = unit_hash(block, rng.uniform(0, 50))
    lc = np.where(noise(rng, 1.2) > -0.35, BUILT, FALLOW).astype(np.uint8)    # buildings with courtyards/yards
    lc[r < 0.10] = TREES                               # parks
    lc[(r >= 0.10) & (r < 0.16)] = FALLOW              # open grounds
    lc[street] = ROAD
    drain, ddist, _ = meander(rng, rng.uniform(1.5, 4.5), amplitude=(20, 70))
    perm = drain.copy()
    details = {"blockM": [round(float(bu * PX_M)), round(float(bv * PX_M))]}
    if any(key in site["place"] for key in ("Bengaluru", "Chennai", "Hyderabad")) or rng.random() < 0.25:
        cy, cx = rng.integers(90, N - 90, 2)           # tank / lake / marsh
        lake = (np.hypot((yy - cy) / rng.uniform(15, 45), (xx - cx) / rng.uniform(15, 45)) + 0.25 * noise(rng, 10)) < 1
        perm |= lake
        details["lake"] = True
    low = 0.9 * noise(rng, 34) - 0.9 * np.exp(-ddist / 25.0)
    cut = np.quantile(low, severity * rng.uniform(0.08, 0.30)) if severity > 0 else -np.inf
    # Water stands in streets, yards and open ground; at 10 m that reads as
    # one contiguous inundated patch with only the larger building blocks dry.
    flood = ndi.binary_closing((low < cut) & (lc != BUILT), iterations=1) & (low < cut) & ~perm
    lc[flood & (street | (lc == BUILT))] = FLOOD_URBAN   # water between buildings: bright double-bounce
    lc[flood & (lc != FLOOD_URBAN)] = FLOOD_OPEN
    lc[perm] = PERM_WATER
    lc[street & drain] = BRIDGE
    return lc, perm, details


BUILDERS = {"meandering": scene_meandering, "braided": scene_braided, "flash": scene_flash,
            "coastal": scene_coastal, "agricultural": scene_agricultural, "urban": scene_urban}


# --------------------------------------------------------------------------
# radar simulation
# --------------------------------------------------------------------------
def simulate_backscatter(rng, lc: np.ndarray) -> np.ndarray:
    """(2, N, N) float32 sigma0 in dB, before real-data calibration."""
    parcel_tone = 1.0 * noise(rng, 3.5)       # field-to-field differences
    moisture = 0.8 * noise(rng, 45)           # broad soil-moisture pattern
    wind = 1.0 / (1.0 + np.exp(-(1.6 * noise(rng, 55) + rng.normal(-0.3, 0.8))))   # 0 calm .. 1 rough
    range_trend = np.linspace(-1, 1, N, dtype=np.float32)[None, :] * rng.uniform(-0.8, 0.8)  # incidence angle
    out = np.empty((2, N, N), np.float32)
    for band in (0, 1):
        mean = np.zeros((N, N), np.float32)
        for code, values in BACKSCATTER.items():
            member = lc == code
            if not member.any():
                continue
            level = np.full(int(member.sum()), values[band], np.float32) + values[2] * rng.standard_normal(int(member.sum())).astype(np.float32) * 0.5
            if code in (PERM_WATER, FLOOD_OPEN):
                level += wind[member] * (ROUGH_WATER[band] - values[band])
            elif code in (BUILT, BRIDGE, FLOOD_URBAN):
                level += np.where(rng.random(level.shape) < 0.08, rng.uniform(5, 11, level.shape), 0)  # corner reflectors
            else:
                level += parcel_tone[member] * (1.0 if code in (PADDY, FALLOW, TALL_CROP) else 0.4) + moisture[member]
            mean[member] = level
        power = 10 ** ((mean + range_trend) / 10.0)
        # multiplicative speckle (gamma, ~4.4 looks), slightly correlated between neighbours as in GRD products
        speckle = ndi.gaussian_filter(rng.gamma(SPECKLE_LOOKS, 1.0 / SPECKLE_LOOKS, (N, N)).astype(np.float32), 0.4)
        floor = 10 ** (NOISE_FLOOR_DB[band] / 10.0) * rng.gamma(SPECKLE_LOOKS, 1.0 / SPECKLE_LOOKS, (N, N)).astype(np.float32)
        out[band] = 10 * np.log10(np.maximum(ndi.gaussian_filter(power, 0.45) * speckle + floor, 1e-6))
    return out


def chip_seed(seed: int, site_index: int, variant: int) -> int:
    return int(seed) * 1_000_003 + site_index * 1009 + variant


def build_chip(seed: int, site_index: int, variant: int) -> dict:
    """Everything about one chip except the real-data calibration."""
    site = FLOOD_SITES[site_index]
    rng = np.random.default_rng(chip_seed(seed, site_index, variant))
    dry = variant == site_index % VARIANTS_PER_SITE
    severity = 0.0 if dry else float(rng.uniform(0.45, 1.0))
    month = int(rng.choice(site["months"])) if not dry else int(rng.choice([1, 2, 3, 4]))
    lc, perm, details = BUILDERS[site["kind"]](rng, site, month, severity)
    s1 = simulate_backscatter(rng, lc)
    label = np.isin(lc, WATER_CLASSES)
    perm = perm & label                                # a bridge deck over the river is not water
    lat = site["lat"] + float(rng.uniform(-0.04, 0.04))
    lon = site["lon"] + float(rng.uniform(-0.04, 0.04))
    dlat = N * PX_M / 1000.0 / 110.574 / 2
    dlon = N * PX_M / 1000.0 / (111.320 * np.cos(np.radians(lat))) / 2
    missing = np.zeros((N, N), bool)
    if rng.random() < 0.06:                            # swath edge: a corner with no radar data
        yy, xx = np.mgrid[0:N, 0:N]
        missing = (xx + yy * rng.uniform(0.4, 1.6)) < rng.uniform(60, 220)
        s1[:, missing] = np.nan
    return {"site": site, "variant": variant, "dry": dry, "severity": severity, "month": month, "landcover": lc,
            "permanent": perm, "label": label, "s1": s1, "missing": missing, "details": details,
            "lat": lat, "lon": lon, "bounds": {"south": lat - dlat, "west": lon - dlon, "north": lat + dlat, "east": lon + dlon}}


# --------------------------------------------------------------------------
# calibration against REAL India training chips
# --------------------------------------------------------------------------
def real_india_training_statistics() -> dict:
    """Mean/std of VV and VH over water and over land in the real
    Sen1Floods11 INDIA chips of the official TRAIN split."""
    from app.services.imagery import flood_data  # noqa: PLC0415

    chips = [c for c in flood_data.real_catalog() if c.split == "train" and c.region == "India"]
    if not chips:
        raise SystemExit("Real Sen1Floods11 India training chips not found -- they anchor the simulator's brightness.")
    sums = {k: [np.zeros(2), np.zeros(2), 0] for k in ("water", "land")}
    for chip in chips:
        s1, label = flood_data.read_s1(chip), flood_data.read_label(chip)
        ok = flood_data.data_mask(s1)
        for key, member in (("water", (label == 1) & ok), ("land", (label == 0) & ok)):
            values = s1[:, member].astype(np.float64)
            sums[key][0] += values.sum(axis=1)
            sums[key][1] += (values ** 2).sum(axis=1)
            sums[key][2] += values.shape[1]
    out = {"chips": len(chips), "chipIds": [c.chip_id for c in chips]}
    for key, (total, total_sq, n) in sums.items():
        mean = total / n
        out[key] = {"mean": mean.tolist(), "std": np.sqrt(total_sq / n - mean ** 2).tolist(), "pixels": int(n)}
    return out


def fit_calibration(seed: int, per_site: int) -> dict:
    real = real_india_training_statistics()
    sums = {k: [np.zeros(2), np.zeros(2), 0] for k in ("water", "land")}
    for site_index in range(len(FLOOD_SITES)):          # pilot: one flooded variant per site
        variant = (site_index + 1) % min(per_site, VARIANTS_PER_SITE)
        chip = build_chip(seed, site_index, variant)
        ok = ~chip["missing"]
        for key, member in (("water", chip["label"] & ok), ("land", ~chip["label"] & ok)):
            values = chip["s1"][:, member].astype(np.float64)
            sums[key][0] += values.sum(axis=1)
            sums[key][1] += (values ** 2).sum(axis=1)
            sums[key][2] += values.shape[1]
    calibration = {"realIndiaTrain": real, "pilot": {}, "scale": {}, "shift": {}}
    for key, (total, total_sq, n) in sums.items():
        mean = total / n
        std = np.sqrt(total_sq / n - mean ** 2)
        # Clamped: a large stretch would turn speckle into something no radar produces. Any
        # spread the clamp leaves unmatched is recorded below, not hidden.
        scale = np.clip(np.asarray(real[key]["std"]) / std, 0.75, 1.4)
        calibration["pilot"][key] = {"mean": mean.tolist(), "std": std.tolist()}
        calibration["scale"][key] = scale.tolist()
        calibration["shift"][key] = (np.asarray(real[key]["mean"]) - mean * scale).tolist()
        calibration.setdefault("calibratedStd", {})[key] = (std * scale).tolist()
    return calibration


def apply_calibration(s1: np.ndarray, label: np.ndarray, calibration: dict) -> np.ndarray:
    out = s1.copy()
    for key, member in (("water", label), ("land", ~label)):
        for band in (0, 1):
            out[band][member] = s1[band][member] * calibration["scale"][key][band] + calibration["shift"][key][band]
    return out


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def split_of(site_index: int, variant: int) -> str:
    slot = (variant + site_index) % VARIANTS_PER_SITE   # rotates which variants are held out per site
    return "valid" if slot == 4 else "test" if slot == 5 else "train"


def metadata_for(chip: dict, chip_id: str, seed: int, site_index: int) -> dict:
    site, label, perm = chip["site"], chip["label"], chip["permanent"]
    flood = label & ~perm
    return {
        "id": chip_id, "synthetic": True, "dataKind": "synthetic simulation -- not a satellite observation",
        "generator": f"ml.hazards.synth_flood v{GENERATOR_VERSION}", "seed": chip_seed(seed, site_index, chip["variant"]),
        "state": site["state"], "region": site["region"], "place": site["place"], "waterFeature": site["feature"],
        "latitude": round(chip["lat"], 5), "longitude": round(chip["lon"], 5),
        "bounds": {k: round(v, 5) for k, v in chip["bounds"].items()},
        "month": chip["month"], "monthName": MONTH_NAMES[chip["month"] - 1], "season": season_of(chip["month"]),
        "scenario": site["scenario"] if not chip["dry"] else "No flood (dry-season baseline)",
        "floodKind": site["kind"], "floodLabel": "flood" if flood.any() else "no-flood",
        "severity": round(chip["severity"], 3),
        "waterFraction": round(float(label.mean()), 4), "floodFraction": round(float(flood.mean()), 4),
        "permanentWaterFraction": round(float(perm.mean()), 4),
        "floodedAreaKm2": round(float(flood.sum()) * PX_M * PX_M / 1e6, 4),
        "missingRadarFraction": round(float(chip["missing"].mean()), 4),
        "split": split_of(site_index, chip["variant"]),
        "image": {"file": f"chips/{chip_id}.npz", "arrays": {
            "s1": "float16 (2, 512, 512): simulated VV, VH backscatter in dB",
            "label": "int8 (512, 512): 1 water, 0 not water (pixel-aligned with s1)",
            "permanent": "uint8 (512, 512): 1 river/sea/pond (water that is there without a flood)",
            "landcover": "uint8 (512, 512): simulator land-cover code",
        }, "pixelSizeM": PX_M},
        "foundation": "procedural terrain, land cover and radar physics; brightness calibrated to the real "
                      "Sen1Floods11 India training chips",
        "details": chip["details"],
    }


def quicklook(rows: list[tuple[dict, dict]], path: Path, title: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.6 * len(rows)), squeeze=False)
    for ax, (meta, arrays) in zip(axes, rows):
        ax[0].imshow(arrays["s1"][0].astype(np.float32), cmap="gray", vmin=-25, vmax=0)
        ax[0].set_title(f"SYNTHETIC radar VV\n{meta['place']}, {meta['state']}", fontsize=8)
        ax[1].imshow(arrays["s1"][1].astype(np.float32), cmap="gray", vmin=-32, vmax=-7)
        ax[1].set_title(f"SYNTHETIC radar VH\n{meta['scenario']} ({meta['monthName']})", fontsize=8)
        mask = np.zeros((*arrays["label"].shape, 3), np.float32) + 0.93
        mask[arrays["label"] == 1] = (0.25, 0.55, 0.95)
        mask[arrays["permanent"] == 1] = (0.05, 0.2, 0.55)
        ax[2].imshow(mask)
        ax[2].set_title(f"mask: flood (light) + permanent water (dark)\nflooded {meta['floodedAreaKm2']:.2f} km2", fontsize=8)
        for a in ax:
            a.axis("off")
    fig.suptitle(f"{title} -- SYNTHETIC SIMULATION, not satellite data", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def generate(out_dir: Path, seed: int = DEFAULT_SEED, per_site: int = VARIANTS_PER_SITE, quicklooks: bool = True,
             calibration: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "chips").mkdir(parents=True, exist_ok=True)
    calibration = calibration or fit_calibration(seed, per_site)
    records, examples = [], {}
    for site_index, site in enumerate(FLOOD_SITES):
        for variant in range(per_site):
            chip = build_chip(seed, site_index, variant)
            s1 = apply_calibration(chip["s1"], chip["label"], calibration)
            chip_id = f"SYNFLOOD_{site['state'].replace(' ', '')}_{site_index:02d}{variant}"
            arrays = {"s1": s1.astype(np.float16), "label": chip["label"].astype(np.int8),
                      "permanent": chip["permanent"].astype(np.uint8), "landcover": chip["landcover"]}
            np.savez_compressed(out_dir / "chips" / f"{chip_id}.npz", **arrays)
            meta = metadata_for(chip, chip_id, seed, site_index)
            records.append(meta)
            if not chip["dry"]:
                examples.setdefault(site["kind"], [])
                if len(examples[site["kind"]]) < 3 and all(m["state"] != meta["state"] for m, _ in examples[site["kind"]]):
                    examples[site["kind"]].append((meta, arrays))
        print(f"  {site_index + 1:2d}/{len(FLOOD_SITES)}  {site['state']:<18} {site['place']}", flush=True)

    (out_dir / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    if quicklooks:
        (out_dir / "quicklooks").mkdir(exist_ok=True)
        for kind, rows in examples.items():
            quicklook(rows, out_dir / "quicklooks" / f"{kind}.png", f"India synthetic flood: {kind}")

    def count(key):
        out: dict = {}
        for r in records:
            out[r[key]] = out.get(r[key], 0) + 1
        return dict(sorted(out.items()))

    summary = {
        "dataset": "AtmoGuard India synthetic flood chips", "synthetic": True,
        "generator": f"ml.hazards.synth_flood v{GENERATOR_VERSION}", "seed": seed,
        "generatedAt": datetime.now(timezone.utc).isoformat(), "chips": len(records),
        "bySplit": count("split"), "byRegion": count("region"), "byState": count("state"),
        "byScenario": count("scenario"), "byFloodKind": count("floodKind"), "byLabel": count("floodLabel"),
        "meanWaterFraction": round(float(np.mean([r["waterFraction"] for r in records])), 4),
        "landcoverCodes": dict(enumerate(LANDCOVER_NAMES)),
        "calibratedAgainst": f"{calibration['realIndiaTrain']['chips']} real Sen1Floods11 India TRAIN chips (no valid/test chip read)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=config.FLOOD_SYNTHETIC_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-site", type=int, default=VARIANTS_PER_SITE)
    parser.add_argument("--no-quicklooks", action="store_true")
    args = parser.parse_args()
    summary = generate(args.out, args.seed, args.per_site, not args.no_quicklooks)
    print(json.dumps({k: v for k, v in summary.items() if k != "landcoverCodes"}, indent=2))


if __name__ == "__main__":
    main()
