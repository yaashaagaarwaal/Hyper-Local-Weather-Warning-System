"""Bake the geographic assets used by the 3D home page into frontend/public/geo/.

    backend/.venv/bin/python frontend/scripts/build-geo-assets.py [--refresh]

Everything is derived from real, openly licensed data and is reproducible.
Downloads are cached in frontend/scripts/.cache/ (gitignored); the baked
outputs in public/geo/ are small and committed.

REAL data baked here
  earth-day.jpg        NASA Blue Marble Next Generation + shaded relief and
                       bathymetry, via NASA GIBS (public domain), 2048x1024.
  earth-night.jpg      NASA/NOAA VIIRS Black Marble night lights, via GIBS.
  india-color.jpg      The same Blue Marble layer at higher resolution over
                       the India window (66-99E, 5-38N).
  india-height.bin     Terrain elevation over that window (Int16, metres),
                       from the AWS Open Data "Terrain Tiles" (Terrarium,
                       SRTM / ETOPO1 / other open DEMs) mosaic at zoom 6.
  india-states.json    State / union-territory boundaries and the national
                       outline: geoBoundaries gbOpen ADM1 for India (source
                       Pathways Data / Local Government Directory, ODbL),
                       simplified to ~1 km. These follow India's official
                       depiction of its territory.
  india-states.png     Raster lookup of those polygons (pixel value = state
                       index + 1, 0 = outside India), used for hover and for
                       dimming everything that is not India.

NOT real data
  earth-clouds.jpg     A procedurally generated, DECORATIVE cloud layer
                       (fractal noise with a plausible latitude pattern).
                       It is not live or historical cloud cover and carries
                       no weather information; the page fades it out before
                       any India weather is shown.
  earth-poster.jpg     A static render of the globe from earth-day/night,
                       used while the 3D scene loads and as the no-WebGL /
                       reduced-motion fallback.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import unicodedata
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
OUT = FRONTEND / "public" / "geo"
CACHE = FRONTEND / "scripts" / ".cache"
STATES_GEOJSON = ROOT / "dataset" / "boundaries" / "geoBoundaries-IND-ADM1.geojson"

GIBS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
TERRAIN = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# The India window. Square in degrees; the 3D patch and every raster below use it.
WEST, EAST, SOUTH, NORTH = 66.0, 99.0, 5.0, 38.0
HEIGHT_RES = 512          # elevation grid is HEIGHT_RES x HEIGHT_RES over the window
ID_RES = 768              # state-id raster resolution
OUTLINE_RES = 2048        # raster used to trace the national outline


def http_get(url: str, tries: int = 3) -> bytes:
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "atmoguard-asset-build/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"download failed: {url} ({last})")


def cached(name: str, url: str, refresh: bool) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists() and not refresh:
        return path.read_bytes()
    print(f"  downloading {name} ...", flush=True)
    data = http_get(url)
    path.write_bytes(data)
    return data


def gibs_image(name: str, layer: str, bbox: tuple, size: tuple, fmt: str, refresh: bool) -> Image.Image:
    west, south, east, north = bbox
    url = (f"{GIBS}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&SRS=EPSG:4326&FORMAT={fmt}"
           f"&LAYERS={layer}&BBOX={west},{south},{east},{north}&WIDTH={size[0]}&HEIGHT={size[1]}")
    return Image.open(io.BytesIO(cached(name, url, refresh)))


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    image.convert("RGB").save(path, "JPEG", quality=quality, optimize=True, progressive=True)
    print(f"  {path.relative_to(FRONTEND)}  {path.stat().st_size / 1024:.0f} KB")


# ----------------------------------------------------------------------------
# Imagery
# ----------------------------------------------------------------------------
def build_imagery(refresh: bool) -> tuple[Image.Image, Image.Image]:
    print("imagery (NASA GIBS)")
    day = gibs_image("bm_global_4096.jpg", "BlueMarble_ShadedRelief_Bathymetry", (-180, -90, 180, 90), (4096, 2048), "image/jpeg", refresh)
    day = day.convert("RGB").resize((2048, 1024), Image.LANCZOS)
    save_jpeg(day, OUT / "earth-day.jpg", 84)

    night = gibs_image("night_global_2048.jpg", "VIIRS_Black_Marble", (-180, -90, 180, 90), (2048, 1024), "image/jpeg", refresh)
    night_gray = night.convert("L")
    # Keep only the lights: crush the faint sensor floor, leave city glow soft.
    arr = np.asarray(night_gray, dtype=np.float32) / 255.0
    arr = np.clip(arr - cv2.GaussianBlur(arr, (0, 0), 7) * 0.9, 0, 1)         # keep point lights, drop the land floor
    arr = np.clip(arr - 0.03, 0, 1)
    arr = np.clip(arr * 5.0, 0, 1) ** 0.8
    arr[int(arr.shape[0] * 0.90):] = 0                                         # Antarctica is not lit
    save_jpeg(Image.fromarray((arr * 255).astype(np.uint8), "L"), OUT / "earth-night.jpg", 80)

    india = gibs_image("bm_india_4096.jpg", "BlueMarble_ShadedRelief_Bathymetry", (WEST, SOUTH, EAST, NORTH), (4096, 4096), "image/jpeg", refresh)
    india = india.convert("RGB").resize((2048, 2048), Image.LANCZOS)
    save_jpeg(india, OUT / "india-color.jpg", 86)
    return day, night_gray


# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------
def tile_xy(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def build_height(refresh: bool) -> None:
    print("terrain (AWS Terrain Tiles / Terrarium, z=6)")
    z = 6
    x0, y0 = tile_xy(WEST, NORTH, z)
    x1, y1 = tile_xy(EAST, SOUTH, z)
    tx0, tx1, ty0, ty1 = int(math.floor(x0)), int(math.floor(x1)), int(math.floor(y0)), int(math.floor(y1))
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            png = Image.open(io.BytesIO(cached(f"terrarium_{z}_{tx}_{ty}.png", TERRAIN.format(z=z, x=tx, y=ty), refresh))).convert("RGB")
            rgb = np.asarray(png, dtype=np.float32)
            elevation = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = elevation

    # Resample the Web-Mercator mosaic onto an equirectangular lon/lat grid.
    lons = WEST + (np.arange(HEIGHT_RES) + 0.5) / HEIGHT_RES * (EAST - WEST)
    lats = NORTH - (np.arange(HEIGHT_RES) + 0.5) / HEIGHT_RES * (NORTH - SOUTH)
    xs = ((lons + 180.0) / 360.0 * 2 ** z - tx0) * 256.0 - 0.5
    ys = np.array([(1.0 - math.asinh(math.tan(math.radians(l))) / math.pi) / 2.0 * 2 ** z for l in lats])
    ys = (ys - ty0) * 256.0 - 0.5
    map_x, map_y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    elevation = cv2.remap(mosaic, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    elevation = cv2.GaussianBlur(elevation, (0, 0), 0.8)          # tidy DEM noise; keeps ridges
    elevation = np.clip(elevation, 0, 8900)                       # the page draws the sea flat; land >= 0
    (OUT / "india-height.bin").write_bytes(elevation.round().astype("<i2").tobytes())
    print(f"  india-height.bin  {HEIGHT_RES}x{HEIGHT_RES} int16  {(OUT / 'india-height.bin').stat().st_size / 1024:.0f} KB  "
          f"max {elevation.max():.0f} m")


# ----------------------------------------------------------------------------
# Boundaries
# ----------------------------------------------------------------------------
def clean_name(name: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def polygons_of(geometry: dict) -> list[list[np.ndarray]]:
    """-> list of polygons, each a list of rings (first = exterior)."""
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return [[np.asarray(ring, dtype=np.float64)[:, :2] for ring in poly] for poly in polys]


def simplify(ring: np.ndarray, epsilon: float) -> np.ndarray | None:
    """Douglas-Peucker. Returns None for specks too small to draw at the page's scale."""
    pts = ring.astype(np.float32).reshape(-1, 1, 2)
    if abs(cv2.contourArea(pts)) < 0.0006:                       # < ~7 km2: invisible at this scale
        return None
    for eps in (epsilon, epsilon / 3, epsilon / 10):
        approx = cv2.approxPolyDP(pts, eps, True).reshape(-1, 2)
        if len(approx) >= 4:
            return approx
    return ring.astype(np.float32)


def to_pixels(ring: np.ndarray, size: int) -> np.ndarray:
    px = (ring[:, 0] - WEST) / (EAST - WEST) * size
    py = (NORTH - ring[:, 1]) / (NORTH - SOUTH) * size
    return np.round(np.stack([px, py], axis=1)).astype(np.int32)


def build_states() -> None:
    print("boundaries (geoBoundaries gbOpen ADM1 India)")
    if not STATES_GEOJSON.exists():
        sys.exit(f"missing {STATES_GEOJSON}; see dataset/scripts/download_geoboundaries_india_adm2.py")
    features = json.loads(STATES_GEOJSON.read_text())["features"]
    features.sort(key=lambda f: clean_name(f["properties"]["shapeName"]))
    states, ids = [], np.zeros((ID_RES, ID_RES), np.uint8)
    outline_raster = np.zeros((OUTLINE_RES, OUTLINE_RES), np.uint8)
    total_points = 0
    for index, feature in enumerate(features, start=1):
        name = clean_name(feature["properties"]["shapeName"])
        rings_out = []
        for poly in polygons_of(feature["geometry"]):
            for r, ring in enumerate(poly):
                cv2.fillPoly(ids if r == 0 else ids, [to_pixels(ring, ID_RES)], index if r == 0 else 0)
                if r == 0:
                    cv2.fillPoly(outline_raster, [to_pixels(ring, OUTLINE_RES)], 255)
                if r == 0:                                         # exterior rings only: interior "holes" are other states
                    simple = simplify(ring, 0.015)
                    if simple is None:
                        continue
                    rings_out.append([round(float(v), 3) for v in simple.reshape(-1)])
                    total_points += len(simple)
        states.append({"id": index, "name": name, "rings": rings_out})

    # National outline = the outer contour of the union of all states.
    outline_raster = cv2.morphologyEx(outline_raster, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(outline_raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = []
    for contour in contours:
        if cv2.contourArea(contour) < 40:                          # drop specks
            continue
        approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2).astype(np.float64)
        lon = WEST + (approx[:, 0] + 0.5) / OUTLINE_RES * (EAST - WEST)
        lat = NORTH - (approx[:, 1] + 0.5) / OUTLINE_RES * (NORTH - SOUTH)
        outline.append([round(float(v), 3) for v in np.stack([lon, lat], axis=1).reshape(-1)])

    payload = {
        "bounds": {"west": WEST, "east": EAST, "south": SOUTH, "north": NORTH},
        "states": states, "outline": outline,
        "source": "geoBoundaries gbOpen ADM1 (India), sourced from Pathways Data / lgdirectory.gov.in; ODbL 1.0",
    }
    (OUT / "india-states.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"  india-states.json  {len(states)} states/UTs, {total_points} points, {len(outline)} outline rings  "
          f"{(OUT / 'india-states.json').stat().st_size / 1024:.0f} KB")

    # Index raster -> mask-friendly PNG (opaque greyscale; value = state id).
    Image.fromarray(ids, "L").save(OUT / "india-states.png", optimize=True)
    print(f"  india-states.png  {ID_RES}x{ID_RES}  {(OUT / 'india-states.png').stat().st_size / 1024:.0f} KB")


# ----------------------------------------------------------------------------
# Decorative clouds (procedural -- see module docstring)
# ----------------------------------------------------------------------------
def value_noise_3d(points: np.ndarray, seed: int) -> np.ndarray:
    """Smooth 3-D value noise at `points` (N, 3), in [0, 1]. The lattice is rotated
    per call so octaves do not line up along the axes."""
    rng = np.random.default_rng(seed)
    table = rng.random((64, 64, 64)).astype(np.float32)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    points = points @ rot.astype(np.float32) + rng.random(3).astype(np.float32) * 17
    i = np.floor(points).astype(np.int64)
    f = points - i
    f = f * f * f * (f * (f * 6 - 15) + 10)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * table[(i[:, 0] + dx) & 63, (i[:, 1] + dy) & 63, (i[:, 2] + dz) & 63]
    return out


def fbm(points: np.ndarray, seed: int, octaves: int, freq: float, gain: float = 0.5) -> np.ndarray:
    total, amp, norm = np.zeros(len(points), np.float32), 1.0, 0.0
    for octave in range(octaves):
        total += amp * value_noise_3d(points * freq, seed + octave)
        norm += amp
        amp *= gain
        freq *= 2.03
    return total / norm


def build_clouds() -> None:
    print("clouds (procedural, decorative)")
    width, height = 2048, 1024
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(lon, lat)
    # Points on the unit sphere, so the pattern is seamless and has no polar pinching.
    pts = np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), -np.cos(lat) * np.sin(lon)], axis=-1).reshape(-1, 3).astype(np.float32)
    latitude = np.degrees(lat.reshape(-1))

    warp = np.stack([fbm(pts, 11 + 7 * k, 3, 1.6) - 0.5 for k in range(3)], axis=1)
    p = pts + warp * 0.30                                         # gentle flow-like distortion
    body = fbm(p, 100, 6, 4.4, 0.52)                              # cloud decks and cells
    detail = fbm(p, 200, 5, 9.0, 0.55)                            # fine texture inside them
    streak = fbm(p * np.array([1.0, 2.6, 1.0], np.float32), 300, 4, 6.0, 0.55)   # east-west wisps

    # Zonal cloudiness: wet band near the equator and the mid-latitude storm tracks,
    # clear subtropical belts, bright polar edge.
    zonal = (0.50 * np.exp(-((latitude - 5) / 10) ** 2) + 0.60 * np.exp(-((np.abs(latitude) - 52) / 13) ** 2)
             + 0.22 + 0.18 * np.exp(-((np.abs(latitude) - 78) / 8) ** 2) - 0.20 * np.exp(-((np.abs(latitude) - 25) / 8) ** 2))
    coverage = np.clip(zonal, 0.08, 1.0)
    threshold = 0.62 - 0.24 * coverage
    deck = np.clip((body - threshold) / 0.14, 0, 1)
    deck = deck * deck * (3 - 2 * deck)
    wisps = np.clip((streak - 0.60) / 0.2, 0, 1) * 0.45 * coverage
    cloud = np.clip(deck * (0.62 + 0.55 * detail) + wisps, 0, 1) ** 1.1
    img = (cloud.reshape(height, width) * 255).astype(np.uint8)
    save_jpeg(Image.fromarray(img, "L"), OUT / "earth-clouds.jpg", 82)


# ----------------------------------------------------------------------------
# Poster
# ----------------------------------------------------------------------------
def sample_equirect(texture: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    h, w = texture.shape[:2]
    x = ((lon + np.pi) / (2 * np.pi) * w - 0.5).astype(np.float32)
    y = ((np.pi / 2 - lat) / np.pi * h - 0.5).astype(np.float32)
    out = cv2.remap(texture, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return out.reshape(lon.shape + (texture.shape[2],))   # remap drops a trailing length-1 channel axis


def build_poster(day: Image.Image, night: Image.Image) -> None:
    print("poster")
    size = 1400
    day_t = np.asarray(day.convert("RGB"), dtype=np.float32) / 255.0
    night_t = np.asarray(night.convert("L"), dtype=np.float32)[..., None] / 255.0
    cloud_t = np.asarray(Image.open(OUT / "earth-clouds.jpg").convert("L"), dtype=np.float32)[..., None] / 255.0

    # Globe of radius R filling most of the frame, viewed from far away, India centred left of the meridian.
    R = 0.46 * size
    cy = cx = size / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx, ny = (xx - cx) / R, -(yy - cy) / R
    r2 = nx ** 2 + ny ** 2
    inside = r2 < 1.0
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    # View rotation: tilt north towards the viewer, centre longitude 66E so India sits left of centre.
    tilt = math.radians(24)
    ex, ey, ez = nx, ny * math.cos(tilt) + nz * math.sin(tilt), -ny * math.sin(tilt) + nz * math.cos(tilt)  # earth-fixed up-axis frame
    lat = np.arcsin(np.clip(ey, -1, 1))
    lon = np.arctan2(ex, ez) + math.radians(66)
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    albedo = sample_equirect(day_t, lon, lat)
    lights = sample_equirect(night_t, lon, lat)
    cloud = sample_equirect(cloud_t, lon, lat)
    # Keep the decorative cloud layer thin over India so the subcontinent reads clearly
    # (the page applies the same clearing; see EarthScene.jsx).
    c_lon, c_lat = math.radians(80.0), math.radians(23.0)
    cos_d = np.sin(lat) * math.sin(c_lat) + np.cos(lat) * math.cos(c_lat) * np.cos(lon - c_lon)
    ang = np.arccos(np.clip(cos_d, -1, 1))
    cloud = cloud * (1 - 0.9 * np.exp(-(ang / 0.42) ** 2))
    sun = np.array([-0.55, 0.45, 0.70], np.float32)
    sun /= np.linalg.norm(sun)
    ndl = nx * sun[0] + ny * sun[1] + nz * sun[2]
    day_amt = np.clip((ndl + 0.10) / 0.5, 0, 1)
    day_amt = day_amt * day_amt * (3 - 2 * day_amt)
    shade = 0.04 + 0.96 * day_amt[..., None]
    surface = albedo * shade
    veil = 0.7 * cloud
    surface = surface * (1 - veil) + veil * 0.92 * (0.04 + 0.96 * day_amt[..., None])      # clouds lit like the surface
    surface += (1 - day_amt)[..., None] * lights * np.array([1.0, 0.78, 0.45], np.float32) * 0.8
    rim = (1 - nz) ** 3.0
    atmosphere = rim[..., None] * np.array([0.30, 0.52, 0.85], np.float32) * (0.25 + 0.9 * day_amt[..., None])
    surface = surface + atmosphere

    bg = np.array([5, 8, 15], np.float32) / 255.0
    radius = np.sqrt(r2)
    # Thin atmospheric halo just outside the limb, brighter on the sunlit side.
    lit = np.clip(nx * sun[0] + ny * sun[1] + 0.3, 0, 1)[..., None]
    halo = (np.exp(-((radius - 1.0) / 0.035) ** 2) * (radius >= 1.0))[..., None]
    backdrop = bg + halo * np.array([0.20, 0.38, 0.68], np.float32) * 0.55 * (0.25 + 0.75 * lit)
    coverage = np.clip((1.0 - radius) * R, 0, 1)[..., None]                                 # anti-aliased limb
    out = backdrop * (1 - coverage) + np.clip(surface, 0, 1) * coverage
    save_jpeg(Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)), OUT / "earth-poster.jpg", 82)


def build_rain_legend(refresh: bool) -> None:
    """NASA's own colour bar for the GIBS IMERG rain-rate layer, sampled into a lookup table.
    The page uses it to turn the colours of the live rainfall image back into mm/h
    (so it can restyle them and report real numbers). Rate = 0.1 * 530**u for u in [0, 1]
    (log scale, 0.1 -> 53 mm/h; the legend's last label is ">= 53")."""
    print("rain legend (NASA GIBS)")
    png = cached("gpm_rain_legend.png", "https://gibs.earthdata.nasa.gov/legends/GPM_Precipitation_Rate_H.png", refresh)
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    row = rgba[29]                                   # inside the "Rain Rate" colour bar (x 32..389)
    bar = row[32:390, :3].astype(np.float32)
    n = 128
    idx = np.linspace(0, len(bar) - 1, n)
    colors = [[int(round(float(np.interp(i, np.arange(len(bar)), bar[:, c])))) for c in range(3)] for i in idx]
    payload = {"unit": "mm/h", "scale": "log", "min": 0.1, "max": 53.0, "colors": colors,
               "source": "NASA GIBS legend GPM_Precipitation_Rate_H.png (layer IMERG_Precipitation_Rate)"}
    (OUT / "rain-legend.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"  rain-legend.json  {n} stops")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download everything instead of using the cache")
    parser.add_argument("--only", choices=("imagery", "height", "states", "clouds", "poster", "rain"), help="rebuild one asset group")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = args.only
    day = night = None
    if only in (None, "imagery", "poster"):
        day, night = build_imagery(args.refresh)
    if only in (None, "height"):
        build_height(args.refresh)
    if only in (None, "states"):
        build_states()
    if only in (None, "rain"):
        build_rain_legend(args.refresh)
    if only in (None, "clouds", "poster"):
        build_clouds()
    if only in (None, "poster"):
        build_poster(day, night)
    total = sum(p.stat().st_size for p in OUT.glob("*") if p.is_file())
    print(f"done. public/geo total {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
