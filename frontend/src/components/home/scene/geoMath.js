// Geography helpers shared by the 3D scene and the flat SVG fallback.
// No three.js import here: the fallback path must not pull the 3D bundle in.

// The India window every baked asset in public/geo/ covers
// (see frontend/scripts/build-geo-assets.py).
export const INDIA_WINDOW = { west: 66, east: 99, south: 5, north: 38 };
// Where the camera settles (roughly the geographic middle of the country).
export const INDIA_FOCUS = { lon: 80.5, lat: 22.5 };
export const EARTH_RADIUS_M = 6371000;

export const DEG = Math.PI / 180;

// Unit vector for a longitude / latitude (degrees), in the globe's local frame:
// lon 0 on +X, lon 90E on -Z, north pole on +Y. This is the same mapping as
// three.js' SphereGeometry with an equirectangular texture.
export function lonLatToXYZ(lon, lat, radius = 1) {
  const lo = lon * DEG;
  const la = lat * DEG;
  return [
    radius * Math.cos(la) * Math.cos(lo),
    radius * Math.sin(la),
    -radius * Math.cos(la) * Math.sin(lo),
  ];
}

export function xyzToLonLat(x, y, z) {
  const r = Math.hypot(x, y, z) || 1;
  return { lon: Math.atan2(-z, x) / DEG, lat: Math.asin(y / r) / DEG };
}

// Normalised position inside the India window: u west->east, v south->north (0..1).
export function windowUV(lon, lat, w = INDIA_WINDOW) {
  return { u: (lon - w.west) / (w.east - w.west), v: (lat - w.south) / (w.north - w.south) };
}

export function insideWindow(lon, lat, w = INDIA_WINDOW) {
  return lon >= w.west && lon <= w.east && lat >= w.south && lat <= w.north;
}

export function smoothstep(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Shortest-arc interpolation between two angles in radians.
export function lerpAngle(a, b, t) {
  let d = (b - a) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}
