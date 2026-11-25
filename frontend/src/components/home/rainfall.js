// Live rainfall layer for the India scene: NASA IMERG rain-rate imagery from
// NASA GIBS (satellite-derived precipitation, public, CORS-enabled).
//
// Nothing here is simulated. The image NASA returns is colour-coded by rain
// rate; we convert those colours back into mm/h with NASA's own legend
// (public/geo/rain-legend.json), restyle them in the page's palette, and
// compute the statistics shown beside the map from the same numbers. If NASA
// does not answer, or has no image for the last few days, the result says so
// (`status: "unavailable"`) and no rainfall is drawn.
import { INDIA_WINDOW } from "./scene/geoMath";

const GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi";
const LAYER = "IMERG_Precipitation_Rate"; // daily mean rain rate
const SIZE = 1024;
const MAX_DAYS_BACK = 4;
const TIMEOUT_MS = 15000;

function isoDay(date) {
  return date.toISOString().slice(0, 10);
}

function gibsUrl(day) {
  const { west, south, east, north } = INDIA_WINDOW;
  return (
    `${GIBS_WMS}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&SRS=EPSG:4326&FORMAT=image/png&TRANSPARENT=TRUE` +
    `&EXCEPTIONS=application/vnd.ogc.se_xml&LAYERS=${LAYER}&BBOX=${west},${south},${east},${north}` +
    `&WIDTH=${SIZE}&HEIGHT=${SIZE}&TIME=${day}`
  );
}

async function fetchImage(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok || !(response.headers.get("content-type") || "").startsWith("image/")) {
    throw new Error(`NASA GIBS responded ${response.status}`);
  }
  return createImageBitmap(await response.blob(), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
}

// Restrained colour ramp for the rain overlay: pale blue for drizzle, deeper
// blue for steady rain, violet then amber for the heaviest rates.
const RAMP = [
  [0.0, [130, 200, 240, 0.62]],
  [0.37, [74, 156, 236, 0.78]], // ~1 mm/h
  [0.62, [92, 118, 240, 0.88]], // ~4 mm/h
  [0.82, [160, 98, 238, 0.94]], // ~16 mm/h
  [1.0, [244, 172, 66, 0.98]], // >= 53 mm/h
];

function ramp(u) {
  for (let i = 1; i < RAMP.length; i += 1) {
    if (u <= RAMP[i][0]) {
      const [u0, c0] = RAMP[i - 1];
      const [u1, c1] = RAMP[i];
      const t = (u - u0) / (u1 - u0);
      return c0.map((v, k) => v + (c1[k] - v) * t);
    }
  }
  return RAMP[RAMP.length - 1][1];
}

// Exposed so the legend in the UI uses exactly the same colours.
export const RAIN_RAMP_STOPS = RAMP.map(([u, [r, g, b]]) => ({ u, color: `rgb(${r}, ${g}, ${b})` }));
export const RAIN_RATE_MIN = 0.1;
export const RAIN_RATE_MAX = 53;
export function rainRateAt(u) {
  return RAIN_RATE_MIN * (RAIN_RATE_MAX / RAIN_RATE_MIN) ** u;
}

function buildLookup(legend) {
  const colors = legend.colors;
  const cache = new Map();
  return (r, g, b) => {
    const key = (r << 16) | (g << 8) | b;
    let u = cache.get(key);
    if (u === undefined) {
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < colors.length; i += 1) {
        const c = colors[i];
        const d = (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      u = best / (colors.length - 1);
      cache.set(key, u);
    }
    return u;
  };
}

// `indiaIds`: optional {data: Uint8Array, size} state raster, row 0 = south (0 = outside India)
// so the statistics can be limited to Indian territory.
export async function loadRainfall({ indiaIds, signal } = {}) {
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), TIMEOUT_MS);
  const merged = signal && typeof AbortSignal.any === "function" ? AbortSignal.any([signal, timeout.signal]) : timeout.signal;
  try {
    const legend = await (await fetch("/geo/rain-legend.json", { signal: merged })).json();
    const toU = buildLookup(legend);

    const today = new Date();
    for (let back = 1; back <= MAX_DAYS_BACK; back += 1) {
      const day = isoDay(new Date(today.getTime() - back * 86400000));
      const bitmap = await fetchImage(gibsUrl(day), merged);
      const canvas = document.createElement("canvas");
      canvas.width = SIZE;
      canvas.height = SIZE;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0);
      const image = ctx.getImageData(0, 0, SIZE, SIZE);
      const px = image.data;

      let wet = 0;
      let peakU = 0;
      let indiaCells = 0;
      let indiaWet = 0;
      let indiaPeakU = 0;
      const idSize = indiaIds?.size;
      for (let i = 0; i < SIZE * SIZE; i += 1) {
        const o = i * 4;
        let inIndia = false;
        if (idSize) {
          const x = Math.floor(((i % SIZE) / SIZE) * idSize);
          const y = idSize - 1 - Math.floor((Math.floor(i / SIZE) / SIZE) * idSize); // image rows run north -> south; ids run south -> north
          inIndia = indiaIds.data[y * idSize + x] > 0;
          if (inIndia) indiaCells += 1;
        }
        if (px[o + 3] < 8) continue;
        const u = toU(px[o], px[o + 1], px[o + 2]);
        const [r, g, b, a] = ramp(u);
        px[o] = r;
        px[o + 1] = g;
        px[o + 2] = b;
        px[o + 3] = Math.round(255 * a);
        wet += 1;
        if (u > peakU) peakU = u;
        if (inIndia) {
          indiaWet += 1;
          if (u > indiaPeakU) indiaPeakU = u;
        }
      }
      if (wet === 0) continue; // nothing for that day yet: try the previous one
      ctx.putImageData(image, 0, 0);
      // IMERG cells are ~11 km: soften their square edges for display (statistics above use the unblurred values).
      const soft = document.createElement("canvas");
      soft.width = SIZE;
      soft.height = SIZE;
      const softCtx = soft.getContext("2d", { willReadFrequently: true });
      softCtx.filter = "blur(2px)";
      softCtx.drawImage(canvas, 0, 0);
      const smooth = softCtx.getImageData(0, 0, SIZE, SIZE).data;
      // Raw RGBA rows, south-first, for a GPU data texture (the canvas serves the flat SVG map).
      const pixels = new Uint8Array(SIZE * SIZE * 4);
      for (let row = 0; row < SIZE; row += 1) pixels.set(smooth.subarray(row * SIZE * 4, (row + 1) * SIZE * 4), (SIZE - 1 - row) * SIZE * 4);
      return {
        status: "ready",
        canvas: soft,
        pixels,
        size: SIZE,
        day,
        layer: LAYER,
        source: "NASA IMERG via NASA GIBS",
        wetAreaPctWindow: (100 * wet) / (SIZE * SIZE),
        peakRateWindow: rainRateAt(peakU),
        wetAreaPctIndia: indiaCells ? (100 * indiaWet) / indiaCells : null,
        peakRateIndia: indiaWet ? rainRateAt(indiaPeakU) : null,
      };
    }
    return { status: "empty", reason: `NASA GIBS returned no IMERG rainfall for the last ${MAX_DAYS_BACK} days.` };
  } catch (err) {
    if (signal?.aborted) return { status: "aborted" };
    return { status: "unavailable", reason: err?.name === "AbortError" ? "NASA GIBS did not answer in time." : err?.message || "NASA GIBS unreachable." };
  } finally {
    clearTimeout(timer);
  }
}
