// Loads the baked India assets from public/geo/ (see scripts/build-geo-assets.py):
// terrain heights, state polygons and the state-id raster. Shared by the 3D
// scene and the flat fallback map. All pure data; no three.js import.
const HEIGHT_SIZE = 512;

let pending = null;

async function loadHeights(signal) {
  const buffer = await (await fetch("/geo/india-height.bin", { signal })).arrayBuffer();
  const raw = new Int16Array(buffer);
  if (raw.length !== HEIGHT_SIZE * HEIGHT_SIZE) throw new Error("terrain file has an unexpected size");
  // The file is stored north-first; the scene wants row 0 = south (v increases northward).
  const flipped = new Int16Array(raw.length);
  for (let row = 0; row < HEIGHT_SIZE; row += 1) {
    flipped.set(raw.subarray(row * HEIGHT_SIZE, (row + 1) * HEIGHT_SIZE), (HEIGHT_SIZE - 1 - row) * HEIGHT_SIZE);
  }
  return { data: flipped, size: HEIGHT_SIZE };
}

async function loadIds(signal) {
  const blob = await (await fetch("/geo/india-states.png", { signal })).blob();
  const bitmap = await createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const size = bitmap.width;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  // PNG rows run north -> south; flip so row 0 = south like the terrain grid (v increases northward).
  const ids = new Uint8Array(size * size);
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) ids[(size - 1 - row) * size + col] = rgba[(row * size + col) * 4];
  }
  return { data: ids, size };
}

// Soft 0..255 mask of "inside India" (box-blurred twice) for dimming the
// neighbouring countries without a hard edge.
function softMask(ids) {
  const { data, size } = ids;
  let a = new Float32Array(size * size);
  for (let i = 0; i < a.length; i += 1) a[i] = data[i] > 0 ? 1 : 0;
  const radius = 3;
  for (let pass = 0; pass < 2; pass += 1) {
    const b = new Float32Array(a.length);
    for (let y = 0; y < size; y += 1) {
      let sum = 0;
      for (let x = -radius; x <= radius; x += 1) sum += a[y * size + Math.min(size - 1, Math.max(0, x))];
      for (let x = 0; x < size; x += 1) {
        b[y * size + x] = sum / (2 * radius + 1);
        sum += a[y * size + Math.min(size - 1, x + radius + 1)] - a[y * size + Math.max(0, x - radius)];
      }
    }
    const c = new Float32Array(a.length);
    for (let x = 0; x < size; x += 1) {
      let sum = 0;
      for (let y = -radius; y <= radius; y += 1) sum += b[Math.min(size - 1, Math.max(0, y)) * size + x];
      for (let y = 0; y < size; y += 1) {
        c[y * size + x] = sum / (2 * radius + 1);
        sum += b[Math.min(size - 1, y + radius + 1) * size + x] - b[Math.max(0, y - radius) * size + x];
      }
    }
    a = c;
  }
  const out = new Uint8Array(size * size);
  for (let i = 0; i < out.length; i += 1) out[i] = Math.round(255 * a[i]);
  return { data: out, size };
}

// Everything the flat map needs (boundaries + ids). Cheap; no terrain.
export async function loadStates(signal) {
  return (await fetch("/geo/india-states.json", { signal })).json();
}

let idsPending = null;
// The state-id raster alone (cheap) -- enough for hover and for limiting rainfall
// statistics to Indian territory on devices that never load the terrain.
export function loadIdRaster() {
  if (!idsPending) {
    idsPending = loadIds().catch((err) => {
      idsPending = null;
      throw err;
    });
  }
  return idsPending;
}

// Everything the 3D India layer needs. Cached for the page's lifetime.
export function loadIndiaAssets() {
  if (!pending) {
    pending = (async () => {
      const [heights, states, ids] = await Promise.all([loadHeights(), loadStates(), loadIdRaster()]);
      return { heights, states, ids, mask: softMask(ids) };
    })().catch((err) => {
      pending = null; // let a later attempt retry
      throw err;
    });
  }
  return pending;
}

// Elevation in metres at lon/lat (bilinear), or 0 outside the window.
export function sampleHeight(heights, u, v) {
  if (u < 0 || u > 1 || v < 0 || v > 1) return 0;
  const { data, size } = heights;
  const x = u * (size - 1);
  const y = v * (size - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(size - 1, x0 + 1);
  const y1 = Math.min(size - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const top = data[y0 * size + x0] * (1 - fx) + data[y0 * size + x1] * fx;
  const bottom = data[y1 * size + x0] * (1 - fx) + data[y1 * size + x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

// State index (1-based, 0 = not India) at u,v inside the window; v increases northward (row 0 = south).
export function stateIdAt(ids, u, v) {
  if (u < 0 || u > 1 || v < 0 || v > 1) return 0;
  const x = Math.min(ids.size - 1, Math.floor(u * ids.size));
  const y = Math.min(ids.size - 1, Math.floor(v * ids.size));
  return ids.data[y * ids.size + x];
}
