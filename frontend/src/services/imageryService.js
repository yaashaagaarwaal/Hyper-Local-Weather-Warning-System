// Real backend client for the satellite/precipitation imagery + video
// processing module (see backend/app/api/v1/endpoints/imagery.py).
//
// Unlike weatherService/riskService/alertService, this has NO mock
// fallback -- it's a new, backend-only capability with nothing in
// src/data/*.js to fall back to. A failure here means the feature is
// genuinely unavailable right now; callers must show that plainly
// ("unavailable", "failed"), never fabricate a processed image or sample.
//
// Processing (denoise/optical-flow/video-encode) is slower than a typical
// weather/risk call, so this module uses its own longer timeout rather
// than apiClient's 3s BACKEND_TIMEOUT_MS (tuned for fast calls).
import { API_BASE_URL, BACKEND_ENABLED } from "./apiClient";

const IMAGERY_TIMEOUT_MS = 20000;
// apiClient's API_BASE_URL already carries the versioned /api/v1 prefix;
// media is served unversioned, from the same backend origin.
const BACKEND_ORIGIN = API_BASE_URL.replace(/\/api\/v1\/?$/, "");

export function mediaUrl(path) {
  if (!path) return null;
  return path.startsWith("http") ? path : `${BACKEND_ORIGIN}${path}`;
}

// Exported for hazardModelService.js, which talks to the same backend with the
// same rules (no mock fallback, per-call timeout).
export async function request(path, { method = "GET", params, body, formData, timeoutMs = IMAGERY_TIMEOUT_MS } = {}) {
  if (!BACKEND_ENABLED) throw new Error("Backend disabled (VITE_USE_BACKEND=false)");

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const query = params
      ? new URLSearchParams(
          Object.entries(params).flatMap(([key, value]) =>
            Array.isArray(value) ? value.map((v) => [key, v]) : value === undefined || value === null ? [] : [[key, value]]
          )
        ).toString()
      : "";
    const url = `${API_BASE_URL}${path}${query ? `?${query}` : ""}`;

    const init = { method, signal: controller.signal };
    if (formData) {
      init.body = formData; // browser sets the multipart Content-Type boundary itself
    } else if (body !== undefined) {
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify(body);
    }

    const response = await fetch(url, init);
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.error || `Imagery backend responded with ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

// The small, real, pre-acquired NASA GIBS sample set (see
// dataset/imagery/metadata/SOURCES.md) -- lets the viewer work with zero
// uploads. Returns [] (not a thrown error) on failure since this backs an
// always-visible gallery section; the page shows an "unavailable" banner
// instead of blocking the rest of the page.
export async function fetchImagerySamples() {
  try {
    return await request("/imagery/samples");
  } catch (err) {
    console.warn("[imageryService] samples unavailable:", err?.message || err);
    return [];
  }
}

export async function uploadImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/imagery/upload", { method: "POST", formData });
}

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/imagery/video/upload", { method: "POST", formData });
}

export async function processImage(imageId, operations) {
  return request(`/imagery/process/${encodeURIComponent(imageId)}`, {
    method: "POST",
    params: { operations },
  });
}

export async function compareImages(beforeId, afterId) {
  return request("/imagery/compare", { method: "POST", params: { beforeId, afterId } });
}

export async function analyzeMotion(beforeId, afterId) {
  return request("/imagery/motion", { method: "POST", params: { beforeId, afterId } });
}

export async function extractVideoFrames(videoId, maxFrames = 12) {
  return request(`/imagery/video/${encodeURIComponent(videoId)}/frames`, {
    method: "POST",
    params: { maxFrames },
  });
}

export async function generateVideo(imageIds, fps = 2) {
  return request("/imagery/video/generate", { method: "POST", body: { imageIds, fps } });
}

// --- Trained deep-learning nowcast (INSAT-3DR infrared) --------------------
//
// A separate family of endpoints from everything above: these are backed by
// a trained ConvLSTM (backend/app/services/imagery/nowcast_service.py), not
// classical OpenCV. Same no-mock rule -- if the backend or the model is
// unavailable the UI says so; nothing is fabricated in the browser.

// First call loads PyTorch + the model, and a full-sector run renders ~30
// large PNGs, so this needs far longer than a normal imagery call.
const NOWCAST_TIMEOUT_MS = 120000;

// Never throws: `{ available: false, reason }` covers both "backend
// unreachable" and "backend up but no usable model artifact".
export async function fetchMlStatus() {
  try {
    return await request("/imagery/ml/status");
  } catch (err) {
    console.warn("[AtmoGuard] ML imagery model status unavailable:", err.message);
    return { available: false, reason: "Backend unreachable.", backendUnreachable: true };
  }
}

// Catalog of real INSAT-3DR frames on disk. Never throws.
export async function fetchInsatFrames() {
  try {
    return await request("/imagery/insat/frames");
  } catch (err) {
    console.warn("[AtmoGuard] INSAT-3DR frame catalog unavailable:", err.message);
    return { available: false, reason: "Backend unreachable.", frames: [], regions: [] };
  }
}

export function runMlNowcast(anchorFrameId, region) {
  return request("/imagery/ml/nowcast", {
    method: "POST",
    body: { anchorFrameId, region },
    timeoutMs: NOWCAST_TIMEOUT_MS,
  });
}
