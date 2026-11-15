// Thin mock-latency wrapper. Used by anything still backed by src/data/*.js
// generators (the AI nowcast, and whatever real-data fetch below it falls
// back to).
const MOCK_LATENCY_MS = 350;

export function mockRequest(factory, { latency = MOCK_LATENCY_MS } = {}) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      try {
        resolve(factory());
      } catch (err) {
        reject(err);
      }
    }, latency);
  });
}

// Matches the FastAPI backend's versioned prefix (settings.API_V1_PREFIX,
// see backend/app/core/config.py) — not just the bare host.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

// Set VITE_USE_BACKEND=false to skip the backend entirely and always use
// each service's direct/mock fallback (e.g. while the backend is known to
// be down).
export const BACKEND_ENABLED = import.meta.env.VITE_USE_BACKEND !== "false";

const BACKEND_TIMEOUT_MS = 3000;

// Shared backend-call helper — every service module that talks to the real
// FastAPI backend (weatherService, riskService, predictionService,
// alertService) uses this, so the "backend first, short timeout, visible
// fallback warning" contract is defined exactly once.
export async function fetchBackendJson(path, params = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);
  try {
    const query = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))
    );
    const qs = query.toString();
    const response = await fetch(`${API_BASE_URL}${path}${qs ? `?${qs}` : ""}`, {
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Backend responded with ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function postBackendJson(path) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { method: "POST", signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Backend responded with ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

export function warnFallback(what, err) {
  // Intentionally visible (not silent): a developer/tester can always see,
  // in the console, that a given call fell back to mock/direct data and why.
  console.warn(`[apiClient] backend unavailable for ${what}, falling back:`, err?.message || err);
}
