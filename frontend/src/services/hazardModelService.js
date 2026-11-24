// Real backend client for the two trained hazard-imagery models -- flood
// mapping (Sentinel-1 radar) and lightning detection (infrared cloud tops).
// See backend/app/api/v1/endpoints/hazard_models.py.
//
// Same rule as imageryService: NO mock fallback. If the backend or a model
// is unavailable the UI says so; no prediction, mask or metric is ever
// fabricated in the browser.
import { request } from "./imageryService";

// The first call loads PyTorch and the model in the backend's worker process.
const PREDICT_TIMEOUT_MS = 120000;

const UNREACHABLE = { available: false, reason: "Backend unreachable.", backendUnreachable: true };

// Never throws.
export async function fetchHazardModelStatus() {
  try {
    return await request("/hazard-models/status", { timeoutMs: PREDICT_TIMEOUT_MS });
  } catch (err) {
    console.warn("[AtmoGuard] hazard model status unavailable:", err.message);
    return { flood: UNREACHABLE, lightning: UNREACHABLE };
  }
}

// Stored evaluation report (real test scores, baselines, synthetic scores). Never throws.
export async function fetchHazardModelMetrics(model) {
  try {
    return await request(`/hazard-models/${model}/metrics`);
  } catch (err) {
    console.warn(`[AtmoGuard] ${model} model metrics unavailable:`, err.message);
    return { available: false, reason: "Backend unreachable." };
  }
}

// Sample catalogs. Never throw: an empty catalog is shown as "unavailable".
export async function fetchFloodSamples() {
  try {
    return await request("/hazard-models/flood/samples");
  } catch (err) {
    console.warn("[AtmoGuard] flood sample catalog unavailable:", err.message);
    return { real: [], synthetic: [], unreachable: true };
  }
}

export async function fetchLightningSamples() {
  try {
    return await request("/hazard-models/lightning/samples");
  } catch (err) {
    console.warn("[AtmoGuard] lightning sample catalog unavailable:", err.message);
    return { real: [], synthetic: [], insat: [], insatRegions: [], unreachable: true };
  }
}

export function runFloodPrediction(sampleId) {
  return request("/hazard-models/flood/predict", { method: "POST", body: { sampleId }, timeoutMs: PREDICT_TIMEOUT_MS });
}

export function uploadFloodImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/hazard-models/flood/predict/upload", { method: "POST", formData, timeoutMs: PREDICT_TIMEOUT_MS });
}

export function runLightningPrediction(sampleId, region) {
  return request("/hazard-models/lightning/predict", { method: "POST", body: { sampleId, region }, timeoutMs: PREDICT_TIMEOUT_MS });
}
