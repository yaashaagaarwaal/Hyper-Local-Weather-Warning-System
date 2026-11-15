import { mockRequest, BACKEND_ENABLED, fetchBackendJson, warnFallback } from "./apiClient";
import { getShortTermForecast } from "../data/weatherData";
import { LOCATIONS } from "../data/locations";
import {
  getRiskSummary,
  getOverallRisk,
  findNearestGridCell,
  getGridById as getMockGridById,
  getGridHazardProbabilities as getMockGridHazards,
} from "../data/riskData";
import {
  fetchRiskGrid,
  fetchRiskGridStats,
  fetchGridById as fetchRealGridById,
  fetchGridHazardProbabilities as fetchRealGridHazards,
  findNearestGridId,
} from "./riskService";

// GET /predictions/nowcast?locationId=...
// Still mock: no trained model exists at 0-60min/15-min-step granularity
// (see docs/model_report.md — the real pipeline's shortest horizon is 1h).
// Clearly labeled "Prototype Prediction" in the UI (Predictions.jsx) — never
// presented as real.
export function fetchShortTermForecast(locationId) {
  return mockRequest(() => getShortTermForecast(locationId));
}

// GET /risk/grid, GET /risk/summary (all grids) — real backend-first (see
// services/riskService.js), re-exported under their original names so
// existing call sites don't need to change.
export { fetchRiskGrid, fetchRiskGridStats };

function locationCoords(locationId) {
  const loc = LOCATIONS.find((l) => l.id === locationId);
  return loc ? { lat: loc.lat, lng: loc.lng } : null;
}

// GET /risk/summary?locationId=... — real rule-based risk engine's hazard
// breakdown for the grid cell nearest this location (same nearest-coordinate
// honesty rule the backend itself uses — see grid_mapping_service.py).
export async function fetchRiskSummary(locationId) {
  if (BACKEND_ENABLED) {
    const coords = locationCoords(locationId);
    if (coords) {
      try {
        const gridId = await findNearestGridId(coords.lat, coords.lng);
        if (gridId) return await fetchRealGridHazards(gridId);
      } catch (err) {
        warnFallback(`risk summary for ${locationId}`, err);
      }
    }
  }
  return mockRequest(() => getRiskSummary(locationId));
}

// GET /risk/overall?locationId=...
export async function fetchOverallRisk(locationId) {
  if (BACKEND_ENABLED) {
    const coords = locationCoords(locationId);
    if (coords) {
      try {
        const gridId = await findNearestGridId(coords.lat, coords.lng);
        if (gridId) {
          const cell = await fetchRealGridById(gridId);
          return { score: cell.riskScore, level: cell.riskLevel };
        }
      } catch (err) {
        warnFallback(`overall risk for ${locationId}`, err);
      }
    }
  }
  return mockRequest(() => getOverallRisk(locationId));
}

// GET /risk/grid/nearest?lat=...&lng=...
export function fetchNearestGridCell(lat, lng) {
  return mockRequest(() => findNearestGridCell(lat, lng));
}

// GET /risk/grid/:id — real backend-first (see services/riskService.js).
export async function fetchGridById(gridId) {
  if (BACKEND_ENABLED) {
    try {
      return await fetchRealGridById(gridId);
    } catch (err) {
      warnFallback(`grid ${gridId}`, err);
    }
  }
  return mockRequest(() => getMockGridById(gridId));
}

// GET /risk/grid/:id/hazards — the real RULE-BASED risk engine's own hazard
// breakdown (see services/riskService.js). This is deliberately a separate
// signal from fetchGridMlForecast() below — never presented as the same
// thing (one is a documented rule-based score, the other a trained model's
// output).
export async function fetchGridHazardProbabilities(gridId) {
  if (BACKEND_ENABLED) {
    try {
      return await fetchRealGridHazards(gridId);
    } catch (err) {
      warnFallback(`hazard probabilities for grid ${gridId}`, err);
    }
  }
  return mockRequest(() => getMockGridHazards(gridId));
}

// GET /predictions/{gridId} — the real Phase 4 ML pipeline (baselines +
// XGBoost, trained on dataset/delhi_ncr_labeled_weather.csv — see
// docs/model_report.md). No mock fallback: if the backend or the trained
// artifacts aren't available, this resolves to `null` and the UI shows its
// own "model unavailable" state rather than presenting a fabricated
// prediction as if it were real (the whole reason this function exists
// separately from the rule-based fetchGridHazardProbabilities above).
export async function fetchGridMlForecast(gridId) {
  if (!BACKEND_ENABLED) return null;
  try {
    return await fetchBackendJson(`/predictions/${gridId}`);
  } catch (err) {
    warnFallback(`ML forecast for grid ${gridId}`, err);
    return null;
  }
}

// GET /analytics/series?locationId=...&range=...
// Backed by real weather data now (see services/weatherService.js) — kept
// here under its original name so Analytics.jsx doesn't need to change.
export { fetchAnalyticsSeries } from "./weatherService";
