// Real risk-grid client (Phase 5 frontend integration).
//
// Tries the FastAPI backend's `/grids` + `/risk/grids` endpoints first (the
// real, backend-computed Hyper-Local Grid and rule-based risk assessment —
// see backend/app/services/risk_engine/), with the same backend-first /
// short-timeout / visible-fallback-warning contract as weatherService.js,
// falling back to the deterministic mock grid in `src/data/riskData.js` the
// instant the backend errors or times out. Every adapted cell carries a
// `source` field (`"risk-engine-v1"` for real, `"simulated"` for mock) so a
// component can tell which one actually answered, and `dataAvailable` /
// `confidence` from the real path's own honesty fields.
import { BACKEND_ENABLED, fetchBackendJson, warnFallback } from "./apiClient";
import {
  RISK_GRID,
  getRiskGridStats,
  getGridById as getMockGridById,
  getGridHazardProbabilities as getMockGridHazards,
  findNearestGridCell as findNearestMockGridCell,
} from "../data/riskData";

// Two different hazard-key vocabularies meet here: `dominantHazard` (on
// RiskAssessmentOut) uses the risk engine's own snake_case hazard names
// ("heavy_rainfall", "thunderstorm", "lightning", "flood"), while each
// HazardAssessmentOut's `hazard` field in the `hazards` list uses the same
// snake_case names too — both map through this one table.
const HAZARD_LABELS = {
  heavy_rainfall: "Heavy Rainfall",
  thunderstorm: "Thunderstorm",
  lightning: "Lightning Potential",
  flood: "Flood Risk",
};

function hazardLabel(key) {
  return HAZARD_LABELS[key] ?? key;
}

// A real RiskAssessmentOut (see backend/app/schemas/risk.py) merged with its
// matching GridCellOut geometry, reshaped into the exact object shape
// src/data/riskData.js's mock cells use, so every existing component
// (RiskMap, GridDetailPanel, Predictions.jsx) works unmodified.
function adaptCell(geometry, assessment) {
  const basis = assessment?.weatherBasis ?? {};
  return {
    id: geometry.id,
    bounds: geometry.bounds,
    center: [Number(geometry.centerLat), Number(geometry.centerLng)],
    riskScore: assessment?.overallScore ?? 0,
    riskLevel: assessment?.riskLevel ?? "low",
    // Rounded for display — the backend keeps full precision (these are
    // sometimes an average across >=2 nearby stations, see
    // grid_mapping_service.py), rounding is purely a presentation choice.
    temperature: basis.temperatureC != null ? Math.round(basis.temperatureC * 10) / 10 : null,
    rainfall: basis.rainfallMm != null ? Math.round(basis.rainfallMm * 10) / 10 : null,
    humidity: basis.humidityPct != null ? Math.round(basis.humidityPct) : null,
    wind: basis.windSpeedKmh != null ? Math.round(basis.windSpeedKmh * 10) / 10 : null,
    hazardType: hazardLabel(assessment?.dominantHazard),
    assessedAt: assessment?.assessedAt ?? null,
    confidence: assessment ? Math.round((assessment.confidence ?? 0) * 100) : null,
    dataAvailable: basis.dataAvailable ?? false,
    source: "risk-engine-v1",
  };
}

async function fetchRealGrid(resolution = "7x7") {
  const [geometries, assessments] = await Promise.all([
    fetchBackendJson("/grids", { resolution }),
    fetchBackendJson("/risk/grids", { resolution }),
  ]);
  const byId = new Map(assessments.map((a) => [a.gridCellId, a]));
  return geometries.map((g) => adaptCell(g, byId.get(g.id)));
}

export async function fetchRiskGrid() {
  if (BACKEND_ENABLED) {
    try {
      return await fetchRealGrid();
    } catch (err) {
      warnFallback("risk grid", err);
    }
  }
  return RISK_GRID;
}

export async function fetchRiskGridStats() {
  if (BACKEND_ENABLED) {
    try {
      const summary = await fetchBackendJson("/risk/summary", { resolution: "7x7" });
      return summary.countsByLevel;
    } catch (err) {
      warnFallback("risk grid stats", err);
    }
  }
  return getRiskGridStats();
}

export async function fetchGridById(gridId) {
  if (BACKEND_ENABLED) {
    try {
      const [geometry, assessment] = await Promise.all([
        fetchBackendJson(`/grids/${gridId}`),
        fetchBackendJson(`/risk/grids/${gridId}`),
      ]);
      return adaptCell(geometry, assessment);
    } catch (err) {
      warnFallback(`grid ${gridId}`, err);
    }
  }
  return getMockGridById(gridId);
}

// Per-grid hazard probability breakdown (Heavy Rain / Thunderstorm /
// Lightning / Flood) — the REAL rule-based risk engine's own hazard scores,
// not an ML prediction (see predictionService.js's fetchGridMlForecast for
// the actual trained-model output). Labeled "Rule-Based Risk Engine" in the
// UI, never "AI"/"ML", per CLAUDE.md's real-vs-simulated boundary.
export async function fetchGridHazardProbabilities(gridId) {
  if (BACKEND_ENABLED) {
    try {
      const assessment = await fetchBackendJson(`/risk/grids/${gridId}`);
      return assessment.hazards.map((h) => ({
        key: h.hazard === "heavy_rainfall" ? "heavyRain" : h.hazard,
        label: hazardLabel(h.hazard),
        score: h.score,
        // Passed through unmodified — each indicator's own `category`
        // ("measured" | "forecast" | "official_warning" |
        // "historical_vulnerability") is what lets the UI show e.g. the
        // flood hazard's static India-Flood-Inventory signal as visibly
        // distinct from a live reading, never conflated. See
        // backend/app/services/risk_engine/types.py.
        indicators: h.indicators ?? [],
      }));
    } catch (err) {
      warnFallback(`hazard probabilities for grid ${gridId}`, err);
    }
  }
  return getMockGridHazards(gridId);
}

// The 20 verified Delhi-NCR districts with India Flood Inventory stats
// (backend/app/services/risk_engine/flood_vulnerability.py), as a GeoJSON
// FeatureCollection for the risk map's optional overlay layer. Static/
// historical data, never a live measurement — no mock fallback exists for
// this (it's a new, backend-only capability), so a failure here just means
// the layer silently isn't offered, never a fabricated substitute.
export async function fetchNcrFloodDistricts() {
  if (!BACKEND_ENABLED) return null;
  try {
    return await fetchBackendJson("/risk/districts");
  } catch (err) {
    warnFallback("NCR flood districts", err);
    return null;
  }
}

// Same static historical signal as above, but for one arbitrary point
// (e.g. a monitored Location) rather than the whole district layer —
// `GET /risk/districts/lookup`, backed by the same point-in-polygon lookup
// the risk engine itself uses. Resolves to `{ available: false }` both on
// a genuine "outside the 20 verified districts" result and on a backend
// failure, so callers render one "not available" state either way rather
// than needing to distinguish them.
export async function fetchDistrictVulnerability(lat, lng) {
  if (!BACKEND_ENABLED) return { available: false };
  try {
    return await fetchBackendJson("/risk/districts/lookup", { lat, lon: lng });
  } catch (err) {
    warnFallback(`district vulnerability for ${lat},${lng}`, err);
    return { available: false };
  }
}

// Pure, synchronous nearest-cell lookup over an already-fetched cell list
// (from fetchRiskGrid()) — for call sites that need "which cell is closest
// to this location" for many points at once without one network call per
// point (e.g. Locations.jsx rendering every monitored station's risk).
export function findNearestCellInList(cells, lat, lng) {
  if (!cells?.length) return null;
  let nearest = cells[0];
  let minDist = Infinity;
  for (const cell of cells) {
    const d = Math.hypot(cell.center[0] - lat, cell.center[1] - lng);
    if (d < minDist) {
      minDist = d;
      nearest = cell;
    }
  }
  return nearest;
}

export async function findNearestGridId(lat, lng) {
  if (BACKEND_ENABLED) {
    try {
      const geometries = await fetchBackendJson("/grids", { resolution: "7x7" });
      let nearest = geometries[0];
      let minDist = Infinity;
      for (const g of geometries) {
        const d = Math.hypot(Number(g.centerLat) - lat, Number(g.centerLng) - lng);
        if (d < minDist) {
          minDist = d;
          nearest = g;
        }
      }
      return nearest?.id ?? null;
    } catch (err) {
      warnFallback("nearest grid lookup", err);
    }
  }
  return findNearestMockGridCell(lat, lng)?.id ?? null;
}
