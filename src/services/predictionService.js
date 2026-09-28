import { mockRequest } from "./apiClient";
import { getShortTermForecast, getAnalyticsSeries } from "../data/weatherData";
import {
  RISK_GRID,
  getRiskGridStats,
  getRiskSummary,
  getOverallRisk,
  findNearestGridCell,
  getGridById,
  getGridHazardProbabilities,
} from "../data/riskData";

// GET /predictions/nowcast?locationId=...
export function fetchShortTermForecast(locationId) {
  return mockRequest(() => getShortTermForecast(locationId));
}

// GET /risk/grid
export function fetchRiskGrid() {
  return mockRequest(() => RISK_GRID, { latency: 450 });
}

// GET /risk/grid/stats
export function fetchRiskGridStats() {
  return mockRequest(() => getRiskGridStats());
}

// GET /risk/summary?locationId=...
export function fetchRiskSummary(locationId) {
  return mockRequest(() => getRiskSummary(locationId));
}

// GET /risk/overall?locationId=...
export function fetchOverallRisk(locationId) {
  return mockRequest(() => getOverallRisk(locationId));
}

// GET /risk/grid/nearest?lat=...&lng=...
export function fetchNearestGridCell(lat, lng) {
  return mockRequest(() => findNearestGridCell(lat, lng));
}

// GET /risk/grid/:id
export function fetchGridById(gridId) {
  return mockRequest(() => getGridById(gridId));
}

// GET /risk/grid/:id/hazards
export function fetchGridHazardProbabilities(gridId) {
  return mockRequest(() => getGridHazardProbabilities(gridId));
}

// GET /analytics/series?locationId=...&range=...
export function fetchAnalyticsSeries(locationId, rangeKey) {
  return mockRequest(() => getAnalyticsSeries(locationId, rangeKey), { latency: 300 });
}
