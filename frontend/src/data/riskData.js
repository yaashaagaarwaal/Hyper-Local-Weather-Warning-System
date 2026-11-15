import { seededRandom, randomInRange, pick } from "../utils/seededRandom";
import { riskLevelFromScore } from "../utils/risk";
import { LOCATIONS } from "./locations";

export const RISK_LEVELS = {
  low: { key: "low", label: "Low Risk", color: "#22c55e", textColor: "text-emerald-400", bg: "bg-emerald-500/15" },
  moderate: { key: "moderate", label: "Moderate Risk", color: "#eab308", textColor: "text-amber-400", bg: "bg-amber-500/15" },
  high: { key: "high", label: "High Risk", color: "#f97316", textColor: "text-orange-400", bg: "bg-orange-500/15" },
  severe: { key: "severe", label: "Severe Risk", color: "#ef4444", textColor: "text-red-400", bg: "bg-red-500/15" },
};

const HAZARD_TYPES = [
  "Heavy Rainfall",
  "Thunderstorm",
  "Flash Flood",
  "Lightning Strike",
  "Urban Waterlogging",
  "High Wind Gusts",
];

// Builds an N x N lat/lng grid over the NCR bounding box, each cell a small
// square polygon carrying a mock risk score + weather snapshot.
function buildRiskGrid() {
  const bounds = { south: 28.32, north: 28.88, west: 76.82, east: 77.56 };
  const rows = 7;
  const cols = 7;
  const latStep = (bounds.north - bounds.south) / rows;
  const lngStep = (bounds.east - bounds.west) / cols;
  const cells = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const id = `GRID-${String(r * cols + c + 1).padStart(3, "0")}`;
      const rng = seededRandom(id);
      const south = bounds.south + r * latStep;
      const north = south + latStep;
      const west = bounds.west + c * lngStep;
      const east = west + lngStep;

      const score = Math.round(randomInRange(rng, 5, 97));
      const level = riskLevelFromScore(score);

      cells.push({
        id,
        bounds: [
          [south, west],
          [south, east],
          [north, east],
          [north, west],
        ],
        center: [(south + north) / 2, (west + east) / 2],
        riskScore: score,
        riskLevel: level,
        temperature: Number(randomInRange(rng, 24, 40).toFixed(1)),
        rainfall: Number(randomInRange(rng, 0, 65).toFixed(1)),
        humidity: Math.round(randomInRange(rng, 35, 95)),
        wind: Number(randomInRange(rng, 3, 42).toFixed(1)),
        hazardType: pick(rng, HAZARD_TYPES),
        predictionTime: `${Math.round(randomInRange(rng, 15, 90))} min`,
      });
    }
  }
  return cells;
}

export const RISK_GRID = buildRiskGrid();

export function getRiskGridStats() {
  const counts = { low: 0, moderate: 0, high: 0, severe: 0 };
  RISK_GRID.forEach((cell) => counts[cell.riskLevel]++);
  return counts;
}

// Per-location hazard risk summary (Heavy Rain / Thunderstorm / Lightning / Flood).
export function getRiskSummary(locationId) {
  const rng = seededRandom(`risk-summary-${locationId}`);
  return [
    { key: "heavyRain", label: "Heavy Rain", score: Math.round(randomInRange(rng, 10, 95)) },
    { key: "thunderstorm", label: "Thunderstorm", score: Math.round(randomInRange(rng, 10, 95)) },
    { key: "lightning", label: "Lightning", score: Math.round(randomInRange(rng, 5, 90)) },
    { key: "flood", label: "Flood", score: Math.round(randomInRange(rng, 5, 90)) },
  ];
}

export function getOverallRisk(locationId) {
  const summary = getRiskSummary(locationId);
  const avg = Math.round(summary.reduce((sum, s) => sum + s.score, 0) / summary.length);
  return { score: avg, level: riskLevelFromScore(avg) };
}

export function findNearestGridCell(lat, lng) {
  let nearest = RISK_GRID[0];
  let minDist = Infinity;
  RISK_GRID.forEach((cell) => {
    const d = Math.hypot(cell.center[0] - lat, cell.center[1] - lng);
    if (d < minDist) {
      minDist = d;
      nearest = cell;
    }
  });
  return nearest;
}

export function getGridById(gridId) {
  return RISK_GRID.find((cell) => cell.id === gridId) ?? null;
}

// Nearest monitored settlement name for a grid cell, used to give map
// popups and detail panels a human-readable place instead of just lat/lng.
export function findNearestLocationLabel(lat, lng) {
  let nearest = LOCATIONS[0];
  let minDist = Infinity;
  LOCATIONS.forEach((loc) => {
    const d = Math.hypot(loc.lat - lat, loc.lng - lng);
    if (d < minDist) {
      minDist = d;
      nearest = loc;
    }
  });
  return `Near ${nearest.name}, ${nearest.region}`;
}

// Per-grid hazard probability breakdown, correlated with the grid's overall
// riskScore so a severe grid reads as severe across every hazard type.
export function getGridHazardProbabilities(gridId) {
  const cell = getGridById(gridId);
  if (!cell) return [];
  const rng = seededRandom(`grid-hazard-${gridId}`);
  const clamp = (v) => Math.min(97, Math.max(3, Math.round(v)));

  return [
    { key: "heavyRain", label: "Heavy Rain", score: clamp(cell.riskScore * 0.95 + randomInRange(rng, -6, 6)) },
    { key: "thunderstorm", label: "Thunderstorm", score: clamp(cell.riskScore * 0.8 + randomInRange(rng, -8, 8)) },
    { key: "lightning", label: "Lightning", score: clamp(cell.riskScore * 0.65 + randomInRange(rng, -8, 8)) },
    { key: "flood", label: "Flood", score: clamp(cell.riskScore * 0.85 + randomInRange(rng, -8, 8)) },
  ];
}

export { LOCATIONS, riskLevelFromScore };
