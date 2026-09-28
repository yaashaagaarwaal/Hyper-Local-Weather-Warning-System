export const ALERT_SEVERITY = {
  advisory: { key: "advisory", label: "Advisory", color: "#eab308", bg: "bg-amber-500/15", text: "text-amber-400", ring: "ring-amber-500/30" },
  warning: { key: "warning", label: "Warning", color: "#f97316", bg: "bg-orange-500/15", text: "text-orange-400", ring: "ring-orange-500/30" },
  severe: { key: "severe", label: "Severe", color: "#ef4444", bg: "bg-red-500/15", text: "text-red-400", ring: "ring-red-500/30" },
};

// Mock active alerts feed. Replace with a FastAPI /alerts endpoint later.
export const ALERTS = [
  {
    id: "ALT-1042",
    title: "Heavy Rainfall Warning",
    severity: "warning",
    location: "South Delhi & Saket",
    issuedAt: "2026-09-29T05:10:00+05:30",
    expiresAt: "2026-09-29T11:00:00+05:30",
    status: "active",
    hazardType: "Heavy Rainfall",
    riskScore: 68,
    description:
      "Very heavy rainfall (65-90 mm) expected over the next 4-6 hours. Localized waterlogging likely on arterial roads near Saket and Mehrauli.",
    recommendedAction:
      "Avoid low-lying underpasses on the Saket-Mehrauli stretch. Keep extra travel time and monitor drainage advisories.",
    affectedGrids: ["GRID-024", "GRID-025", "GRID-031"],
    confidence: 87,
  },
  {
    id: "ALT-1041",
    title: "Thunderstorm Warning",
    severity: "severe",
    location: "Gurugram & NH-48 Corridor",
    issuedAt: "2026-09-29T04:35:00+05:30",
    expiresAt: "2026-09-29T09:30:00+05:30",
    status: "active",
    hazardType: "Thunderstorm",
    riskScore: 88,
    description:
      "Severe thunderstorm cell moving northeast at 22 km/h with wind gusts up to 65 km/h and frequent lightning. Avoid open areas and elevated structures.",
    recommendedAction:
      "Move indoors immediately. Avoid open areas, elevated structures, and NH-48 travel until the cell passes.",
    affectedGrids: ["GRID-010", "GRID-011", "GRID-017"],
    confidence: 92,
  },
  {
    id: "ALT-1040",
    title: "Flash Flood Risk",
    severity: "severe",
    location: "Najafgarh Drain Basin",
    issuedAt: "2026-09-29T03:50:00+05:30",
    expiresAt: "2026-09-29T14:00:00+05:30",
    status: "active",
    hazardType: "Flash Flood",
    riskScore: 81,
    description:
      "Rapid water-level rise detected in low-lying catchments along the Najafgarh drain. Flash flooding possible in adjoining residential pockets within 2-3 hours.",
    recommendedAction:
      "Residents in low-lying pockets should move valuables to higher floors and avoid drain-adjacent roads.",
    affectedGrids: ["GRID-002", "GRID-003"],
    confidence: 78,
  },
  {
    id: "ALT-1039",
    title: "Lightning Advisory",
    severity: "advisory",
    location: "Noida & Greater Noida",
    issuedAt: "2026-09-29T02:15:00+05:30",
    expiresAt: "2026-09-29T08:00:00+05:30",
    status: "active",
    hazardType: "Lightning Strike",
    riskScore: 54,
    description:
      "Scattered lightning activity detected across eastern NCR. Outdoor workers and commuters advised to seek covered shelter during active cells.",
    recommendedAction:
      "Outdoor workers should shelter during active cells. Avoid open fields and metal structures.",
    affectedGrids: ["GRID-034", "GRID-041"],
    confidence: 71,
  },
  {
    id: "ALT-1035",
    title: "High Wind Gust Advisory",
    severity: "advisory",
    location: "Dwarka & IGI Airport Belt",
    issuedAt: "2026-09-28T22:40:00+05:30",
    expiresAt: "2026-09-29T06:00:00+05:30",
    status: "resolved",
    hazardType: "High Wind Gusts",
    riskScore: 41,
    description:
      "Gusty winds of 40-50 km/h recorded overnight, now subsiding. Minor disruption to loose outdoor structures reported near the airport belt.",
    recommendedAction:
      "No further action required. Secure loose outdoor items as a precaution during the tail end of the gusts.",
    affectedGrids: ["GRID-005"],
    confidence: 64,
  },
  {
    id: "ALT-1031",
    title: "Urban Waterlogging Notice",
    severity: "warning",
    location: "Rohini & Outer Delhi",
    issuedAt: "2026-09-28T19:05:00+05:30",
    expiresAt: "2026-09-29T02:00:00+05:30",
    status: "resolved",
    hazardType: "Urban Waterlogging",
    riskScore: 47,
    description:
      "Waterlogging cleared from major underpasses near Rohini sector roads following overnight drainage operations.",
    recommendedAction: "Roads are clear. Routine monitoring only — no action required.",
    affectedGrids: ["GRID-045", "GRID-046"],
    confidence: 69,
  },
];

export function getActiveAlerts() {
  return ALERTS.filter((a) => a.status === "active");
}

export function getAlertById(id) {
  return ALERTS.find((a) => a.id === id);
}

const RANGE_TO_HOURS = { "1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7 };

// Count of alerts issued within the given Analytics range filter window,
// relative to the most recently issued alert (stands in for "now" since the
// mock data is dated around a fixed demo day).
export function getAlertsCountInRange(rangeKey) {
  const hours = RANGE_TO_HOURS[rangeKey] ?? 24;
  const latest = Math.max(...ALERTS.map((a) => new Date(a.issuedAt).getTime()));
  const cutoff = latest - hours * 60 * 60 * 1000;
  return ALERTS.filter((a) => new Date(a.issuedAt).getTime() >= cutoff).length;
}
