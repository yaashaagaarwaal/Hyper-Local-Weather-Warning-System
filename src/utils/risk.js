// Derives a 0-100 composite risk score from raw weather signals so
// "AI risk" numbers stay internally consistent across the app instead of
// being independently randomized.
export function deriveRiskScore(rainfall, humidity) {
  const score = rainfall * 1.3 + Math.max(0, humidity - 50) * 0.6;
  return Math.round(Math.min(98, Math.max(2, score)));
}

export function riskLevelFromScore(score) {
  if (score >= 80) return "severe";
  if (score >= 55) return "high";
  if (score >= 30) return "moderate";
  return "low";
}
