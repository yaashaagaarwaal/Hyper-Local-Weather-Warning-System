import { seededRandom, randomInRange } from "../utils/seededRandom";
import { LOCATIONS } from "./locations";

// Deterministic seed used ONLY to ground the prototype AI nowcast's starting
// point (see predictionService.fetchShortTermForecast). This is intentionally
// kept separate from real weather data — the nowcast trajectory itself is a
// synthetic, seeded stand-in for a trained model, not real observations.
function buildNowcastSeed(locationId) {
  const rng = seededRandom(`current-${locationId}`);
  return {
    temperature: Number(randomInRange(rng, 24, 39).toFixed(1)),
    rainfall: Number(randomInRange(rng, 0, 42).toFixed(1)),
  };
}

// 60-minute short-term nowcast used by the AI Prediction panel. Prototype
// only — no trained model is connected. Seeded per location so it stays
// stable across reloads instead of re-randomizing on every render.
export function getShortTermForecast(locationId) {
  const seed = buildNowcastSeed(locationId);
  const rng = seededRandom(`forecast-${locationId}`);
  const steps = [0, 15, 30, 45, 60];
  let rainfall = seed.rainfall;
  let risk = Number(randomInRange(rng, 15, 45).toFixed(0));

  return steps.map((minute, idx) => {
    if (idx > 0) {
      rainfall = Math.max(0, rainfall + randomInRange(rng, -3, 9));
      risk = Math.min(98, Math.max(4, risk + randomInRange(rng, -6, 14)));
    }
    return {
      minute,
      label: minute === 0 ? "Now" : `+${minute} min`,
      rainfall: Number(rainfall.toFixed(1)),
      riskScore: Math.round(risk),
      temperature: Number((seed.temperature - minute * 0.02 + randomInRange(rng, -0.4, 0.4)).toFixed(1)),
    };
  });
}

export { LOCATIONS };
