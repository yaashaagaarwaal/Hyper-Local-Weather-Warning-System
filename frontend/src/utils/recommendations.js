const ACTIONS_BY_LEVEL = {
  severe: {
    headline: "Take immediate precaution",
    action:
      "Avoid travel through affected zones. Move valuables and vehicles away from low-lying and flood-prone areas. Monitor official updates every 15 minutes.",
  },
  high: {
    headline: "Exercise heightened caution",
    action:
      "Postpone non-essential outdoor activity. Keep emergency contacts and a charged phone ready. Re-check conditions before commuting.",
  },
  moderate: {
    headline: "Stay alert and informed",
    action:
      "Carry rain protection and allow extra travel time. No immediate action required, but conditions may escalate — check back within the hour.",
  },
  low: {
    headline: "No action needed",
    action: "Conditions are within normal range. Routine monitoring is sufficient.",
  },
};

export function getRecommendedAction(level) {
  return ACTIONS_BY_LEVEL[level] ?? ACTIONS_BY_LEVEL.low;
}
