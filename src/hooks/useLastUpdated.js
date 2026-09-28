import { useEffect, useState } from "react";
import { useSettings } from "../context/SettingsContext";
import { formatSecondsAgo } from "../utils/formatters";

// Simulates a periodic data refresh cadence driven by the user's configured
// refresh interval (Settings page). Only the "last updated" timestamp moves —
// this never claims the underlying mock data is a live feed.
export function useLastUpdated() {
  const { refreshInterval } = useSettings();
  const [lastUpdated, setLastUpdated] = useState(() => new Date());
  const [, forceTick] = useState(0);

  useEffect(() => {
    const refreshId = setInterval(() => setLastUpdated(new Date()), refreshInterval * 1000);
    return () => clearInterval(refreshId);
  }, [refreshInterval]);

  useEffect(() => {
    const tickId = setInterval(() => forceTick((n) => n + 1), 1000);
    return () => clearInterval(tickId);
  }, []);

  return { lastUpdated, label: formatSecondsAgo(lastUpdated) };
}
