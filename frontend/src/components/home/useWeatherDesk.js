// Data for the "weather desk" section: everything comes from the app's own
// services, and nothing is invented. Two rules matter here:
//   * The risk and alert services quietly fall back to simulated data when the
//     backend is down. That fallback is detected (`source === "simulated"`, or a
//     hazard list without its `indicators`) and reported as unavailable instead.
//   * The risk grid covers Delhi-NCR only. A location far from every cell gets
//     weather readings but no model or risk figures, and the UI says why.
import { LOCATIONS } from "../../data/locations";
import { useAsync } from "../../hooks/useAsync";
import { fetchCurrentWeather, fetchHourlyTrend, fetchWeeklyOutlook } from "../../services/weatherService";
import { fetchRiskGrid, fetchGridHazardProbabilities, findNearestCellInList } from "../../services/riskService";
import { fetchGridMlForecast } from "../../services/predictionService";
import { fetchActiveAlerts } from "../../services/alertService";
import { memo } from "./memo";

export const DESK_LOCATION_IDS = ["loc-ndls", "loc-agra", "loc-noida", "loc-ggn", "loc-fbd", "loc-ghz"];
export const DESK_LOCATIONS = DESK_LOCATION_IDS.map((id) => LOCATIONS.find((l) => l.id === id)).filter(Boolean);
const MAX_CELL_DISTANCE_KM = 12; // grid cells are ~9 km across

function haversineKm(lat1, lon1, lat2, lon2) {
  const r = Math.PI / 180;
  const a = Math.sin(((lat2 - lat1) * r) / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lon2 - lon1) * r) / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(a));
}

async function loadModelBlock(location) {
  const cells = await memo("risk-grid", fetchRiskGrid);
  if (!cells?.length || cells.some((c) => c.source === "simulated")) return { status: "unavailable" };
  const cell = findNearestCellInList(cells, location.lat, location.lng);
  const distanceKm = haversineKm(location.lat, location.lng, cell.center[0], cell.center[1]);
  if (distanceKm > MAX_CELL_DISTANCE_KM) return { status: "outside", distanceKm };
  const [ml, hazards] = await Promise.all([
    memo(`ml:${cell.id}`, () => fetchGridMlForecast(cell.id)),
    memo(`hazards:${cell.id}`, () => fetchGridHazardProbabilities(cell.id)),
  ]);
  const realHazards = Array.isArray(hazards) && hazards.length > 0 && hazards.every((h) => Array.isArray(h.indicators));
  return {
    status: "ready",
    cell,
    distanceKm,
    ml: ml?.dataAvailable ? ml : null,
    hazards: realHazards ? hazards : null,
  };
}

export function useWeatherDesk(locationId) {
  const location = LOCATIONS.find((l) => l.id === locationId) ?? DESK_LOCATIONS[0];
  const weather = useAsync(() => memo(`weather:${location.id}`, () => fetchCurrentWeather(location.id)), [location.id], { refreshMs: 5 * 60 * 1000 });
  const daily = useAsync(() => memo(`daily:${location.id}`, () => fetchWeeklyOutlook(location.id)), [location.id]);
  const hourly = useAsync(() => memo(`hourly:${location.id}`, () => fetchHourlyTrend(location.id)), [location.id]);
  const model = useAsync(() => memo(`model:${location.id}`, () => loadModelBlock(location)), [location.id]);
  const alerts = useAsync(async () => {
    const list = await memo("alerts", fetchActiveAlerts);
    const real = (list ?? []).filter((a) => a.source && a.source !== "simulated");
    if ((list ?? []).length > 0 && real.length === 0) return { status: "unavailable", items: [] };
    return { status: "ready", items: real };
  }, []);
  return { location, weather, daily, hourly, model, alerts };
}
