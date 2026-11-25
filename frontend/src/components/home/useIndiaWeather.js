// The real data behind the India scene's overlays, gathered from the same
// services the rest of the app uses -- never from a generator in this file.
//
//   stations  current readings at the monitored places (weatherService)
//   warnings  active alerts from the AtmoGuard engine, placed at their grid cell
//   rain      NASA IMERG satellite rainfall (rainfall.js)
//
// Each part reports its own status so the UI can show a proper "unavailable"
// state. In particular the alert and risk services silently fall back to
// simulated data when the backend is down; that fallback is detected here
// (`source === "simulated"`) and treated as unavailable, never plotted.
import { useEffect, useState } from "react";
import { LOCATIONS } from "../../data/locations";
import { fetchAllCurrentWeather } from "../../services/weatherService";
import { fetchActiveAlerts } from "../../services/alertService";
import { fetchRiskGrid } from "../../services/riskService";
import { loadRainfall } from "./rainfall";
import { loadIdRaster } from "./indiaData";
import { memo } from "./memo";

// Delhi-NCR is a cluster of nearby points; at country scale it is one place.
const NCR_IDS = LOCATIONS.filter((l) => l.id !== "loc-agra").map((l) => l.id);

export function stationsFrom(weatherById) {
  const ncr = LOCATIONS.filter((l) => NCR_IDS.includes(l.id));
  const agra = LOCATIONS.find((l) => l.id === "loc-agra");
  const delhi = LOCATIONS.find((l) => l.isPrimary) ?? ncr[0];
  const fmt = (w) => (w && Number.isFinite(w.temperature) ? `${Math.round(w.temperature)}°C` : null);
  const out = [];
  if (delhi) {
    const lat = ncr.reduce((a, l) => a + l.lat, 0) / ncr.length;
    const lon = ncr.reduce((a, l) => a + l.lng, 0) / ncr.length;
    out.push({ id: "ncr", name: "Delhi–NCR", lat, lon, value: fmt(weatherById?.[delhi.id]) });
  }
  if (agra) out.push({ id: agra.id, name: agra.name, lat: agra.lat, lon: agra.lng, value: fmt(weatherById?.[agra.id]), below: true });
  return out;
}

export function useIndiaWeather(enabled = true) {
  const [stations, setStations] = useState({ status: "loading", items: [] });
  const [warnings, setWarnings] = useState({ status: "loading", items: [], count: 0 });
  const [rain, setRain] = useState({ status: "loading" });

  useEffect(() => {
    if (!enabled) return undefined;
    const controller = new AbortController();
    let alive = true;

    memo("all-weather", fetchAllCurrentWeather)
      .then((byId) => alive && setStations({ status: "ready", items: stationsFrom(byId) }))
      .catch(() => alive && setStations({ status: "unavailable", items: stationsFrom(null) }));

    Promise.all([memo("alerts", fetchActiveAlerts), memo("risk-grid", fetchRiskGrid)])
      .then(([alerts, cells]) => {
        if (!alive) return;
        const real = (alerts ?? []).filter((a) => a.source && a.source !== "simulated");
        const simulated = (alerts ?? []).length > 0 && real.length === 0;
        const gridReal = (cells ?? []).every((c) => c.source !== "simulated");
        if (simulated || !gridReal) {
          setWarnings({ status: "unavailable", items: [], count: 0 });
          return;
        }
        const items = real.flatMap((a) => {
          const cell = cells.find((c) => c.id === a.affectedGrids?.[0]);
          return cell ? [{ id: a.id, name: a.title, lat: cell.center[0], lon: cell.center[1], severity: a.severity }] : [];
        });
        setWarnings({ status: "ready", items, count: real.length });
      })
      .catch(() => alive && setWarnings({ status: "unavailable", items: [], count: 0 }));

    loadIdRaster()
      .catch(() => null)
      .then((ids) => loadRainfall({ indiaIds: ids, signal: controller.signal }))
      .then((result) => alive && result.status !== "aborted" && setRain(result));

    return () => {
      alive = false;
      controller.abort();
    };
  }, [enabled]);

  return { stations, warnings, rain };
}
