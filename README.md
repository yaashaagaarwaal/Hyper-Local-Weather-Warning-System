# MausamAI — Hyper-Local Weather Intelligence & Early Warning System

Frontend prototype for **SIH Problem Statement 26077**. This build is a fully working
React + Vite dashboard running on realistic mock data, ready to demo now and to wire
up to a FastAPI + ML backend later without touching component code. It runs in a
clearly labelled **Demo Mode** — every screen states that weather/prediction data is
simulated, not live.

## Tech Stack

- React 19 + Vite
- Tailwind CSS v4
- React Router
- Leaflet / React-Leaflet (risk map)
- Recharts (nowcast, trend, outlook, distribution charts)
- Lucide React (icons)

## Getting Started

```bash
npm install
npm run dev
```

Open the printed local URL (defaults to `http://localhost:5173`).

```bash
npm run build     # production build
npm run preview   # preview the production build
npm run lint      # oxlint
```

## Project Structure

```
src/
  components/
    layout/     Sidebar, Navbar, AppLayout
    dashboard/  WeatherCard, RiskCard, RiskSummary, PredictionPanel,
                SystemStatusBar, SevereScenarioBanner
    map/        RiskMap (Leaflet), GridDetailPanel
    weather/    CurrentWeatherPanel, LocationSelector
    charts/     PredictionChart, TrendChart, WeeklyBarChart,
                RiskDistributionChart, SingleMetricChart
    alerts/     AlertCard, NotificationPanel
    common/     GlassCard, Badge, RiskIndicator, Toggle, Skeleton,
                LoadingSpinner, Temp (unit-aware temperature)
  pages/        Dashboard, LiveWeather, RiskMapPage, Predictions, Alerts,
                Analytics, Locations, Settings, About
  data/         weatherData.js, riskData.js, alertsData.js, locations.js — mock data
  services/     weatherService.js, predictionService.js, alertService.js — API layer
  context/      LocationContext (active location), SettingsContext (units, theme,
                refresh interval, alert filters, demo mode), GridFocusContext
                (cross-page "jump to this grid" state + the severe-scenario banner)
  hooks/        useAsync, useDateTime, useLastUpdated
```

## Demo Flow (works end-to-end, no backend needed)

1. **Dashboard** — note the system status bar (AI Engine / Weather Data / Prediction
   Engine / Last updated) and the DEMO MODE badge.
2. Click a **red (Severe)** grid cell on the map. A scripted "Severe Weather" banner
   appears with fixed headline numbers (rainfall 87%, flood risk 81%, overall risk
   84/100) — a repeatable, memorable talking point.
3. Click **"View Detailed Prediction"** — jumps to Predictions, focused on that grid:
   current conditions, the 60-minute nowcast (labelled "Prototype Prediction"),
   hazard probability breakdown, risk score, and a recommended action.
4. Open **Alerts**, expand the Thunderstorm Warning, click **"View on Map"** — jumps
   to Risk Map, auto-pans and selects the affected grid, with a banner confirming
   where you came from.
5. **Settings** — Temperature unit (C/F), Alert Severity filters (try turning off
   "Severe" and watch the notification bell count drop live), Theme (accent color),
   Data Refresh Interval, and Demo Mode toggle all take effect immediately.

## Mock Data → Real Backend

All screens read data through `src/services/*.js`, never straight from `src/data/*.js`.
Each service function is a thin wrapper (`mockRequest`) around the mock data that already
returns a `Promise`, so swapping in FastAPI later is a body-only change:

```js
// before
export function fetchCurrentWeather(locationId) {
  return mockRequest(() => getCurrentWeather(locationId));
}

// after
export function fetchCurrentWeather(locationId) {
  return fetch(`${API_BASE_URL}/weather/current?locationId=${locationId}`).then((r) => r.json());
}
```

No component or page needs to change. `VITE_API_BASE_URL` (see `src/services/apiClient.js`)
already defaults to `http://localhost:8000/api` for the future FastAPI service.

## Current Limitations (frontend-only prototype)

- All weather, risk-grid, and alert data is deterministically generated mock data
  (seeded per location/grid, so numbers are stable across renders but vary by location).
- No authentication — the profile shown is static.
- No persistence — Settings choices reset on reload (in-memory only).
- Charts render raw Celsius values regardless of the Temperature Unit setting; only
  numeric readouts (stat cards, panels, popups) convert to °F.
- The risk map uses standard OpenStreetMap tiles with a CSS dark-mode filter (no API
  key required); swap for a hosted vector basemap when available.
