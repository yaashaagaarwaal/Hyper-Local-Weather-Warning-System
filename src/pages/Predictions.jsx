import { Sparkles, CalendarDays, MapPin, X, Gauge, ClipboardList, Info } from "lucide-react";
import { useLocationContext } from "../context/LocationContext";
import { useGridFocus } from "../context/GridFocusContext";
import { useAsync } from "../hooks/useAsync";
import { fetchShortTermForecast, fetchGridById, fetchGridHazardProbabilities, fetchOverallRisk, fetchRiskSummary } from "../services/predictionService";
import { fetchWeeklyOutlook, fetchCurrentWeather } from "../services/weatherService";
import PredictionChart from "../components/charts/PredictionChart";
import WeeklyBarChart from "../components/charts/WeeklyBarChart";
import RiskIndicator from "../components/common/RiskIndicator";
import Temp from "../components/common/Temp";
import GlassCard from "../components/common/GlassCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import { RISK_LEVELS, findNearestLocationLabel } from "../data/riskData";
import { getRecommendedAction } from "../utils/recommendations";

function riskLevelFromScore(score) {
  if (score >= 80) return RISK_LEVELS.severe;
  if (score >= 55) return RISK_LEVELS.high;
  if (score >= 30) return RISK_LEVELS.moderate;
  return RISK_LEVELS.low;
}

export default function Predictions() {
  const { selectedLocationId, selectedLocation } = useLocationContext();
  const { focusedGridId, clearFocusedGrid } = useGridFocus();

  const { data: forecast, isLoading: forecastLoading } = useAsync(
    () => fetchShortTermForecast(selectedLocationId),
    [selectedLocationId]
  );
  const { data: currentWeather } = useAsync(() => fetchCurrentWeather(selectedLocationId), [selectedLocationId]);
  const { data: weekly, isLoading: weeklyLoading } = useAsync(
    () => fetchWeeklyOutlook(selectedLocationId),
    [selectedLocationId]
  );
  const { data: focusedGrid, isLoading: gridLoading } = useAsync(
    () => (focusedGridId ? fetchGridById(focusedGridId) : Promise.resolve(null)),
    [focusedGridId]
  );
  const { data: gridHazards } = useAsync(
    () => (focusedGridId ? fetchGridHazardProbabilities(focusedGridId) : Promise.resolve(null)),
    [focusedGridId]
  );
  const { data: locationRisk } = useAsync(() => fetchOverallRisk(selectedLocationId), [selectedLocationId]);
  const { data: locationHazards } = useAsync(() => fetchRiskSummary(selectedLocationId), [selectedLocationId]);

  const isGridFocused = Boolean(focusedGridId && focusedGrid);
  const hazards = isGridFocused ? gridHazards : locationHazards;
  const riskScore = isGridFocused ? focusedGrid?.riskScore : locationRisk?.score;
  const riskLevel = isGridFocused ? focusedGrid?.riskLevel : locationRisk?.level;
  const risk = riskLevelFromScore(riskScore ?? 0);
  const recommendation = getRecommendedAction(riskLevel ?? "low");

  const headerTitle = isGridFocused
    ? `Grid ${focusedGrid.id}`
    : `${selectedLocation.name}, ${selectedLocation.region}`;
  const headerSubtitle = isGridFocused
    ? findNearestLocationLabel(focusedGrid.center[0], focusedGrid.center[1])
    : "Active monitoring location";

  return (
    <div className="space-y-6">
      <GlassCard className="flex flex-wrap items-center justify-between gap-3 p-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 ring-1 ring-cyan-500/20">
            <MapPin className="h-5 w-5 text-cyan-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">
              {isGridFocused ? "Focused Grid" : "Selected Location"}
            </p>
            <h2 className="text-lg font-bold text-white">{headerTitle}</h2>
            <p className="text-xs text-slate-500">{headerSubtitle}</p>
          </div>
        </div>
        {isGridFocused && (
          <button
            onClick={clearFocusedGrid}
            className="flex items-center gap-1.5 rounded-lg bg-navy-800/60 px-3 py-1.5 text-xs font-medium text-slate-300 ring-1 ring-white/5 hover:bg-navy-800 hover:text-cyan-300"
          >
            <X className="h-3.5 w-3.5" /> Back to my location
          </button>
        )}
      </GlassCard>

      {isGridFocused && gridLoading ? (
        <LoadingSpinner label="Loading grid data" />
      ) : (
        <>
          {/* Current Conditions */}
          <GlassCard className="p-5">
            <h3 className="mb-4 text-sm font-semibold text-slate-200">Current Conditions</h3>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl bg-navy-800/50 p-3.5 ring-1 ring-white/5">
                <p className="text-[11px] text-slate-500">Temperature</p>
                <p className="mt-1 text-lg font-bold text-white">
                  <Temp value={isGridFocused ? focusedGrid.temperature : currentWeather?.temperature ?? 0} />
                </p>
              </div>
              <div className="rounded-xl bg-navy-800/50 p-3.5 ring-1 ring-white/5">
                <p className="text-[11px] text-slate-500">Rainfall</p>
                <p className="mt-1 text-lg font-bold text-white">
                  {isGridFocused ? focusedGrid.rainfall : currentWeather?.rainfall ?? "—"} mm
                </p>
              </div>
              <div className="rounded-xl bg-navy-800/50 p-3.5 ring-1 ring-white/5">
                <p className="text-[11px] text-slate-500">Humidity</p>
                <p className="mt-1 text-lg font-bold text-white">
                  {isGridFocused ? focusedGrid.humidity : currentWeather?.humidity ?? "—"}%
                </p>
              </div>
              <div className="rounded-xl bg-navy-800/50 p-3.5 ring-1 ring-white/5">
                <p className="text-[11px] text-slate-500">Wind</p>
                <p className="mt-1 text-lg font-bold text-white">
                  {isGridFocused ? focusedGrid.wind : currentWeather?.windSpeed ?? "—"} km/h
                </p>
              </div>
            </div>
          </GlassCard>

          {/* Prediction Timeline */}
          <GlassCard className="p-5">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500/20 to-indigo-500/20 ring-1 ring-cyan-500/20">
                  <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
                </div>
                <h3 className="text-sm font-semibold text-slate-200">Prediction Timeline — Next 60 Minutes</h3>
              </div>
              <span className="rounded-full bg-amber-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-amber-400 ring-1 ring-amber-500/25">
                Prototype Prediction
              </span>
            </div>
            <p className="mb-4 flex items-start gap-1.5 text-xs text-slate-500">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              AI prediction based on recent meteorological conditions and spatial weather patterns.
              The production model will be connected via the FastAPI backend.
            </p>
            {forecastLoading ? <LoadingSpinner label="Running nowcast model" /> : <PredictionChart data={forecast} height={300} />}
          </GlassCard>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
            {/* Hazard Probability */}
            <GlassCard className="p-5 lg:col-span-2">
              <h3 className="mb-4 text-sm font-semibold text-slate-200">Hazard Probability</h3>
              {!hazards ? (
                <LoadingSpinner label="Loading hazard breakdown" />
              ) : (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  {hazards.map((h) => (
                    <RiskIndicator key={h.key} label={h.label} score={h.score} />
                  ))}
                </div>
              )}
            </GlassCard>

            {/* Risk Score */}
            <GlassCard className="flex flex-col items-center justify-center p-5 text-center">
              <div className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-500">
                <Gauge className="h-3.5 w-3.5" /> Risk Score
              </div>
              <p className="text-4xl font-bold text-white">{riskScore ?? "--"}</p>
              <p className="text-xs text-slate-500">out of 100</p>
              <span
                className="mt-3 rounded-full px-3 py-1 text-xs font-semibold"
                style={{ color: risk.color, backgroundColor: `${risk.color}22` }}
              >
                {risk.label}
              </span>
            </GlassCard>
          </div>

          {/* Recommended Action */}
          <GlassCard className="flex items-start gap-3 p-5">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-cyan-500/10 ring-1 ring-cyan-500/20">
              <ClipboardList className="h-4 w-4 text-cyan-400" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-200">Recommended Action — {recommendation.headline}</p>
              <p className="mt-1 text-sm leading-relaxed text-slate-400">{recommendation.action}</p>
            </div>
          </GlassCard>
        </>
      )}

      {/* Extended outlook (location-based, always shown) */}
      <GlassCard className="p-5">
        <div className="mb-1 flex items-center gap-2">
          <CalendarDays className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">7-Day Risk Outlook — {selectedLocation.name}</h3>
        </div>
        <p className="mb-4 text-xs text-slate-500">Composite hazard risk score projected across the week</p>
        {weeklyLoading ? <LoadingSpinner label="Loading outlook" /> : <WeeklyBarChart data={weekly} />}
      </GlassCard>
    </div>
  );
}
