import { useState } from "react";
import { Thermometer, CloudRain, Gauge, Bell, PieChart as PieIcon } from "lucide-react";
import { useLocationContext } from "../context/LocationContext";
import { useAsync } from "../hooks/useAsync";
import { fetchAnalyticsSeries, fetchRiskGridStats } from "../services/predictionService";
import { fetchAlertsCountInRange } from "../services/alertService";
import SingleMetricChart from "../components/charts/SingleMetricChart";
import RiskDistributionChart from "../components/charts/RiskDistributionChart";
import GlassCard from "../components/common/GlassCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import Temp from "../components/common/Temp";

const RANGES = [
  { key: "1h", label: "Last 1 Hour" },
  { key: "6h", label: "Last 6 Hours" },
  { key: "24h", label: "Last 24 Hours" },
  { key: "7d", label: "Last 7 Days" },
];

export default function Analytics() {
  const { selectedLocationId, selectedLocation } = useLocationContext();
  const [range, setRange] = useState("24h");

  const { data: series, isLoading: seriesLoading } = useAsync(
    () => fetchAnalyticsSeries(selectedLocationId, range),
    [selectedLocationId, range]
  );
  const { data: gridStats, isLoading: statsLoading } = useAsync(fetchRiskGridStats, []);
  const { data: alertCount, isLoading: alertCountLoading } = useAsync(
    () => fetchAlertsCountInRange(range),
    [range]
  );

  const avgTemp = series ? series.reduce((s, h) => s + h.temperature, 0) / series.length : null;
  const totalRain = series ? series.reduce((s, h) => s + h.rainfall, 0).toFixed(0) : "--";
  const avgRisk = series ? Math.round(series.reduce((s, h) => s + h.riskScore, 0) / series.length) : "--";

  return (
    <div className="space-y-6">
      <GlassCard className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div>
          <p className="text-sm font-semibold text-slate-200">Analytics — {selectedLocation.name}</p>
          <p className="text-xs text-slate-500">Rainfall, temperature, and risk trends over time</p>
        </div>
        <div className="flex items-center gap-1 rounded-xl bg-navy-800/60 p-1 ring-1 ring-white/5">
          {RANGES.map((r) => (
            <button
              key={r.key}
              onClick={() => setRange(r.key)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                range === r.key ? "bg-cyan-500/15 text-cyan-300" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard className="p-5">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <Thermometer className="h-3.5 w-3.5 text-orange-400" /> Avg. Temperature
          </div>
          <p className="text-2xl font-bold text-white">{avgTemp === null ? "--" : <Temp value={avgTemp} />}</p>
        </GlassCard>
        <GlassCard className="p-5">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <CloudRain className="h-3.5 w-3.5 text-blue-400" /> Total Rainfall
          </div>
          <p className="text-2xl font-bold text-white">{totalRain} mm</p>
        </GlassCard>
        <GlassCard className="p-5">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <Gauge className="h-3.5 w-3.5 text-red-400" /> Avg. Risk Score
          </div>
          <p className="text-2xl font-bold text-white">{avgRisk}/100</p>
        </GlassCard>
        <GlassCard className="p-5">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <Bell className="h-3.5 w-3.5 text-amber-400" /> Alerts Issued
          </div>
          <p className="text-2xl font-bold text-white">{alertCountLoading ? "--" : alertCount}</p>
        </GlassCard>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <GlassCard className="p-5">
          <h3 className="mb-1 text-sm font-semibold text-slate-200">Rainfall Trend</h3>
          <p className="mb-3 text-xs text-slate-500">mm over selected range</p>
          {seriesLoading ? <LoadingSpinner label="Loading" /> : <SingleMetricChart data={series} dataKey="rainfall" color="#22d3ee" unit=" mm" />}
        </GlassCard>
        <GlassCard className="p-5">
          <h3 className="mb-1 text-sm font-semibold text-slate-200">Temperature Trend</h3>
          <p className="mb-3 text-xs text-slate-500">°C over selected range</p>
          {seriesLoading ? <LoadingSpinner label="Loading" /> : <SingleMetricChart data={series} dataKey="temperature" color="#f97316" unit="°C" />}
        </GlassCard>
        <GlassCard className="p-5">
          <h3 className="mb-1 text-sm font-semibold text-slate-200">Risk Trend</h3>
          <p className="mb-3 text-xs text-slate-500">composite score over selected range</p>
          {seriesLoading ? <LoadingSpinner label="Loading" /> : <SingleMetricChart data={series} dataKey="riskScore" color="#ef4444" unit="" />}
        </GlassCard>
      </div>

      <GlassCard className="p-5">
        <div className="mb-1 flex items-center gap-2">
          <PieIcon className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">Hazard Distribution</h3>
        </div>
        <p className="mb-2 text-xs text-slate-500">
          Live NCR grid classification, {statsLoading ? "…" : Object.values(gridStats).reduce((a, b) => a + b, 0)} zones
          monitored (independent of the time filter above)
        </p>
        {statsLoading ? <LoadingSpinner label="Loading grid stats" /> : <RiskDistributionChart stats={gridStats} />}
      </GlassCard>
    </div>
  );
}
