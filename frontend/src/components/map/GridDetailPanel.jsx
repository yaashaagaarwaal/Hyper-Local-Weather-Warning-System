import { useNavigate } from "react-router-dom";
import { Thermometer, CloudRain, Droplets, Wind, Clock, TriangleAlert, LocateFixed, MapPin, TrendingUp } from "lucide-react";
import GlassCard from "../common/GlassCard";
import Temp from "../common/Temp";
import { RISK_LEVELS, findNearestLocationLabel } from "../../data/riskData";
import { useGridFocus } from "../../context/GridFocusContext";
import { useSettings } from "../../context/SettingsContext";

const STATS = [
  { key: "rainfall", label: "Rainfall", unit: "mm", icon: CloudRain, accent: "text-blue-400" },
  { key: "humidity", label: "Humidity", unit: "%", icon: Droplets, accent: "text-cyan-400" },
  { key: "wind", label: "Wind", unit: "km/h", icon: Wind, accent: "text-slate-300" },
];

export default function GridDetailPanel({ cell }) {
  const navigate = useNavigate();
  const { setFocusedGridId } = useGridFocus();
  const { accent } = useSettings();

  if (!cell) {
    return (
      <GlassCard className="flex h-full flex-col items-center justify-center p-8 text-center">
        <LocateFixed className="mb-3 h-8 w-8 text-slate-600" />
        <p className="text-sm font-medium text-slate-400">No grid selected</p>
        <p className="mt-1 text-xs text-slate-600">Click any colored cell on the map to inspect it.</p>
      </GlassCard>
    );
  }

  const risk = RISK_LEVELS[cell.riskLevel];
  const locationLabel = findNearestLocationLabel(cell.center[0], cell.center[1]);

  function handleViewPrediction() {
    setFocusedGridId(cell.id);
    navigate("/predictions");
  }

  return (
    <GlassCard className="p-5">
      <div className="mb-1 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Selected Grid</p>
          <h3 className="text-lg font-bold text-white">{cell.id}</h3>
        </div>
        <span
          className="rounded-full px-2.5 py-1 text-xs font-semibold"
          style={{ color: risk.color, backgroundColor: `${risk.color}22` }}
        >
          {risk.label}
        </span>
      </div>

      <div className="mb-4 flex items-center gap-1.5 text-xs text-slate-500">
        <MapPin className="h-3.5 w-3.5" /> {locationLabel}
      </div>

      <div className="mb-4 rounded-xl bg-navy-800/50 p-3.5 ring-1 ring-white/5">
        <div className="mb-1 flex items-center justify-between text-xs">
          <span className="text-slate-500">Risk Score · Predicted Severity</span>
          <span className="font-semibold text-slate-200">{cell.riskScore}/100</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-navy-700/80">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${cell.riskScore}%`, backgroundColor: risk.color }}
          />
        </div>
      </div>

      <div className="mb-4 flex items-center gap-2 rounded-xl bg-amber-500/10 px-3.5 py-2.5 text-sm text-amber-300 ring-1 ring-amber-500/20">
        <TriangleAlert className="h-4 w-4 shrink-0" />
        <span className="font-medium">Main hazard: {cell.hazardType}</span>
      </div>

      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Current Conditions</p>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-navy-800/50 p-3 ring-1 ring-white/5">
          <Thermometer className="mb-1.5 h-4 w-4 text-orange-400" />
          <p className="text-base font-semibold text-slate-100">
            <Temp value={cell.temperature} />
          </p>
          <p className="mt-0.5 text-[11px] text-slate-500">Temperature</p>
        </div>
        {STATS.map(({ key, label, unit, icon: Icon, accent }) => (
          <div key={key} className="rounded-xl bg-navy-800/50 p-3 ring-1 ring-white/5">
            <Icon className={`mb-1.5 h-4 w-4 ${accent}`} />
            <p className="text-base font-semibold text-slate-100">
              {cell[key]}
              <span className="ml-0.5 text-xs font-normal text-slate-500">{unit}</span>
            </p>
            <p className="mt-0.5 text-[11px] text-slate-500">{label}</p>
          </div>
        ))}
      </div>

      <div className="mt-3.5 flex items-center gap-1.5 text-xs text-slate-500">
        <Clock className="h-3.5 w-3.5" /> Prediction window: {cell.predictionTime}
      </div>

      <button
        onClick={handleViewPrediction}
        style={{ background: `linear-gradient(90deg, ${accent.color}, ${accent.colorTo})` }}
        className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl py-2.5 text-sm font-semibold text-white shadow-lg transition-transform hover:scale-[1.01] active:scale-[0.98]"
      >
        <TrendingUp className="h-4 w-4" />
        View Detailed Prediction
      </button>
    </GlassCard>
  );
}
