import { MapPin, CheckCircle2, Plus, Crosshair, Home, Briefcase, GraduationCap } from "lucide-react";
import { useLocationContext } from "../context/LocationContext";
import { CURRENT_WEATHER_BY_LOCATION } from "../data/weatherData";
import { getOverallRisk, RISK_LEVELS } from "../data/riskData";
import { SAVED_LOCATION_META, getSavedLocations } from "../data/locations";
import GlassCard from "../components/common/GlassCard";
import Temp from "../components/common/Temp";

const SAVED_ICONS = { home: Home, work: Briefcase, college: GraduationCap };

export default function Locations() {
  const { locations, selectedLocationId, setSelectedLocationId } = useLocationContext();
  const savedLocations = getSavedLocations();

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-3 text-sm font-semibold text-slate-200">Saved Locations</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {savedLocations.map((loc) => {
            const meta = SAVED_LOCATION_META[loc.savedAs];
            const Icon = SAVED_ICONS[loc.savedAs];
            const isSelected = loc.id === selectedLocationId;
            const weather = CURRENT_WEATHER_BY_LOCATION[loc.id];
            return (
              <button
                key={loc.id}
                onClick={() => setSelectedLocationId(loc.id)}
                className={`flex items-center gap-3 rounded-2xl p-4 text-left transition-all duration-200 ${
                  isSelected
                    ? "glass-panel-solid ring-1 ring-cyan-500/40"
                    : "glass-panel hover:-translate-y-0.5 hover:ring-1 hover:ring-white/10"
                }`}
              >
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-cyan-500/10 ring-1 ring-cyan-500/20">
                  <Icon className="h-5 w-5 text-cyan-400" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-cyan-400">{meta.label}</p>
                  <p className="truncate text-sm font-medium text-slate-200">
                    {loc.name}, {loc.region}
                  </p>
                  <p className="text-xs text-slate-500">
                    <Temp value={weather.temperature} /> · {weather.condition}
                  </p>
                </div>
                {isSelected && <CheckCircle2 className="h-4 w-4 shrink-0 text-cyan-400" />}
              </button>
            );
          })}
        </div>
      </div>

      <GlassCard className="flex flex-wrap items-center justify-between gap-4 p-5">
        <div>
          <p className="text-sm font-semibold text-slate-200">All Monitored Stations</p>
          <p className="text-xs text-slate-500">{locations.length} locations across Delhi/NCR</p>
        </div>
        <button
          disabled
          title="Connect the backend to add live monitoring stations"
          className="flex cursor-not-allowed items-center gap-2 rounded-xl bg-navy-800/60 px-4 py-2.5 text-sm font-medium text-slate-500 ring-1 ring-white/5"
        >
          <Plus className="h-4 w-4" /> Add Station
        </button>
      </GlassCard>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {locations.map((loc) => {
          const weather = CURRENT_WEATHER_BY_LOCATION[loc.id];
          const risk = getOverallRisk(loc.id);
          const riskMeta = RISK_LEVELS[risk.level];
          const isSelected = loc.id === selectedLocationId;

          return (
            <GlassCard
              key={loc.id}
              className={`p-5 transition-all duration-200 ${isSelected ? "ring-1 ring-cyan-500/40" : "hover:-translate-y-0.5"}`}
            >
              <div className="mb-3 flex items-start justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-800 ring-1 ring-white/5">
                    <MapPin className="h-4 w-4 text-cyan-400" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-slate-100">{loc.name}</p>
                    <p className="text-xs text-slate-500">{loc.region}</p>
                  </div>
                </div>
                {isSelected && <CheckCircle2 className="h-4 w-4 text-cyan-400" />}
              </div>

              <div className="mb-4 flex items-center justify-between rounded-xl bg-navy-800/50 px-3.5 py-2.5 ring-1 ring-white/5">
                <span className="text-xl font-bold text-white">
                  <Temp value={weather.temperature} />
                </span>
                <span
                  className="rounded-full px-2 py-0.5 text-[11px] font-semibold"
                  style={{ color: riskMeta.color, backgroundColor: `${riskMeta.color}22` }}
                >
                  {riskMeta.label}
                </span>
              </div>

              <div className="mb-4 flex items-center gap-1.5 text-[11px] text-slate-600">
                <Crosshair className="h-3 w-3" />
                {loc.lat.toFixed(4)}, {loc.lng.toFixed(4)}
              </div>

              <button
                onClick={() => setSelectedLocationId(loc.id)}
                disabled={isSelected}
                className={`w-full rounded-lg py-2 text-xs font-semibold transition-colors ${
                  isSelected
                    ? "cursor-default bg-cyan-500/10 text-cyan-300"
                    : "bg-navy-800 text-slate-300 hover:bg-navy-700 hover:text-white"
                }`}
              >
                {isSelected ? "Currently Viewing" : "Set as Active"}
              </button>
            </GlassCard>
          );
        })}
      </div>
    </div>
  );
}
