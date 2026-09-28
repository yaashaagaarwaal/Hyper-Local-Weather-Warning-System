import { Thermometer, Droplets, Wind, Gauge, Eye, CloudRain } from "lucide-react";
import GlassCard from "../common/GlassCard";
import Skeleton from "../common/Skeleton";
import Temp from "../common/Temp";

const METRICS = [
  { key: "humidity", label: "Humidity", icon: Droplets, unit: "%", accent: "text-cyan-400" },
  { key: "windSpeed", label: "Wind", icon: Wind, unit: "km/h", accent: "text-slate-300" },
  { key: "pressure", label: "Pressure", icon: Gauge, unit: "hPa", accent: "text-indigo-400" },
  { key: "visibility", label: "Visibility", icon: Eye, unit: "km", accent: "text-emerald-400" },
  { key: "rainProbability", label: "Rain Chance", icon: CloudRain, unit: "%", accent: "text-blue-400" },
];

export default function CurrentWeatherPanel({ weather, loading }) {
  return (
    <GlassCard className="p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Current Weather</h3>
        {!loading && (
          <span className="rounded-full bg-navy-800 px-2.5 py-1 text-xs text-slate-400 ring-1 ring-white/5">
            {weather.condition}
          </span>
        )}
      </div>

      {loading ? (
        <Skeleton className="h-16 w-32" />
      ) : (
        <div className="mb-5 flex items-end gap-3">
          <span className="text-5xl font-bold tracking-tight text-white">
            <Temp value={weather.temperature} decimals={0} />
          </span>
          <span className="mb-1.5 text-sm text-slate-500">{weather.locationName}, {weather.region}</span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div className="rounded-xl bg-navy-800/50 p-3 ring-1 ring-white/5">
          <Thermometer className="mb-1.5 h-4 w-4 text-orange-400" />
          {loading ? (
            <Skeleton className="h-5 w-12" />
          ) : (
            <p className="text-base font-semibold text-slate-100">
              <Temp value={weather.feelsLike} />
            </p>
          )}
          <p className="mt-0.5 text-[11px] text-slate-500">Feels Like</p>
        </div>
        {METRICS.map(({ key, label, icon: Icon, unit, accent }) => (
          <div key={key} className="rounded-xl bg-navy-800/50 p-3 ring-1 ring-white/5">
            <Icon className={`mb-1.5 h-4 w-4 ${accent}`} />
            {loading ? (
              <Skeleton className="h-5 w-12" />
            ) : (
              <p className="text-base font-semibold text-slate-100">
                {weather[key]}
                <span className="ml-0.5 text-xs font-normal text-slate-500">{unit}</span>
              </p>
            )}
            <p className="mt-0.5 text-[11px] text-slate-500">{label}</p>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}
