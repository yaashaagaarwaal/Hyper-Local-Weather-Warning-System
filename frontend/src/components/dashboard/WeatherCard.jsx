import GlassCard from "../common/GlassCard";
import Temp from "../common/Temp";

export default function WeatherCard({ icon: Icon, label, value, unit, trend, accent = "cyan", loading = false, isTemp = false }) {
  const accentMap = {
    cyan: "from-cyan-500/20 to-cyan-500/0 text-cyan-400",
    blue: "from-blue-500/20 to-blue-500/0 text-blue-400",
    indigo: "from-indigo-500/20 to-indigo-500/0 text-indigo-400",
    amber: "from-amber-500/20 to-amber-500/0 text-amber-400",
    red: "from-red-500/20 to-red-500/0 text-red-400",
    emerald: "from-emerald-500/20 to-emerald-500/0 text-emerald-400",
  };

  return (
    <GlassCard className="group relative overflow-hidden p-4 transition-transform duration-200 hover:-translate-y-0.5 sm:p-5">
      <div className={`absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br opacity-40 blur-2xl ${accentMap[accent]}`} />
      <div className="relative flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
          {loading ? (
            <div className="mt-2 h-8 w-20 animate-pulse rounded-lg bg-navy-700/60" />
          ) : isTemp ? (
            <p className="mt-1.5 text-2xl font-bold text-white">
              <Temp value={value} decimals={1} />
            </p>
          ) : (
            <p className="mt-1.5 flex items-baseline gap-1 text-2xl font-bold text-white">
              {value}
              {unit && <span className="text-sm font-medium text-slate-400">{unit}</span>}
            </p>
          )}
          {trend && !loading && <p className="mt-1.5 text-xs text-slate-500">{trend}</p>}
        </div>
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-navy-800/80 ring-1 ring-white/5 ${accentMap[accent].split(" ").pop()}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
    </GlassCard>
  );
}
