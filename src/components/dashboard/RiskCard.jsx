import { ShieldAlert } from "lucide-react";
import GlassCard from "../common/GlassCard";
import { RISK_LEVELS } from "../../data/riskData";

export default function RiskCard({ score, level, loading = false }) {
  const risk = RISK_LEVELS[level] ?? RISK_LEVELS.low;

  return (
    <GlassCard className="group relative overflow-hidden p-4 transition-transform duration-200 hover:-translate-y-0.5 sm:p-5">
      <div
        className="absolute -right-6 -top-6 h-24 w-24 rounded-full opacity-30 blur-2xl"
        style={{ backgroundColor: risk.color }}
      />
      <div className="relative flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium uppercase tracking-wide text-slate-500">Overall Risk</p>
          {loading ? (
            <div className="mt-2 h-8 w-20 animate-pulse rounded-lg bg-navy-700/60" />
          ) : (
            <p className="mt-1.5 flex items-baseline gap-1 text-2xl font-bold text-white">
              {score}
              <span className="text-sm font-medium text-slate-400">/100</span>
            </p>
          )}
          {!loading && (
            <span
              className="mt-1.5 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold"
              style={{ color: risk.color, backgroundColor: `${risk.color}22` }}
            >
              {risk.label}
            </span>
          )}
        </div>
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ring-1 ring-white/5"
          style={{ backgroundColor: `${risk.color}1a`, color: risk.color }}
        >
          <ShieldAlert className="h-5 w-5" />
        </div>
      </div>
    </GlassCard>
  );
}
