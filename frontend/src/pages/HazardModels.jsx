// Flood & Lightning AI: the two trained hazard-imagery models.
//
// Everything on this page comes from the backend's /hazard-models endpoints.
// Two lines are kept visible throughout: trained model vs. classical method
// (violet vs. cyan), and real observation vs. synthetic simulation (emerald
// vs. amber). Neither model feeds the risk engine or alerts.
import { useState } from "react";
import { BrainCircuit, Info, Waves, Zap } from "lucide-react";
import GlassCard from "../components/common/GlassCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import FloodModelPanel from "../components/hazards/FloodModelPanel";
import LightningModelPanel from "../components/hazards/LightningModelPanel";
import { ML_BADGE, REAL_BADGE, SYNTHETIC_BADGE, CLASSICAL_BADGE } from "../components/hazards/shared";
import { useAsync } from "../hooks/useAsync";
import { fetchHazardModelStatus } from "../services/hazardModelService";

const TABS = [
  { key: "flood", label: "Flood mapping", icon: Waves, sub: "Sentinel-1 radar → water mask" },
  { key: "lightning", label: "Lightning detection", icon: Zap, sub: "Infrared cloud tops → flash probability" },
];

function StatusDot({ status }) {
  const ok = status?.available;
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] ${ok ? "text-emerald-400" : "text-amber-400"}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-amber-400"}`} />
      {ok ? `model loaded · ${String(status.device).toUpperCase()}` : "model unavailable"}
    </span>
  );
}

export default function HazardModels() {
  const { data: status, isLoading } = useAsync(() => fetchHazardModelStatus(), []);
  const [tab, setTab] = useState("flood");

  return (
    <div className="space-y-6">
      <GlassCard className="flex items-start gap-3 border border-violet-500/20 bg-violet-500/5 p-4">
        <BrainCircuit className="mt-0.5 h-4 w-4 shrink-0 text-violet-300" />
        <div className="min-w-0 space-y-2 text-xs leading-snug text-slate-300">
          <p>
            Two trained image models. <span className="font-semibold text-slate-100">Flood mapping</span> marks water in
            satellite radar images; <span className="font-semibold text-slate-100">lightning detection</span> estimates
            where lightning is likely from infrared cloud-top images. Both analyse an image that already exists — neither
            forecasts, and neither feeds the Risk Map or Alerts, which still use the rule-based engine.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <span className={ML_BADGE}>Trained ML model</span>
            <span className={CLASSICAL_BADGE}>Classical method</span>
            <span className={REAL_BADGE}>Real observation</span>
            <span className={SYNTHETIC_BADGE}>Synthetic simulation</span>
          </div>
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {TABS.map(({ key, label, icon: Icon, sub }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex min-w-0 items-center gap-3 rounded-2xl p-4 text-left ring-1 transition-colors ${tab === key ? "bg-violet-500/10 ring-violet-500/40" : "glass-panel ring-white/5 hover:bg-white/5"}`}
          >
            <Icon className={`h-5 w-5 shrink-0 ${tab === key ? "text-violet-300" : "text-slate-500"}`} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold text-slate-100">{label}</span>
              <span className="block truncate text-[11px] text-slate-500">{sub}</span>
            </span>
            {!isLoading && <StatusDot status={status?.[key]} />}
          </button>
        ))}
      </div>

      <GlassCard className="p-5">
        {isLoading ? (
          <LoadingSpinner label="Loading model status" />
        ) : tab === "flood" ? (
          <FloodModelPanel status={status?.flood} />
        ) : (
          <LightningModelPanel status={status?.lightning} />
        )}
      </GlassCard>

      <GlassCard className="flex items-start gap-3 p-4">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
        <p className="min-w-0 text-[11px] leading-snug text-slate-500">
          Synthetic India scenes are computer simulations built for development (Indian places, seasons and storm or
          flood types), not satellite data. Scores on them show only how well a model agrees with the simulator and are
          always reported apart from scores on real data. Details: docs/flood_lightning_model_report.md.
        </p>
      </GlassCard>
    </div>
  );
}
