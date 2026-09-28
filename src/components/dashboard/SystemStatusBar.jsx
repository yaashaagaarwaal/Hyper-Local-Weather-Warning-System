import { BrainCircuit, Radio, Activity, Clock, FlaskConical } from "lucide-react";
import GlassCard from "../common/GlassCard";
import { useLastUpdated } from "../../hooks/useLastUpdated";
import { useSettings } from "../../context/SettingsContext";

const STATUS_ITEMS = [
  { icon: BrainCircuit, label: "AI Engine", value: "Online", color: "text-emerald-400" },
  { icon: Radio, label: "Weather Data", value: "Live", color: "text-cyan-400" },
  { icon: Activity, label: "Prediction Engine", value: "Active", color: "text-emerald-400" },
];

export default function SystemStatusBar() {
  const { label } = useLastUpdated();
  const { demoMode } = useSettings();

  return (
    <GlassCard className="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3.5">
      {STATUS_ITEMS.map(({ icon: Icon, label: itemLabel, value, color }) => (
        <div key={itemLabel} className="flex items-center gap-2 text-xs">
          <span className="relative flex h-2 w-2">
            <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${color.replace("text-", "bg-")}`} />
            <span className={`relative inline-flex h-2 w-2 rounded-full ${color.replace("text-", "bg-")}`} />
          </span>
          <Icon className="h-3.5 w-3.5 text-slate-500" />
          <span className="text-slate-500">{itemLabel}:</span>
          <span className={`font-semibold ${color}`}>{value}</span>
        </div>
      ))}

      <div className="ml-auto flex items-center gap-4">
        <div className="flex items-center gap-1.5 text-xs text-slate-500">
          <Clock className="h-3.5 w-3.5" />
          Last updated <span className="font-medium text-slate-300">{label}</span>
        </div>
        {demoMode && (
          <div className="flex items-center gap-1.5 rounded-full bg-amber-500/10 px-2.5 py-1 text-[11px] font-semibold text-amber-400 ring-1 ring-amber-500/25">
            <FlaskConical className="h-3 w-3" />
            DEMO MODE
          </div>
        )}
      </div>
    </GlassCard>
  );
}
