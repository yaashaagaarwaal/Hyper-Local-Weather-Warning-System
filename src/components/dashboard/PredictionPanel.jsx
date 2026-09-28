import { Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import GlassCard from "../common/GlassCard";
import LoadingSpinner from "../common/LoadingSpinner";
import PredictionChart from "../charts/PredictionChart";

export default function PredictionPanel({ forecast, loading }) {
  return (
    <GlassCard className="p-5">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500/20 to-indigo-500/20 ring-1 ring-cyan-500/20">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">AI Prediction — Next 60 Minutes</h3>
          <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-amber-400 ring-1 ring-amber-500/25">
            Prototype
          </span>
        </div>
        <Link to="/predictions" className="text-xs font-medium text-cyan-400 hover:text-cyan-300">
          Full breakdown
        </Link>
      </div>
      <p className="mb-4 text-xs text-slate-500">
        AI prediction based on recent meteorological conditions and spatial weather patterns.
      </p>

      {loading ? <LoadingSpinner label="Running nowcast model" /> : <PredictionChart data={forecast} />}
    </GlassCard>
  );
}
