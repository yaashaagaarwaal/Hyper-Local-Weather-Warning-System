import { AlertOctagon, X, CloudRain, Waves, Gauge } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useGridFocus } from "../../context/GridFocusContext";

export default function SevereScenarioBanner() {
  const { severeScenario, dismissSevereScenario, setFocusedGridId } = useGridFocus();
  const navigate = useNavigate();

  if (!severeScenario) return null;

  function handleViewPrediction() {
    setFocusedGridId(severeScenario.gridId);
    navigate("/predictions");
  }

  return (
    <div className="animate-fade-in relative overflow-hidden rounded-2xl border border-red-500/30 bg-gradient-to-r from-red-500/15 via-red-500/10 to-orange-500/10 p-5 shadow-lg shadow-red-500/10">
      <button
        onClick={dismissSevereScenario}
        className="absolute right-4 top-4 rounded-lg p-1.5 text-red-300/70 hover:bg-white/5 hover:text-red-200"
        aria-label="Dismiss scenario"
      >
        <X className="h-4 w-4" />
      </button>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-red-500/20 ring-1 ring-red-500/30">
          <AlertOctagon className="h-6 w-6 text-red-400" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-base font-bold text-red-200">Severe Weather — {severeScenario.gridId}</h3>
            <span className="rounded-full bg-red-500/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-red-300 ring-1 ring-red-500/30">
              Demo Scenario
            </span>
          </div>
          <p className="mt-0.5 text-xs text-red-300/70">
            Simulated {severeScenario.hazardType.toLowerCase()} event for demonstration purposes.
          </p>

          <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
            <div className="flex items-center gap-1.5">
              <CloudRain className="h-4 w-4 text-red-400" />
              <span className="text-red-300/70">Rainfall probability</span>
              <span className="font-bold text-red-200">{severeScenario.rainfallProbability}%</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Waves className="h-4 w-4 text-red-400" />
              <span className="text-red-300/70">Flood risk</span>
              <span className="font-bold text-red-200">{severeScenario.floodRisk}%</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Gauge className="h-4 w-4 text-red-400" />
              <span className="text-red-300/70">Overall risk</span>
              <span className="font-bold text-red-200">{severeScenario.overallRisk}/100</span>
            </div>
          </div>
        </div>

        <button
          onClick={handleViewPrediction}
          className="shrink-0 rounded-xl bg-red-500 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-red-500/30 transition-transform hover:scale-[1.02] active:scale-[0.98]"
        >
          View Detailed Prediction
        </button>
      </div>
    </div>
  );
}
