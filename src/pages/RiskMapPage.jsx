import { useEffect, useState } from "react";
import { Target } from "lucide-react";
import RiskMap from "../components/map/RiskMap";
import GridDetailPanel from "../components/map/GridDetailPanel";
import GlassCard from "../components/common/GlassCard";
import { RISK_LEVELS, getRiskGridStats, getGridById } from "../data/riskData";
import { useGridFocus } from "../context/GridFocusContext";

export default function RiskMapPage() {
  const [selectedCell, setSelectedCell] = useState(null);
  const [focusCell, setFocusCell] = useState(null);
  const { focusedGridId, clearFocusedGrid } = useGridFocus();
  const stats = getRiskGridStats();

  useEffect(() => {
    if (!focusedGridId) return;
    const cell = getGridById(focusedGridId);
    if (cell) {
      setSelectedCell(cell);
      setFocusCell(cell);
    }
    clearFocusedGrid();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusedGridId]);

  return (
    <div className="space-y-5">
      {focusCell && (
        <div className="flex items-center gap-2 rounded-xl bg-cyan-500/10 px-4 py-2.5 text-xs text-cyan-300 ring-1 ring-cyan-500/20">
          <Target className="h-3.5 w-3.5" />
          Showing affected area for <span className="font-semibold">{focusCell.id}</span> — jumped here from an alert.
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Object.values(RISK_LEVELS).map((risk) => (
          <GlassCard key={risk.key} className="flex items-center gap-3 p-4">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: risk.color }} />
            <div>
              <p className="text-lg font-bold text-white">{stats[risk.key]}</p>
              <p className="text-xs text-slate-500">{risk.label} zones</p>
            </div>
          </GlassCard>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-4">
        <GlassCard className="overflow-hidden p-0 xl:col-span-3">
          <div className="h-[640px] p-4">
            <RiskMap
              onCellSelect={(cell) => {
                setSelectedCell(cell);
                if (focusCell && cell.id !== focusCell.id) setFocusCell(null);
              }}
              focusCell={focusCell}
            />
          </div>
        </GlassCard>

        <GridDetailPanel cell={selectedCell} />
      </div>
    </div>
  );
}
