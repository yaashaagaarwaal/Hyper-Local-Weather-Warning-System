import { useEffect, useState } from "react";
import { Target, History } from "lucide-react";
import RiskMap from "../components/map/RiskMap";
import GridDetailPanel from "../components/map/GridDetailPanel";
import GlassCard from "../components/common/GlassCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import { RISK_LEVELS } from "../data/riskData";
import { fetchRiskGridStats, fetchGridById, fetchNcrFloodDistricts } from "../services/riskService";
import { useAsync } from "../hooks/useAsync";
import { useGridFocus } from "../context/GridFocusContext";

// Static historical comparison across the 20 verified NCR districts —
// item 7's "historical flood visualization", using real India Flood
// Inventory data (same source as the map's district popups). A plain
// ranked bar list, not a new chart library dependency — proportionate to
// a 20-row comparison and consistent with this project's "no unnecessary
// features" instruction.
function DistrictFloodSeverityChart({ geojson }) {
  const ranked = [...geojson.features]
    .filter((f) => f.properties.DFSI != null)
    .sort((a, b) => b.properties.DFSI - a.properties.DFSI);

  return (
    <div className="space-y-2">
      {ranked.map((f) => {
        const { DFSI, lgd_district_name, geoBoundaries_shapeName } = f.properties;
        const name = lgd_district_name ?? geoBoundaries_shapeName;
        const pct = Math.min(100, (DFSI / 20) * 100);
        return (
          <div key={f.properties.geoBoundaries_shapeID} className="flex items-center gap-3 text-xs">
            <span className="w-28 shrink-0 truncate text-slate-400">{name}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-navy-800/80">
              <div
                className="h-full rounded-full bg-gradient-to-r from-amber-500 to-red-500"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="w-10 shrink-0 text-right font-semibold text-slate-300">{DFSI.toFixed(1)}</span>
          </div>
        );
      })}
    </div>
  );
}

export default function RiskMapPage() {
  const [selectedCell, setSelectedCell] = useState(null);
  const [focusCell, setFocusCell] = useState(null);
  const { focusedGridId, clearFocusedGrid } = useGridFocus();
  const { data: stats } = useAsync(() => fetchRiskGridStats(), []);
  const { data: districtGeoJson, isLoading: districtsLoading } = useAsync(() => fetchNcrFloodDistricts(), []);

  useEffect(() => {
    if (!focusedGridId) return;
    fetchGridById(focusedGridId).then((cell) => {
      if (cell) {
        setSelectedCell(cell);
        setFocusCell(cell);
      }
    });
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
              <p className="text-lg font-bold text-white">{stats?.[risk.key] ?? "—"}</p>
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

      <GlassCard className="p-5">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <History className="h-4 w-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-slate-200">District Flood Severity (Historical)</h3>
          </div>
          <span className="rounded-full bg-amber-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-amber-400 ring-1 ring-amber-500/25">
            India Flood Inventory
          </span>
        </div>
        <p className="mb-4 text-xs text-slate-500">
          District Flood Severity Index (0-20) across the 20 verified Delhi-NCR districts — static
          historical data, not a live reading or forecast.
        </p>
        {districtsLoading ? (
          <LoadingSpinner label="Loading district flood history" />
        ) : districtGeoJson ? (
          <DistrictFloodSeverityChart geojson={districtGeoJson} />
        ) : (
          <p className="py-4 text-center text-xs text-slate-500">
            District flood history is unavailable right now (backend unreachable).
          </p>
        )}
      </GlassCard>
    </div>
  );
}
