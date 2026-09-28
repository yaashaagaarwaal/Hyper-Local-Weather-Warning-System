import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, MapPin, Clock, ChevronDown, CheckCircle2, Radio, Gauge, ClipboardList, Map } from "lucide-react";
import { ALERT_SEVERITY } from "../../data/alertsData";
import { formatRelativeTime, formatClockTime } from "../../utils/formatters";
import { useGridFocus } from "../../context/GridFocusContext";
import GlassCard from "../common/GlassCard";

export default function AlertCard({ alert, compact = false }) {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();
  const { setFocusedGridId } = useGridFocus();
  const sev = ALERT_SEVERITY[alert.severity];
  const isActive = alert.status === "active";

  function handleViewOnMap(e) {
    e.stopPropagation();
    setFocusedGridId(alert.affectedGrids[0]);
    navigate("/risk-map");
  }

  return (
    <GlassCard
      className={`overflow-hidden ring-1 transition-all duration-200 ${
        expanded ? "ring-white/10" : "ring-transparent"
      }`}
    >
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-start gap-3.5 p-4 text-left transition-colors hover:bg-white/[0.03] sm:p-5"
      >
        <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${sev.bg}`}>
          <AlertTriangle className={`h-5 w-5 ${sev.text}`} />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-sm font-semibold text-slate-100 sm:text-base">{alert.title}</h4>
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${sev.bg} ${sev.text}`}>
              {sev.label}
            </span>
            {isActive ? (
              <span className="flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
                <Radio className="h-2.5 w-2.5 animate-pulse" /> Active
              </span>
            ) : (
              <span className="flex items-center gap-1 rounded-full bg-slate-500/10 px-2 py-0.5 text-[10px] font-semibold text-slate-400">
                <CheckCircle2 className="h-2.5 w-2.5" /> Resolved
              </span>
            )}
          </div>

          <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <MapPin className="h-3.5 w-3.5" /> {alert.location}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" /> {formatRelativeTime(alert.issuedAt)}
            </span>
            {!compact && (
              <span className="flex items-center gap-1">
                <Gauge className="h-3.5 w-3.5" /> Risk score {alert.riskScore}
              </span>
            )}
          </div>
        </div>

        <ChevronDown className={`h-4 w-4 shrink-0 text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>

      {expanded && (
        <div className="animate-fade-in border-t border-white/5 px-4 pb-4 pt-3.5 sm:px-5">
          <p className="text-sm leading-relaxed text-slate-400">{alert.description}</p>

          <div className="mt-3 flex items-start gap-2 rounded-xl bg-navy-800/50 p-3 ring-1 ring-white/5">
            <ClipboardList className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400" />
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Recommended Action</p>
              <p className="mt-0.5 text-xs leading-relaxed text-slate-300">{alert.recommendedAction}</p>
            </div>
          </div>

          <div className="mt-3.5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div>
              <p className="text-[11px] text-slate-500">Issued</p>
              <p className="text-xs font-medium text-slate-300">{formatClockTime(alert.issuedAt)}</p>
            </div>
            <div>
              <p className="text-[11px] text-slate-500">Expires</p>
              <p className="text-xs font-medium text-slate-300">{formatClockTime(alert.expiresAt)}</p>
            </div>
            <div>
              <p className="text-[11px] text-slate-500">Risk Score</p>
              <p className="text-xs font-medium text-slate-300">{alert.riskScore}/100</p>
            </div>
            <div>
              <p className="text-[11px] text-slate-500">Confidence</p>
              <p className="text-xs font-medium text-slate-300">{alert.confidence}%</p>
            </div>
          </div>

          <button
            onClick={handleViewOnMap}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-navy-800 py-2.5 text-xs font-semibold text-slate-200 ring-1 ring-white/10 transition-colors hover:bg-navy-700 hover:text-cyan-300 sm:w-auto sm:px-5"
          >
            <Map className="h-3.5 w-3.5" />
            View on Map
          </button>
        </div>
      )}
    </GlassCard>
  );
}
