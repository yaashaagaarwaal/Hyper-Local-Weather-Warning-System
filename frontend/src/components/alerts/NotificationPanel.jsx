import { useEffect, useMemo, useRef } from "react";
import { AlertTriangle, CloudRain, Zap, Waves, X } from "lucide-react";
import { useAsync } from "../../hooks/useAsync";
import { fetchActiveAlerts } from "../../services/alertService";
import { ALERT_SEVERITY } from "../../data/alertsData";
import { formatRelativeTime } from "../../utils/formatters";
import { useSettings } from "../../context/SettingsContext";
import LoadingSpinner from "../common/LoadingSpinner";
import { Link } from "react-router-dom";

const HAZARD_ICONS = {
  "Heavy Rainfall": CloudRain,
  Thunderstorm: Zap,
  "Flash Flood": Waves,
  "Lightning Strike": Zap,
};

export default function NotificationPanel({ onClose }) {
  const { data: rawAlerts, isLoading } = useAsync(fetchActiveAlerts, []);
  const { alertSeverityPrefs } = useSettings();
  const panelRef = useRef(null);

  const alerts = useMemo(
    () => (rawAlerts ?? []).filter((a) => alertSeverityPrefs[a.severity]),
    [rawAlerts, alertSeverityPrefs]
  );

  useEffect(() => {
    function handleClickOutside(e) {
      if (panelRef.current && !panelRef.current.contains(e.target)) onClose();
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onClose]);

  return (
    <div
      ref={panelRef}
      className="absolute right-0 top-full z-50 mt-3 w-[22rem] overflow-hidden rounded-2xl border border-white/5 bg-navy-850 shadow-2xl shadow-black/50 animate-fade-in"
    >
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3.5">
        <div>
          <p className="text-sm font-semibold text-slate-100">Notifications</p>
          <p className="text-xs text-slate-500">{alerts?.length ?? 0} active alerts</p>
        </div>
        <button onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-white/5 hover:text-slate-300">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="max-h-96 overflow-y-auto scrollbar-thin">
        {isLoading ? (
          <LoadingSpinner label="Loading alerts" />
        ) : alerts.length === 0 ? (
          <p className="px-4 py-8 text-center text-sm text-slate-500">No active alerts right now.</p>
        ) : (
          alerts.map((alert) => {
            const sev = ALERT_SEVERITY[alert.severity];
            const Icon = HAZARD_ICONS[alert.hazardType] ?? AlertTriangle;
            return (
              <Link
                key={alert.id}
                to="/alerts"
                onClick={onClose}
                className="flex gap-3 border-b border-white/5 px-4 py-3.5 transition-colors last:border-0 hover:bg-white/5"
              >
                <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${sev.bg}`}>
                  <Icon className={`h-4 w-4 ${sev.text}`} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <p className="truncate text-sm font-medium text-slate-200">{alert.title}</p>
                    <span className={`shrink-0 text-[10px] font-semibold uppercase tracking-wide ${sev.text}`}>
                      {sev.label}
                    </span>
                  </div>
                  <p className="mt-0.5 truncate text-xs text-slate-500">{alert.location}</p>
                  <p className="mt-1 text-[11px] text-slate-600">{formatRelativeTime(alert.issuedAt)}</p>
                </div>
              </Link>
            );
          })
        )}
      </div>

      <Link
        to="/alerts"
        onClick={onClose}
        className="block border-t border-white/5 px-4 py-3 text-center text-xs font-medium text-cyan-400 hover:bg-white/5"
      >
        View all alerts
      </Link>
    </div>
  );
}
