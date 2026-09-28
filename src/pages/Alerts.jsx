import { useMemo } from "react";
import { Radio, CheckCircle2 } from "lucide-react";
import { useAsync } from "../hooks/useAsync";
import { fetchAllAlerts } from "../services/alertService";
import AlertCard from "../components/alerts/AlertCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import GlassCard from "../components/common/GlassCard";

export default function Alerts() {
  const { data: alerts, isLoading } = useAsync(fetchAllAlerts, []);

  const { active, resolved } = useMemo(() => {
    if (!alerts) return { active: [], resolved: [] };
    return {
      active: alerts.filter((a) => a.status === "active"),
      resolved: alerts.filter((a) => a.status === "resolved"),
    };
  }, [alerts]);

  return (
    <div className="space-y-6">
      <GlassCard className="flex flex-wrap items-center justify-between gap-4 p-5">
        <div>
          <p className="text-sm font-semibold text-slate-200">Warning Bulletins</p>
          <p className="text-xs text-slate-500">
            {isLoading ? "Loading…" : `${active.length} active · ${resolved.length} resolved in last 24 hours`}
          </p>
        </div>
      </GlassCard>

      {isLoading ? (
        <LoadingSpinner label="Loading alerts" />
      ) : (
        <>
          <section>
            <div className="mb-3 flex items-center gap-2">
              <Radio className="h-4 w-4 text-red-400" />
              <h3 className="text-sm font-semibold text-slate-200">Active Alerts</h3>
              <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[11px] font-semibold text-red-400">
                {active.length}
              </span>
            </div>
            {active.length === 0 ? (
              <GlassCard className="p-8 text-center text-sm text-slate-500">No active alerts right now.</GlassCard>
            ) : (
              <div className="space-y-3">
                {active.map((alert) => (
                  <AlertCard key={alert.id} alert={alert} />
                ))}
              </div>
            )}
          </section>

          <section>
            <div className="mb-3 flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-slate-500" />
              <h3 className="text-sm font-semibold text-slate-200">Resolved Alerts</h3>
              <span className="rounded-full bg-slate-500/15 px-2 py-0.5 text-[11px] font-semibold text-slate-400">
                {resolved.length}
              </span>
            </div>
            {resolved.length === 0 ? (
              <GlassCard className="p-8 text-center text-sm text-slate-500">No resolved alerts yet.</GlassCard>
            ) : (
              <div className="space-y-3 opacity-80">
                {resolved.map((alert) => (
                  <AlertCard key={alert.id} alert={alert} />
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
