import { Radio, CheckCircle2 } from "lucide-react";
import { useAsync } from "../hooks/useAsync";
import { fetchActiveAlerts, fetchAlertHistory, acknowledgeAlert, resolveAlert } from "../services/alertService";
import AlertCard from "../components/alerts/AlertCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import GlassCard from "../components/common/GlassCard";

export default function Alerts() {
  const { data: active, isLoading: activeLoading, refetch: refetchActive } = useAsync(fetchActiveAlerts, []);
  const { data: resolved, isLoading: historyLoading, refetch: refetchHistory } = useAsync(fetchAlertHistory, []);
  const isLoading = activeLoading || historyLoading;

  async function handleAcknowledge(id) {
    await acknowledgeAlert(id);
    refetchActive();
  }

  async function handleResolve(id) {
    await resolveAlert(id);
    refetchActive();
    refetchHistory();
  }

  return (
    <div className="space-y-6">
      <GlassCard className="flex flex-wrap items-center justify-between gap-4 p-5">
        <div>
          <p className="text-sm font-semibold text-slate-200">Warning Bulletins</p>
          <p className="text-xs text-slate-500">
            {isLoading
              ? "Loading…"
              : `${active?.length ?? 0} active · ${resolved?.length ?? 0} resolved/expired`}
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
                {active?.length ?? 0}
              </span>
            </div>
            {!active?.length ? (
              <GlassCard className="p-8 text-center text-sm text-slate-500">No active alerts right now.</GlassCard>
            ) : (
              <div className="space-y-3">
                {active.map((alert) => (
                  <AlertCard
                    key={alert.id}
                    alert={alert}
                    onAcknowledge={() => handleAcknowledge(alert.id)}
                    onResolve={() => handleResolve(alert.id)}
                  />
                ))}
              </div>
            )}
          </section>

          <section>
            <div className="mb-3 flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-slate-500" />
              <h3 className="text-sm font-semibold text-slate-200">Resolved &amp; Expired Alerts</h3>
              <span className="rounded-full bg-slate-500/15 px-2 py-0.5 text-[11px] font-semibold text-slate-400">
                {resolved?.length ?? 0}
              </span>
            </div>
            {!resolved?.length ? (
              <GlassCard className="p-8 text-center text-sm text-slate-500">No resolved or expired alerts yet.</GlassCard>
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
