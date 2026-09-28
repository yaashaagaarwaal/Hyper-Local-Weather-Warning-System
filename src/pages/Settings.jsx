import { useState } from "react";
import { Bell, Shield, Globe, Info, Check, Thermometer, RefreshCw, Palette, MapPin, ShieldAlert } from "lucide-react";
import GlassCard from "../components/common/GlassCard";
import Toggle from "../components/common/Toggle";
import { useSettings } from "../context/SettingsContext";
import { useLocationContext } from "../context/LocationContext";
import { ALERT_SEVERITY } from "../data/alertsData";

const NOTIFICATION_OPTIONS = [
  { key: "severeAlerts", label: "Severe Weather Alerts", desc: "Immediate push for severe/extreme hazard warnings" },
  { key: "warningAlerts", label: "Warning-Level Alerts", desc: "Notify for moderate and warning-tier bulletins" },
  { key: "dailyDigest", label: "Daily Digest", desc: "Summary of forecast and risk trend every morning" },
  { key: "gridUpdates", label: "Grid Risk Updates", desc: "Alert when a monitored grid crosses into High/Severe" },
];

function ToggleRow({ label, desc, checked, onChange }) {
  return (
    <div className="flex items-center justify-between gap-4 py-3.5">
      <div className="min-w-0">
        <p className="text-sm font-medium text-slate-200">{label}</p>
        <p className="mt-0.5 text-xs text-slate-500">{desc}</p>
      </div>
      <Toggle checked={checked} onChange={onChange} />
    </div>
  );
}

export default function Settings() {
  const {
    tempUnit,
    toggleTempUnit,
    refreshInterval,
    setRefreshInterval,
    refreshIntervalOptions,
    accentTheme,
    setAccentTheme,
    accentThemeOptions,
    accent,
    demoMode,
    setDemoMode,
    alertSeverityPrefs,
    setAlertSeverityPrefs,
    notificationPrefs,
    setNotificationPrefs,
  } = useSettings();
  const { locations, selectedLocationId, setSelectedLocationId } = useLocationContext();
  const [saved, setSaved] = useState(false);

  function handleSave() {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <GlassCard className="p-5">
        <div className="mb-1 flex items-center gap-2">
          <Bell className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200">Notification Preferences</h3>
        </div>
        <p className="mb-1 text-xs text-slate-500">Choose what triggers a push or in-app notification</p>
        <div className="divide-y divide-white/5">
          {NOTIFICATION_OPTIONS.map((opt) => (
            <ToggleRow
              key={opt.key}
              label={opt.label}
              desc={opt.desc}
              checked={notificationPrefs[opt.key]}
              onChange={(v) => setNotificationPrefs((s) => ({ ...s, [opt.key]: v }))}
            />
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <div className="mb-1 flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-orange-400" />
          <h3 className="text-sm font-semibold text-slate-200">Alert Severity Preferences</h3>
        </div>
        <p className="mb-1 text-xs text-slate-500">
          Controls which severities appear in the notification bell across the app
        </p>
        <div className="divide-y divide-white/5">
          {Object.values(ALERT_SEVERITY).map((sev) => (
            <ToggleRow
              key={sev.key}
              label={sev.label}
              desc={`Show ${sev.label.toLowerCase()}-level alerts in notifications`}
              checked={alertSeverityPrefs[sev.key]}
              onChange={(v) => setAlertSeverityPrefs((s) => ({ ...s, [sev.key]: v }))}
            />
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <div className="mb-4 flex items-center gap-2">
          <Thermometer className="h-4 w-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-slate-200">Temperature Unit</h3>
        </div>
        <div className="flex items-center gap-2 rounded-xl bg-navy-800/60 p-1 ring-1 ring-white/5">
          {["C", "F"].map((unit) => (
            <button
              key={unit}
              onClick={() => unit !== tempUnit && toggleTempUnit()}
              className={`flex-1 rounded-lg py-2 text-sm font-semibold transition-colors ${
                tempUnit === unit ? "bg-cyan-500/15 text-cyan-300" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              °{unit === "C" ? "Celsius" : "Fahrenheit"}
            </button>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <div className="mb-4 flex items-center gap-2">
          <MapPin className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200">Location Settings</h3>
        </div>
        <p className="mb-2 text-xs text-slate-500">Default location shown when the app opens</p>
        <select
          value={selectedLocationId}
          onChange={(e) => setSelectedLocationId(e.target.value)}
          className="w-full rounded-xl bg-navy-800/60 px-3.5 py-2.5 text-sm text-slate-200 ring-1 ring-white/5 focus:outline-none focus:ring-cyan-500/40"
        >
          {locations.map((loc) => (
            <option key={loc.id} value={loc.id}>
              {loc.name}, {loc.region}
            </option>
          ))}
        </select>
      </GlassCard>

      <GlassCard className="p-5">
        <div className="mb-4 flex items-center gap-2">
          <RefreshCw className="h-4 w-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-200">Data Refresh Interval</h3>
        </div>
        <p className="mb-2 text-xs text-slate-500">
          How often the "Last updated" indicator on the Dashboard refreshes
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {refreshIntervalOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setRefreshInterval(opt.value)}
              className={`rounded-lg py-2 text-xs font-semibold transition-colors ${
                refreshInterval === opt.value
                  ? "bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30"
                  : "bg-navy-800/60 text-slate-400 ring-1 ring-white/5 hover:text-slate-200"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <div className="mb-4 flex items-center gap-2">
          <Palette className="h-4 w-4 text-violet-400" />
          <h3 className="text-sm font-semibold text-slate-200">Theme</h3>
        </div>
        <p className="mb-3 text-xs text-slate-500">Accent color used across the sidebar, buttons, and highlights</p>
        <div className="grid grid-cols-3 gap-2">
          {Object.values(accentThemeOptions).map((theme) => (
            <button
              key={theme.key}
              onClick={() => setAccentTheme(theme.key)}
              className={`flex flex-col items-center gap-2 rounded-xl p-3 ring-1 transition-colors ${
                accentTheme === theme.key ? "bg-white/5 ring-white/20" : "ring-white/5 hover:bg-white/[0.03]"
              }`}
            >
              <span
                className="h-6 w-6 rounded-full"
                style={{ background: `linear-gradient(135deg, ${theme.color}, ${theme.colorTo})` }}
              />
              <span className="text-[11px] font-medium text-slate-300">{theme.label}</span>
            </button>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <div className="mb-1 flex items-center gap-2">
          <Shield className="h-4 w-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-200">System Status</h3>
        </div>
        <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          {[
            { label: "AI Engine", value: "Online", color: "text-emerald-400" },
            { label: "Data Feed", value: "Connected", color: "text-emerald-400" },
            { label: "Model Version", value: "v2.4.1-nowcast", color: "text-slate-300" },
          ].map((s) => (
            <div key={s.label} className="rounded-xl bg-navy-800/50 p-3.5 ring-1 ring-white/5">
              <p className="text-[11px] text-slate-500">{s.label}</p>
              <p className={`mt-1 text-sm font-semibold ${s.color}`}>{s.value}</p>
            </div>
          ))}
        </div>
        <ToggleRow
          label="Demo Mode"
          desc="Show the DEMO MODE badge and banner while backend/ML are not yet connected"
          checked={demoMode}
          onChange={setDemoMode}
        />
      </GlassCard>

      <GlassCard className="p-5">
        <div className="mb-2 flex items-center gap-2">
          <Globe className="h-4 w-4 text-slate-400" />
          <h3 className="text-sm font-semibold text-slate-200">About MausamAI</h3>
        </div>
        <p className="text-xs leading-relaxed text-slate-500">
          MausamAI is a hyper-local weather intelligence and early warning system built for SIH Problem
          Statement 26077. This build is a frontend prototype running on mock data — production
          deployments connect to a FastAPI backend and ML nowcasting pipeline.
        </p>
        <div className="mt-3 flex items-center gap-1.5 text-xs text-slate-600">
          <Info className="h-3.5 w-3.5" /> Version 0.1.0-prototype
        </div>
      </GlassCard>

      <div className="flex justify-end">
        <button
          onClick={handleSave}
          style={{ background: `linear-gradient(90deg, ${accent.color}, ${accent.colorTo})` }}
          className="flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold text-white shadow-lg transition-transform hover:scale-[1.02] active:scale-[0.98]"
        >
          {saved ? <Check className="h-4 w-4" /> : null}
          {saved ? "Saved" : "Save Changes"}
        </button>
      </div>
    </div>
  );
}
