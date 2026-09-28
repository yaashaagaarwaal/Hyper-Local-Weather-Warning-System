import { createContext, useContext, useMemo, useState } from "react";

const SettingsContext = createContext(null);

const ACCENT_THEMES = {
  cyan: { key: "cyan", label: "Cyan Ocean", color: "#22d3ee", colorTo: "#3b82f6" },
  violet: { key: "violet", label: "Violet Storm", color: "#a78bfa", colorTo: "#6366f1" },
  emerald: { key: "emerald", label: "Emerald Radar", color: "#34d399", colorTo: "#0ea5e9" },
};

const REFRESH_INTERVALS = [
  { value: 30, label: "30 seconds" },
  { value: 60, label: "1 minute" },
  { value: 300, label: "5 minutes" },
  { value: 900, label: "15 minutes" },
];

export function SettingsProvider({ children }) {
  const [tempUnit, setTempUnit] = useState("C");
  const [refreshInterval, setRefreshInterval] = useState(60);
  const [accentTheme, setAccentTheme] = useState("cyan");
  const [demoMode, setDemoMode] = useState(true);
  const [alertSeverityPrefs, setAlertSeverityPrefs] = useState({
    advisory: true,
    warning: true,
    severe: true,
  });
  const [notificationPrefs, setNotificationPrefs] = useState({
    severeAlerts: true,
    warningAlerts: true,
    dailyDigest: false,
    gridUpdates: true,
  });

  const value = useMemo(
    () => ({
      tempUnit,
      toggleTempUnit: () => setTempUnit((u) => (u === "C" ? "F" : "C")),
      refreshInterval,
      setRefreshInterval,
      refreshIntervalOptions: REFRESH_INTERVALS,
      accentTheme,
      setAccentTheme,
      accentThemeOptions: ACCENT_THEMES,
      accent: ACCENT_THEMES[accentTheme],
      demoMode,
      setDemoMode,
      alertSeverityPrefs,
      setAlertSeverityPrefs,
      notificationPrefs,
      setNotificationPrefs,
    }),
    [tempUnit, refreshInterval, accentTheme, demoMode, alertSeverityPrefs, notificationPrefs]
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within a SettingsProvider");
  return ctx;
}
