import { useState } from "react";
import { Bell, Menu } from "lucide-react";
import LocationSelector from "../weather/LocationSelector";
import NotificationPanel from "../alerts/NotificationPanel";
import { useDateTime } from "../../hooks/useDateTime";
import { formatDate, formatTime } from "../../utils/formatters";
import { getActiveAlerts } from "../../data/alertsData";
import { useSettings } from "../../context/SettingsContext";

export default function Navbar({ title, subtitle, onMenuClick }) {
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const now = useDateTime();
  const { alertSeverityPrefs } = useSettings();
  const activeCount = getActiveAlerts().filter((a) => alertSeverityPrefs[a.severity]).length;

  return (
    <header className="sticky top-0 z-30 border-b border-white/5 bg-navy-950/80 backdrop-blur-xl">
      <div className="flex items-center gap-4 px-4 py-4 lg:px-8">
        <button
          onClick={onMenuClick}
          className="rounded-lg p-2 text-slate-400 hover:bg-white/5 hover:text-slate-200 lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="min-w-0 flex-1">
          <h1 className="truncate text-lg font-bold tracking-tight text-white sm:text-xl">{title}</h1>
          {subtitle && <p className="hidden truncate text-xs text-slate-500 sm:block">{subtitle}</p>}
        </div>

        <LocationSelector className="hidden w-64 md:block" />

        <div className="hidden flex-col items-end text-right lg:flex">
          <span className="text-sm font-medium text-slate-200">{formatTime(now)}</span>
          <span className="text-[11px] text-slate-500">{formatDate(now)}</span>
        </div>

        <div className="relative">
          <button
            onClick={() => setNotificationsOpen((v) => !v)}
            className="relative rounded-xl bg-navy-800/60 p-2.5 text-slate-400 ring-1 ring-white/5 transition-colors hover:bg-navy-800 hover:text-slate-200"
          >
            <Bell className="h-[18px] w-[18px]" />
            {activeCount > 0 && (
              <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[9px] font-bold text-white ring-2 ring-navy-950">
                {activeCount}
              </span>
            )}
          </button>
          {notificationsOpen && <NotificationPanel onClose={() => setNotificationsOpen(false)} />}
        </div>

        <button className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 text-sm font-semibold text-white ring-2 ring-white/5 transition-transform hover:scale-105">
          YA
        </button>
      </div>
    </header>
  );
}
