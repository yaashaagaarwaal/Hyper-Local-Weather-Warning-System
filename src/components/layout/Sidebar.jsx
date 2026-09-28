import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  CloudSun,
  Map,
  TrendingUp,
  AlertTriangle,
  BarChart3,
  MapPin,
  Settings,
  Radar,
  Satellite,
  Wifi,
  Info,
  FlaskConical,
} from "lucide-react";
import { useSettings } from "../../context/SettingsContext";
import { hexToRgba } from "../../utils/colors";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/weather", label: "Live Weather", icon: CloudSun },
  { to: "/risk-map", label: "Risk Map", icon: Map },
  { to: "/predictions", label: "Predictions", icon: TrendingUp },
  { to: "/alerts", label: "Alerts", icon: AlertTriangle },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/locations", label: "Locations", icon: MapPin },
  { to: "/about", label: "About", icon: Info },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar({ mobileOpen, onNavigate }) {
  const { accent, demoMode } = useSettings();

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-white/5 bg-navy-900/95 backdrop-blur-xl transition-transform duration-300 lg:static lg:translate-x-0 ${
        mobileOpen ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      <div className="flex items-center gap-2.5 px-5 py-6">
        <div
          className="flex h-10 w-10 items-center justify-center rounded-xl shadow-lg"
          style={{
            background: `linear-gradient(135deg, ${accent.color}, ${accent.colorTo})`,
            boxShadow: `0 8px 20px ${hexToRgba(accent.color, 0.25)}`,
          }}
        >
          <Radar className="h-5 w-5 text-navy-950" strokeWidth={2.5} />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <p className="truncate text-base font-bold leading-tight tracking-tight text-white">MausamAI</p>
            {demoMode && (
              <span className="shrink-0 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-amber-400 ring-1 ring-amber-500/30">
                Demo
              </span>
            )}
          </div>
          <p className="text-[11px] leading-tight text-slate-400">Weather Intelligence</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-2 scrollbar-thin">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onNavigate}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all duration-150 ${
                isActive ? "shadow-inner" : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
              }`
            }
            style={({ isActive }) =>
              isActive
                ? {
                    background: `linear-gradient(90deg, ${hexToRgba(accent.color, 0.15)}, ${hexToRgba(accent.colorTo, 0.08)})`,
                    color: accent.color,
                    boxShadow: `inset 0 0 0 1px ${hexToRgba(accent.color, 0.2)}`,
                  }
                : undefined
            }
          >
            {({ isActive }) => (
              <>
                <Icon
                  className="h-[18px] w-[18px] transition-colors"
                  style={{ color: isActive ? accent.color : undefined }}
                />
                <span>{label}</span>
                {isActive && <span className="ml-auto h-1.5 w-1.5 rounded-full" style={{ backgroundColor: accent.color }} />}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="space-y-3 border-t border-white/5 px-4 py-4">
        {demoMode && (
          <div className="flex items-start gap-2 rounded-xl bg-amber-500/10 p-3 ring-1 ring-amber-500/20">
            <FlaskConical className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
            <p className="text-[11px] leading-snug text-amber-300/90">
              <span className="font-semibold">Demo Mode</span> — using simulated weather &amp; prediction data.
            </p>
          </div>
        )}

        <div className="space-y-2 rounded-xl bg-navy-800/60 p-3 ring-1 ring-white/5">
          <div className="flex items-center gap-2 text-xs">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            </span>
            <span className="flex items-center gap-1.5 text-slate-300">
              <Satellite className="h-3.5 w-3.5 text-slate-500" />
              AI Engine Online
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-400" />
            </span>
            <span className="flex items-center gap-1.5 text-slate-300">
              <Wifi className="h-3.5 w-3.5 text-slate-500" />
              Live Data Connected
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 rounded-xl px-1 py-1">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 text-sm font-semibold text-white">
            YA
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-slate-200">Yash Agarwal</p>
            <p className="truncate text-xs text-slate-500">SIH Team · Analyst</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
