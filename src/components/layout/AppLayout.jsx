import { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";

const PAGE_META = {
  "/dashboard": { title: "Dashboard", subtitle: "Real-time hyper-local weather intelligence overview" },
  "/weather": { title: "Live Weather", subtitle: "Current conditions across monitored stations" },
  "/risk-map": { title: "Risk Map", subtitle: "Grid-level hazard risk visualization for NCR" },
  "/predictions": { title: "Predictions", subtitle: "AI-driven short and mid-term forecasts" },
  "/alerts": { title: "Alerts", subtitle: "Active and historical warning bulletins" },
  "/analytics": { title: "Analytics", subtitle: "Trends, patterns and historical performance" },
  "/locations": { title: "Locations", subtitle: "Manage monitored stations and coverage" },
  "/settings": { title: "Settings", subtitle: "Configure system and notification preferences" },
  "/about": { title: "About MausamAI", subtitle: "The problem, the approach, and the AI pipeline" },
};

export default function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const meta = PAGE_META[location.pathname] ?? { title: "MausamAI", subtitle: "" };

  return (
    <div className="flex min-h-screen bg-navy-950">
      <Sidebar mobileOpen={mobileOpen} onNavigate={() => setMobileOpen(false)} />

      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <div className="flex min-h-screen flex-1 flex-col lg:pl-0">
        <Navbar title={meta.title} subtitle={meta.subtitle} onMenuClick={() => setMobileOpen((v) => !v)} />
        <main className="flex-1 px-4 py-6 lg:px-8 lg:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
