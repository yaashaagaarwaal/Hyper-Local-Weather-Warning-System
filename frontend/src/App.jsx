import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { LocationProvider } from "./context/LocationContext";
import { SettingsProvider } from "./context/SettingsContext";
import { GridFocusProvider } from "./context/GridFocusContext";
import AppLayout from "./components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import LiveWeather from "./pages/LiveWeather";
import RiskMapPage from "./pages/RiskMapPage";
import Imagery from "./pages/Imagery";
import HazardModels from "./pages/HazardModels";
import Predictions from "./pages/Predictions";
import Alerts from "./pages/Alerts";
import Analytics from "./pages/Analytics";
import Locations from "./pages/Locations";
import Settings from "./pages/Settings";
import About from "./pages/About";

// The public home page carries three.js and GSAP; it is its own chunk so the app pages never pay for it.
const Home = lazy(() => import("./pages/Home"));

export default function App() {
  return (
    <SettingsProvider>
      <LocationProvider>
        <GridFocusProvider>
          <BrowserRouter>
            <Routes>
              <Route
                path="/"
                element={(
                  <Suspense fallback={<div style={{ minHeight: "100vh", background: "#05080f" }} />}>
                    <Home />
                  </Suspense>
                )}
              />
              <Route element={<AppLayout />}>
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/weather" element={<LiveWeather />} />
                <Route path="/risk-map" element={<RiskMapPage />} />
                <Route path="/imagery" element={<Imagery />} />
                <Route path="/hazard-models" element={<HazardModels />} />
                <Route path="/predictions" element={<Predictions />} />
                <Route path="/alerts" element={<Alerts />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/locations" element={<Locations />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/about" element={<About />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </GridFocusProvider>
      </LocationProvider>
    </SettingsProvider>
  );
}
