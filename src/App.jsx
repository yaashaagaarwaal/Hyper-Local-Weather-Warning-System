import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { LocationProvider } from "./context/LocationContext";
import { SettingsProvider } from "./context/SettingsContext";
import { GridFocusProvider } from "./context/GridFocusContext";
import AppLayout from "./components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import LiveWeather from "./pages/LiveWeather";
import RiskMapPage from "./pages/RiskMapPage";
import Predictions from "./pages/Predictions";
import Alerts from "./pages/Alerts";
import Analytics from "./pages/Analytics";
import Locations from "./pages/Locations";
import Settings from "./pages/Settings";
import About from "./pages/About";

export default function App() {
  return (
    <SettingsProvider>
      <LocationProvider>
        <GridFocusProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<AppLayout />}>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/weather" element={<LiveWeather />} />
                <Route path="/risk-map" element={<RiskMapPage />} />
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
