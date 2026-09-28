import { useState } from "react";
import { Thermometer, CloudRain, Droplets, Wind, Maximize2 } from "lucide-react";
import { Link } from "react-router-dom";
import { useLocationContext } from "../context/LocationContext";
import { useGridFocus } from "../context/GridFocusContext";
import { useSettings } from "../context/SettingsContext";
import { formatTemperature } from "../utils/formatters";
import { useAsync } from "../hooks/useAsync";
import { fetchCurrentWeather } from "../services/weatherService";
import { fetchShortTermForecast, fetchOverallRisk, fetchRiskSummary } from "../services/predictionService";
import { fetchActiveAlerts } from "../services/alertService";

import WeatherCard from "../components/dashboard/WeatherCard";
import RiskCard from "../components/dashboard/RiskCard";
import RiskSummary from "../components/dashboard/RiskSummary";
import PredictionPanel from "../components/dashboard/PredictionPanel";
import SystemStatusBar from "../components/dashboard/SystemStatusBar";
import SevereScenarioBanner from "../components/dashboard/SevereScenarioBanner";
import CurrentWeatherPanel from "../components/weather/CurrentWeatherPanel";
import RiskMap from "../components/map/RiskMap";
import GridDetailPanel from "../components/map/GridDetailPanel";
import AlertCard from "../components/alerts/AlertCard";
import GlassCard from "../components/common/GlassCard";
import LoadingSpinner from "../components/common/LoadingSpinner";

export default function Dashboard() {
  const { selectedLocationId } = useLocationContext();
  const { triggerSevereScenario } = useGridFocus();
  const { tempUnit } = useSettings();
  const [selectedCell, setSelectedCell] = useState(null);

  const { data: weather, isLoading: weatherLoading } = useAsync(
    () => fetchCurrentWeather(selectedLocationId),
    [selectedLocationId]
  );
  const { data: forecast, isLoading: forecastLoading } = useAsync(
    () => fetchShortTermForecast(selectedLocationId),
    [selectedLocationId]
  );
  const { data: overallRisk, isLoading: riskLoading } = useAsync(
    () => fetchOverallRisk(selectedLocationId),
    [selectedLocationId]
  );
  const { data: riskSummary, isLoading: summaryLoading } = useAsync(
    () => fetchRiskSummary(selectedLocationId),
    [selectedLocationId]
  );
  const { data: alerts, isLoading: alertsLoading } = useAsync(fetchActiveAlerts, []);

  function handleCellSelect(cell) {
    setSelectedCell(cell);
    if (cell.riskLevel === "severe") triggerSevereScenario(cell);
  }

  return (
    <div className="space-y-6">
      <SystemStatusBar />
      <SevereScenarioBanner />

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <WeatherCard
          icon={Thermometer}
          label="Temperature"
          value={weather?.temperature}
          isTemp
          trend={weather ? `Feels like ${formatTemperature(weather.feelsLike, tempUnit)}` : undefined}
          accent="amber"
          loading={weatherLoading}
        />
        <WeatherCard
          icon={CloudRain}
          label="Rainfall"
          value={weather?.rainfall}
          unit="mm"
          trend={weather ? `${weather.rainProbability}% chance` : undefined}
          accent="blue"
          loading={weatherLoading}
        />
        <WeatherCard
          icon={Droplets}
          label="Humidity"
          value={weather?.humidity}
          unit="%"
          trend={weather ? weather.condition : undefined}
          accent="cyan"
          loading={weatherLoading}
        />
        <WeatherCard
          icon={Wind}
          label="Wind Speed"
          value={weather?.windSpeed}
          unit="km/h"
          trend={weather ? `Direction ${weather.windDirection}` : undefined}
          accent="indigo"
          loading={weatherLoading}
        />
        <div className="col-span-2 sm:col-span-1">
          <RiskCard score={overallRisk?.score ?? 0} level={overallRisk?.level ?? "low"} loading={riskLoading} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <GlassCard className="overflow-hidden p-0 xl:col-span-2">
          <div className="flex items-center justify-between border-b border-white/5 p-5 pb-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Interactive Risk Map</h3>
              <p className="text-xs text-slate-500">Grid-level hazard risk across Delhi/NCR · click any zone</p>
            </div>
            <Link
              to="/risk-map"
              className="flex items-center gap-1.5 rounded-lg bg-navy-800/60 px-3 py-1.5 text-xs font-medium text-slate-300 ring-1 ring-white/5 transition-colors hover:bg-navy-800 hover:text-cyan-300"
            >
              <Maximize2 className="h-3.5 w-3.5" /> Full View
            </Link>
          </div>
          <div className="h-[540px] p-4">
            <RiskMap onCellSelect={handleCellSelect} />
          </div>
        </GlassCard>

        <GridDetailPanel cell={selectedCell} />
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <CurrentWeatherPanel weather={weather} loading={weatherLoading} />
        </div>

        <GlassCard className="p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200">Active Alerts</h3>
            <Link to="/alerts" className="text-xs font-medium text-cyan-400 hover:text-cyan-300">
              View all
            </Link>
          </div>
          {alertsLoading ? (
            <LoadingSpinner label="Loading alerts" />
          ) : alerts.length === 0 ? (
            <p className="py-8 text-center text-sm text-slate-500">No active alerts right now.</p>
          ) : (
            <div className="space-y-3">
              {alerts.slice(0, 3).map((alert) => (
                <AlertCard key={alert.id} alert={alert} compact />
              ))}
            </div>
          )}
        </GlassCard>
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <PredictionPanel forecast={forecast ?? []} loading={forecastLoading} />
        </div>
        <RiskSummary summary={riskSummary ?? []} loading={summaryLoading} />
      </div>
    </div>
  );
}
