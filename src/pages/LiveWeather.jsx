import { Thermometer, CloudRain, Droplets, Wind, Gauge, Eye } from "lucide-react";
import { useLocationContext } from "../context/LocationContext";
import { useAsync } from "../hooks/useAsync";
import { fetchCurrentWeather, fetchHourlyTrend } from "../services/weatherService";
import CurrentWeatherPanel from "../components/weather/CurrentWeatherPanel";
import Temp from "../components/common/Temp";
import TrendChart from "../components/charts/TrendChart";
import GlassCard from "../components/common/GlassCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import { LOCATIONS } from "../data/locations";
import { CURRENT_WEATHER_BY_LOCATION } from "../data/weatherData";

const QUICK_STATS = [
  { key: "temperature", label: "Temp", icon: Thermometer, unit: "°C" },
  { key: "rainfall", label: "Rain", icon: CloudRain, unit: "mm" },
  { key: "humidity", label: "Humidity", icon: Droplets, unit: "%" },
  { key: "windSpeed", label: "Wind", icon: Wind, unit: "km/h" },
];

export default function LiveWeather() {
  const { selectedLocationId, setSelectedLocationId } = useLocationContext();
  const { data: weather, isLoading: weatherLoading } = useAsync(
    () => fetchCurrentWeather(selectedLocationId),
    [selectedLocationId]
  );
  const { data: hourly, isLoading: hourlyLoading } = useAsync(
    () => fetchHourlyTrend(selectedLocationId),
    [selectedLocationId]
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <CurrentWeatherPanel weather={weather} loading={weatherLoading} />
        </div>

        <GlassCard className="p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Gauge className="h-4 w-4 text-cyan-400" /> Air &amp; Sky
          </div>
          <ul className="space-y-3 text-sm">
            <li className="flex items-center justify-between">
              <span className="text-slate-500">UV Index</span>
              <span className="font-medium text-slate-200">{weather?.uvIndex ?? "--"}</span>
            </li>
            <li className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-slate-500">
                <Eye className="h-3.5 w-3.5" /> Visibility
              </span>
              <span className="font-medium text-slate-200">{weather?.visibility ?? "--"} km</span>
            </li>
            <li className="flex items-center justify-between">
              <span className="text-slate-500">Condition</span>
              <span className="font-medium text-slate-200">{weather?.condition ?? "--"}</span>
            </li>
            <li className="flex items-center justify-between">
              <span className="text-slate-500">Wind Direction</span>
              <span className="font-medium text-slate-200">{weather?.windDirection ?? "--"}</span>
            </li>
          </ul>
        </GlassCard>
      </div>

      <GlassCard className="p-5">
        <h3 className="mb-1 text-sm font-semibold text-slate-200">24-Hour Trend</h3>
        <p className="mb-4 text-xs text-slate-500">Temperature and rainfall observed over the last 24 hours</p>
        {hourlyLoading ? (
          <LoadingSpinner label="Loading trend data" />
        ) : (
          <TrendChart
            data={hourly}
            lines={[
              { dataKey: "temperature", name: "Temperature (°C)", color: "#f97316" },
              { dataKey: "rainfall", name: "Rainfall (mm)", color: "#22d3ee", axis: "right" },
            ]}
          />
        )}
      </GlassCard>

      <div>
        <h3 className="mb-4 text-sm font-semibold text-slate-200">All Monitored Stations</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {LOCATIONS.map((loc) => {
            const w = CURRENT_WEATHER_BY_LOCATION[loc.id];
            const isSelected = loc.id === selectedLocationId;
            return (
              <button
                key={loc.id}
                onClick={() => setSelectedLocationId(loc.id)}
                className={`rounded-2xl p-4 text-left transition-all duration-200 ${
                  isSelected
                    ? "glass-panel-solid ring-1 ring-cyan-500/40"
                    : "glass-panel hover:-translate-y-0.5 hover:ring-1 hover:ring-white/10"
                }`}
              >
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-sm font-semibold text-slate-100">{loc.name}</p>
                  {isSelected && <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />}
                </div>
                <p className="mb-3 text-2xl font-bold text-white">
                  <Temp value={w.temperature} decimals={0} />
                </p>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs text-slate-500">
                  {QUICK_STATS.slice(1).map(({ key, label, unit }) => (
                    <span key={key}>
                      {label}: <span className="font-medium text-slate-300">{w[key]}{unit}</span>
                    </span>
                  ))}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
