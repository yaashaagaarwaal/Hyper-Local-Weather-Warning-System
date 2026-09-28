import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, Polygon, Popup, Marker, useMap } from "react-leaflet";
import { useNavigate } from "react-router-dom";
import L from "leaflet";
import { RISK_GRID, RISK_LEVELS, findNearestLocationLabel } from "../../data/riskData";
import { useLocationContext } from "../../context/LocationContext";
import { useGridFocus } from "../../context/GridFocusContext";
import { useSettings } from "../../context/SettingsContext";
import Temp from "../common/Temp";
import { Thermometer, CloudRain, Droplets, Wind, Clock, TriangleAlert, MapPin, TrendingUp } from "lucide-react";

const NCR_CENTER = [28.58, 77.2];

function buildLocationIcon(color) {
  return L.divIcon({
    className: "",
    html: `<div style="width:14px;height:14px;border-radius:9999px;background:${color};border:2px solid #0a1120;box-shadow:0 0 0 4px ${color}40"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
}

function FlyToCell({ cell }) {
  const map = useMap();
  useEffect(() => {
    if (cell) map.flyTo(cell.center, 12, { duration: 0.8 });
  }, [cell, map]);
  return null;
}

function GridPopupContent({ cell }) {
  const navigate = useNavigate();
  const { setFocusedGridId } = useGridFocus();
  const risk = RISK_LEVELS[cell.riskLevel];
  const locationLabel = findNearestLocationLabel(cell.center[0], cell.center[1]);

  return (
    <div className="min-w-[240px] p-3.5">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-sm font-bold text-white">{cell.id}</span>
        <span
          className="rounded-full px-2 py-0.5 text-[11px] font-semibold"
          style={{ color: risk.color, backgroundColor: `${risk.color}22` }}
        >
          {risk.label}
        </span>
      </div>

      <div className="mb-2.5 flex items-center gap-1.5 text-[11px] text-slate-400">
        <MapPin className="h-3 w-3" /> {locationLabel}
      </div>

      <div className="mb-2.5 flex items-center gap-1.5 text-xs text-amber-300">
        <TriangleAlert className="h-3.5 w-3.5" />
        <span className="font-medium">{cell.hazardType}</span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="flex items-center gap-1.5 text-slate-300">
          <Thermometer className="h-3.5 w-3.5 text-orange-400" /> <Temp value={cell.temperature} />
        </div>
        <div className="flex items-center gap-1.5 text-slate-300">
          <CloudRain className="h-3.5 w-3.5 text-blue-400" /> {cell.rainfall} mm
        </div>
        <div className="flex items-center gap-1.5 text-slate-300">
          <Droplets className="h-3.5 w-3.5 text-cyan-400" /> {cell.humidity}%
        </div>
        <div className="flex items-center gap-1.5 text-slate-300">
          <Wind className="h-3.5 w-3.5 text-slate-400" /> {cell.wind} km/h
        </div>
      </div>

      <div className="mt-2.5 flex items-center justify-between border-t border-white/10 pt-2.5">
        <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
          <Clock className="h-3 w-3" /> ETA {cell.predictionTime}
        </div>
        <div className="text-[11px] font-semibold text-slate-300">Score: {cell.riskScore}</div>
      </div>

      <button
        onClick={() => {
          setFocusedGridId(cell.id);
          navigate("/predictions");
        }}
        className="mt-2.5 flex w-full items-center justify-center gap-1.5 rounded-lg bg-cyan-500/15 py-2 text-xs font-semibold text-cyan-300 transition-colors hover:bg-cyan-500/25"
      >
        <TrendingUp className="h-3.5 w-3.5" />
        View Detailed Prediction
      </button>
    </div>
  );
}

export default function RiskMap({ onCellSelect, height = "100%", focusCell = null }) {
  const [activeCellId, setActiveCellId] = useState(focusCell?.id ?? null);
  const { selectedLocation } = useLocationContext();
  const { accent } = useSettings();
  const locationIcon = useMemo(() => buildLocationIcon(accent.color), [accent.color]);

  return (
    <div style={{ height }} className="relative w-full overflow-hidden rounded-2xl">
      <MapContainer
        center={focusCell?.center ?? NCR_CENTER}
        zoom={focusCell ? 12 : 10}
        scrollWheelZoom={true}
        className="h-full w-full"
        attributionControl={true}
      >
        <TileLayer
          className="map-tiles-dark"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {focusCell && <FlyToCell cell={focusCell} />}

        {RISK_GRID.map((cell) => {
          const risk = RISK_LEVELS[cell.riskLevel];
          const isActive = cell.id === activeCellId;
          return (
            <Polygon
              key={cell.id}
              positions={cell.bounds}
              pathOptions={{
                color: risk.color,
                weight: isActive ? 2.5 : 1,
                fillColor: risk.color,
                fillOpacity: isActive ? 0.55 : 0.28,
                opacity: isActive ? 1 : 0.6,
              }}
              eventHandlers={{
                click: () => {
                  setActiveCellId(cell.id);
                  onCellSelect?.(cell);
                },
                mouseover: (e) => e.target.setStyle({ fillOpacity: 0.5 }),
                mouseout: (e) =>
                  e.target.setStyle({ fillOpacity: isActive ? 0.55 : 0.28 }),
              }}
            >
              <Popup>
                <GridPopupContent cell={cell} />
              </Popup>
            </Polygon>
          );
        })}

        <Marker position={[selectedLocation.lat, selectedLocation.lng]} icon={locationIcon} />
      </MapContainer>

      <div className="pointer-events-none absolute bottom-4 left-4 z-[1000] flex flex-wrap gap-2">
        {Object.values(RISK_LEVELS).map((r) => (
          <div
            key={r.key}
            className="flex items-center gap-1.5 rounded-lg bg-navy-900/90 px-2.5 py-1.5 text-[11px] font-medium text-slate-300 ring-1 ring-white/10 backdrop-blur"
          >
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: r.color }} />
            {r.label}
          </div>
        ))}
      </div>
    </div>
  );
}
