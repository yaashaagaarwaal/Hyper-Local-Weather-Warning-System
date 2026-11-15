import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, Polygon, Popup, Marker, GeoJSON, useMap } from "react-leaflet";
import { useNavigate } from "react-router-dom";
import L from "leaflet";
import { RISK_LEVELS, findNearestLocationLabel } from "../../data/riskData";
import { fetchRiskGrid, fetchNcrFloodDistricts } from "../../services/riskService";
import { useAsync } from "../../hooks/useAsync";
import { formatRelativeTime } from "../../utils/formatters";
import { useLocationContext } from "../../context/LocationContext";
import { useGridFocus } from "../../context/GridFocusContext";
import { useSettings } from "../../context/SettingsContext";
import Temp from "../common/Temp";
import LoadingSpinner from "../common/LoadingSpinner";
import { Thermometer, CloudRain, Droplets, Wind, Clock, TriangleAlert, MapPin, TrendingUp, FlaskConical, Radio, Layers } from "lucide-react";

const NCR_CENTER = [28.58, 77.2];

// DFSI (District Flood Severity Index, 0-20ish observed nationally) ->
// a 3-band amber/orange/red scale for the district-vulnerability overlay —
// deliberately a different palette from RISK_LEVELS' green/yellow/orange/red
// so this static historical layer is never visually confused with the
// live, dynamic risk-grid coloring above it.
function dfsiColor(dfsi) {
  if (dfsi == null) return "#64748b"; // slate — no data for this district
  if (dfsi < 10) return "#d97706"; // amber
  if (dfsi < 15) return "#ea580c"; // orange
  return "#dc2626"; // red
}

// react-leaflet's <GeoJSON> doesn't support nested JSX <Popup> children per
// feature the way <Polygon> does above — Leaflet's own imperative
// `layer.bindPopup()` (via onEachFeature) is the standard pattern here, so
// this builds a plain HTML string rather than a React component.
function districtPopupHtml(properties) {
  const name = properties.lgd_district_name ?? properties.geoBoundaries_shapeName;
  const dfsi = properties.DFSI;
  const pctFlooded = properties.Percent_Flooded_Area;
  return `
    <div style="min-width:220px;padding:4px 2px;font-family:inherit;">
      <div style="font-weight:700;color:#fff;font-size:13px;margin-bottom:6px;">${name}</div>
      <span style="display:inline-block;border-radius:9999px;background:rgba(245,158,11,0.12);padding:2px 8px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;color:#f59e0b;">
        Historical — India Flood Inventory
      </span>
      <div style="margin-top:8px;font-size:12px;color:#cbd5e1;">
        <div style="display:flex;justify-content:space-between;"><span style="color:#64748b;">Flood Severity Index</span><span style="font-weight:600;">${dfsi != null ? `${dfsi.toFixed(1)}/20` : "Not available"}</span></div>
        <div style="display:flex;justify-content:space-between;margin-top:2px;"><span style="color:#64748b;">Historically flooded area</span><span style="font-weight:600;">${pctFlooded != null ? `${pctFlooded.toFixed(2)}%` : "Not available"}</span></div>
      </div>
      <p style="margin-top:8px;font-size:10px;line-height:1.3;color:#64748b;">
        Static, non-meteorological data — not a live reading or forecast. Source: India Flood Inventory-Impacts v4 (CC-BY-NC-4.0).
      </p>
    </div>
  `;
}

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
          <Thermometer className="h-3.5 w-3.5 text-orange-400" />
          {cell.temperature != null ? <Temp value={cell.temperature} /> : "—"}
        </div>
        <div className="flex items-center gap-1.5 text-slate-300">
          <CloudRain className="h-3.5 w-3.5 text-blue-400" /> {cell.rainfall != null ? `${cell.rainfall} mm` : "—"}
        </div>
        <div className="flex items-center gap-1.5 text-slate-300">
          <Droplets className="h-3.5 w-3.5 text-cyan-400" /> {cell.humidity != null ? `${cell.humidity}%` : "—"}
        </div>
        <div className="flex items-center gap-1.5 text-slate-300">
          <Wind className="h-3.5 w-3.5 text-slate-400" /> {cell.wind != null ? `${cell.wind} km/h` : "—"}
        </div>
      </div>

      <div className="mt-2.5 flex items-center justify-between border-t border-white/10 pt-2.5">
        <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
          <Clock className="h-3 w-3" />
          {cell.assessedAt ? `Assessed ${formatRelativeTime(cell.assessedAt)}` : `ETA ${cell.predictionTime ?? "—"}`}
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
  const [showDistricts, setShowDistricts] = useState(false);
  const { selectedLocation } = useLocationContext();
  const { accent } = useSettings();
  const locationIcon = useMemo(() => buildLocationIcon(accent.color), [accent.color]);
  const { data: cells } = useAsync(() => fetchRiskGrid(), []);
  const { data: districtGeoJson, isLoading: districtsLoading } = useAsync(() => fetchNcrFloodDistricts(), []);
  const isLive = cells?.[0]?.source === "risk-engine-v1";
  // fetchNcrFloodDistricts() catches its own errors and resolves to `null`
  // rather than throwing (no fabricated fallback for this backend-only
  // layer) — so useAsync's own `error` is never set for a real failure.
  // `isLoading` is what actually distinguishes "still fetching" from
  // "fetch finished and genuinely has nothing" (backend down/unreachable).
  const districtsUnavailable = !districtsLoading && !districtGeoJson;

  const districtStyle = (feature) => ({
    color: dfsiColor(feature.properties.DFSI),
    weight: 1.5,
    fillColor: dfsiColor(feature.properties.DFSI),
    fillOpacity: 0.12,
    dashArray: "4",
  });

  if (!cells) {
    return (
      <div style={{ height }} className="flex items-center justify-center overflow-hidden rounded-2xl">
        <LoadingSpinner label="Loading risk grid" />
      </div>
    );
  }

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

        {cells.map((cell) => {
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

        {showDistricts && districtGeoJson && (
          <GeoJSON
            key="ncr-flood-districts"
            data={districtGeoJson}
            style={districtStyle}
            onEachFeature={(feature, layer) => layer.bindPopup(districtPopupHtml(feature.properties))}
          />
        )}

        <Marker position={[selectedLocation.lat, selectedLocation.lng]} icon={locationIcon} />
      </MapContainer>

      <button
        onClick={() => setShowDistricts((v) => !v)}
        disabled={!districtGeoJson}
        className={`absolute right-4 top-4 z-[1000] flex items-center gap-1.5 rounded-lg px-3 py-2 text-[11px] font-semibold backdrop-blur ring-1 transition-colors ${
          showDistricts
            ? "bg-amber-500/20 text-amber-300 ring-amber-500/40"
            : "bg-navy-900/90 text-slate-300 ring-white/10 hover:bg-navy-800"
        } disabled:cursor-not-allowed disabled:opacity-50`}
      >
        <Layers className="h-3.5 w-3.5" />
        {districtGeoJson
          ? "Flood Vulnerability Districts"
          : districtsUnavailable
            ? "Districts unavailable"
            : "Loading districts…"}
      </button>

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
        {isLive ? (
          <div className="flex items-center gap-1.5 rounded-lg bg-emerald-500/10 px-2.5 py-1.5 text-[11px] font-semibold text-emerald-400 ring-1 ring-emerald-500/25 backdrop-blur">
            <Radio className="h-3 w-3" />
            Risk Zones: Live Rule-Based Engine
          </div>
        ) : (
          <div className="flex items-center gap-1.5 rounded-lg bg-amber-500/10 px-2.5 py-1.5 text-[11px] font-semibold text-amber-400 ring-1 ring-amber-500/25 backdrop-blur">
            <FlaskConical className="h-3 w-3" />
            Risk Zones: Simulated
          </div>
        )}
        {showDistricts && (
          <div className="flex items-center gap-1.5 rounded-lg bg-navy-900/90 px-2.5 py-1.5 text-[11px] font-medium text-slate-300 ring-1 ring-white/10 backdrop-blur">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: dfsiColor(16) }} />
            High historical flood severity (IFI, static)
          </div>
        )}
      </div>
    </div>
  );
}
