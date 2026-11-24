// Small Leaflet map for a model result: where the scene is, the model's
// output laid over it when the scene's grid allows that, and point markers.
// Same dark basemap as the Risk Map (OSM tiles + CSS filter).
import { useEffect } from "react";
import { MapContainer, TileLayer, Rectangle, ImageOverlay, CircleMarker, Tooltip, useMap } from "react-leaflet";

function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(bounds, { padding: [24, 24], maxZoom: 13 });
  }, [map, bounds]);
  return null;
}

export default function HazardMap({ bounds, overlayUrl, points = [], truthPoints = [], footprintColor = "#a78bfa", height = 320 }) {
  if (!bounds) {
    return (
      <div className="flex items-center justify-center rounded-lg bg-navy-800/50 p-4 text-center text-[11px] text-slate-500 ring-1 ring-white/5" style={{ height }}>
        This image carries no geographic reference, so it cannot be placed on a map.
      </div>
    );
  }
  const box = [[bounds.south, bounds.west], [bounds.north, bounds.east]];
  return (
    <div className="overflow-hidden rounded-lg ring-1 ring-white/10" style={{ height }}>
      <MapContainer bounds={box} scrollWheelZoom={false} style={{ height: "100%", width: "100%", background: "#0b1220" }}>
        <TileLayer
          className="map-tiles-dark"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FitBounds bounds={box} />
        <Rectangle bounds={box} pathOptions={{ color: footprintColor, weight: 1.5, fill: false, dashArray: "4 3" }} />
        {overlayUrl && <ImageOverlay url={overlayUrl} bounds={box} opacity={0.85} />}
        {truthPoints.map((p, i) => (
          <CircleMarker key={`t${i}`} center={[p.latitude, p.longitude]} radius={2} pathOptions={{ color: "#fde047", weight: 0, fillOpacity: 0.8 }} />
        ))}
        {points.map((p, i) => (
          <CircleMarker
            key={`p${i}`}
            center={[p.latitude, p.longitude]}
            radius={3 + 4 * (p.probability ?? 0.5)}
            pathOptions={{ color: "#c084fc", weight: 1, fillColor: "#a855f7", fillOpacity: 0.25 + 0.5 * (p.probability ?? 0.5) }}
          >
            <Tooltip>
              {p.latitude.toFixed(2)}°N, {p.longitude.toFixed(2)}°E · model probability {(100 * (p.probability ?? 0)).toFixed(0)}%
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}
