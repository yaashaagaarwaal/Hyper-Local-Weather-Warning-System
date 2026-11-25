// Flat SVG map of India: the lightweight stand-in for the 3D India scene on
// phones, with WebGL off, or with reduced motion. Same real overlays (NASA
// IMERG rainfall, station readings, active warnings) over the same NASA
// imagery and the same boundaries; just no camera move and no terrain.
import { useEffect, useMemo, useState } from "react";
import { INDIA_WINDOW } from "./scene/geoMath";
import { loadStates } from "./indiaData";

const { west, east, south, north } = INDIA_WINDOW;
// Equirectangular, scaled in x by cos(mid-latitude) so shapes are not stretched.
const KX = Math.cos(21.5 * (Math.PI / 180));
const W = (east - west) * KX;
const H = north - south;

const px = (lon) => (lon - west) * KX;
const py = (lat) => north - lat;

function ringPath(flat) {
  let d = "";
  for (let i = 0; i + 1 < flat.length; i += 2) d += `${i ? "L" : "M"}${px(flat[i]).toFixed(3)} ${py(flat[i + 1]).toFixed(3)}`;
  return `${d}Z`;
}

export default function IndiaFlatMap({ rain, stations, warnings, layers, hover, onHover }) {
  const [states, setStates] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const c = new AbortController();
    loadStates(c.signal)
      .then((data) => { setStates(data); setFailed(false); })
      .catch((err) => { if (err?.name !== "AbortError") setFailed(true); }); // an aborted request is just a re-mount
    return () => c.abort();
  }, []);

  const outsideMask = useMemo(() => {
    if (!states) return null;
    const frame = `M0 0H${W}V${H}H0Z`;
    return frame + states.outline.map(ringPath).join("");
  }, [states]);
  const statePaths = useMemo(() => states?.states.map((s) => ({ id: s.id, name: s.name, d: s.rings.map(ringPath).join("") })) ?? [], [states]);
  const rainHref = useMemo(() => (rain?.status === "ready" ? rain.canvas.toDataURL("image/png") : null), [rain]);

  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Map of India with state boundaries, rainfall, station readings and warnings" className="block h-auto w-full" style={{ background: "#08101c" }}>
        <image href="/geo/india-color.jpg" x="0" y="0" width={W} height={H} preserveAspectRatio="none" />
        {outsideMask && <path d={outsideMask} fillRule="evenodd" fill="#05080f" fillOpacity="0.62" />}
        {rainHref && layers.rain && <image href={rainHref} x="0" y="0" width={W} height={H} preserveAspectRatio="none" opacity="0.95" />}
        {layers.borders && (
          <g fill="none" strokeLinejoin="round">
            {statePaths.map((s) => (
              <path
                key={s.id} d={s.d} stroke="#e8e3d6" strokeOpacity={hover?.id === s.id ? 0.95 : 0.4} strokeWidth={hover?.id === s.id ? 0.16 : 0.07}
                fill={hover?.id === s.id ? "rgba(232,227,214,0.12)" : "transparent"} style={{ pointerEvents: "all" }}
                onMouseEnter={() => onHover?.({ id: s.id, name: s.name })} onMouseLeave={() => onHover?.(null)}
              />
            ))}
            {states && <path d={states.outline.map(ringPath).join("")} stroke="#f4efe4" strokeOpacity="0.9" strokeWidth="0.12" />}
          </g>
        )}
        {layers.stations && stations.items.map((m) => (
          <g key={m.id} transform={`translate(${px(m.lon)} ${py(m.lat)})`}>
            <circle r="0.42" fill="none" stroke="#f4efe4" strokeWidth="0.09" />
            <circle r="0.17" fill="#f4efe4" />
            <text x="0.7" y="-0.5" fontSize="0.78" fill="#f4efe4" style={{ paintOrder: "stroke", stroke: "#05080f", strokeWidth: 0.2 }}>
              {m.name}{m.value ? ` ${m.value}` : ""}
            </text>
          </g>
        ))}
        {layers.warnings && warnings.items.map((m) => (
          <g key={m.id} transform={`translate(${px(m.lon)} ${py(m.lat)})`}>
            <circle r="0.55" fill="none" stroke="#e0a03a" strokeWidth="0.12" />
            <circle r="0.2" fill="#e0a03a" />
          </g>
        ))}
      </svg>
      {failed && <figcaption className="mt-2 text-xs text-[var(--hm-ink-dim)]">State boundaries could not be loaded.</figcaption>}
    </figure>
  );
}
