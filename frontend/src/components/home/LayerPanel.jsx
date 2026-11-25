// The "Weather over India" panel: layer switches, each with the live status of
// the data behind it, a rainfall legend and the hovered state. Every line here
// is read from the data hook -- if a source is down the line says so.
import { RAIN_RAMP_STOPS, rainRateAt } from "./rainfall";

function formatDay(iso) {
  if (!iso) return "";
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

function Row({ id, label, status, pressed, onToggle, children }) {
  return (
    <div>
      <button type="button" className="hm-toggle" aria-pressed={pressed} onClick={() => onToggle(id)}>
        <span className="hm-box" />
        <span className="min-w-0 flex-1">{label}</span>
      </button>
      <p className="pb-2.5 pl-[22px] text-[11.5px] leading-snug text-[var(--hm-ink-dim)]">{status}</p>
      {children}
    </div>
  );
}

function RainLegend() {
  const stops = RAIN_RAMP_STOPS.map((s) => `${s.color} ${s.u * 100}%`).join(", ");
  const ticks = [0, 0.37, 0.62, 1];
  return (
    <div className="mb-3 ml-[22px]" aria-label="Rainfall rate legend">
      <div className="h-1.5 w-full max-w-[220px]" style={{ background: `linear-gradient(90deg, ${stops})` }} />
      <div className="relative mt-1 h-3 max-w-[220px] font-[var(--hm-mono)] text-[10px] text-[var(--hm-ink-faint)]">
        {ticks.map((t) => (
          <span key={t} className="absolute" style={{ left: `${t * 100}%`, transform: t === 1 ? "translateX(-100%)" : t === 0 ? "none" : "translateX(-50%)" }}>
            {t === 1 ? "53+" : rainRateAt(t) < 1 ? rainRateAt(t).toFixed(1) : Math.round(rainRateAt(t))}
          </span>
        ))}
      </div>
      <p className="mt-0.5 text-[10px] text-[var(--hm-ink-faint)]">mm per hour</p>
    </div>
  );
}

export default function LayerPanel({ data, layers, onToggle, hover, className = "", compact = false }) {
  const { stations, warnings, rain } = data;

  const rainStatus =
    rain.status === "loading" ? "Contacting NASA…"
    : rain.status === "ready"
      ? `NASA IMERG · daily mean · ${formatDay(rain.day)}${
          rain.wetAreaPctIndia != null ? ` · rain over ${rain.wetAreaPctIndia < 1 ? "<1" : Math.round(rain.wetAreaPctIndia)}% of India, peak ${rain.peakRateIndia ? rain.peakRateIndia.toFixed(rain.peakRateIndia < 10 ? 1 : 0) : "0"} mm/h` : ""}`
      : `Unavailable — ${rain.reason ?? "no data"}`;

  const stationLine =
    stations.status === "loading" ? "Reading the weather service…"
    : stations.items.some((s) => s.value) ? stations.items.map((s) => `${s.name} ${s.value ?? "—"}`).join(" · ") + " · AtmoGuard weather service"
    : "Unavailable — the weather service did not answer.";

  const warningLine =
    warnings.status === "loading" ? "Reading the alert engine…"
    : warnings.status === "unavailable" ? "Unavailable — the alert service did not answer."
    : warnings.count === 0 ? "No active warnings from the AtmoGuard engine."
    : `${warnings.count} active warning${warnings.count > 1 ? "s" : ""} from the AtmoGuard engine${warnings.items.length < warnings.count ? " (some outside the mapped grid)" : ""}.`;

  return (
    <div className={className}>
      <p className="hm-eyebrow mb-3">Layers</p>
      <Row id="rain" label="Rainfall" status={rainStatus} pressed={layers.rain} onToggle={onToggle}>
        {rain.status === "ready" && layers.rain && <RainLegend />}
      </Row>
      <Row id="stations" label="Station readings" status={stationLine} pressed={layers.stations} onToggle={onToggle} />
      <Row id="warnings" label="Warnings" status={warningLine} pressed={layers.warnings} onToggle={onToggle} />
      <Row id="borders" label="State boundaries" status="36 states and union territories · geoBoundaries (ODbL)" pressed={layers.borders} onToggle={onToggle} />
      {!compact && <Row id="regions" label="Geographic regions" status="Named physical features only." pressed={layers.regions} onToggle={onToggle} />}
      <p className="mt-1 min-h-[2.4em] border-t border-[var(--hm-rule)] pt-3 text-[12.5px] text-[var(--hm-ink)]" aria-live="polite">
        {hover?.name ? <><span className="text-[var(--hm-ink-dim)]">State · </span>{hover.name}</> : <span className="text-[var(--hm-ink-faint)]">Hover a state to read its name.</span>}
      </p>
    </div>
  );
}
