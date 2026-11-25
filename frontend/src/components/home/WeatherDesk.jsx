// "Weather intelligence" section: a live read-out for the monitored locations.
// Typeset as a ledger, not a dashboard of cards. Every figure is fetched from
// the app's services; each block shows a plain unavailable state when its source
// does not answer.
import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { useSettings } from "../../context/SettingsContext";
import { formatTemperature } from "../../utils/formatters";
import { DESK_LOCATIONS, useWeatherDesk } from "./useWeatherDesk";
import { RainBars } from "./charts";
import { Reveal } from "./Reveal";

const LEVEL_COLOUR = { low: "var(--hm-low)", moderate: "var(--hm-moderate)", high: "var(--hm-high)", severe: "var(--hm-severe)" };

function pct(p) {
  if (p === null || p === undefined) return "—";
  if (p < 0.01) return "<1%";
  return `${Math.round(p * 100)}%`;
}

function Unavailable({ children }) {
  return <p className="border-t border-[var(--hm-rule)] py-4 text-[13px] leading-relaxed text-[var(--hm-ink-dim)]">{children}</p>;
}

function Skeleton({ className = "" }) {
  return <div className={`hm-skeleton ${className}`} aria-hidden="true" />;
}

function formatUtc(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " UTC";
}

function Current({ state, unit }) {
  const { data: w, isLoading, error } = state;
  if (isLoading && !w) return <div className="space-y-4"><Skeleton className="h-28 w-56" /><Skeleton className="h-4 w-40" /><Skeleton className="h-24 w-full" /></div>;
  if (error || !w) return <Unavailable>Current conditions are unavailable: the weather service did not answer.</Unavailable>;
  const rows = [
    ["Feels like", formatTemperature(w.feelsLike, unit, 0)],
    ["Humidity", `${Math.round(w.humidity)}%`],
    ["Wind", `${Math.round(w.windSpeed)} km/h${w.windDirection ? ` ${w.windDirection}` : ""}`],
    ["Rain now", `${(w.rainfall ?? 0).toFixed(1)} mm`],
    ["Pressure", w.pressure ? `${Math.round(w.pressure)} hPa` : "—"],
  ];
  return (
    <div>
      <p className="hm-num text-[clamp(88px,11vw,150px)]">{formatTemperature(w.temperature, unit, 0)}</p>
      <p className="mt-3 text-[17px] text-[var(--hm-ink)]">{w.condition}</p>
      <dl className="mt-7">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between border-t border-[var(--hm-rule)] py-2.5 text-[13px]">
            <dt className="text-[var(--hm-ink-dim)]">{k}</dt>
            <dd className="hm-mono">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-[11.5px] text-[var(--hm-ink-faint)]">
        Observed {formatUtc(w.lastUpdated)} · {w.source === "Open-Meteo" ? "Open-Meteo (direct)" : "Open-Meteo via the AtmoGuard weather service"}
      </p>
    </div>
  );
}

function ModelTable({ model, location }) {
  const { data, isLoading } = model;
  if (isLoading && !data) return <div className="space-y-3"><Skeleton className="h-6 w-48" /><Skeleton className="h-40 w-full" /></div>;
  if (!data || data.status === "unavailable") return <Unavailable>Risk and model figures are unavailable: the AtmoGuard backend did not answer.</Unavailable>;
  if (data.status === "outside") {
    return (
      <Unavailable>
        {location.name} lies {Math.round(data.distanceKm)} km from the nearest cell of the Delhi–NCR risk grid, so AtmoGuard computes no model or
        rule-based risk for it — the readings and forecast on this page are the whole picture here. Choose a Delhi–NCR location to see the risk ledger.
      </Unavailable>
    );
  }
  const { ml, hazards, cell } = data;
  const color = LEVEL_COLOUR[cell.riskLevel] ?? "var(--hm-ink)";
  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="hm-eyebrow">Risk · grid cell {cell.id}</p>
        <p className="text-[13px]">
          <span className="text-[var(--hm-ink-dim)]">Rule-based overall </span>
          <span className="hm-mono" style={{ color }}>{cell.riskLevel.toUpperCase()} · {Math.round(cell.riskScore)}/100</span>
        </p>
      </div>

      {ml ? (
        <table className="hm-ledger mt-5">
          <caption className="sr-only">Model forecasts for the next 1, 3 and 6 hours</caption>
          <thead><tr><th scope="col">Trained model</th>{ml.heavyRainRisk.map((p) => <th key={p.horizonHours} scope="col">+{p.horizonHours} h</th>)}</tr></thead>
          <tbody>
            <tr>
              <td>Heavy-rain risk{ml.heavyRainRisk.some((p) => p.lowConfidence) && <span className="ml-2 text-[11px] text-[var(--hm-amber)]">low confidence</span>}</td>
              {ml.heavyRainRisk.map((p) => <td key={p.horizonHours} style={p.risk ? { color: "var(--hm-amber)" } : undefined}>{pct(p.probability)}{p.risk ? " ▲" : ""}</td>)}
            </tr>
            <tr>
              <td>Thunderstorm risk</td>
              {ml.thunderstormRisk.map((p) => <td key={p.horizonHours} style={p.risk ? { color: "var(--hm-amber)" } : undefined}>{pct(p.probability)}{p.risk ? " ▲" : ""}</td>)}
            </tr>
            <tr>
              <td>Rainfall, mm/h</td>
              {ml.rainfall.map((p) => <td key={p.horizonHours}>{p.value.toFixed(1)}</td>)}
            </tr>
            <tr>
              <td>Temperature</td>
              {ml.temperature.map((p) => <td key={p.horizonHours}>{Math.round(p.value)}°C</td>)}
            </tr>
          </tbody>
        </table>
      ) : (
        <Unavailable>The trained-model forecast is unavailable for this cell right now.</Unavailable>
      )}

      {hazards ? (
        <table className="hm-ledger mt-8">
          <caption className="sr-only">Rule-based hazard scores out of 100</caption>
          <thead><tr><th scope="col">Rule-based engine</th><th scope="col">Score / 100</th></tr></thead>
          <tbody>
            {hazards.map((h) => (
              <tr key={h.key}>
                <td>{h.label}</td>
                <td>{Math.round(h.score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Unavailable>The per-hazard breakdown is unavailable right now.</Unavailable>
      )}
      <p className="mt-4 text-[11.5px] leading-relaxed text-[var(--hm-ink-faint)]">
        ▲ = the model flags the event. Model figures are probabilities from XGBoost / linear models trained on historical Delhi–NCR data; the heavy-rain model
        rarely sees the event it predicts and is marked low-confidence. Lightning is a potential score from convective indices, not a detection.
      </p>
    </div>
  );
}

function Forecast({ daily, unit }) {
  const { data, isLoading } = daily;
  if (isLoading && !data) return <Skeleton className="h-28 w-full" />;
  if (!data?.length) return <Unavailable>The 7-day forecast is unavailable: the weather service did not answer.</Unavailable>;
  return (
    <ol className="grid grid-cols-2 border-t border-[var(--hm-rule-strong)] sm:grid-cols-4 lg:grid-cols-7">
      {data.slice(0, 7).map((d) => (
        <li key={d.day} className="border-b border-[var(--hm-rule)] py-4 pr-3 lg:border-b-0">
          <p className="hm-eyebrow">{d.day}</p>
          <p className="hm-num mt-3 text-[34px]">{formatTemperature(d.high, unit, 0)}</p>
          <p className="hm-mono mt-1 text-[12px] text-[var(--hm-ink-dim)]">{formatTemperature(d.low, unit, 0)} low</p>
          <p className="mt-2 text-[12.5px] leading-snug text-[var(--hm-ink)]">{d.condition}</p>
          <p className="hm-mono mt-1 text-[11.5px] text-[var(--hm-ink-dim)]">{d.rainfall > 0 ? `${d.rainfall.toFixed(1)} mm` : "dry"}</p>
        </li>
      ))}
    </ol>
  );
}

function Warnings({ alerts }) {
  const { data, isLoading } = alerts;
  if (isLoading && !data) return <Skeleton className="h-12 w-full" />;
  if (!data || data.status === "unavailable") return <Unavailable>Active warnings are unavailable: the alert service did not answer.</Unavailable>;
  if (data.items.length === 0) {
    return <Unavailable>No active warnings. AtmoGuard issues a bulletin only when its engine rates a hazard <em>high</em> or <em>severe</em>.</Unavailable>;
  }
  return (
    <ul className="border-t border-[var(--hm-rule-strong)]">
      {data.items.slice(0, 4).map((a) => (
        <li key={a.id} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-[var(--hm-rule)] py-3.5">
          <span className="hm-mono text-[11px] uppercase tracking-wider" style={{ color: a.severity === "severe" ? "var(--hm-severe)" : "var(--hm-high)" }}>{a.severity}</span>
          <span className="min-w-0 flex-1 text-[14px]">{a.title}</span>
          <span className="text-[12px] text-[var(--hm-ink-dim)]">{a.location}</span>
        </li>
      ))}
    </ul>
  );
}

export default function WeatherDesk() {
  const [locationId, setLocationId] = useState(DESK_LOCATIONS[0].id);
  const { tempUnit } = useSettings();
  const desk = useWeatherDesk(locationId);

  return (
    <section id="weather" className="hm-section" aria-labelledby="weather-title">
      <div className="hm-wrap">
        <Reveal className="grid items-end gap-8 lg:grid-cols-[1fr_auto]">
          <div>
            <p className="hm-eyebrow mb-5">Weather intelligence</p>
            <h2 id="weather-title" className="hm-display text-[clamp(38px,5.2vw,72px)]">Live readings. Model output.<br /><em>Nothing staged.</em></h2>
          </div>
          <p className="hm-body max-w-[26em]">
            Every figure below is fetched when you load the page. If a source does not answer, the figure is replaced by a plain note, never by a guess.
          </p>
        </Reveal>

        <div role="tablist" aria-label="Location" className="mt-12 flex flex-wrap gap-x-7 gap-y-2 border-b border-[var(--hm-rule)]">
          {DESK_LOCATIONS.map((l) => (
            <button
              key={l.id} role="tab" aria-selected={l.id === locationId} onClick={() => setLocationId(l.id)}
              className="-mb-px border-b px-0 pb-3 pt-1 text-[14px] transition-colors"
              style={{ borderColor: l.id === locationId ? "var(--hm-ink)" : "transparent", color: l.id === locationId ? "var(--hm-ink)" : "var(--hm-ink-dim)" }}
            >
              {l.name}{l.region && l.region !== l.name ? <span className="ml-1.5 text-[11.5px] text-[var(--hm-ink-faint)]">{l.region}</span> : null}
            </button>
          ))}
        </div>

        <div className="mt-12 grid gap-14 lg:grid-cols-[5fr_7fr] lg:gap-20">
          <div>
            <Current state={desk.weather} unit={tempUnit} />
            <div className="mt-10">
              <p className="hm-eyebrow mb-3">Hourly rainfall</p>
              <RainBars state={desk.hourly} />
            </div>
          </div>
          <ModelTable model={desk.model} location={desk.location} />
        </div>

        <div className="mt-16">
          <p className="hm-eyebrow mb-4">Seven-day outlook · {desk.location.name}</p>
          <Forecast daily={desk.daily} unit={tempUnit} />
        </div>

        <div className="mt-16 grid gap-8 lg:grid-cols-[5fr_7fr] lg:gap-20">
          <div>
            <p className="hm-eyebrow mb-4">Active warnings</p>
            <p className="hm-body mb-6 max-w-[28em]">Bulletins from the alert engine, across every monitored location.</p>
            <Link to="/alerts" className="hm-link inline-flex items-center gap-2 text-[14px]">Open the alert feed <ArrowRight size={15} strokeWidth={1.6} /></Link>
          </div>
          <Warnings alerts={desk.alerts} />
        </div>
      </div>
    </section>
  );
}
