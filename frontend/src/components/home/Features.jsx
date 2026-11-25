// Four features, each shown with a visual that is itself real data from the
// system (the actual risk grid, the actual rainfall series, the actual model
// output, the actual alert feed) rather than a decorative card.
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { LOCATIONS } from "../../data/locations";
import { fetchRiskGrid } from "../../services/riskService";
import { useWeatherDesk } from "./useWeatherDesk";
import { memo } from "./memo";
import { LevelScale, RainBars } from "./charts";
import { Reveal } from "./Reveal";

const LEVEL_FILL = { low: "#7fae8f", moderate: "#cdb25a", high: "#d98c4a", severe: "#cf5b4a" };

function Feature({ n, title, children, visual, flip, link }) {
  return (
    <Reveal as="article" className="grid items-center gap-10 border-t border-[var(--hm-rule)] py-16 lg:grid-cols-2 lg:gap-24 lg:py-24">
      <div className={flip ? "lg:order-2" : ""}>
        <p className="hm-mono text-[12px] text-[var(--hm-ink-faint)]">{n}</p>
        <h3 className="hm-display mt-2 text-[clamp(34px,4vw,56px)]">{title}</h3>
        <div className="hm-body mt-6 max-w-[30em] space-y-4">{children}</div>
        {link && <Link to={link.to} className="hm-link mt-8 inline-flex items-center gap-2 text-[14px]">{link.label} <ArrowRight size={15} strokeWidth={1.6} /></Link>}
      </div>
      <div className={flip ? "lg:order-1" : ""}>{visual}</div>
    </Reveal>
  );
}

// 1. The real 7x7 grid over Delhi-NCR with the monitored stations on it.
function GridVisual() {
  const [state, setState] = useState({ status: "loading", cells: [] });
  useEffect(() => {
    let alive = true;
    memo("risk-grid", fetchRiskGrid).then((cells) => alive && setState({ status: "ready", cells })).catch(() => alive && setState({ status: "unavailable", cells: [] }));
    return () => { alive = false; };
  }, []);

  const view = useMemo(() => {
    const cells = state.cells;
    if (!cells.length) return null;
    const lats = cells.flatMap((c) => c.bounds.map((b) => b[0]));
    const lons = cells.flatMap((c) => c.bounds.map((b) => b[1]));
    const south = Math.min(...lats), north = Math.max(...lats), west = Math.min(...lons), east = Math.max(...lons);
    const k = Math.cos(((south + north) / 2) * (Math.PI / 180));
    const W = (east - west) * k, H = north - south;
    const x = (lon) => (lon - west) * k, y = (lat) => north - lat;
    return { W, H, x, y, south, north, west, east };
  }, [state.cells]);

  if (state.status === "loading") return <div className="hm-skeleton aspect-square w-full max-w-[520px]" aria-hidden="true" />;
  if (!view) return <p className="border-t border-[var(--hm-rule)] py-4 text-[13px] text-[var(--hm-ink-dim)]">The risk grid is unavailable: the AtmoGuard backend did not answer.</p>;
  const live = state.cells.every((c) => c.source !== "simulated");
  const stations = LOCATIONS.filter((l) => l.lat >= view.south && l.lat <= view.north && l.lng >= view.west && l.lng <= view.east);
  return (
    <figure className="m-0">
      <svg viewBox={`${-0.4} ${-0.1} ${view.W + 0.8} ${view.H + 0.2}`} className="block w-full max-w-[560px]" role="img" aria-label="The 7 by 7 risk grid over Delhi-NCR with monitored locations">
        {state.cells.map((c) => {
          const lats = c.bounds.map((b) => b[0]), lons = c.bounds.map((b) => b[1]);
          const x0 = view.x(Math.min(...lons)), x1 = view.x(Math.max(...lons)), y0 = view.y(Math.max(...lats)), y1 = view.y(Math.min(...lats));
          const fill = live ? LEVEL_FILL[c.riskLevel] ?? "transparent" : "transparent";
          return <rect key={c.id} x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill={fill} fillOpacity={live ? 0.26 : 0} stroke="var(--hm-ink)" strokeOpacity="0.28" strokeWidth="0.004" />;
        })}
        {stations.map((l) => (
          <g key={l.id} transform={`translate(${view.x(l.lng)} ${view.y(l.lat)})`}>
            <circle r="0.017" fill="var(--hm-ink)" />
            <text x="0.03" y="-0.02" fontSize="0.05" fill="var(--hm-ink)" style={{ paintOrder: "stroke", stroke: "var(--hm-bg)", strokeWidth: 0.012 }}>{l.name}</text>
          </g>
        ))}
      </svg>
      <figcaption className="mt-3 text-[11.5px] leading-relaxed text-[var(--hm-ink-faint)]">
        {state.cells.length} cells over Delhi–NCR, each about 9 km across. {live ? "Tint = each cell's current rule-based risk level." : "Live risk tint unavailable; geometry only."} Dots are monitored locations.
      </figcaption>
    </figure>
  );
}

// 2. Rain: the real hourly series plus the trained model's numbers for it.
function RainVisual({ desk }) {
  const ml = desk.model.data?.ml;
  return (
    <div>
      <p className="hm-eyebrow mb-3">{desk.location.name} · next 24 hours</p>
      <RainBars state={desk.hourly} />
      {ml ? (
        <table className="hm-ledger mt-7">
          <thead><tr><th scope="col">Trained model</th>{ml.rainfall.map((p) => <th key={p.horizonHours} scope="col">+{p.horizonHours} h</th>)}</tr></thead>
          <tbody>
            <tr><td>Rainfall, mm/h</td>{ml.rainfall.map((p) => <td key={p.horizonHours}>{p.value.toFixed(1)}</td>)}</tr>
            <tr><td>Heavy-rain flag</td>{ml.heavyRainRisk.map((p) => <td key={p.horizonHours} style={p.risk ? { color: "var(--hm-amber)" } : undefined}>{p.risk ? "raised" : "none"}</td>)}</tr>
          </tbody>
        </table>
      ) : (
        <p className="mt-6 border-t border-[var(--hm-rule)] py-4 text-[13px] text-[var(--hm-ink-dim)]">Model forecast unavailable right now.</p>
      )}
    </div>
  );
}

// 3. Thunderstorm: rule-based scores on the four-level scale + the classifier's probabilities.
function StormVisual({ desk }) {
  const hazards = desk.model.data?.hazards;
  const ml = desk.model.data?.ml;
  if (!hazards && !ml) return <p className="border-t border-[var(--hm-rule)] py-4 text-[13px] text-[var(--hm-ink-dim)]">Thunderstorm figures are unavailable right now.</p>;
  const rows = (hazards ?? []).filter((h) => h.key === "thunderstorm" || h.key === "lightning");
  return (
    <div>
      <p className="hm-eyebrow mb-6">{desk.location.name} · rule-based score</p>
      <div className="space-y-7">
        {rows.map((h) => (
          <div key={h.key}>
            <div className="mb-2 flex items-baseline justify-between text-[13px]"><span>{h.key === "lightning" ? "Lightning potential" : "Thunderstorm"}</span><span className="hm-mono">{Math.round(h.score)}/100</span></div>
            <LevelScale score={h.score} labelled={h.key === "thunderstorm"} />
          </div>
        ))}
      </div>
      {ml && (
        <table className="hm-ledger mt-8">
          <thead><tr><th scope="col">Trained classifier</th>{ml.thunderstormRisk.map((p) => <th key={p.horizonHours} scope="col">+{p.horizonHours} h</th>)}</tr></thead>
          <tbody><tr><td>Chance of a storm</td>{ml.thunderstormRisk.map((p) => <td key={p.horizonHours}>{p.probability < 0.01 ? "<1%" : `${Math.round(p.probability * 100)}%`}</td>)}</tr></tbody>
        </table>
      )}
    </div>
  );
}

// 4. Warnings: the lifecycle, and the latest real bulletin if there is one.
function WarningVisual({ desk }) {
  const alerts = desk.alerts.data;
  const steps = ["Hazard rated high or severe", "Bulletin issued", "Kept current while it lasts", "Resolved or expired"];
  const latest = alerts?.status === "ready" ? alerts.items[0] : null;
  return (
    <div>
      <ol className="relative ml-2 border-l border-[var(--hm-rule-strong)]">
        {steps.map((s, i) => (
          <li key={s} className="relative pb-7 pl-7 last:pb-0">
            <span className="absolute -left-[5px] top-[7px] h-[9px] w-[9px] rounded-full border border-[var(--hm-ink)]" style={{ background: i === 1 ? "var(--hm-amber)" : "var(--hm-bg)" }} />
            <p className="hm-mono text-[11px] text-[var(--hm-ink-faint)]">{`0${i + 1}`}</p>
            <p className="text-[15px]">{s}</p>
          </li>
        ))}
      </ol>
      <p className="mt-8 border-t border-[var(--hm-rule)] pt-4 text-[13px] leading-relaxed text-[var(--hm-ink-dim)]">
        {desk.alerts.isLoading && !alerts ? "Reading the alert engine…"
          : alerts?.status !== "ready" ? "The alert service did not answer, so no live bulletin is shown."
          : latest ? <>Latest active bulletin: <span className="text-[var(--hm-ink)]">{latest.title}</span> — {latest.location}.</>
          : "No active warnings at the moment."}
      </p>
    </div>
  );
}

export default function Features() {
  const desk = useWeatherDesk("loc-ndls");
  return (
    <section id="features" className="hm-section pb-0" aria-labelledby="features-title">
      <div className="hm-wrap">
        <Reveal>
          <p className="hm-eyebrow mb-5">What it does</p>
          <h2 id="features-title" className="hm-display max-w-[13em] text-[clamp(38px,5.2vw,72px)]">Four things, <em>done carefully.</em></h2>
        </Reveal>
        <div className="mt-14">
          <Feature n="01" title="Hyper-local forecasting" visual={<GridVisual />} link={{ to: "/risk-map", label: "Open the live map" }}>
            <p>Delhi–NCR is divided into a 7 × 7 grid and every cell is assessed on its own, using the nearest real observing location and recording how far away it is.</p>
            <p>City-scale stations are never presented as street-level truth: each assessment carries its distance and the age of the reading behind it.</p>
          </Feature>
          <Feature n="02" title="Heavy-rainfall prediction" flip visual={<RainVisual desk={desk} />} link={{ to: "/predictions", label: "See the model forecasts" }}>
            <p>Rainfall in millimetres per hour, and a heavy-rain flag, one, three and six hours ahead.</p>
            <p>Heavy rain is rare in the training record, so this model is marked low-confidence everywhere it appears. Treat its flag as a prompt to look closer, not a verdict.</p>
          </Feature>
          <Feature n="03" title="Thunderstorm-risk detection" visual={<StormVisual desk={desk} />} link={{ to: "/risk-map", label: "Inspect hazards on the map" }}>
            <p>A rule-based score built from convective indices (CAPE, lifted index) sits beside a trained classifier&apos;s chance of a storm in the next one to six hours.</p>
            <p>Lightning is shown as a potential, never a detection: no lightning-sensor feed is connected to the warning system.</p>
          </Feature>
          <Feature n="04" title="Early weather warnings" flip visual={<WarningVisual desk={desk} />} link={{ to: "/alerts", label: "Open the alert feed" }}>
            <p>Only hazards rated high or severe become bulletins. A database rule allows one active bulletin per hazard and grid cell, so repeated checks update it rather than repeating it.</p>
            <p>Every bulletin states whether it came from the rule-based engine, the model, or both.</p>
          </Feature>
        </div>
      </div>
    </section>
  );
}
