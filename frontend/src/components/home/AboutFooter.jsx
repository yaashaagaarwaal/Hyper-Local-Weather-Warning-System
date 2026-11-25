// About (#about) and the footer with contact details (#contact).
import { Link } from "react-router-dom";
import { Reveal } from "./Reveal";
import { scrollToSection } from "./HomeNav";
import { SITE } from "../../config/site";

const OBJECTIVES = [
  ["Local, not citywide", "Warnings that reflect the conditions over a neighbourhood-sized cell, with the limits of the data stated beside every number."],
  ["Early, not instant", "Forecasts at one, three and six hours, so there is time to act before the weather arrives."],
  ["Honest about uncertainty", "Models are labelled as models, rules as rules, simulations as simulations, and weak results as weak."],
];

const YEAR = new Date().getFullYear();

const CREDITS = [
  ["Imagery & rainfall", "NASA Blue Marble, VIIRS Black Marble and IMERG precipitation, via NASA GIBS (public domain)."],
  ["Terrain", "AWS Open Data Terrain Tiles (SRTM and other open elevation sources)."],
  ["State boundaries", "geoBoundaries gbOpen, from the Local Government Directory (ODbL)."],
  ["Weather data", "Open-Meteo (CC BY 4.0)."],
  ["Type", "Instrument Serif (SIL Open Font License)."],
];

export function About() {
  return (
    <section id="about" className="hm-section" aria-labelledby="about-title">
      <div className="hm-wrap">
        <div className="hm-cols hm-cols--12">
          <Reveal>
            <p className="hm-eyebrow mb-5">About</p>
            <h2 id="about-title" className="hm-display text-[clamp(38px,5vw,68px)]">Weather warnings that know <em>where</em> you are.</h2>
          </Reveal>
          <Reveal className="space-y-6">
            <p className="hm-lede">
              AtmoGuard turns live weather readings, trained forecasting models and a transparent rule-based risk engine into early warnings for small areas,
              starting with a 49-cell grid over Delhi–NCR. Satellite imagery work extends the same approach across India.
            </p>
            <p className="hm-body">
              It began as a Smart India Hackathon entry (Problem Statement 26077, hyper-local weather intelligence and early warning) and is a working prototype:
              a FastAPI backend, a PostgreSQL store, a React front end and a set of documented models. It is <strong className="font-medium text-[var(--hm-ink)]">not an official warning service</strong>;
              for official warnings, follow the India Meteorological Department.
            </p>
          </Reveal>
        </div>

        <Reveal className="mt-20 border-t border-[var(--hm-rule-strong)]">
          <p className="hm-eyebrow mb-2 mt-6">Purpose and objectives</p>
          <div className="grid md:grid-cols-3">
            {OBJECTIVES.map(([t, b], i) => (
              <div key={t} className="border-b border-[var(--hm-rule)] py-8 md:border-b-0 md:pr-10" style={{ borderLeft: i ? "0" : undefined }}>
                <p className="hm-mono text-[11px] text-[var(--hm-ink-faint)]">{`0${i + 1}`}</p>
                <h3 className="hm-display mt-2 text-[26px] leading-[1.1]">{t}</h3>
                <p className="hm-body mt-3">{b}</p>
              </div>
            ))}
          </div>
        </Reveal>

        <Reveal className="mt-14 grid gap-8 md:grid-cols-[14rem_1fr] md:gap-10">
          <p className="hm-eyebrow">Technology overview</p>
          <p className="hm-body max-w-[46em]">
            React and Vite on the front end, with Three.js for the globe and Leaflet for the live map. FastAPI and PostgreSQL behind it; XGBoost and linear models for forecasting;
            a rule-based risk engine with documented thresholds; an alert pipeline with database-enforced de-duplication; and PyTorch models for satellite-image work, kept clearly apart from the warning logic.
          </p>
        </Reveal>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer id="contact" className="border-t border-[var(--hm-rule-strong)]" style={{ background: "var(--hm-bg-2)" }}>
      <div className="hm-wrap py-20">
        <div className="grid gap-14 md:grid-cols-[1.3fr_1fr_1fr]">
          <div>
            <p className="hm-display text-[clamp(40px,5vw,64px)]">AtmoGuard</p>
            <p className="hm-body mt-4 max-w-[26em]">{SITE.tagline}. {SITE.problemStatement}.</p>
          </div>

          <div>
            <p className="hm-eyebrow mb-4">Contact</p>
            <p className="hm-body">
              There is no public mailbox for this prototype. Questions, issues and the source code live on GitHub.
            </p>
            {SITE.githubUrl && (
              <a href={SITE.githubUrl} target="_blank" rel="noreferrer noopener" className="hm-link mt-5 inline-block text-[14px]">GitHub repository ↗</a>
            )}
          </div>

          <div>
            <p className="hm-eyebrow mb-4">Navigate</p>
            <ul className="space-y-2.5 text-[14px]">
              <li><Link to="/risk-map" className="hm-link">Live map</Link></li>
              <li><Link to="/alerts" className="hm-link">Alerts</Link></li>
              <li><Link to="/predictions" className="hm-link">Predictions</Link></li>
              <li><Link to="/imagery" className="hm-link">Satellite imagery</Link></li>
              <li><Link to="/dashboard" className="hm-link">Dashboard</Link></li>
              <li><a href="#about" className="hm-link" onClick={(e) => { e.preventDefault(); scrollToSection("about"); }}>About</a></li>
            </ul>
          </div>
        </div>

        <details className="mt-16 border-t border-[var(--hm-rule)] pt-6">
          <summary className="hm-eyebrow cursor-pointer">Data and credits</summary>
          <dl className="mt-5 grid gap-x-12 gap-y-3 text-[13px] md:grid-cols-2">
            {CREDITS.map(([k, v]) => (
              <div key={k}><dt className="text-[var(--hm-ink-dim)]">{k}</dt><dd className="text-[var(--hm-ink)]">{v}</dd></div>
            ))}
          </dl>
          <p className="mt-5 text-[12px] text-[var(--hm-ink-faint)]">The clouds on the globe are procedurally generated decoration, not live cloud cover.</p>
        </details>

        <p className="mt-10 text-[12px] text-[var(--hm-ink-faint)]">© {YEAR} AtmoGuard · A prototype, not an official warning service.</p>
      </div>
    </footer>
  );
}
