// "How it works": the four stages the backend actually runs, and the real
// technology behind them. No accuracy figures are quoted here on purpose.
import { Reveal } from "./Reveal";

const G = { fill: "none", stroke: "currentColor", strokeWidth: 1.2, strokeLinecap: "round", strokeLinejoin: "round" };

const STEPS = [
  {
    n: "01", title: "Weather data collection",
    glyph: <svg viewBox="0 0 48 48" width="44" height="44" aria-hidden="true"><path {...G} d="M4 14c6-6 10 6 16 0s10 6 16 0 6 0 8-2M4 26c6-6 10 6 16 0s10 6 16 0 6 0 8-2M4 38c6-6 10 6 16 0s10 6 16 0 6 0 8-2" /></svg>,
    body: "Hourly readings for every monitored location: temperature, rainfall, humidity, wind, pressure and the convective indices CAPE and lifted index. Satellite imagery adds context from above.",
  },
  {
    n: "02", title: "Machine-learning prediction",
    glyph: <svg viewBox="0 0 48 48" width="44" height="44" aria-hidden="true"><path {...G} d="M6 38 18 22l8 8 16-20" /><circle {...G} cx="18" cy="22" r="2.6" /><circle {...G} cx="26" cy="30" r="2.6" /><circle {...G} cx="42" cy="10" r="2.6" /></svg>,
    body: "Gradient-boosted trees (XGBoost) and linear baselines, trained on a historical Delhi–NCR record, forecast temperature, rainfall, heavy-rain risk and thunderstorm risk one, three and six hours ahead.",
  },
  {
    n: "03", title: "Weather risk assessment",
    glyph: <svg viewBox="0 0 48 48" width="44" height="44" aria-hidden="true"><path {...G} d="M8 40h32M12 40V28M20 40V20M28 40V12M36 40V24" /></svg>,
    body: "A transparent rule-based engine scores four hazards from 0 to 100 against published rainfall and instability thresholds. The overall score is the single highest hazard, so a severe one is never averaged away.",
  },
  {
    n: "04", title: "Early warning generation",
    glyph: <svg viewBox="0 0 48 48" width="44" height="44" aria-hidden="true"><path {...G} d="M24 6 42 38H6Z" /><path {...G} d="M24 20v9" /><circle cx="24" cy="33.5" r="1.1" fill="currentColor" /></svg>,
    body: "When a hazard reaches high or severe, one bulletin is raised per hazard and grid cell, kept current while it lasts, then resolved or expired. Bulletins carry their source: rule-based, model, or both.",
  },
];

const TECH = [
  ["XGBoost", "Supervised forecasting of temperature, rainfall, heavy-rain and thunderstorm risk; 12 trained models, each compared with simpler baselines on held-out data."],
  ["Atmospheric data", "Surface weather and convective indices through Open-Meteo, retrieved live and cached."],
  ["Numerical weather prediction", "Forecast inputs come from NWP models served by Open-Meteo. AtmoGuard does not run its own NWP."],
  ["Satellite imagery", "NASA GIBS imagery and IMERG rainfall; an INSAT-3DR infrared nowcast model; Sentinel-1 radar flood mapping. Each is labelled with what it can and cannot show."],
  ["Machine-learning forecasting", "Validation results and known weaknesses are written up in the project reports rather than summarised as a single accuracy number."],
];

export default function HowItWorks() {
  return (
    <section id="how" className="hm-section" style={{ background: "var(--hm-bg-2)" }} aria-labelledby="how-title">
      <div className="hm-wrap">
        <Reveal>
          <p className="hm-eyebrow mb-5">How it works</p>
          <h2 id="how-title" className="hm-display max-w-[14em] text-[clamp(38px,5.2vw,72px)]">From raw readings to a warning, in four <em>deliberate</em> steps.</h2>
        </Reveal>

        <ol className="relative mt-16 grid gap-12 md:grid-cols-4 md:gap-8">
          <span aria-hidden="true" className="absolute left-0 right-0 top-[22px] hidden h-px bg-[var(--hm-rule-strong)] md:block" />
          {STEPS.map((s, i) => (
            <Reveal as="li" key={s.n} className="relative" style={{ transitionDelay: `${i * 90}ms` }}>
              <div className="relative z-10 inline-block pr-4" style={{ background: "var(--hm-bg-2)", color: "var(--hm-sky)" }}>{s.glyph}</div>
              <p className="hm-mono mt-6 text-[12px] text-[var(--hm-ink-faint)]">{s.n}</p>
              <h3 className="hm-display mt-1 text-[28px] leading-[1.05]">{s.title}</h3>
              <p className="hm-body mt-4">{s.body}</p>
            </Reveal>
          ))}
        </ol>

        <Reveal className="mt-24">
          <p className="hm-eyebrow mb-5">The technology</p>
          <dl className="border-t border-[var(--hm-rule-strong)]">
            {TECH.map(([k, v]) => (
              <div key={k} className="grid gap-2 border-b border-[var(--hm-rule)] py-5 md:grid-cols-[14rem_1fr] md:gap-10">
                <dt className="text-[15px]">{k}</dt>
                <dd className="hm-body">{v}</dd>
              </div>
            ))}
          </dl>
        </Reveal>
      </div>
    </section>
  );
}
