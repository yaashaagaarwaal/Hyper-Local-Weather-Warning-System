// The hero + the Earth -> India journey.
//
//   full / lite   one viewport pinned for ~4 screens of scroll. GSAP ScrollTrigger
//                 scrubs a single timeline; its progress (0..1) drives both the
//                 3D camera (EarthScene) and the text on top of it.
//   static        no WebGL, phone, or reduced motion: a pre-rendered Earth, then a
//                 flat SVG map of India with the same live overlays.
import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useRenderTier } from "./useRenderTier";
import { useIndiaWeather } from "./useIndiaWeather";
import LayerPanel from "./LayerPanel";
import IndiaFlatMap from "./IndiaFlatMap";
import { scrollToSection } from "./HomeNav";

gsap.registerPlugin(ScrollTrigger);

const EarthScene = lazy(() => import("./scene/EarthScene"));

const DEFAULT_LAYERS = { rain: true, borders: true, stations: true, warnings: true, regions: true };

function useMinWidth(px) {
  const [match, setMatch] = useState(() => typeof window !== "undefined" && window.innerWidth >= px);
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${px}px)`);
    const on = () => setMatch(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [px]);
  return match;
}

function HeroCopy({ onDiscover }) {
  return (
    <div className="max-w-[560px]">
      <p className="hm-eyebrow mb-6">Hyper-local weather warning system</p>
      <h1 className="hm-display text-[clamp(50px,7.2vw,108px)]">
        Know the Weather.<br /><em>Stay Prepared.</em>
      </h1>
      <p className="hm-lede mt-7">
        AI-powered hyper-local weather intelligence that helps predict severe weather conditions and deliver timely warnings across India.
      </p>
      <div className="mt-9 flex flex-wrap items-center gap-3">
        <Link to="/risk-map" className="hm-btn hm-btn--solid">Explore Live Map <ArrowRight size={16} strokeWidth={1.6} /></Link>
        <button type="button" className="hm-btn hm-btn--line" onClick={onDiscover}>Discover AtmoGuard</button>
      </div>
    </div>
  );
}

function Chapter({ index, title, children, className = "" }) {
  return (
    <div className={className}>
      <p className="hm-eyebrow mb-3 flex items-center gap-3"><span className="text-[var(--hm-ink)]">{index}</span><span className="h-px w-8 bg-[var(--hm-rule-strong)]" />{title}</p>
      <p className="hm-display text-[clamp(26px,3vw,40px)] leading-[1.08]">{children}</p>
    </div>
  );
}

const CHAPTER_ONE = "Weather begins in the atmosphere. AtmoGuard reads the air above India.";
const CHAPTER_TWO = "Thirty-six states and union territories, from the Himalaya to the Deccan, between two coastlines.";

// ---------------------------------------------------------------------------
// Pinned 3D journey
// ---------------------------------------------------------------------------
function PinnedJourney({ tier, data }) {
  const wide = useMinWidth(1024);
  const layout = wide ? "wide" : "narrow";
  const trigger = useRef(null);
  const stage = useRef(null);
  const progress = useRef(0);
  const scrollTrigger = useRef(null);
  const [active, setActive] = useState(true);
  const [ready, setReady] = useState(false);
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [hover, setHover] = useState(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const length = Math.round(window.innerHeight * 4.4);
      const tl = gsap.timeline({
        defaults: { ease: "none" },
        scrollTrigger: {
          trigger: trigger.current, start: "top top", end: `+=${length}`, pin: stage.current, scrub: 0.8, anticipatePin: 1,
          invalidateOnRefresh: true,
        },
        onUpdate: () => { progress.current = tl.progress(); },
      });
      scrollTrigger.current = tl.scrollTrigger;

      tl.to(".j-hero", { autoAlpha: 0, y: -36, duration: 0.10 }, 0.035)
        .to(".j-hint", { autoAlpha: 0, duration: 0.04 }, 0.01)
        .fromTo(".j-ch1", { autoAlpha: 0, y: 24 }, { autoAlpha: 1, y: 0, duration: 0.07 }, 0.15)
        .to(".j-ch1", { autoAlpha: 0, y: -18, duration: 0.06 }, 0.34)
        .fromTo(".j-ch2", { autoAlpha: 0, y: 24 }, { autoAlpha: 1, y: 0, duration: 0.07 }, 0.42)
        .to(".j-ch2", { autoAlpha: 0, y: -18, duration: 0.06 }, 0.64)
        .fromTo(".j-panel", { autoAlpha: 0, y: 24 }, { autoAlpha: 1, y: 0, duration: 0.08 }, 0.72)
        .fromTo(".j-scrim", { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.12 }, 0.12)
        .to({}, { duration: 0.001 }, 1); // pad so the timeline is exactly 1 long

      ScrollTrigger.create({
        trigger: trigger.current, start: "top bottom", end: "bottom top", onToggle: (self) => setActive(self.isActive),
      });
    }, trigger);
    return () => ctx.revert();
  }, []);

  // A hidden tab should not render.
  useEffect(() => {
    const on = () => setActive(!document.hidden);
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);

  const discover = useCallback(() => {
    const st = scrollTrigger.current;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (st) window.scrollTo({ top: st.end, behavior: reduce ? "auto" : "smooth" });
  }, []);
  const toggle = useCallback((id) => setLayers((l) => ({ ...l, [id]: !l[id] })), []);

  const markers = { stations: data.stations.items, warnings: data.warnings.items };

  return (
    <section id="journey" ref={trigger} aria-label="AtmoGuard: from Earth to India" className="relative">
      <div ref={stage} className="relative h-[100svh] w-full overflow-hidden" style={{ background: "var(--hm-bg)" }}>
        <div className="absolute inset-0" style={{ opacity: ready ? 1 : 0, transition: "opacity 1.4s ease" }}>
          <Suspense fallback={null}>
            <EarthScene
              progress={progress} layout={layout} tier={tier} active={active} layers={layers} rain={data.rain}
              markers={markers} hover={hover} onHover={setHover} onReady={() => setReady(true)}
            />
          </Suspense>
        </div>

        {/* soft darkening behind the copy so it stays legible over the globe */}
        <div className="j-scrim pointer-events-none absolute inset-y-0 left-0 w-[62%] opacity-0" style={{ background: "linear-gradient(90deg, rgba(5,8,15,0.78), rgba(5,8,15,0.45) 55%, transparent)" }} />

        <div className="hm-wrap pointer-events-none absolute inset-0 flex items-center">
          <div className="j-hero pointer-events-auto pt-14 max-lg:self-start max-lg:pt-28"><HeroCopy onDiscover={discover} /></div>
        </div>

        <div className="hm-wrap pointer-events-none absolute inset-x-0 bottom-0 pb-10 max-lg:pb-8">
          <p className="j-hint hm-eyebrow flex items-center gap-3"><span className="h-px w-10 bg-[var(--hm-rule-strong)]" />Scroll</p>
          <Chapter index="01" title="Observe" className="j-ch1 absolute bottom-10 max-w-[440px] opacity-0 max-lg:bottom-8 max-lg:max-w-none">{CHAPTER_ONE}</Chapter>
          <Chapter index="02" title="India" className="j-ch2 absolute bottom-10 max-w-[460px] opacity-0 max-lg:bottom-8 max-lg:max-w-none">{CHAPTER_TWO}</Chapter>
        </div>

        <div className="hm-wrap pointer-events-none absolute inset-0 flex items-end pb-9 max-lg:pb-5">
          <div className="j-panel pointer-events-auto w-[340px] opacity-0 max-lg:w-full max-lg:max-w-[420px]" style={{ background: "linear-gradient(180deg, rgba(5,8,15,0) 0%, rgba(5,8,15,0.72) 14%)", padding: "28px 0 0" }}>
            <p className="hm-eyebrow mb-4 flex items-center gap-3"><span className="text-[var(--hm-ink)]">03</span><span className="h-px w-8 bg-[var(--hm-rule-strong)]" />Weather over India</p>
            <LayerPanel data={data} layers={layers} onToggle={toggle} hover={hover} compact={!wide} />
            <p className="mt-2 text-[10.5px] leading-snug text-[var(--hm-ink-faint)]">Terrain exaggerated 36×. Imagery: NASA Blue Marble · Elevation: open terrain tiles.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// No-3D journey
// ---------------------------------------------------------------------------
function StaticJourney({ data }) {
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [hover, setHover] = useState(null);
  const toggle = useCallback((id) => setLayers((l) => ({ ...l, [id]: !l[id] })), []);
  return (
    <section id="journey" aria-label="AtmoGuard: from Earth to India">
      <div className="hm-wrap grid items-center gap-10 pb-16 pt-28 lg:grid-cols-[1.05fr_1fr] lg:pb-24 lg:pt-32">
        <HeroCopy onDiscover={() => scrollToSection("india")} />
        <img
          src="/geo/earth-poster.jpg" alt="The Earth from space, India on the sunlit side" width="1400" height="1400"
          className="mx-auto w-full max-w-[560px] lg:max-w-none" decoding="async" fetchPriority="high"
        />
      </div>

      <div id="india" className="hm-wrap pb-20 pt-4">
        <Chapter index="01" title="Observe" className="max-w-[560px]">{CHAPTER_ONE}</Chapter>
        <Chapter index="02" title="India" className="mt-14 max-w-[560px]">{CHAPTER_TWO}</Chapter>
        <div className="mt-12 grid items-start gap-8 lg:grid-cols-[1.25fr_1fr]">
          <IndiaFlatMap rain={data.rain} stations={data.stations} warnings={data.warnings} layers={layers} hover={hover} onHover={setHover} />
          <div>
            <p className="hm-eyebrow mb-4 flex items-center gap-3"><span className="text-[var(--hm-ink)]">03</span><span className="h-px w-8 bg-[var(--hm-rule-strong)]" />Weather over India</p>
            <LayerPanel data={data} layers={layers} onToggle={toggle} hover={hover} compact />
            <p className="mt-2 text-[10.5px] leading-snug text-[var(--hm-ink-faint)]">Imagery: NASA Blue Marble. The interactive 3D globe is shown on larger screens when motion is allowed.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Journey() {
  const tier = useRenderTier();
  const data = useIndiaWeather(true);
  return tier === "static" ? <StaticJourney data={data} /> : <PinnedJourney key={tier} tier={tier} data={data} />;
}
