import { CloudRain, Grid3x3, BrainCircuit, ShieldAlert, Gauge, BellRing, ArrowDown, Target, Lightbulb } from "lucide-react";
import GlassCard from "../components/common/GlassCard";

const PIPELINE_STEPS = [
  { icon: CloudRain, label: "Weather Data", desc: "Multi-source meteorological signals ingested per region" },
  { icon: Grid3x3, label: "Spatial Grid", desc: "Region divided into fine-grained hyper-local cells" },
  { icon: BrainCircuit, label: "AI Prediction", desc: "Nowcasting model projects near-term conditions per cell" },
  { icon: ShieldAlert, label: "Hazard Detection", desc: "Cell-level output classified into hazard types" },
  { icon: Gauge, label: "Risk Score", desc: "Hazard likelihood combined into a single 0-100 score" },
  { icon: BellRing, label: "Early Warning", desc: "Threshold breaches trigger targeted alerts" },
];

export default function About() {
  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <GlassCard className="p-6 sm:p-8">
        <p className="text-xs font-semibold uppercase tracking-wide text-cyan-400">SIH Problem Statement 26077</p>
        <h2 className="mt-2 text-2xl font-bold text-white sm:text-3xl">MausamAI</h2>
        <p className="mt-1 text-sm text-slate-400">Hyper-Local Weather Intelligence &amp; Early Warning System</p>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-slate-400">
          MausamAI is a prototype for turning coarse, city-wide weather forecasts into
          street-level, actionable early warnings — built spatial grid by spatial grid instead
          of one number for an entire city.
        </p>
      </GlassCard>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <GlassCard className="p-6">
          <div className="mb-3 flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-500/10 ring-1 ring-red-500/20">
              <Target className="h-4 w-4 text-red-400" />
            </div>
            <h3 className="text-sm font-semibold text-slate-200">The Problem</h3>
          </div>
          <p className="text-sm leading-relaxed text-slate-400">
            Traditional weather forecasts are often too broad geographically — a single
            "city-level" forecast can miss the fact that one neighborhood is flooding while
            another, a few kilometers away, stays dry. This gap means residents get warnings
            that are either too late or not specific enough to act on.
          </p>
        </GlassCard>

        <GlassCard className="p-6">
          <div className="mb-3 flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/10 ring-1 ring-emerald-500/20">
              <Lightbulb className="h-4 w-4 text-emerald-400" />
            </div>
            <h3 className="text-sm font-semibold text-slate-200">The Solution</h3>
          </div>
          <p className="text-sm leading-relaxed text-slate-400">
            MausamAI divides a region into smaller spatial grids — roughly neighborhood-sized —
            and independently scores each one for rainfall, thunderstorm, lightning, and flood
            risk. This turns "it might rain in Delhi" into "Sector 62 is at severe flood risk in
            the next 30 minutes," giving people and authorities warnings they can actually act on.
          </p>
        </GlassCard>
      </div>

      <GlassCard className="p-6 sm:p-8">
        <h3 className="mb-1 text-sm font-semibold text-slate-200">AI Pipeline</h3>
        <p className="mb-6 text-xs text-slate-500">How raw weather data becomes a hyper-local warning</p>

        <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:gap-0">
          {PIPELINE_STEPS.map((step, idx) => (
            <div key={step.label} className="flex flex-1 items-center">
              <div className="flex flex-1 flex-col items-center rounded-xl bg-navy-800/50 p-4 text-center ring-1 ring-white/5">
                <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500/20 to-indigo-500/20 ring-1 ring-cyan-500/20">
                  <step.icon className="h-5 w-5 text-cyan-400" />
                </div>
                <p className="text-xs font-semibold text-slate-200">{step.label}</p>
                <p className="mt-1 text-[11px] leading-snug text-slate-500">{step.desc}</p>
              </div>
              {idx < PIPELINE_STEPS.length - 1 && (
                <div className="flex shrink-0 items-center justify-center px-1 py-1 sm:rotate-[-90deg]">
                  <ArrowDown className="h-4 w-4 text-slate-600 sm:hidden" />
                  <ArrowDown className="hidden h-4 w-4 text-slate-600 sm:block" />
                </div>
              )}
            </div>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-6 sm:p-8">
        <h3 className="mb-3 text-sm font-semibold text-slate-200">Prototype Status</h3>
        <ul className="space-y-2 text-sm text-slate-400">
          <li className="flex items-start gap-2">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
            Frontend dashboard, risk map, prediction &amp; alert UI — complete, running on
            realistic mock data.
          </li>
          <li className="flex items-start gap-2">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400" />
            FastAPI backend and trained nowcasting model — planned next phase, designed for
            drop-in replacement via the existing service layer.
          </li>
          <li className="flex items-start gap-2">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-500" />
            This build runs in <span className="font-medium text-slate-300">Demo Mode</span> —
            all weather, prediction and alert data is simulated for demonstration purposes.
          </li>
        </ul>
      </GlassCard>
    </div>
  );
}
