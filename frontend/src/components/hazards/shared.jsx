// Pieces shared by the flood and lightning model panels: badges, the model
// card, the "what kind of data is this" banner, and small formatters.
//
// Colour convention (same as the nowcast panel): violet = trained model,
// cyan = classical method, yellow/emerald = ground truth, amber = synthetic.
import { AlertTriangle, FlaskConical, Info, Satellite } from "lucide-react";

export const ML_BADGE = "rounded-full bg-violet-500/15 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-violet-300 ring-1 ring-violet-500/30";
export const CLASSICAL_BADGE = "rounded-full bg-cyan-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-cyan-300 ring-1 ring-cyan-500/25";
export const REAL_BADGE = "rounded-full bg-emerald-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-emerald-400 ring-1 ring-emerald-500/25";
export const SYNTHETIC_BADGE = "rounded-full bg-amber-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-amber-400 ring-1 ring-amber-500/25";
export const UNVERIFIED_BADGE = "rounded-full bg-rose-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-rose-300 ring-1 ring-rose-500/25";

export const SELECT_CLASS = "mt-1 w-full rounded-lg bg-navy-800 px-2.5 py-2 text-xs text-slate-200 ring-1 ring-white/10";
export const RUN_BUTTON = "flex shrink-0 items-center gap-2 rounded-lg bg-violet-500/20 px-4 py-2 text-xs font-semibold text-violet-200 ring-1 ring-violet-500/40 transition-colors hover:bg-violet-500/30 disabled:cursor-not-allowed disabled:opacity-40";

const SPLIT_LABELS = { test: "held-out test", valid: "validation", val: "validation", train: "training", unused: "unused" };

export function splitLabel(split) {
  return SPLIT_LABELS[split] ?? split ?? "";
}

export function fmt(value, digits = 2) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

export function pct(value, digits = 1) {
  return value === null || value === undefined ? "—" : `${(100 * Number(value)).toFixed(digits)}%`;
}

export function formatBytes(bytes) {
  return bytes ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : "—";
}

export function formatUtc(iso) {
  if (!iso) return "—";
  const text = new Date(iso).toLocaleString("en-GB", {
    timeZone: "UTC", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
  return `${text} UTC`;
}

export function sourceBadge(source) {
  if (source === "synthetic") return { label: "Synthetic simulation", className: SYNTHETIC_BADGE };
  if (source === "real-insat") return { label: "Real image · no ground truth", className: UNVERIFIED_BADGE };
  if (source === "upload") return { label: "Uploaded image", className: REAL_BADGE };
  return { label: "Real observation", className: REAL_BADGE };
}

// What the user is looking at, said plainly, before any picture is shown.
export function ProvenanceBanner({ sample }) {
  const synthetic = sample.source === "synthetic";
  const unverified = sample.source === "real-insat";
  const tone = synthetic
    ? "border-amber-500/30 bg-amber-500/5 text-amber-200/90"
    : unverified
      ? "border-rose-500/30 bg-rose-500/5 text-rose-200/90"
      : sample.split === "train"
        ? "border-amber-500/25 bg-amber-500/5 text-amber-200/90"
        : "border-emerald-500/25 bg-emerald-500/5 text-emerald-200/90";
  const Icon = synthetic ? FlaskConical : unverified ? AlertTriangle : Satellite;
  return (
    <div className={`flex items-start gap-2 rounded-xl border p-3 text-xs ${tone}`}>
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <p className="min-w-0 leading-snug">
        <span className="font-semibold">{sample.sourceLabel}.</span>{" "}
        {sample.splitNote}
      </p>
    </div>
  );
}

export function Stat({ label, value, hint }) {
  return (
    <div className="min-w-0 rounded-lg bg-navy-800/50 p-2 ring-1 ring-white/5">
      <dt className="truncate text-[11px] text-slate-500">{label}</dt>
      <dd className="truncate text-sm font-semibold text-slate-200">{value}</dd>
      {hint && <p className="mt-0.5 text-[10px] leading-snug text-slate-600">{hint}</p>}
    </div>
  );
}

export function Column({ badge, badgeClass, title, url, caption, children }) {
  return (
    <div className="min-w-0 space-y-1.5">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className={`shrink-0 ${badgeClass}`}>{badge}</span>
        <span className="min-w-0 text-[11px] text-slate-400">{title}</span>
      </div>
      {url ? (
        <img src={url} alt={`${badge}: ${title}`} className="w-full rounded-lg ring-1 ring-white/10" />
      ) : (
        <div className="flex aspect-[4/3] items-center justify-center rounded-lg bg-navy-800/50 p-3 text-center text-[11px] text-slate-500 ring-1 ring-white/5">
          {caption}
        </div>
      )}
      {children}
    </div>
  );
}

// Model card: what the model is, how it scored on REAL held-out data, how
// that compares with the classical baseline, and -- separately -- how it
// scored on synthetic India samples. `rows` describes the score table.
export function ModelCard({ status, fallbackName, scoreCaption, columns, rows, syntheticLine }) {
  if (!status?.available) {
    return (
      <div className="flex items-start gap-2 rounded-xl border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-200/90">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        <p className="min-w-0 leading-snug">
          <span className="font-semibold">Trained model unavailable.</span> {status?.reason ?? "Status unavailable."}{" "}
          {!status?.backendUnreachable && `The classical ${fallbackName} still runs and is labelled as such.`}
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-3 rounded-xl bg-violet-500/5 p-3 ring-1 ring-violet-500/20">
      <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-400">
        <span className="font-semibold text-violet-200">{status.name} v{status.version}</span>
        <span className="min-w-0">{status.architecture}</span>
        <span>{status.parameters?.toLocaleString()} parameters</span>
        <span>{formatBytes(status.artifactSizeBytes)}</span>
        <span>running on <span className="font-semibold uppercase text-slate-300">{status.device}</span></span>
        <span>trained on <span className="font-semibold text-slate-300">{status.trainedOn}</span> data</span>
      </div>

      {rows?.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-left text-[11px]">
            <caption className="mb-1.5 text-left text-[11px] text-slate-500">{scoreCaption}</caption>
            <thead className="text-slate-500">
              <tr>
                <th className="py-1 pr-3 font-medium">Method</th>
                {columns.map((c) => <th key={c.key} className="py-1 pr-3 font-medium">{c.label}</th>)}
              </tr>
            </thead>
            <tbody className="tabular-nums text-slate-300">
              {rows.filter((r) => r.metrics).map((r) => (
                <tr key={r.label} className="border-t border-white/5">
                  <td className={`py-1 pr-3 font-medium ${r.tone}`}>{r.label}</td>
                  {columns.map((c) => <td key={c.key} className="py-1 pr-3">{fmt(r.metrics[c.key], 3)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {syntheticLine && (
        <p className="flex items-start gap-1.5 text-[11px] leading-snug text-amber-200/80">
          <FlaskConical className="mt-0.5 h-3 w-3 shrink-0 text-amber-400" />
          <span className="min-w-0">{syntheticLine}</span>
        </p>
      )}

      {status.limitations?.length > 0 && (
        <details className="text-[11px] text-slate-500">
          <summary className="cursor-pointer text-slate-400">What this model can and cannot tell you</summary>
          <ul className="mt-1.5 list-disc space-y-1 pl-4 leading-snug">
            {status.limitations.map((line) => <li key={line}>{line}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}

export function Notes({ lines }) {
  return (
    <div className="space-y-1 text-[11px] leading-snug text-slate-500">
      {lines.filter(Boolean).map((line) => (
        <p key={line} className="flex items-start gap-1.5">
          <Info className="mt-0.5 h-3 w-3 shrink-0 text-slate-600" />
          <span className="min-w-0">{line}</span>
        </p>
      ))}
    </div>
  );
}
