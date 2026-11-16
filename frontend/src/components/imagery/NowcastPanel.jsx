// Trained deep-learning nowcast panel (INSAT-3DR infrared).
//
// This is the ONE part of the Imagery page backed by a trained model, and
// everything in it is built to keep that visibly separate from the
// classical OpenCV tools on the rest of the page: violet = trained model,
// cyan = classical method, yellow outline / emerald = what the satellite
// actually observed. If the backend has no usable model it says so and
// shows only the classical extrapolation -- nothing is invented here.
import { useMemo, useState } from "react";
import { Loader2, Play, AlertTriangle, CheckCircle2, Info } from "lucide-react";
import LoadingSpinner from "../common/LoadingSpinner";
import { useAsync } from "../../hooks/useAsync";
import { fetchMlStatus, fetchInsatFrames, runMlNowcast, mediaUrl } from "../../services/imageryService";

const ML_BADGE = "rounded-full bg-violet-500/15 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-violet-300 ring-1 ring-violet-500/30";
const CLASSICAL_BADGE = "rounded-full bg-cyan-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-cyan-300 ring-1 ring-cyan-500/25";
const ACTUAL_BADGE = "rounded-full bg-emerald-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-emerald-400 ring-1 ring-emerald-500/25";

const SPLIT_LABELS = { test: "held-out test", val: "validation", train: "training", unseen: "unseen" };
const SPLIT_STYLES = {
  test: "border-emerald-500/25 bg-emerald-500/5 text-emerald-200/90",
  val: "border-sky-500/25 bg-sky-500/5 text-sky-200/90",
  train: "border-amber-500/25 bg-amber-500/5 text-amber-200/90",
  unseen: "border-white/10 bg-white/5 text-slate-300",
};
const VIEWS = [
  { key: "image", label: "Predicted image" },
  { key: "cold", label: "Cold-cloud overlay" },
  { key: "error", label: "Error vs. actual" },
];

function formatUtc(iso) {
  if (!iso) return "—";
  const text = new Date(iso).toLocaleString("en-GB", {
    timeZone: "UTC", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
  return `${text} UTC`;
}

function fmt(value, digits = 2) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function formatBytes(bytes) {
  return bytes ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : "—";
}

// --- Model card: what the model is and how it scored when tested ----------

function ModelCard({ status }) {
  if (!status.available) {
    return (
      <div className="flex items-start gap-2 rounded-xl border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-200/90">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        <p className="min-w-0 leading-snug">
          <span className="font-semibold">Trained model unavailable.</span> {status.reason}{" "}
          {!status.backendUnreachable && "The classical OpenCV optical-flow extrapolation below still works and is labelled as such."}
        </p>
      </div>
    );
  }

  const leads = Object.entries(status.testMetrics ?? {});
  return (
    <div className="space-y-3 rounded-xl bg-violet-500/5 p-3 ring-1 ring-violet-500/20">
      <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-400">
        <span className="font-semibold text-violet-200">{status.name} v{status.version}</span>
        <span className="min-w-0">{status.architecture}</span>
        <span>{status.parameters?.toLocaleString()} parameters</span>
        <span>{formatBytes(status.artifactSizeBytes)}</span>
        <span>running on <span className="font-semibold uppercase text-slate-300">{status.device}</span></span>
      </div>

      {leads.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-left text-[11px]">
            <caption className="mb-1.5 text-left text-[11px] text-slate-500">
              Held-out test set — average error in kelvin (lower is better) and cold-cloud overlap (IoU, higher is better).
            </caption>
            <thead className="text-slate-500">
              <tr>
                <th className="py-1 pr-3 font-medium">Lead time</th>
                <th className="py-1 pr-3 font-medium text-violet-300">Trained model</th>
                <th className="py-1 pr-3 font-medium text-cyan-300">OpenCV optical flow</th>
                <th className="py-1 pr-3 font-medium">No-change guess</th>
                <th className="py-1 pr-3 font-medium">Model vs. OpenCV</th>
                <th className="py-1 font-medium">Cold-cloud IoU (model / OpenCV)</th>
              </tr>
            </thead>
            <tbody className="tabular-nums text-slate-300">
              {leads.map(([minutes, m]) => (
                <tr key={minutes} className="border-t border-white/5">
                  <td className="py-1 pr-3">+{minutes} min <span className="text-slate-600">({m.frames} frames)</span></td>
                  <td className="py-1 pr-3 font-semibold text-violet-200">{fmt(m.model.maeK)} K</td>
                  <td className="py-1 pr-3">{fmt(m.opticalFlow.maeK)} K</td>
                  <td className="py-1 pr-3">{fmt(m.persistence.maeK)} K</td>
                  <td className="py-1 pr-3">
                    {m.skill.maeReductionVsOpticalFlowPct >= 0 ? "" : "+"}
                    {fmt(-m.skill.maeReductionVsOpticalFlowPct, 1)}% error
                  </td>
                  <td className="py-1">{fmt(m.model.coldCloudIou, 3)} / {fmt(m.opticalFlow.coldCloudIou, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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

// --- One column of the comparison (model / classical / actual) ------------

function Column({ badge, badgeClass, title, url, caption, children }) {
  return (
    <div className="min-w-0 space-y-1.5">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className={`shrink-0 ${badgeClass}`}>{badge}</span>
        <span className="min-w-0 text-[11px] text-slate-400">{title}</span>
      </div>
      {url ? (
        <img src={mediaUrl(url)} alt={`${badge}: ${title}`} className="w-full rounded-lg ring-1 ring-white/10" />
      ) : (
        <div className="flex aspect-[4/3] items-center justify-center rounded-lg bg-navy-800/50 p-3 text-center text-[11px] text-slate-500 ring-1 ring-white/5">
          {caption}
        </div>
      )}
      {children}
    </div>
  );
}

function VerificationTable({ verification }) {
  const rows = [
    { key: "ml", label: "Trained model", tone: "text-violet-200" },
    { key: "classical", label: "OpenCV optical flow", tone: "text-cyan-200" },
    { key: "persistence", label: "No-change guess", tone: "text-slate-300" },
  ].filter((r) => verification[r.key]);
  const best = Math.min(...rows.map((r) => verification[r.key].maeK));

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[460px] text-left text-[11px]">
        <caption className="mb-1.5 text-left text-[11px] text-slate-500">
          Checked against the actual satellite frame for this time and area.
        </caption>
        <thead className="text-slate-500">
          <tr>
            <th className="py-1 pr-3 font-medium">Method</th>
            <th className="py-1 pr-3 font-medium">Mean error</th>
            <th className="py-1 pr-3 font-medium">RMSE</th>
            <th className="py-1 pr-3 font-medium">Cold-cloud IoU</th>
            <th className="py-1 pr-3 font-medium">Precision</th>
            <th className="py-1 font-medium">Recall</th>
          </tr>
        </thead>
        <tbody className="tabular-nums text-slate-300">
          {rows.map(({ key, label, tone }) => {
            const m = verification[key];
            return (
              <tr key={key} className="border-t border-white/5">
                <td className={`py-1 pr-3 font-medium ${tone}`}>{label}</td>
                <td className="py-1 pr-3">
                  {fmt(m.maeK)} K {m.maeK === best && <CheckCircle2 className="ml-0.5 inline h-3 w-3 text-emerald-400" aria-label="lowest error" />}
                </td>
                <td className="py-1 pr-3">{fmt(m.rmseK)} K</td>
                <td className="py-1 pr-3">{fmt(m.coldCloud.iou, 3)}</td>
                <td className="py-1 pr-3">{fmt(m.coldCloud.precision, 3)}</td>
                <td className="py-1">{fmt(m.coldCloud.recall, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function NowcastResult({ result }) {
  const [leadIndex, setLeadIndex] = useState(0);
  const [view, setView] = useState("image");
  const lead = result.leads[leadIndex];
  const hasActual = Boolean(lead.actual);
  const activeView = view === "error" && !hasActual ? "image" : view;
  const pick = (side) => {
    if (!side) return null;
    if (activeView === "cold") return side.coldCloudUrl;
    if (activeView === "error") return side.errorUrl;
    return side.predictionUrl;
  };
  const confidence = lead.ml?.confidence;

  return (
    <div className="space-y-4">
      <div className={`flex items-start gap-2 rounded-xl border p-3 text-xs ${SPLIT_STYLES[result.dataSplit] ?? SPLIT_STYLES.unseen}`}>
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <p className="min-w-0 leading-snug">
          <span className="font-semibold capitalize">{SPLIT_LABELS[result.dataSplit] ?? result.dataSplit} data.</span>{" "}
          {result.dataSplitNote}
        </p>
      </div>

      <div>
        <p className="mb-1.5 text-[11px] text-slate-500">
          Input — the three real frames the prediction starts from ({result.regionLabel}, {result.grid.widthPx}×{result.grid.heightPx} px at 4 km;
          white box = Delhi-NCR)
        </p>
        <div className="grid grid-cols-3 gap-2">
          {result.inputs.map((input) => (
            <figure key={input.frameId} className="min-w-0">
              <img src={mediaUrl(input.url)} alt={`Observed ${formatUtc(input.timestamp)}`} className="w-full rounded-lg ring-1 ring-white/10" />
              <figcaption className="mt-1 truncate text-[10px] text-slate-500">{formatUtc(input.timestamp)}</figcaption>
            </figure>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {result.leads.map((l, i) => (
          <button
            key={l.leadMinutes}
            onClick={() => setLeadIndex(i)}
            className={`rounded-lg px-3 py-1.5 font-semibold ring-1 ring-white/10 transition-colors ${i === leadIndex ? "bg-violet-500/20 text-violet-200" : "text-slate-400 hover:bg-white/5"}`}
          >
            +{l.leadMinutes} min
          </button>
        ))}
        <span className="mx-1 hidden h-4 w-px bg-white/10 sm:block" />
        {VIEWS.map((v) => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            disabled={v.key === "error" && !hasActual}
            className={`rounded-lg px-3 py-1.5 font-medium ring-1 ring-white/10 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${v.key === activeView ? "bg-white/10 text-slate-100" : "text-slate-400 hover:bg-white/5"}`}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Column
          badge="Trained ML model"
          badgeClass={ML_BADGE}
          title={`predicted for ${formatUtc(lead.validTime)}`}
          url={pick(lead.ml)}
          caption={`No trained-model output: ${result.model.reason ?? "model unavailable"}`}
        >
          {confidence && (
            <dl className="grid grid-cols-3 gap-2 text-[11px]">
              <div className="rounded-lg bg-navy-800/50 p-2 ring-1 ring-white/5">
                <dt className="text-slate-500">Expected error</dt>
                <dd className="font-semibold text-slate-200">±{fmt(confidence.expectedErrorK)} K</dd>
              </div>
              <div className="rounded-lg bg-navy-800/50 p-2 ring-1 ring-white/5">
                <dt className="text-slate-500">Cold-cloud area</dt>
                <dd className="font-semibold text-slate-200">{fmt(confidence.coldCloudAreaPct, 1)}%</dd>
              </div>
              <div className="rounded-lg bg-navy-800/50 p-2 ring-1 ring-white/5">
                <dt className="text-slate-500">Mask confidence</dt>
                <dd className="font-semibold text-slate-200">
                  {confidence.coldCloudConfidencePct === null ? "—" : `${fmt(confidence.coldCloudConfidencePct, 0)}%`}
                </dd>
              </div>
            </dl>
          )}
        </Column>

        <Column
          badge="Classical OpenCV"
          badgeClass={CLASSICAL_BADGE}
          title="optical-flow extrapolation (no training)"
          url={pick(lead.classical)}
          caption="No classical output"
        >
          <p className="text-[11px] leading-snug text-slate-500">
            Cold-cloud area {fmt(lead.classical.coldCloudAreaPct, 1)}%. No confidence estimate — this method cannot produce one.
          </p>
        </Column>

        <Column
          badge="Actual observation"
          badgeClass={ACTUAL_BADGE}
          title={hasActual ? `INSAT-3DR at ${formatUtc(lead.validTime)}` : "not available"}
          url={hasActual && activeView !== "error" ? lead.actual.url : null}
          caption={
            hasActual
              ? "Error maps compare each prediction with this frame: blue = predicted too cold, red = too warm, white = correct."
              : "No satellite frame exists in the dataset for this time, so this prediction cannot be checked."
          }
        >
          {hasActual && (
            <p className="text-[11px] leading-snug text-slate-500">
              Cold-cloud area {fmt(lead.actual.coldCloudAreaPct, 1)}%
              {activeView === "cold" && " — outlined in yellow on both predictions."}
            </p>
          )}
        </Column>
      </div>

      {lead.ml?.uncertaintyUrl && (
        <details className="text-[11px] text-slate-500">
          <summary className="cursor-pointer text-slate-400">Where the model is least sure (uncertainty map)</summary>
          <div className="mt-2 grid grid-cols-1 gap-3 md:grid-cols-3">
            <img src={mediaUrl(lead.ml.uncertaintyUrl)} alt="Model uncertainty map" className="w-full rounded-lg ring-1 ring-white/10" />
            <p className="leading-snug md:col-span-2">
              Brighter = larger expected error (scale 0–8 K). The model predicts this alongside the image itself.
              {lead.verification?.ml?.within90PctInterval !== undefined && (
                <> For this image, {fmt(100 * lead.verification.ml.within90PctInterval, 1)}% of pixels fell inside the model&apos;s
                  stated 90% range (90% would be perfectly honest).</>
              )}
            </p>
          </div>
        </details>
      )}

      {lead.verification && <VerificationTable verification={lead.verification} />}

      <div className="space-y-1 text-[11px] leading-snug text-slate-500">
        <p>{result.notes.task}</p>
        <p>{result.notes.coldCloud}</p>
        {result.model.available && (
          <p>
            Model inference: {fmt(result.model.inferenceMs, 0)} ms on {String(result.model.device).toUpperCase()}.
          </p>
        )}
      </div>
    </div>
  );
}

// --- Panel -------------------------------------------------------------

export default function NowcastPanel() {
  const { data: status, isLoading: statusLoading } = useAsync(() => fetchMlStatus(), []);
  const { data: catalog, isLoading: catalogLoading } = useAsync(() => fetchInsatFrames(), []);
  const [chosenAnchorId, setAnchorId] = useState("");
  const [region, setRegion] = useState("north_india");
  const [run, setRun] = useState({ status: "idle", data: null, error: null });

  const anchors = useMemo(() => (catalog?.frames ?? []).filter((f) => f.canAnchorNowcast), [catalog]);

  // Default to a held-out test frame whose three lead times can all be
  // checked against a real later observation -- the most honest demo.
  // Derived during render (not stored) until the user picks one themselves.
  const defaultAnchorId = useMemo(() => {
    const preferred =
      anchors.find((f) => f.split === "test" && f.verifiableLeadMinutes.length === 3) ??
      anchors.find((f) => f.verifiableLeadMinutes.length === 3) ??
      anchors[0];
    return preferred?.frameId ?? "";
  }, [anchors]);
  const anchorId = chosenAnchorId || defaultAnchorId;

  const start = async () => {
    if (!anchorId) return;
    setRun({ status: "loading", data: null, error: null });
    try {
      setRun({ status: "done", data: await runMlNowcast(anchorId, region), error: null });
    } catch (err) {
      setRun({ status: "idle", data: null, error: err.name === "AbortError" ? "The nowcast took too long and was cancelled." : err.message });
    }
  };

  if (statusLoading || catalogLoading) return <LoadingSpinner label="Loading model status and INSAT-3DR frames" />;

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-slate-400">
        A ConvLSTM trained on real INSAT-3DR infrared frames predicts what the cloud-top image will look like 30, 60
        and 90 minutes later. Each prediction is shown next to the classical OpenCV method and, where it exists, the
        frame the satellite actually recorded.
      </p>

      <ModelCard status={status ?? { available: false, reason: "Status unavailable." }} />

      {!catalog?.available ? (
        <p className="rounded-xl bg-navy-800/50 p-3 text-xs text-slate-500 ring-1 ring-white/5">
          INSAT-3DR frames are unavailable: {catalog?.reason ?? "unknown reason"}
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-3">
            <label className="min-w-0 flex-1 basis-64 text-[11px] text-slate-500">
              Latest observed frame (prediction starts here)
              <select
                value={anchorId}
                onChange={(e) => setAnchorId(e.target.value)}
                className="mt-1 w-full rounded-lg bg-navy-800 px-2.5 py-2 text-xs text-slate-200 ring-1 ring-white/10"
              >
                {anchors.map((f) => (
                  <option key={f.frameId} value={f.frameId}>
                    {formatUtc(f.timestamp)} · {SPLIT_LABELS[f.split] ?? f.split}
                    {f.verifiableLeadMinutes.length ? ` · checkable +${f.verifiableLeadMinutes.join("/+")} min` : " · not checkable"}
                  </option>
                ))}
              </select>
            </label>
            <label className="min-w-0 flex-1 basis-48 text-[11px] text-slate-500">
              Area
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="mt-1 w-full rounded-lg bg-navy-800 px-2.5 py-2 text-xs text-slate-200 ring-1 ring-white/10"
              >
                {catalog.regions.map((r) => (
                  <option key={r.key} value={r.key}>{r.label} ({r.widthPx}×{r.heightPx} px)</option>
                ))}
              </select>
            </label>
            <button
              onClick={start}
              disabled={!anchorId || run.status === "loading"}
              className="flex shrink-0 items-center gap-2 rounded-lg bg-violet-500/20 px-4 py-2 text-xs font-semibold text-violet-200 ring-1 ring-violet-500/40 transition-colors hover:bg-violet-500/30 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {run.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {run.status === "loading" ? "Running…" : status?.available ? "Run nowcast" : "Run classical extrapolation"}
            </button>
          </div>
          <p className="text-[11px] text-slate-600">
            {catalog.validCount} usable frames, {catalog.corruptCount} corrupt (truncated downloads, excluded) · {catalog.source}
          </p>
        </>
      )}

      {run.error && <p className="text-xs text-red-400">{run.error}</p>}
      {run.data && <NowcastResult key={run.data.resultId} result={run.data} />}
    </div>
  );
}
