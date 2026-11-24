// Flood-mapping panel: a trained U-Net on Sentinel-1 radar, shown next to
// the classical backscatter threshold and -- where one exists -- the label.
// Real observations and synthetic simulations are selected from separate
// lists and tagged on every result. Nothing here is computed in the browser.
import { useMemo, useRef, useState } from "react";
import { Loader2, Play, UploadCloud } from "lucide-react";
import LoadingSpinner from "../common/LoadingSpinner";
import HazardMap from "./HazardMap";
import {
  CLASSICAL_BADGE, ML_BADGE, REAL_BADGE, RUN_BUTTON, SELECT_CLASS, SYNTHETIC_BADGE,
  Column, ModelCard, Notes, ProvenanceBanner, Stat, fmt, sourceBadge, splitLabel,
} from "./shared";
import { useAsync } from "../../hooks/useAsync";
import { mediaUrl } from "../../services/imageryService";
import { fetchFloodSamples, runFloodPrediction, uploadFloodImage } from "../../services/hazardModelService";

const VIEWS = [
  { key: "overlay", label: "Water overlay" },
  { key: "probability", label: "Probability map" },
  { key: "error", label: "Error vs. label" },
];
const SCORE_COLUMNS = [
  { key: "iou", label: "IoU" }, { key: "f1", label: "F1" }, { key: "precision", label: "Precision" }, { key: "recall", label: "Recall" },
];

function km2(value) {
  return value === null || value === undefined ? "—" : `${fmt(value, value < 10 ? 2 : 1)} km²`;
}

function FloodResult({ result }) {
  const [view, setView] = useState("overlay");
  const { ml, classical, verification, sample, geo } = result;
  const hasLabel = Boolean(verification);
  const activeView = (view === "error" && !hasLabel) || (view === "probability" && !ml) ? "overlay" : view;
  const mlUrl = !ml ? null : activeView === "probability" ? ml.probabilityUrl : activeView === "error" ? ml.errorUrl : ml.overlayUrl;
  const classicalUrl = activeView === "error" ? classical.errorUrl : classical.overlayUrl;
  const badge = sourceBadge(sample.source);
  const rows = [
    { label: "Trained U-Net", tone: "text-violet-200", metrics: verification?.ml },
    { label: "VH threshold", tone: "text-cyan-200", metrics: verification?.classical },
    { label: "Dataset's own Otsu mask", tone: "text-slate-300", metrics: verification?.datasetOtsuReference },
  ].filter((r) => r.metrics);

  return (
    <div className="space-y-4">
      <ProvenanceBanner sample={sample} />

      <div className="flex min-w-0 flex-wrap items-center gap-2 text-[11px] text-slate-400">
        <span className={badge.className}>{badge.label}</span>
        {sample.place && <span>{sample.place}, {sample.state} · {sample.scenario} · {sample.monthName}</span>}
        {sample.event && <span>{sample.country} · {sample.eventNote} · radar acquired {sample.acquired}</span>}
        {geo.centre && <span>{fmt(geo.centre.latitude, 3)}°N, {fmt(geo.centre.longitude, 3)}°E</span>}
        <span>{geo.widthPx}×{geo.heightPx} px at {geo.pixelSizeM} m</span>
      </div>

      <dl className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label={ml?.floodAreaKm2 !== undefined ? "Estimated flooded area" : "Estimated water area"}
          value={ml ? km2(ml.floodAreaKm2 ?? ml.waterAreaKm2) : "—"}
          hint={ml ? (ml.floodAreaKm2 !== undefined ? `outside permanent water (${sample.permanentWaterSource})` : "no permanent-water layer for this image") : "trained model unavailable"}
        />
        <Stat label="Water share of image" value={ml ? `${fmt(ml.waterAreaPct, 1)}%` : `${fmt(classical.waterAreaPct, 1)}%`} hint={ml ? `threshold method: ${fmt(classical.waterAreaPct, 1)}%` : "threshold method"} />
        <Stat label="Model confidence in its water pixels" value={ml?.meanWaterConfidencePct != null ? `${fmt(ml.meanWaterConfidencePct, 0)}%` : "—"} hint={ml ? `${fmt(ml.uncertainAreaPct, 1)}% of the image is uncertain (30–70%)` : undefined} />
        <Stat label="Scene footprint" value={km2(geo.footprintKm2)} hint={result.radar.missingPct > 0 ? `${fmt(result.radar.missingPct, 1)}% has no radar data` : undefined} />
      </dl>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {VIEWS.map((v) => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            disabled={(v.key === "error" && !hasLabel) || (v.key === "probability" && !ml)}
            className={`rounded-lg px-3 py-1.5 font-medium ring-1 ring-white/10 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${v.key === activeView ? "bg-white/10 text-slate-100" : "text-slate-400 hover:bg-white/5"}`}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Column badge="Trained ML model" badgeClass={ML_BADGE} title={activeView === "probability" ? "water probability (dark = 0, yellow = 1)" : "U-Net water mask"} url={mediaUrl(mlUrl)} caption={`No trained-model output: ${result.model.reason ?? "model unavailable"}`} />
        <Column badge="Classical threshold" badgeClass={CLASSICAL_BADGE} title={`VH darker than ${fmt(classical.thresholdDb, 1)} dB (${classical.thresholdSource})`} url={mediaUrl(activeView === "probability" ? classical.overlayUrl : classicalUrl)} caption="No classical output" />
        <Column
          badge={hasLabel ? (sample.source === "synthetic" ? "Simulated truth" : "Human label") : "Radar image"}
          badgeClass={sample.source === "synthetic" ? SYNTHETIC_BADGE : REAL_BADGE}
          title={hasLabel ? `water in ${fmt(verification.labelWaterPct, 1)}% of labelled pixels` : "VH backscatter"}
          url={mediaUrl(hasLabel ? result.images.labelUrl : result.images.radarVhUrl)}
          caption=""
        >
          {hasLabel && (
            <p className="text-[11px] leading-snug text-slate-500">
              {activeView === "error"
                ? "Error maps: blue = water found, red = false alarm, orange = missed water, dark grey = not labelled."
                : "Outlined in yellow on both results."}{" "}
              {verification.labelledPct < 100 && `${fmt(verification.labelledPct, 0)}% of the image is labelled.`}
            </p>
          )}
        </Column>
      </div>

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[460px] text-left text-[11px]">
            <caption className="mb-1.5 text-left text-[11px] text-slate-500">
              {sample.source === "synthetic"
                ? "Checked against the SIMULATOR's own mask — agreement with a simulation, not real-world accuracy."
                : "Checked against the human-drawn label for this chip (water class, labelled pixels only)."}
            </caption>
            <thead className="text-slate-500">
              <tr>
                <th className="py-1 pr-3 font-medium">Method</th>
                {SCORE_COLUMNS.map((c) => <th key={c.key} className="py-1 pr-3 font-medium">{c.label}</th>)}
              </tr>
            </thead>
            <tbody className="tabular-nums text-slate-300">
              {rows.map((r) => (
                <tr key={r.label} className="border-t border-white/5">
                  <td className={`py-1 pr-3 font-medium ${r.tone}`}>{r.label}</td>
                  {SCORE_COLUMNS.map((c) => <td key={c.key} className="py-1 pr-3">{fmt(r.metrics[c.key], 3)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <p className="mb-1.5 text-[11px] text-slate-500">
            Where this scene is{ml?.maskUrl ? " — the model's water mask is laid over the map in violet" : ""}
            {sample.source === "synthetic" && " (a simulated scene placed at a real location; the map underneath is real, the flood is not)"}
          </p>
          <HazardMap bounds={geo.bounds} overlayUrl={mediaUrl(ml?.maskUrl)} footprintColor={sample.source === "synthetic" ? "#fbbf24" : "#a78bfa"} />
        </div>
        <Notes
          lines={[
            result.notes.task,
            result.notes.classical,
            result.notes.area,
            result.model.available && `Model inference: ${fmt(result.model.inferenceMs, 0)} ms on ${String(result.model.device).toUpperCase()}.`,
          ]}
        />
      </div>
    </div>
  );
}

export default function FloodModelPanel({ status }) {
  const { data: catalog, isLoading } = useAsync(() => fetchFloodSamples(), []);
  const [source, setSource] = useState("real");
  const [chosen, setChosen] = useState({ real: "", synthetic: "" });
  const [run, setRun] = useState({ status: "idle", data: null, error: null });
  const fileInput = useRef(null);

  const list = useMemo(() => catalog?.[source] ?? [], [catalog, source]);
  // Default to a held-out test sample (an Indian one when there is one): the honest demo.
  const defaultId = useMemo(() => {
    const preferred = list.find((s) => s.split === "test" && (s.event === "India" || s.floodLabel === "flood")) ?? list.find((s) => s.split === "test") ?? list[0];
    return preferred?.sampleId ?? "";
  }, [list]);
  const sampleId = chosen[source] || defaultId;

  const execute = async (job) => {
    setRun({ status: "loading", data: null, error: null });
    try {
      setRun({ status: "done", data: await job(), error: null });
    } catch (err) {
      setRun({ status: "idle", data: null, error: err.name === "AbortError" ? "The request took too long and was cancelled." : err.message });
    }
  };

  const test = status?.testMetrics;
  if (isLoading) return <LoadingSpinner label="Loading flood samples" />;

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-slate-400">
        A U-Net trained on real Sentinel-1 radar chips with human-drawn water masks marks which pixels are under water.
        Radar sees through monsoon cloud, which is why it is used for flood mapping. Each result is shown next to the
        classical threshold method and, where it exists, the label.
      </p>

      <ModelCard
        status={status}
        fallbackName="backscatter threshold"
        scoreCaption={`Real held-out test chips (${test?.real?.chips ?? "—"} Sen1Floods11 chips never used in training) — water class, higher is better.`}
        columns={SCORE_COLUMNS}
        rows={[
          { label: "Trained U-Net — all test chips", tone: "text-violet-200", metrics: test?.real },
          { label: `Trained U-Net — India chips only (${test?.realIndiaOnly?.chips ?? "—"})`, tone: "text-violet-200", metrics: test?.realIndiaOnly },
          { label: "VH threshold (tuned on training chips)", tone: "text-cyan-200", metrics: test?.baselineGlobalThresholdVH },
          { label: "Dataset's own Otsu mask", tone: "text-slate-300", metrics: test?.baselineDatasetOtsu },
        ]}
        syntheticLine={status?.syntheticMetrics?.model && `On ${status.syntheticMetrics.model.chips} synthetic India test chips the model scores IoU ${fmt(status.syntheticMetrics.model.iou, 3)} (threshold method ${fmt(status.syntheticMetrics.baselineGlobalThresholdVH?.iou, 3)}). That is agreement with a simulator, kept separate from the real scores above.`}
      />

      {catalog?.unreachable ? (
        <p className="rounded-xl bg-navy-800/50 p-3 text-xs text-slate-500 ring-1 ring-white/5">Flood samples are unavailable: backend unreachable.</p>
      ) : (
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex shrink-0 gap-1 rounded-lg bg-navy-800/60 p-1 ring-1 ring-white/10">
            {[{ key: "real", label: `Real radar (${catalog?.real?.length ?? 0})` }, { key: "synthetic", label: `Synthetic India (${catalog?.synthetic?.length ?? 0})` }].map((s) => (
              <button
                key={s.key}
                onClick={() => setSource(s.key)}
                className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${source === s.key ? (s.key === "real" ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-500/15 text-amber-300") : "text-slate-400 hover:bg-white/5"}`}
              >
                {s.label}
              </button>
            ))}
          </div>
          <label className="min-w-0 flex-1 basis-64 text-[11px] text-slate-500">
            {source === "real" ? "Real Sentinel-1 chip (Sen1Floods11)" : "Synthetic India scene (simulation)"}
            <select value={sampleId} onChange={(e) => setChosen((c) => ({ ...c, [source]: e.target.value }))} className={SELECT_CLASS} disabled={!list.length}>
              {list.map((s) => (
                <option key={s.sampleId} value={s.sampleId}>
                  {source === "real" ? `${s.title} · ${s.acquired}` : `${s.state}: ${s.title} · ${s.month}`} · {splitLabel(s.split)}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => execute(() => runFloodPrediction(sampleId))} disabled={!sampleId || run.status === "loading"} className={RUN_BUTTON}>
            {run.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            {run.status === "loading" ? "Running…" : status?.available ? "Map flood water" : "Run threshold method"}
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".tif,.tiff"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) execute(() => uploadFloodImage(file));
            }}
          />
          <button onClick={() => fileInput.current?.click()} disabled={run.status === "loading"} className="flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-slate-300 ring-1 ring-white/10 transition-colors hover:bg-white/5 disabled:opacity-40">
            <UploadCloud className="h-3.5 w-3.5" /> Upload Sentinel-1 GeoTIFF
          </button>
        </div>
      )}
      {!catalog?.unreachable && (
        <p className="text-[11px] text-slate-600">
          {source === "real" ? catalog?.realSource : catalog?.syntheticSource} Uploads must be 2-band VV/VH backscatter in dB.
        </p>
      )}

      {run.error && <p className="text-xs text-red-400">{run.error}</p>}
      {run.data && <FloodResult key={run.data.resultId} result={run.data} />}
    </div>
  );
}
