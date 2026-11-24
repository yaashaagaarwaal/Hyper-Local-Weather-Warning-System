// Lightning-detection panel: a trained U-Net that estimates where lightning
// is likely from two infrared cloud-top images, shown next to the classical
// cold-cloud threshold. Three clearly separated kinds of input: real GOES-16
// over the Americas (checked against real GLM flashes), synthetic India
// scenes (checked against simulated flashes) and real INSAT-3DR over India
// (no ground truth -- unverified). Nothing here is computed in the browser.
import { useMemo, useState } from "react";
import { AlertTriangle, Loader2, Play } from "lucide-react";
import LoadingSpinner from "../common/LoadingSpinner";
import HazardMap from "./HazardMap";
import {
  CLASSICAL_BADGE, ML_BADGE, REAL_BADGE, RUN_BUTTON, SELECT_CLASS, SYNTHETIC_BADGE, UNVERIFIED_BADGE,
  Column, ModelCard, Notes, ProvenanceBanner, Stat, fmt, formatUtc, sourceBadge, splitLabel,
} from "./shared";
import { useAsync } from "../../hooks/useAsync";
import { mediaUrl } from "../../services/imageryService";
import { fetchLightningSamples, runLightningPrediction } from "../../services/hazardModelService";

const SOURCES = [
  { key: "synthetic", label: "Synthetic India", active: "bg-amber-500/15 text-amber-300" },
  { key: "real", label: "Real GOES-16 (Americas)", active: "bg-emerald-500/15 text-emerald-300" },
  { key: "insat", label: "Real INSAT-3DR (India, unverified)", active: "bg-rose-500/15 text-rose-300" },
];
const SCOPE_LABELS = {
  heldOutTestBlocks: "Held-out test blocks only (the honest check)",
  allCells: "Whole image",
};
const SCORE_COLUMNS = [
  { key: "iou", label: "CSI" }, { key: "recall", label: "Detection rate" }, { key: "falseAlarmRatio", label: "False-alarm ratio" }, { key: "precision", label: "Precision" },
];

function LightningResult({ result }) {
  const [view, setView] = useState("overlay");
  const { ml, classical, verification, sample, geo } = result;
  const badge = sourceBadge(sample.source);
  const hasTruth = Boolean(verification);
  const synthetic = sample.source === "synthetic";
  const mlUrl = !ml ? null : view === "probability" ? ml.probabilityUrl : ml.overlayUrl;

  return (
    <div className="space-y-4">
      <ProvenanceBanner sample={sample} />

      <div className="flex min-w-0 flex-wrap items-center gap-2 text-[11px] text-slate-400">
        <span className={badge.className}>{badge.label}</span>
        {sample.place && <span>{sample.place}, {sample.state} · {sample.scenario} · {sample.monthName} ({sample.season})</span>}
        {sample.timestamp && <span>{formatUtc(sample.timestamp)}</span>}
        {sample.regionLabel && <span>{sample.regionLabel}</span>}
        <span>{geo.widthCells}×{geo.heightCells} cells of {geo.cellKm} km</span>
        <span className={UNVERIFIED_BADGE}>Not validated for India</span>
      </div>

      <dl className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label="Lightning likely?" value={ml ? (ml.lightningDetected ? "Yes" : "No") : "—"} hint={ml ? `${ml.flaggedCells} cells above the ${fmt(ml.threshold, 2)} threshold` : "trained model unavailable"} />
        <Stat label="Area flagged" value={ml ? `${ml.flaggedAreaKm2.toLocaleString()} km²` : `${classical.flaggedAreaKm2.toLocaleString()} km²`} hint={ml ? `threshold method: ${classical.flaggedAreaKm2.toLocaleString()} km²` : "threshold method"} />
        <Stat label="Highest probability" value={ml?.maxProbability != null ? `${fmt(100 * ml.maxProbability, 0)}%` : "—"} hint={ml?.meanFlaggedConfidencePct != null ? `mean over flagged cells ${fmt(ml.meanFlaggedConfidencePct, 0)}%` : undefined} />
        <Stat label="Coldest cloud top" value={`${fmt(result.cloud.minBtK, 0)} K`} hint={`${fmt(result.cloud.coldCloudPct, 1)}% of the image below 235 K`} />
      </dl>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {[{ key: "overlay", label: "Lightning overlay" }, { key: "probability", label: "Probability map" }].map((v) => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            disabled={v.key === "probability" && !ml}
            className={`rounded-lg px-3 py-1.5 font-medium ring-1 ring-white/10 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${v.key === view ? "bg-white/10 text-slate-100" : "text-slate-400 hover:bg-white/5"}`}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Column badge="Trained ML model" badgeClass={ML_BADGE} title={view === "probability" ? "flash probability per 8 km cell" : "cells where lightning is likely"} url={mediaUrl(mlUrl)} caption={`No trained-model output: ${result.model.reason ?? "model unavailable"}`} />
        <Column badge="Classical threshold" badgeClass={CLASSICAL_BADGE} title={`cloud top colder than ${fmt(classical.thresholdK, 0)} K (${classical.thresholdSource})`} url={mediaUrl(classical.overlayUrl)} caption="No classical output" />
        <Column
          badge={hasTruth ? (synthetic ? "Simulated lightning" : "Real GLM lightning") : "No ground truth"}
          badgeClass={hasTruth ? (synthetic ? SYNTHETIC_BADGE : REAL_BADGE) : UNVERIFIED_BADGE}
          title={hasTruth ? `${verification.truthCells} cells with a flash within ±5 min` : "infrared image only"}
          url={mediaUrl(hasTruth ? result.images.truthUrl : result.images.infraredUrl)}
          caption=""
        >
          <p className="text-[11px] leading-snug text-slate-500">
            {hasTruth
              ? `Outlined in yellow on both results.${result.truthFlashes ? ` ${result.truthFlashes.count.toLocaleString()} ${synthetic ? "simulated" : "detected"} flashes.` : ""}`
              : "Nothing here can be checked: no lightning sensor data exists for this image."}
          </p>
        </Column>
      </div>

      {hasTruth && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-left text-[11px]">
            <caption className="mb-1.5 text-left text-[11px] text-slate-500">
              {synthetic
                ? "Checked against SIMULATED flashes — agreement with a simulator, not real-world accuracy."
                : "Checked against real GLM flashes for this time (Americas). CSI = hits ÷ (hits + misses + false alarms)."}
            </caption>
            <thead className="text-slate-500">
              <tr>
                <th className="py-1 pr-3 font-medium">Scored on</th>
                <th className="py-1 pr-3 font-medium">Method</th>
                {SCORE_COLUMNS.map((c) => <th key={c.key} className="py-1 pr-3 font-medium">{c.label}</th>)}
              </tr>
            </thead>
            <tbody className="tabular-nums text-slate-300">
              {Object.entries(verification.scopes).flatMap(([scope, s]) =>
                [{ key: "ml", label: "Trained U-Net", tone: "text-violet-200" }, { key: "classical", label: "Cold-cloud threshold", tone: "text-cyan-200" }]
                  .filter((m) => s[m.key])
                  .map((m, i) => (
                    <tr key={`${scope}-${m.key}`} className="border-t border-white/5">
                      <td className="py-1 pr-3 text-slate-400">{i === 0 ? `${SCOPE_LABELS[scope] ?? scope} (${s.truthCells} lightning cells)` : ""}</td>
                      <td className={`py-1 pr-3 font-medium ${m.tone}`}>{m.label}</td>
                      {SCORE_COLUMNS.map((c) => <td key={c.key} className="py-1 pr-3">{fmt(s[m.key][c.key], 3)}</td>)}
                    </tr>
                  ))
              )}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <p className="mb-1.5 text-[11px] text-slate-500">
            {ml?.locations?.length
              ? `Where lightning is likely — the ${ml.locations.length} highest-probability cells (violet${result.truthFlashes ? `; yellow dots = ${synthetic ? "simulated" : "real"} flashes` : ""})`
              : "Scene location — no cell is above the model's threshold"}
            {synthetic && ". The map underneath is real; the storm is simulated."}
          </p>
          <HazardMap
            bounds={geo.bounds}
            overlayUrl={geo.mapOverlaySupported ? mediaUrl(ml?.mapOverlayUrl) : null}
            points={ml?.locations ?? []}
            truthPoints={(result.truthFlashes?.points ?? []).slice(0, 300)}
            footprintColor={synthetic ? "#fbbf24" : sample.source === "real-insat" ? "#fb7185" : "#a78bfa"}
          />
        </div>
        <Notes
          lines={[
            result.notes.task,
            result.notes.classical,
            ml?.locationsNote,
            result.model.available && `Model inference: ${fmt(result.model.inferenceMs, 0)} ms on ${String(result.model.device).toUpperCase()}.`,
          ]}
        />
      </div>
    </div>
  );
}

export default function LightningModelPanel({ status }) {
  const { data: catalog, isLoading } = useAsync(() => fetchLightningSamples(), []);
  const [source, setSource] = useState("synthetic");
  const [chosen, setChosen] = useState({ real: "", synthetic: "", insat: "" });
  const [region, setRegion] = useState("north_india");
  const [run, setRun] = useState({ status: "idle", data: null, error: null });

  const list = useMemo(() => catalog?.[source] ?? [], [catalog, source]);
  const defaultId = useMemo(() => {
    const preferred = list.find((s) => s.split === "test" && s.lightningLabel !== "no-lightning") ?? list.find((s) => s.split === "test") ?? list[0];
    return preferred?.sampleId ?? "";
  }, [list]);
  const sampleId = chosen[source] || defaultId;

  const start = async () => {
    if (!sampleId) return;
    setRun({ status: "loading", data: null, error: null });
    try {
      setRun({ status: "done", data: await runLightningPrediction(sampleId, region), error: null });
    } catch (err) {
      setRun({ status: "idle", data: null, error: err.name === "AbortError" ? "The request took too long and was cancelled." : err.message });
    }
  };

  const test = status?.testMetrics;
  if (isLoading) return <LoadingSpinner label="Loading lightning samples" />;

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-slate-400">
        A U-Net estimates where lightning is likely from two infrared cloud-top images taken 30 minutes apart — how
        cold the storm tops are and how fast they are cooling. It was trained on real GOES-16 imagery and real GLM
        lightning detections over the Americas.
      </p>

      <div className="flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/5 p-3 text-xs text-rose-200/90">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
        <p className="min-w-0 leading-snug">
          <span className="font-semibold">Not validated for India.</span> No lightning ground truth for India is in this
          project (GOES does not see India; NASA&apos;s ISS-LIS data needs an account). The Indian scenes below are either
          simulations or real INSAT-3DR images with nothing to check the answer against.
        </p>
      </div>

      <ModelCard
        status={status}
        fallbackName="cold-cloud threshold"
        scoreCaption="Real held-out test cells (GOES-16 + GLM, Americas; later times in map blocks never trained on). CSI: higher is better; false-alarm ratio: lower is better."
        columns={SCORE_COLUMNS}
        rows={[
          { label: "Trained U-Net", tone: "text-violet-200", metrics: test?.real },
          { label: "Cold-cloud threshold (tuned on training cells)", tone: "text-cyan-200", metrics: test?.baseline },
        ]}
        syntheticLine={status?.syntheticMetrics?.model && `On the synthetic India test scenes the model scores CSI ${fmt(status.syntheticMetrics.model.iou, 3)} (threshold method ${fmt(status.syntheticMetrics.baseline?.iou, 3)}). That is agreement with a simulator, kept separate from the real scores above, and is not accuracy over India.`}
      />

      {catalog?.unreachable ? (
        <p className="rounded-xl bg-navy-800/50 p-3 text-xs text-slate-500 ring-1 ring-white/5">Lightning samples are unavailable: backend unreachable.</p>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex min-w-0 flex-wrap gap-1 rounded-lg bg-navy-800/60 p-1 ring-1 ring-white/10">
              {SOURCES.map((s) => (
                <button
                  key={s.key}
                  onClick={() => setSource(s.key)}
                  className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${source === s.key ? s.active : "text-slate-400 hover:bg-white/5"}`}
                >
                  {s.label} ({catalog?.[s.key]?.length ?? 0})
                </button>
              ))}
            </div>
            <label className="min-w-0 flex-1 basis-64 text-[11px] text-slate-500">
              {source === "synthetic" ? "Synthetic India scene (simulation)" : source === "real" ? "Real GOES-16 image time" : "Real INSAT-3DR image time"}
              <select value={sampleId} onChange={(e) => setChosen((c) => ({ ...c, [source]: e.target.value }))} className={SELECT_CLASS} disabled={!list.length}>
                {list.map((s) => (
                  <option key={s.sampleId} value={s.sampleId}>
                    {source === "synthetic" ? `${s.state}: ${s.title} · ${s.month} · ${s.lightningLabel}` : s.title}
                    {s.split ? ` · ${splitLabel(s.split)}` : ""}
                  </option>
                ))}
              </select>
            </label>
            {source === "insat" && (
              <label className="min-w-0 basis-40 text-[11px] text-slate-500">
                Area
                <select value={region} onChange={(e) => setRegion(e.target.value)} className={SELECT_CLASS}>
                  {(catalog?.insatRegions ?? []).map((r) => <option key={r.key} value={r.key}>{r.label}</option>)}
                </select>
              </label>
            )}
            <button onClick={start} disabled={!sampleId || run.status === "loading"} className={RUN_BUTTON}>
              {run.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {run.status === "loading" ? "Running…" : status?.available ? "Detect lightning" : "Run threshold method"}
            </button>
          </div>
          <p className="text-[11px] text-slate-600">
            {source === "synthetic" ? catalog?.syntheticSource : source === "real" ? catalog?.realSource : catalog?.insatSource}
          </p>
        </>
      )}

      {run.error && <p className="text-xs text-red-400">{run.error}</p>}
      {run.data && <LightningResult key={run.data.resultId} result={run.data} />}
    </div>
  );
}
