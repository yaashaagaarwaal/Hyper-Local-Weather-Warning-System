import { useMemo, useState } from "react";
import {
  Satellite, CloudRain, Radar, Info, UploadCloud, Video, Wand2,
  GitCompare, Wind, Film, Loader2,
} from "lucide-react";
import GlassCard from "../components/common/GlassCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import { useAsync } from "../hooks/useAsync";
import {
  fetchImagerySamples, uploadImage, uploadVideo, processImage,
  compareImages, analyzeMotion, extractVideoFrames, generateVideo, mediaUrl,
} from "../services/imageryService";

const PROTOTYPE_BADGE = "rounded-full bg-amber-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-amber-400 ring-1 ring-amber-500/25";
const REAL_BADGE = "rounded-full bg-emerald-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-emerald-400 ring-1 ring-emerald-500/25";

const OPERATIONS = [
  { key: "denoise", label: "Denoise" },
  { key: "contrast", label: "Contrast Enhance (CLAHE)" },
  { key: "cloud_detection", label: "Cloud Detection" },
  { key: "storm_regions", label: "Storm-Region Candidates" },
  { key: "segment_otsu", label: "Segmentation (Otsu)" },
  { key: "segment_kmeans", label: "Segmentation (k-means)" },
];

function SectionHeader({ icon: Icon, title, badge, badgeClass }) {
  return (
    <div className="mb-1 flex min-w-0 flex-wrap items-center justify-between gap-2">
      <div className="flex min-w-0 items-center gap-2">
        <Icon className="h-4 w-4 shrink-0 text-cyan-400" />
        <h3 className="min-w-0 text-sm font-semibold text-slate-200">{title}</h3>
      </div>
      {badge && <span className={`shrink-0 ${badgeClass ?? PROTOTYPE_BADGE}`}>{badge}</span>}
    </div>
  );
}

function OutputImage({ label, url, note }) {
  return (
    <div className="space-y-1.5">
      <img src={mediaUrl(url)} alt={label} className="w-full rounded-lg ring-1 ring-white/10" />
      <p className="text-xs font-medium text-slate-300">{label}</p>
      {note && <p className="text-[11px] leading-snug text-slate-500">{note}</p>}
    </div>
  );
}

// --- Satellite / precipitation sample gallery (real NASA GIBS data) ------

function SampleGallery({ samples, loading, selected, onSelect }) {
  const [filter, setFilter] = useState("all");
  const filtered = samples.filter((s) => filter === "all" || s.category === filter);

  if (loading) return <LoadingSpinner label="Loading NASA GIBS sample set" />;
  if (!samples.length) {
    return <p className="py-4 text-center text-xs text-slate-500">Sample imagery is unavailable right now (backend unreachable).</p>;
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2 text-xs">
        {[
          { key: "all", label: "All" },
          { key: "satellite_truecolor", label: "True-Color Satellite" },
          { key: "precipitation_rate_sequence_frame", label: "Precipitation Sequence" },
        ].map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={`rounded-lg px-2.5 py-1 font-medium ring-1 ring-white/10 transition-colors ${filter === f.key ? "bg-cyan-500/15 text-cyan-300" : "text-slate-400 hover:bg-white/5"
              }`}
          >
            {f.label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {filtered.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={`group overflow-hidden rounded-xl text-left ring-1 transition-all ${selected === s.id ? "ring-2 ring-cyan-400" : "ring-white/10 hover:ring-white/30"
              }`}
          >
            <img src={mediaUrl(s.url)} alt={s.filename} className="aspect-[4/3] w-full object-cover" />
            <div className="space-y-0.5 bg-navy-800/80 p-2">
              <p className="truncate text-[11px] font-medium text-slate-300">
                {s.sensor ?? s.product ?? "GIBS"}
              </p>
              <p className="truncate text-[10px] text-slate-500">
                {s.acquisition_date ?? s.timestamp_utc ?? ""}
              </p>
            </div>
          </button>
        ))}
      </div>
      <p className="flex items-start gap-1.5 text-[11px] leading-snug text-slate-500">
        <Info className="mt-0.5 h-3 w-3 shrink-0" />
        <span className="min-w-0">
          Real NASA GIBS browse imagery (public domain/CC0), Delhi-NCR bbox. See{" "}
          <code className="break-all text-slate-400">dataset/imagery/metadata/SOURCES.md</code> for exact source URLs,
          timestamps and licensing. Click a tile to use it below.
        </span>
      </p>
    </div>
  );
}

// --- Upload + process a single image --------------------------------------

function ProcessPanel({ workingImageId, workingImageLabel, onNewUpload }) {
  const [ops, setOps] = useState(new Set(["denoise", "contrast", "cloud_detection"]));
  const [state, setState] = useState({ status: "idle", data: null, error: null });
  const [uploadState, setUploadState] = useState({ status: "idle", error: null });

  const toggleOp = (key) => {
    setOps((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const handleFile = async (file) => {
    if (!file) return;
    setUploadState({ status: "loading", error: null });
    try {
      const result = await uploadImage(file);
      onNewUpload(result.assetId, result.originalFilename, result.url);
      setUploadState({ status: "idle", error: null });
    } catch (err) {
      setUploadState({ status: "idle", error: err.message });
    }
  };

  const runProcessing = async () => {
    if (!workingImageId || ops.size === 0) return;
    setState({ status: "loading", data: null, error: null });
    try {
      const result = await processImage(workingImageId, [...ops]);
      setState({ status: "idle", data: result, error: null });
    } catch (err) {
      setState({ status: "idle", data: null, error: err.message });
    }
  };

  return (
    <div className="space-y-4">
      <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-white/15 bg-navy-800/40 px-4 py-6 text-sm text-slate-400 transition-colors hover:border-cyan-500/40 hover:text-cyan-300">
        <UploadCloud className="h-4 w-4" />
        {uploadState.status === "loading" ? "Uploading…" : "Upload an image to process (or pick a sample above)"}
        <input type="file" accept="image/*" className="hidden" onChange={(e) => handleFile(e.target.files?.[0])} />
      </label>
      {uploadState.error && <p className="text-xs text-red-400">{uploadState.error}</p>}

      <div className="flex flex-wrap items-center gap-3 rounded-xl bg-navy-800/40 p-3 ring-1 ring-white/5">
        <span className="text-xs text-slate-500">Selected image:</span>
        <span className="text-xs font-medium text-cyan-300">{workingImageLabel ?? "none — pick a sample or upload one"}</span>
      </div>

      <div className="flex flex-wrap gap-2">
        {OPERATIONS.map((op) => (
          <button
            key={op.key}
            onClick={() => toggleOp(op.key)}
            className={`rounded-lg px-2.5 py-1.5 text-xs font-medium ring-1 ring-white/10 transition-colors ${ops.has(op.key) ? "bg-cyan-500/15 text-cyan-300" : "text-slate-400 hover:bg-white/5"
              }`}
          >
            {op.label}
          </button>
        ))}
      </div>

      <button
        onClick={runProcessing}
        disabled={!workingImageId || ops.size === 0 || state.status === "loading"}
        className="flex items-center gap-2 rounded-lg bg-cyan-500/15 px-4 py-2 text-xs font-semibold text-cyan-300 ring-1 ring-cyan-500/30 transition-colors hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {state.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
        Run Processing
      </button>

      {state.error && <p className="text-xs text-red-400">Processing failed: {state.error}</p>}

      {state.data && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {state.data.outputs.map((o) => (
              <OutputImage key={o.operation} label={OPERATIONS.find((op) => op.key === o.operation)?.label ?? o.operation} url={o.url} note={o.note} />
            ))}
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            {state.data.stats.cloudCoveragePct != null && (
              <Stat label="Cloud Coverage" value={`${state.data.stats.cloudCoveragePct}%`} />
            )}
            {state.data.stats.otsuForegroundPct != null && (
              <Stat label="Otsu Foreground" value={`${state.data.stats.otsuForegroundPct}%`} />
            )}
            {state.data.stats.stormCandidateRegions && (
              <Stat label="Storm-Region Candidates" value={state.data.stats.stormCandidateRegions.length} />
            )}
            <Stat label="Mean Brightness" value={state.data.stats.meanBrightness} />
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg bg-navy-800/50 p-2.5 ring-1 ring-white/5">
      <p className="text-[10px] text-slate-500">{label}</p>
      <p className="text-sm font-bold text-white">{value}</p>
    </div>
  );
}

// --- Compare + motion (optical flow) between two images --------------------

function ComparePanel({ options, beforeId, afterId, setBeforeId, setAfterId }) {
  const [compareState, setCompareState] = useState({ status: "idle", data: null, error: null });
  const [motionState, setMotionState] = useState({ status: "idle", data: null, error: null });

  const runCompare = async () => {
    if (!beforeId || !afterId) return;
    setCompareState({ status: "loading", data: null, error: null });
    try {
      const result = await compareImages(beforeId, afterId);
      setCompareState({ status: "idle", data: result, error: null });
    } catch (err) {
      setCompareState({ status: "idle", data: null, error: err.message });
    }
  };

  const runMotion = async () => {
    if (!beforeId || !afterId) return;
    setMotionState({ status: "loading", data: null, error: null });
    try {
      const result = await analyzeMotion(beforeId, afterId);
      setMotionState({ status: "idle", data: result, error: null });
    } catch (err) {
      setMotionState({ status: "idle", data: null, error: err.message });
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-[11px] text-slate-500">Image A (before)</label>
          <select
            value={beforeId ?? ""}
            onChange={(e) => setBeforeId(e.target.value)}
            className="w-full rounded-lg bg-navy-800/60 px-3 py-2 text-xs text-slate-200 ring-1 ring-white/10"
          >
            <option value="">— select —</option>
            {options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-[11px] text-slate-500">Image B (after)</label>
          <select
            value={afterId ?? ""}
            onChange={(e) => setAfterId(e.target.value)}
            className="w-full rounded-lg bg-navy-800/60 px-3 py-2 text-xs text-slate-200 ring-1 ring-white/10"
          >
            <option value="">— select —</option>
            {options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={runCompare}
          disabled={!beforeId || !afterId || compareState.status === "loading"}
          className="flex items-center gap-2 rounded-lg bg-cyan-500/15 px-4 py-2 text-xs font-semibold text-cyan-300 ring-1 ring-cyan-500/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {compareState.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <GitCompare className="h-3.5 w-3.5" />}
          Compare (Pixel Diff)
        </button>
        <button
          onClick={runMotion}
          disabled={!beforeId || !afterId || motionState.status === "loading"}
          className="flex items-center gap-2 rounded-lg bg-indigo-500/15 px-4 py-2 text-xs font-semibold text-indigo-300 ring-1 ring-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {motionState.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wind className="h-3.5 w-3.5" />}
          Analyze Motion (Optical Flow)
        </button>
      </div>

      {compareState.error && <p className="text-xs text-red-400">{compareState.error}</p>}
      {compareState.data && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-[200px_1fr]">
          <img src={mediaUrl(compareState.data.heatmapUrl)} alt="diff heatmap" className="rounded-lg ring-1 ring-white/10" />
          <div className="space-y-1 text-xs text-slate-400">
            <p>Mean absolute pixel difference: <span className="font-semibold text-slate-200">{compareState.data.meanAbsDiff}</span></p>
            <p>Changed pixels: <span className="font-semibold text-slate-200">{compareState.data.changedPixelPct}%</span></p>
          </div>
        </div>
      )}

      {motionState.error && <p className="text-xs text-red-400">{motionState.error}</p>}
      {motionState.data && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-[200px_1fr]">
          <img src={mediaUrl(motionState.data.visualizationUrl)} alt="optical flow" className="rounded-lg ring-1 ring-white/10" />
          <div className="space-y-1 text-xs text-slate-400">
            <p>Mean motion magnitude: <span className="font-semibold text-slate-200">{motionState.data.meanMagnitudePx} px</span></p>
            <p>Dominant direction: <span className="font-semibold text-slate-200">{motionState.data.dominantDirectionLabel} ({motionState.data.dominantDirectionDeg}°)</span></p>
            <p className="text-[11px] leading-snug text-slate-500">{motionState.data.note}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// --- Upload + process a video ----------------------------------------------

function VideoPanel({ onFramesExtracted }) {
  const [upload, setUpload] = useState({ status: "idle", data: null, error: null });
  const [frames, setFrames] = useState({ status: "idle", data: null, error: null });

  const handleFile = async (file) => {
    if (!file) return;
    setUpload({ status: "loading", data: null, error: null });
    setFrames({ status: "idle", data: null, error: null });
    try {
      const result = await uploadVideo(file);
      setUpload({ status: "idle", data: result, error: null });
    } catch (err) {
      setUpload({ status: "idle", data: null, error: err.message });
    }
  };

  const runExtract = async () => {
    if (!upload.data) return;
    setFrames({ status: "loading", data: null, error: null });
    try {
      const result = await extractVideoFrames(upload.data.assetId, 8);
      setFrames({ status: "idle", data: result, error: null });
      onFramesExtracted(result.frames);
    } catch (err) {
      setFrames({ status: "idle", data: null, error: err.message });
    }
  };

  return (
    <div className="space-y-4">
      <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-white/15 bg-navy-800/40 px-4 py-6 text-sm text-slate-400 transition-colors hover:border-cyan-500/40 hover:text-cyan-300">
        <Video className="h-4 w-4" />
        {upload.status === "loading" ? "Uploading…" : "Upload a short video (mp4/avi/mov/mkv, max 60MB)"}
        <input type="file" accept="video/*" className="hidden" onChange={(e) => handleFile(e.target.files?.[0])} />
      </label>
      {upload.error && <p className="text-xs text-red-400">{upload.error}</p>}

      {upload.data && (
        <div className="space-y-3">
          <video controls src={mediaUrl(upload.data.url)} className="w-full max-w-md rounded-lg ring-1 ring-white/10" />
          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <Stat label="Frames" value={upload.data.frameCount} />
            <Stat label="FPS" value={upload.data.fps ?? "—"} />
            <Stat label="Duration" value={upload.data.durationSeconds != null ? `${upload.data.durationSeconds}s` : "—"} />
            <Stat label="Size" value={`${(upload.data.sizeBytes / 1024).toFixed(0)} KB`} />
          </div>
          <button
            onClick={runExtract}
            disabled={frames.status === "loading"}
            className="flex items-center gap-2 rounded-lg bg-cyan-500/15 px-4 py-2 text-xs font-semibold text-cyan-300 ring-1 ring-cyan-500/30 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {frames.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Film className="h-3.5 w-3.5" />}
            Extract Frames
          </button>
        </div>
      )}
      {frames.error && <p className="text-xs text-red-400">{frames.error}</p>}
      {frames.data && (
        <div className="space-y-2">
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
            {frames.data.frames.map((f) => (
              <div key={f.imageId} className="space-y-1">
                <img src={mediaUrl(f.url)} alt={`frame ${f.index}`} className="aspect-square w-full rounded-lg object-cover ring-1 ring-white/10" />
                <p className="text-center text-[10px] text-slate-500">{f.timestampSeconds}s</p>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-500">
            Extracted frames are now usable below in Compare/Motion as "Uploaded: frame_….png".
          </p>
        </div>
      )}
    </div>
  );
}

// --- Time-lapse from the real precipitation sequence ------------------------

function TimelapsePanel({ precipSamples }) {
  const [state, setState] = useState({ status: "idle", data: null, error: null });

  const run = async () => {
    if (precipSamples.length < 2) return;
    setState({ status: "loading", data: null, error: null });
    try {
      const ids = [...precipSamples].sort((a, b) => a.sequence_index - b.sequence_index).map((s) => s.id);
      const result = await generateVideo(ids, 2);
      setState({ status: "idle", data: result, error: null });
    } catch (err) {
      setState({ status: "idle", data: null, error: err.message });
    }
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-500">
        Encodes a real {precipSamples.length}-frame time-lapse directly from the genuinely time-ordered
        GPM IMERG precipitation-rate sequence above (2026-08-05, 07:00–11:30 UTC) — not a synthetic animation.
      </p>
      <button
        onClick={run}
        disabled={precipSamples.length < 2 || state.status === "loading"}
        className="flex items-center gap-2 rounded-lg bg-cyan-500/15 px-4 py-2 text-xs font-semibold text-cyan-300 ring-1 ring-cyan-500/30 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {state.status === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Film className="h-3.5 w-3.5" />}
        Generate Time-Lapse
      </button>
      {state.error && <p className="text-xs text-red-400">{state.error}</p>}
      {state.data && (
        <video controls autoPlay loop src={mediaUrl(state.data.url)} className="w-full max-w-md rounded-lg ring-1 ring-white/10" />
      )}
    </div>
  );
}

// --- Page --------------------------------------------------------------

export default function Imagery() {
  const { data: samples, isLoading: samplesLoading } = useAsync(() => fetchImagerySamples(), []);
  const allSamples = samples ?? [];
  const precipSamples = allSamples.filter((s) => s.category === "precipitation_rate_sequence_frame");

  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [uploadedImages, setUploadedImages] = useState([]); // [{id, label, url}]
  const [beforeId, setBeforeId] = useState(null);
  const [afterId, setAfterId] = useState(null);

  const combinedOptions = useMemo(() => {
    const fromSamples = (samples ?? []).map((s) => ({ id: s.id, label: `Sample: ${s.sensor ?? s.product ?? s.filename}` }));
    const fromUploads = uploadedImages.map((u) => ({ id: u.id, label: `Uploaded: ${u.label}` }));
    return [...fromSamples, ...fromUploads];
  }, [samples, uploadedImages]);

  const addUploadedImage = (id, label, url) => {
    setUploadedImages((prev) => [...prev, { id, label, url }]);
    setSelectedSampleId(id);
  };

  const addExtractedFrames = (frames) => {
    setUploadedImages((prev) => [
      ...prev,
      ...frames.map((f) => ({ id: f.imageId, label: `frame_${f.index}.png`, url: f.url })),
    ]);
  };

  const selectedLabel = combinedOptions.find((o) => o.id === selectedSampleId)?.label ?? null;

  return (
    <div className="space-y-5">
      <GlassCard className="flex items-start gap-3 border border-amber-500/20 bg-amber-500/5 p-4">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        <p className="min-w-0 text-xs leading-snug text-amber-200/90">
          <span className="font-semibold">Prototype image/video processing module.</span> Every detection/
          segmentation operation below is a classical OpenCV technique (thresholding, connected components,
          k-means, dense optical flow) — <span className="font-semibold">none of this is a trained cloud, storm
          or flood-prediction model.</span> See <code className="break-all text-amber-300">dataset/imagery/metadata/SOURCES.md</code> and
          <code className="ml-1 break-all text-amber-300">backend/app/services/imagery/detection.py</code> for exactly why,
          and what real imagery this was built and tested against.
        </p>
      </GlassCard>

      <GlassCard className="p-5">
        <SectionHeader icon={Satellite} title="Satellite & Precipitation Imagery (Real NASA GIBS Data)" badge="Real Data" badgeClass={REAL_BADGE} />
        <SampleGallery samples={allSamples} loading={samplesLoading} selected={selectedSampleId} onSelect={setSelectedSampleId} />
      </GlassCard>

      <GlassCard className="flex items-start gap-3 p-4">
        <Radar className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
        <p className="min-w-0 text-xs leading-snug text-slate-500">
          <span className="font-semibold text-slate-400">Ground radar viewer:</span> no public, unauthenticated
          IMD (India Meteorological Department) radar imagery source was found for Delhi-NCR — see the sources
          table in <code className="break-all text-slate-400">dataset/imagery/metadata/SOURCES.md</code>. Shown above instead:
          GPM IMERG satellite-derived precipitation-rate imagery, which is real but not ground radar, and is
          labeled as such everywhere in this module.
        </p>
      </GlassCard>

      <GlassCard className="p-5">
        <SectionHeader icon={CloudRain} title="Time-Lapse from a Genuine Precipitation Sequence" />
        <TimelapsePanel precipSamples={precipSamples} />
      </GlassCard>

      <GlassCard className="p-5">
        <SectionHeader icon={Wand2} title="Image Upload & Processing" />
        <ProcessPanel workingImageId={selectedSampleId} workingImageLabel={selectedLabel} onNewUpload={addUploadedImage} />
      </GlassCard>

      <GlassCard className="p-5">
        <SectionHeader icon={GitCompare} title="Compare & Motion Analysis" />
        <ComparePanel options={combinedOptions} beforeId={beforeId} afterId={afterId} setBeforeId={setBeforeId} setAfterId={setAfterId} />
      </GlassCard>

      <GlassCard className="p-5">
        <SectionHeader icon={Video} title="Video Upload, Playback & Frame Extraction" />
        <VideoPanel onFramesExtracted={addExtractedFrames} />
      </GlassCard>
    </div>
  );
}
