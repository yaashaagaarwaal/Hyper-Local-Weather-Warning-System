import { TriangleAlert, RefreshCw } from "lucide-react";
import GlassCard from "./GlassCard";

// Shown when a real API call (weather) fails — explains what happened and
// offers a retry, instead of leaving the page stuck on a loading skeleton.
export default function ErrorBanner({ message = "Unable to load live weather data.", onRetry }) {
  return (
    <GlassCard className="flex flex-wrap items-center justify-between gap-3 border border-red-500/20 bg-red-500/5 p-4">
      <div className="flex items-center gap-2 text-sm text-red-300">
        <TriangleAlert className="h-4 w-4 shrink-0" />
        {message}
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-1.5 rounded-lg bg-red-500/10 px-3 py-1.5 text-xs font-semibold text-red-300 ring-1 ring-red-500/25 transition-colors hover:bg-red-500/20"
        >
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </button>
      )}
    </GlassCard>
  );
}
