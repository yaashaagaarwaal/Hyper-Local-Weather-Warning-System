function levelFromScore(score) {
  if (score >= 80) return { color: "#ef4444", label: "Severe" };
  if (score >= 55) return { color: "#f97316", label: "High" };
  if (score >= 30) return { color: "#eab308", label: "Moderate" };
  return { color: "#22c55e", label: "Low" };
}

export default function RiskIndicator({ label, score, showLabel = true }) {
  const { color, label: levelLabel } = levelFromScore(score);

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-sm">
        <span className="font-medium text-slate-300">{label}</span>
        <div className="flex items-center gap-2">
          {showLabel && (
            <span className="text-xs font-medium" style={{ color }}>
              {levelLabel}
            </span>
          )}
          <span className="font-semibold tabular-nums text-slate-100">{score}</span>
        </div>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-navy-700/80">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${score}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
