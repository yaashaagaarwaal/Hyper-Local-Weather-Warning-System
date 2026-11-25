// Small hand-drawn SVG charts for the home page (no charting library: the home
// route stays light, and these need a typographic rather than dashboard look).
import { formatTemperature } from "../../utils/formatters";

// 24-hour rainfall bars from the hourly series.
export function RainBars({ state }) {
  const { data, isLoading } = state;
  if (isLoading && !data) return <div className="hm-skeleton h-24 w-full" aria-hidden="true" />;
  if (!data?.length) return <p className="border-t border-[var(--hm-rule)] py-4 text-[13px] text-[var(--hm-ink-dim)]">Hourly data is unavailable.</p>;
  const points = data.slice(0, 24);
  const max = Math.max(1, ...points.map((p) => p.rainfall ?? 0));
  const total = points.reduce((a, p) => a + (p.rainfall ?? 0), 0);
  const w = 480, h = 88, gap = 3;
  const bw = (w - gap * (points.length - 1)) / points.length;
  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${w} ${h + 18}`} className="block w-full" role="img" aria-label={`Hourly rainfall over the next ${points.length} hours, ${total.toFixed(1)} millimetres in total`}>
        <line x1="0" x2={w} y1={h} y2={h} stroke="var(--hm-rule-strong)" strokeWidth="1" />
        {points.map((p, i) => {
          const v = p.rainfall ?? 0;
          const bh = v > 0 ? Math.max(2, (v / max) * (h - 6)) : 0;
          return <rect key={i} x={i * (bw + gap)} y={h - bh} width={bw} height={bh} fill="var(--hm-sky)" opacity="0.9" />;
        })}
        {[0, 6, 12, 18].filter((i) => points[i]).map((i) => (
          <text key={i} x={i * (bw + gap)} y={h + 14} fontSize="10" fill="var(--hm-ink-faint)" fontFamily="var(--hm-mono)">{points[i].label}</text>
        ))}
      </svg>
      <figcaption className="mt-1 text-[11.5px] text-[var(--hm-ink-faint)]">{total > 0 ? `${total.toFixed(1)} mm expected over ${points.length} h · tallest bar ${max.toFixed(1)} mm` : `No rain expected over the next ${points.length} h`}</figcaption>
    </figure>
  );
}

// A strip of tick marks across the four hazard levels, with a marker at `score` (0-100).
export function LevelScale({ score, labelled = true }) {
  const stops = [
    { to: 25, name: "Low", color: "var(--hm-low)" },
    { to: 50, name: "Moderate", color: "var(--hm-moderate)" },
    { to: 75, name: "High", color: "var(--hm-high)" },
    { to: 100, name: "Severe", color: "var(--hm-severe)" },
  ];
  const s = Math.max(0, Math.min(100, score ?? 0));
  return (
    <div>
      <div className="relative h-[18px]">
        <div className="absolute inset-x-0 top-1/2 flex h-[3px] -translate-y-1/2 gap-[3px]">
          {stops.map((st, i) => <div key={st.name} className="flex-1" style={{ background: st.color, opacity: s > (i ? stops[i - 1].to : 0) ? 1 : 0.32 }} />)}
        </div>
        {score != null && <div className="absolute top-0 h-full w-px bg-[var(--hm-ink)]" style={{ left: `${s}%` }} />}
      </div>
      {labelled && (
        <div className="mt-1.5 flex justify-between font-[var(--hm-mono)] text-[10px] uppercase tracking-wider text-[var(--hm-ink-faint)]">
          {stops.map((st) => <span key={st.name}>{st.name}</span>)}
        </div>
      )}
    </div>
  );
}

export { formatTemperature };
