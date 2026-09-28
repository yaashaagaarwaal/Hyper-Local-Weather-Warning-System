import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-navy-850 px-3.5 py-2.5 text-xs shadow-xl">
      <p className="mb-1.5 font-semibold text-slate-200">{label}</p>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 py-0.5">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span className="text-slate-400">{p.name}</span>
          <span className="ml-auto font-semibold text-slate-100">{p.value}</span>
        </div>
      ))}
    </div>
  );
}

// `lines` entries may set `axis: "right"` to plot against a secondary Y-axis —
// use this when combining metrics on very different scales (e.g. °C vs mm).
export default function TrendChart({ data, lines, height = 280 }) {
  const hasRightAxis = lines.some((l) => l.axis === "right");

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: hasRightAxis ? 8 : 12, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="#1b2740" strokeDasharray="4 4" vertical={false} />
        <XAxis dataKey="hour" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={{ stroke: "#1b2740" }} tickLine={false} interval={2} />
        <YAxis
          yAxisId="left"
          tick={{ fill: "#64748b", fontSize: 12 }}
          axisLine={false}
          tickLine={false}
          width={32}
        />
        {hasRightAxis && (
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fill: "#64748b", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={32}
          />
        )}
        <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#33445f", strokeWidth: 1 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} formatter={(value) => <span className="text-slate-400">{value}</span>} />
        {lines.map((line) => (
          <Line
            key={line.dataKey}
            yAxisId={line.axis === "right" ? "right" : "left"}
            type="monotone"
            dataKey={line.dataKey}
            name={line.name}
            stroke={line.color}
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
