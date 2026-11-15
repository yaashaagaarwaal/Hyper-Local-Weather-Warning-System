import {
  AreaChart,
  Area,
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
          <span className="ml-auto font-semibold text-slate-100">
            {p.value}
            {p.dataKey === "rainfall" ? " mm" : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function PredictionChart({ data, height = 260 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <defs>
          <linearGradient id="rainGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.45} />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f97316" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#f97316" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#1b2740" strokeDasharray="4 4" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: "#64748b", fontSize: 12 }}
          axisLine={{ stroke: "#1b2740" }}
          tickLine={false}
        />
        <YAxis tick={{ fill: "#64748b", fontSize: 12 }} axisLine={false} tickLine={false} width={36} />
        <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#33445f", strokeWidth: 1 }} />
        <Legend
          wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
          formatter={(value) => <span className="text-slate-400">{value}</span>}
        />
        <Area
          type="monotone"
          dataKey="rainfall"
          name="Rainfall (mm)"
          stroke="#22d3ee"
          strokeWidth={2.5}
          fill="url(#rainGradient)"
          activeDot={{ r: 5 }}
        />
        <Area
          type="monotone"
          dataKey="riskScore"
          name="Risk Trend"
          stroke="#f97316"
          strokeWidth={2.5}
          fill="url(#riskGradient)"
          activeDot={{ r: 5 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
