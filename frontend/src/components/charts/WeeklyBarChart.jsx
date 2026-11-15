import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { RISK_LEVELS } from "../../data/riskData";

function riskColor(score) {
  if (score >= 80) return RISK_LEVELS.severe.color;
  if (score >= 55) return RISK_LEVELS.high.color;
  if (score >= 30) return RISK_LEVELS.moderate.color;
  return RISK_LEVELS.low.color;
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div className="rounded-xl border border-white/10 bg-navy-850 px-3.5 py-2.5 text-xs shadow-xl">
      <p className="mb-1 font-semibold text-slate-200">{label}</p>
      <p className="text-slate-400">
        Risk score: <span className="font-semibold text-slate-100">{p.value}</span>
      </p>
    </div>
  );
}

export default function WeeklyBarChart({ data, height = 240 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="#1b2740" strokeDasharray="4 4" vertical={false} />
        <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 12 }} axisLine={{ stroke: "#1b2740" }} tickLine={false} />
        <YAxis tick={{ fill: "#64748b", fontSize: 12 }} axisLine={false} tickLine={false} width={32} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(148,163,184,0.06)" }} />
        <Bar dataKey="riskScore" radius={[6, 6, 0, 0]} maxBarSize={36}>
          {data.map((entry) => (
            <Cell key={entry.day} fill={riskColor(entry.riskScore)} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
