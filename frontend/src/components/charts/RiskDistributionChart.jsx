import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { RISK_LEVELS } from "../../data/riskData";

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div className="rounded-xl border border-white/10 bg-navy-850 px-3.5 py-2.5 text-xs shadow-xl">
      <p className="font-semibold text-slate-200">
        {p.name}: <span className="text-slate-100">{p.value} zones</span>
      </p>
    </div>
  );
}

export default function RiskDistributionChart({ stats, height = 260 }) {
  const data = Object.values(RISK_LEVELS).map((level) => ({
    name: level.label,
    value: stats[level.key],
    color: level.color,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius={62} outerRadius={92} paddingAngle={3}>
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} fillOpacity={0.85} stroke="none" />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip />} />
        <Legend
          verticalAlign="bottom"
          iconType="circle"
          iconSize={8}
          wrapperStyle={{ fontSize: 12 }}
          formatter={(value) => <span className="text-slate-400">{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
