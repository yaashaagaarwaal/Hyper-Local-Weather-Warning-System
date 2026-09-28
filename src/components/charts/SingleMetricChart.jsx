import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-navy-850 px-3 py-2 text-xs shadow-xl">
      <p className="mb-1 font-semibold text-slate-200">{label}</p>
      <p className="text-slate-300">
        {payload[0].value}
        {unit}
      </p>
    </div>
  );
}

export default function SingleMetricChart({ data, dataKey, color, unit = "", height = 180 }) {
  const gradientId = `grad-${dataKey}`;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.4} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#1b2740" strokeDasharray="4 4" vertical={false} />
        <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#1b2740" }} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} width={30} />
        <Tooltip content={<ChartTooltip unit={unit} />} cursor={{ stroke: "#33445f", strokeWidth: 1 }} />
        <Area type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} fill={`url(#${gradientId})`} activeDot={{ r: 4 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
