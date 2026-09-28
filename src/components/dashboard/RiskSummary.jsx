import GlassCard from "../common/GlassCard";
import RiskIndicator from "../common/RiskIndicator";
import Skeleton from "../common/Skeleton";

export default function RiskSummary({ summary, loading }) {
  return (
    <GlassCard className="p-5">
      <h3 className="mb-4 text-sm font-semibold text-slate-200">Risk Summary</h3>
      <div className="space-y-4">
        {loading
          ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-8 w-full" />)
          : summary.map((item) => <RiskIndicator key={item.key} label={item.label} score={item.score} />)}
      </div>
    </GlassCard>
  );
}
