export default function Badge({ children, color = "#64748b", bg = "bg-slate-500/15", text = "text-slate-300", className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ring-white/5 ${bg} ${text} ${className}`}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
      {children}
    </span>
  );
}
