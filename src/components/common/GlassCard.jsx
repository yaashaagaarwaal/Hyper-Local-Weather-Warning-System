export default function GlassCard({ children, className = "", solid = false, ...props }) {
  return (
    <div
      className={`rounded-2xl ${solid ? "glass-panel-solid" : "glass-panel"} shadow-xl shadow-black/20 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
