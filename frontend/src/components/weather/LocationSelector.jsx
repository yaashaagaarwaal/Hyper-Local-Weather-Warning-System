import { useEffect, useMemo, useRef, useState } from "react";
import { Search, MapPin, X } from "lucide-react";
import { useLocationContext } from "../../context/LocationContext";

export default function LocationSelector({ className = "" }) {
  const { locations, selectedLocation, setSelectedLocationId } = useLocationContext();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  const filtered = useMemo(() => {
    if (!query.trim()) return locations;
    const q = query.toLowerCase();
    return locations.filter(
      (l) => l.name.toLowerCase().includes(q) || l.region.toLowerCase().includes(q)
    );
  }, [locations, query]);

  useEffect(() => {
    function handleClickOutside(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  function handleSelect(loc) {
    setSelectedLocationId(loc.id);
    setQuery("");
    setOpen(false);
  }

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <div className="flex items-center gap-2 rounded-xl border border-white/5 bg-navy-800/60 px-3.5 py-2.5 transition-colors focus-within:border-cyan-500/40 focus-within:bg-navy-800">
        <Search className="h-4 w-4 shrink-0 text-slate-500" />
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          placeholder={`Search location — ${selectedLocation.name}`}
          className="w-full min-w-[140px] bg-transparent text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none"
        />
        {query && (
          <button onClick={() => setQuery("")} className="shrink-0 text-slate-500 hover:text-slate-300">
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {open && (
        <div className="absolute left-0 right-0 top-full z-50 mt-2 max-h-72 overflow-y-auto rounded-xl border border-white/5 bg-navy-850 p-1.5 shadow-2xl shadow-black/40 scrollbar-thin animate-fade-in">
          {filtered.length === 0 ? (
            <p className="px-3 py-4 text-center text-sm text-slate-500">No locations found</p>
          ) : (
            filtered.map((loc) => (
              <button
                key={loc.id}
                onClick={() => handleSelect(loc)}
                className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
                  loc.id === selectedLocation.id
                    ? "bg-cyan-500/10 text-cyan-300"
                    : "text-slate-300 hover:bg-white/5"
                }`}
              >
                <MapPin className="h-3.5 w-3.5 shrink-0 text-slate-500" />
                <span className="font-medium">{loc.name}</span>
                <span className="ml-auto text-xs text-slate-500">{loc.region}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
