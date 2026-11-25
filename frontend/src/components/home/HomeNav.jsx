// Minimal top navigation for the home page: Home | Live Map | Alerts | About | Contact,
// plus a quiet route into the full application.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { SITE } from "../../config/site";

export function scrollToSection(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

function Mark() {
  // A planet limb with a single orbit line: the whole logo.
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true" className="shrink-0">
      <circle cx="11" cy="11" r="8.5" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <path d="M2.5 11.6c3.2-2.6 8.4-3.6 17-1.2" fill="none" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

export default function HomeNav() {
  const [solid, setSolid] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setSolid(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const jump = (id) => (e) => {
    e.preventDefault();
    setOpen(false);
    scrollToSection(id);
  };
  const top = (e) => {
    e.preventDefault();
    setOpen(false);
    window.scrollTo({ top: 0, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };

  const links = (
    <>
      <a href="/" onClick={top} className="hm-nav-link">Home</a>
      <Link to="/risk-map" className="hm-nav-link" onClick={() => setOpen(false)}>Live Map</Link>
      <Link to="/alerts" className="hm-nav-link" onClick={() => setOpen(false)}>Alerts</Link>
      <a href="#about" onClick={jump("about")} className="hm-nav-link">About</a>
      <a href="#contact" onClick={jump("contact")} className="hm-nav-link">Contact</a>
    </>
  );

  return (
    <header
      className="fixed inset-x-0 top-0 z-50 transition-colors duration-300"
      style={{
        background: solid || open ? "rgba(5,8,15,0.86)" : "transparent",
        backdropFilter: solid || open ? "blur(10px)" : "none",
        borderBottom: `1px solid ${solid || open ? "var(--hm-rule)" : "transparent"}`,
      }}
    >
      <div className="hm-wrap flex h-16 items-center justify-between">
        <a href="/" onClick={top} className="flex items-center gap-2.5" aria-label={`${SITE.name} home`}>
          <Mark />
          <span className="hm-display text-[26px] leading-none">{SITE.name}</span>
        </a>

        <nav aria-label="Primary" className="hidden items-center gap-9 md:flex">{links}</nav>

        <div className="flex items-center gap-5">
          <Link to="/dashboard" className="hm-link hidden text-[13px] sm:inline-block">Open dashboard</Link>
          <button
            type="button" className="grid h-10 w-10 place-items-center md:hidden" aria-label="Menu" aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            <span className="block w-5">
              <span className="block h-px bg-current" style={{ transform: open ? "translateY(3px) rotate(45deg)" : "none", transition: "transform .2s" }} />
              <span className="mt-[6px] block h-px bg-current" style={{ transform: open ? "translateY(-3px) rotate(-45deg)" : "none", transition: "transform .2s" }} />
            </span>
          </button>
        </div>
      </div>

      {open && (
        <nav aria-label="Primary mobile" className="hm-wrap flex flex-col gap-1 pb-6 md:hidden">
          <div className="flex flex-col [&>*]:border-t [&>*]:border-[var(--hm-rule)] [&>*]:py-3.5 [&>*]:text-lg">{links}</div>
          <Link to="/dashboard" className="hm-link mt-3 self-start text-sm" onClick={() => setOpen(false)}>Open dashboard</Link>
        </nav>
      )}

      <style>{`
        .hm-nav-link { font-size: 13.5px; letter-spacing: 0.01em; color: var(--hm-ink-dim); transition: color .18s ease; }
        .hm-nav-link:hover { color: var(--hm-ink); }
      `}</style>
    </header>
  );
}
