// Fades a block in once as it scrolls into view. Honors prefers-reduced-motion
// (the stylesheet shows it immediately) and works without IntersectionObserver.
import { useEffect, useRef, useState } from "react";

export function Reveal({ as: Tag = "div", className = "", children, ...rest }) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") { setShown(true); return undefined; }
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setShown(true); io.disconnect(); }
    }, { rootMargin: "0px 0px -8% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return <Tag ref={ref} className={`hm-reveal ${shown ? "is-in" : ""} ${className}`} {...rest}>{children}</Tag>;
}
