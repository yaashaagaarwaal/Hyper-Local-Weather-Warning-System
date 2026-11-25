// Decides how much 3D this device gets, and keeps deciding as conditions change.
//
//   full    desktop-class machine: full-resolution Earth, 320x320 terrain mesh
//   lite    tablets / smaller or lower-memory laptops: same scene, lighter
//   static  no WebGL2, a phone-sized screen, or prefers-reduced-motion: no 3D at
//           all -- a pre-rendered Earth plus a flat SVG map of India
//
// Override for testing or preference with ?scene=static|lite|full.
import { useEffect, useState } from "react";

function hasWebGL2() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2"));
  } catch {
    return false;
  }
}

export function detectTier() {
  if (typeof window === "undefined") return "static";
  const forced = new URLSearchParams(window.location.search).get("scene");
  if (forced === "static" || forced === "lite" || forced === "full") {
    if (forced !== "static" && !hasWebGL2()) return "static";
    return forced;
  }
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return "static";
  if (!hasWebGL2()) return "static";
  const width = window.innerWidth;
  if (width < 720) return "static";
  const memory = navigator.deviceMemory ?? 8; // Chromium only; assume capable elsewhere
  const cores = navigator.hardwareConcurrency ?? 8;
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  if (width < 1100 || memory <= 4 || cores <= 4 || coarse) return "lite";
  return "full";
}

export function useRenderTier() {
  const [tier, setTier] = useState(detectTier);
  useEffect(() => {
    let timer;
    const recompute = () => {
      clearTimeout(timer);
      timer = setTimeout(() => setTier(detectTier()), 200);
    };
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    window.addEventListener("resize", recompute);
    motion.addEventListener("change", recompute);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", recompute);
      motion.removeEventListener("change", recompute);
    };
  }, []);
  return tier;
}
