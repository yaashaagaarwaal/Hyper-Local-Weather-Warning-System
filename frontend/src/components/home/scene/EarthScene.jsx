/* eslint-disable react/immutability -- three.js/R3F objects (uniforms, refs, camera) are mutated by design */
// The home page's 3D scene: a lit, rotating Earth that the camera descends
// onto India as the page scrolls. Loaded lazily (three.js is large) and only
// on capable devices; see Journey.jsx for the fallbacks.
//
// What is real here: the imagery (NASA Blue Marble / Black Marble), the terrain
// (open elevation data), the state boundaries (geoBoundaries / LGD) and, when
// switched on, NASA IMERG satellite rainfall plus the readings and warnings the
// AtmoGuard backend returns. What is not: the globe's cloud layer is
// procedural decoration (faded out before the India stage).
import { Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, PerformanceMonitor, useTexture } from "@react-three/drei";
import * as THREE from "three";
import {
  atmosphereFragment, cloudFragment, globeFragment, globeVertex,
  lineFragment, lineVertex, terrainFragment, terrainVertex,
} from "./shaders";
import {
  DEG, INDIA_FOCUS, INDIA_WINDOW, lerp, lerpAngle, lonLatToXYZ, smoothstep, windowUV, xyzToLonLat,
} from "./geoMath";
import { loadIndiaAssets, sampleHeight, stateIdAt } from "../indiaData";

const EXAGGERATION = 36; // vertical exaggeration of the terrain once the camera is over India
const SUN = new THREE.Vector3(-0.62, 0.4, 0.68).normalize();
const WIN_RAD = new THREE.Vector4(
  INDIA_WINDOW.west * DEG, INDIA_WINDOW.east * DEG, INDIA_WINDOW.south * DEG, INDIA_WINDOW.north * DEG,
);
const INDIA_DIR = new THREE.Vector3(...lonLatToXYZ(INDIA_FOCUS.lon, INDIA_FOCUS.lat));

// Geographic regions are named only where the names are uncontroversial physical features.
const REGION_LABELS = [
  { name: "Himalaya", lon: 84.5, lat: 28.7 },
  { name: "Indo-Gangetic Plain", lon: 81.5, lat: 26.2 },
  { name: "Thar Desert", lon: 71.3, lat: 27.0 },
  { name: "Deccan Plateau", lon: 77.8, lat: 17.4 },
  { name: "Western Ghats", lon: 74.6, lat: 13.6 },
  { name: "Brahmaputra Valley", lon: 93.2, lat: 26.9 },
  { name: "Arabian Sea", lon: 68.6, lat: 15.5 },
  { name: "Bay of Bengal", lon: 87.8, lat: 14.5 },
];

// ---------------------------------------------------------------------------
// Journey keyframes: everything is a pure function of scroll progress p (0..1).
// ---------------------------------------------------------------------------
export function journeyAt(p, layout) {
  const approach = smoothstep(0.0, 0.40, p);
  const descent = smoothstep(0.38, 0.80, p);
  const wide = layout === "wide";
  // Portrait screens are narrow: pull the camera back so the globe and, later, India still fit across the width.
  const fit = wide ? 1 : lerp(1.5, 1.22, smoothstep(0.0, 0.5, p));
  return {
    align: smoothstep(0.04, 0.50, p), // how far the globe has been turned so India faces the camera
    distance: lerp(lerp(4.0, 2.2, approach), 1.04, descent) * fit,
    tilt: smoothstep(0.46, 0.92, p) * 42 * DEG, // camera swings south of India, looking north over the relief
    targetBlend: smoothstep(0.28, 0.66, p), // look-at moves from the globe's centre to India's surface
    shiftX: wide ? lerp(0.225, 0.075, smoothstep(0.02, 0.40, p)) : 0, // globe sits right of the copy, then centres
    shiftY: wide ? 0 : lerp(0.26, -0.25, smoothstep(0.02, 0.55, p)), // down (below the headline), then up (above the layer panel)
    cloud: 0.74 * (1 - smoothstep(0.16, 0.42, p)),
    patchAlpha: smoothstep(0.30, 0.46, p),
    exag: EXAGGERATION * smoothstep(0.40, 0.80, p),
    maskDim: smoothstep(0.36, 0.62, p),
    atmosphere: lerp(1.0, 0.7, descent),
    borders: smoothstep(0.55, 0.84, p),
    rain: smoothstep(0.62, 0.82, p),
    markers: smoothstep(0.74, 0.90, p),
    labels: smoothstep(0.80, 0.95, p),
  };
}

// ---------------------------------------------------------------------------
// Globe: surface, clouds, atmosphere
// ---------------------------------------------------------------------------
function useEarthTextures() {
  const gl = useThree((s) => s.gl);
  const textures = useTexture(["/geo/earth-day.jpg", "/geo/earth-night.jpg", "/geo/earth-clouds.jpg"]);
  useLayoutEffect(() => {
    const [day, night, clouds] = textures;
    const aniso = Math.min(8, gl.capabilities.getMaxAnisotropy());
    day.colorSpace = THREE.SRGBColorSpace;
    day.anisotropy = aniso;
    night.colorSpace = THREE.NoColorSpace;
    clouds.colorSpace = THREE.NoColorSpace;
    clouds.wrapS = THREE.RepeatWrapping;
    [day, night, clouds].forEach((t) => { t.needsUpdate = true; });
  }, [textures, gl]);
  return textures;
}

function Globe({ materialRefs, segments }) {
  const [day, night, clouds] = useEarthTextures();
  const globeUniforms = useMemo(() => ({
    uDay: { value: day }, uNight: { value: night }, uClouds: { value: clouds },
    uSun: { value: SUN }, uCloudAmount: { value: 0.9 }, uTime: { value: 0 }, uIndiaDir: { value: INDIA_DIR },
  }), [day, night, clouds]);
  const cloudUniforms = useMemo(() => ({
    uClouds: { value: clouds }, uSun: { value: SUN }, uCloudAmount: { value: 0.9 },
    uTime: { value: 0 }, uIndiaDir: { value: INDIA_DIR },
  }), [clouds]);
  const atmoUniforms = useMemo(() => ({ uSun: { value: SUN }, uStrength: { value: 1 } }), []);
  const materials = useMemo(() => ({
    globe: new THREE.ShaderMaterial({ uniforms: globeUniforms, vertexShader: globeVertex, fragmentShader: globeFragment }),
    cloud: new THREE.ShaderMaterial({ uniforms: cloudUniforms, vertexShader: globeVertex, fragmentShader: cloudFragment, transparent: true, depthWrite: false }),
    atmosphere: new THREE.ShaderMaterial({
      uniforms: atmoUniforms, vertexShader: globeVertex, fragmentShader: atmosphereFragment,
      side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }),
  }), [globeUniforms, cloudUniforms, atmoUniforms]);
  useLayoutEffect(() => {
    materialRefs.current = materials;
    return () => Object.values(materials).forEach((m) => m.dispose());
  }, [materials, materialRefs]);
  const [w, h] = segments;

  return (
    <>
      <mesh material={materials.globe}>
        <sphereGeometry args={[1, w, h]} />
      </mesh>
      <mesh renderOrder={2} material={materials.cloud}>
        <sphereGeometry args={[1.007, w, h]} />
      </mesh>
      <mesh renderOrder={1} material={materials.atmosphere}>
        <sphereGeometry args={[1.05, w, h]} />
      </mesh>
    </>
  );
}

function Stars({ count }) {
  const geometry = useMemo(() => {
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const u = rnd() * 2 - 1;
      const a = rnd() * Math.PI * 2;
      const r = Math.sqrt(1 - u * u);
      pos.set([r * Math.cos(a) * 80, u * 80, r * Math.sin(a) * 80], i * 3);
      const b = 0.25 + 0.75 * rnd() ** 3;
      const warm = rnd();
      col.set([b * (0.85 + 0.15 * warm), b * 0.9, b * (1 - 0.15 * warm)], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, [count]);
  return (
    <points geometry={geometry} frustumCulled={false}>
      <pointsMaterial size={1.25} sizeAttenuation={false} vertexColors transparent opacity={0.8} depthWrite={false} />
    </points>
  );
}

// ---------------------------------------------------------------------------
// India layer: terrain patch, boundary lines, markers, labels
// ---------------------------------------------------------------------------
function makeDataTexture(data, size, { type, internalFormat, nearest }) {
  const tex = new THREE.DataTexture(data, size, size, THREE.RedFormat, type);
  if (internalFormat) tex.internalFormat = internalFormat;
  tex.minFilter = nearest ? THREE.NearestFilter : THREE.LinearFilter;
  tex.magFilter = tex.minFilter;
  tex.generateMipmaps = false;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

function buildBorderGeometry(states) {
  // Densify long segments so lines follow the curvature and terrain.
  const MAX_STEP = 0.2;
  const emit = (rings) => {
    const out = [];
    for (const ring of rings) {
      for (let i = 0; i + 3 < ring.length; i += 2) {
        const x0 = ring[i], y0 = ring[i + 1], x1 = ring[i + 2], y1 = ring[i + 3];
        const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / MAX_STEP));
        for (let s = 0; s < steps; s += 1) {
          out.push(lerp(x0, x1, s / steps), lerp(y0, y1, s / steps), lerp(x0, x1, (s + 1) / steps), lerp(y0, y1, (s + 1) / steps));
        }
      }
      // close the ring
      const n = ring.length;
      if (n >= 4 && (ring[0] !== ring[n - 2] || ring[1] !== ring[n - 1])) {
        const x0 = ring[n - 2], y0 = ring[n - 1], x1 = ring[0], y1 = ring[1];
        const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / MAX_STEP));
        for (let s = 0; s < steps; s += 1) {
          out.push(lerp(x0, x1, s / steps), lerp(y0, y1, s / steps), lerp(x0, x1, (s + 1) / steps), lerp(y0, y1, (s + 1) / steps));
        }
      }
    }
    return out;
  };
  const make = (flat) => {
    // each segment = 2 vertices, each vertex = (lon, lat); aLonLat is the only attribute
    const arr = new Float32Array(flat.length);
    for (let i = 0; i < flat.length; i += 1) arr[i] = flat[i];
    const g = new THREE.BufferGeometry();
    g.setAttribute("aLonLat", new THREE.BufferAttribute(arr, 2));
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array((arr.length / 2) * 3), 3)); // unused, needed for drawRange
    return g;
  };
  return {
    states: make(emit(states.states.flatMap((s) => s.rings))),
    outline: make(emit(states.outline)),
  };
}

// NOTE: materials are built here, not as <shaderMaterial uniforms={...}> JSX: R3F gives JSX
// materials a clone of the uniforms object (three.js clones texture uniforms), so values
// mutated every frame -- and textures assigned later, like the rainfall layer -- would
// never reach the GPU.
function Border({ geometry, uniforms, color, opacity, revealRef, name }) {
  const mesh = useRef();
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: lineVertex, fragmentShader: lineFragment, transparent: true, depthWrite: false,
    uniforms: { ...uniforms, uColor: { value: new THREE.Color(color) }, uOpacity: { value: opacity } },
  }), [uniforms, color, opacity]);
  useEffect(() => () => material.dispose(), [material]);
  useFrame(() => {
    const reveal = revealRef.current;
    const total = geometry.getAttribute("aLonLat").count;
    geometry.setDrawRange(0, Math.floor((total * reveal) / 2) * 2);
    material.uniforms.uOpacity.value = opacity * Math.min(1, reveal * 4);
    mesh.current.visible = reveal > 0.001;
  });
  return <lineSegments ref={mesh} geometry={geometry} material={material} frustumCulled={false} name={name} renderOrder={3} />;
}

function Marker({ marker, assets, state, kind }) {
  const group = useRef();
  const ringMat = useRef();
  const dotMat = useRef();
  const label = useRef();
  const { u, v } = windowUV(marker.lon, marker.lat);
  const heightM = useMemo(() => sampleHeight(assets.heights, u, v), [assets, u, v]);
  const dir = useMemo(() => new THREE.Vector3(...lonLatToXYZ(marker.lon, marker.lat)), [marker.lon, marker.lat]);
  const colour = kind === "warning" ? "#e0a03a" : "#f4efe4";

  useFrame(() => {
    const j = state.current;
    const r = 1.004 + (heightM * j.exag) / 6371000 + 0.004;
    group.current.position.copy(dir).multiplyScalar(r);
    group.current.lookAt(dir.clone().multiplyScalar(3));
    const opacity = j.markers * j.layerOn[kind];
    group.current.visible = opacity > 0.01;
    ringMat.current.opacity = opacity * 0.9;
    dotMat.current.opacity = opacity;
    if (label.current) label.current.style.opacity = String(opacity);
  });

  return (
    <group ref={group} renderOrder={5}>
      <mesh>
        <ringGeometry args={[kind === "warning" ? 0.013 : 0.0085, kind === "warning" ? 0.0155 : 0.0102, 40]} />
        <meshBasicMaterial ref={ringMat} color={colour} transparent depthTest={false} />
      </mesh>
      <mesh>
        <circleGeometry args={[kind === "warning" ? 0.005 : 0.0042, 24]} />
        <meshBasicMaterial ref={dotMat} color={colour} transparent depthTest={false} />
      </mesh>
      <Html center zIndexRange={[8, 0]} style={{ pointerEvents: "none" }}>
        <div ref={label} className="home-map-label home-map-label--marker" style={{ opacity: 0, transform: marker.below ? "translate(18px, 6px)" : undefined }}>
          <span>{marker.name}</span>
          {marker.value && <b>{marker.value}</b>}
        </div>
      </Html>
    </group>
  );
}

function RegionLabels({ assets, state }) {
  const refs = useRef([]);
  const items = useMemo(() => REGION_LABELS.map((r) => {
    const { u, v } = windowUV(r.lon, r.lat);
    return { ...r, dir: new THREE.Vector3(...lonLatToXYZ(r.lon, r.lat)), heightM: sampleHeight(assets.heights, u, v) };
  }), [assets]);
  const group = useRef();
  useFrame(() => {
    const j = state.current;
    group.current.children.forEach((child, i) => {
      const it = items[i];
      child.position.copy(it.dir).multiplyScalar(1.004 + (it.heightM * j.exag) / 6371000 + 0.008);
      const el = refs.current[i];
      if (el) el.style.opacity = String(j.labels * j.layerOn.regions);
    });
  });
  return (
    <group ref={group}>
      {items.map((it, i) => (
        <group key={it.name}>
          <Html center zIndexRange={[6, 0]} style={{ pointerEvents: "none" }}>
            <div ref={(el) => { refs.current[i] = el; }} className="home-map-label home-map-label--region" style={{ opacity: 0 }}>
              {it.name}
            </div>
          </Html>
        </group>
      ))}
    </group>
  );
}

function IndiaLayer({ assets, state, rain, markers, hover, tier }) {
  const colour = useTexture("/geo/india-color.jpg");
  const patch = useRef();
  const invalidate = useThree((s) => s.invalidate);
  const gl = useThree((s) => s.gl);

  const textures = useMemo(() => {
    const half = new Uint16Array(assets.heights.data.length);
    for (let i = 0; i < half.length; i += 1) half[i] = THREE.DataUtils.toHalfFloat(assets.heights.data[i]);
    return {
      height: makeDataTexture(half, assets.heights.size, { type: THREE.HalfFloatType, internalFormat: "R16F" }),
      mask: makeDataTexture(assets.mask.data, assets.mask.size, { type: THREE.UnsignedByteType }),
      ids: makeDataTexture(assets.ids.data, assets.ids.size, { type: THREE.UnsignedByteType, nearest: true }),
      none: new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1),
    };
  }, [assets]);
  useEffect(() => { textures.none.needsUpdate = true; }, [textures]);

  useLayoutEffect(() => {
    colour.colorSpace = THREE.SRGBColorSpace;
    colour.anisotropy = Math.min(8, gl.capabilities.getMaxAnisotropy());
    colour.needsUpdate = true;
  }, [colour, gl]);

  const rainTexture = useMemo(() => {
    if (rain?.status !== "ready") return null;
    const t = new THREE.DataTexture(rain.pixels, rain.size, rain.size, THREE.RGBAFormat);
    t.colorSpace = THREE.SRGBColorSpace;
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  }, [rain]);

  const geometry = useMemo(() => new THREE.PlaneGeometry(1, 1, tier === "full" ? 320 : 192, tier === "full" ? 320 : 192), [tier]);
  const borders = useMemo(() => buildBorderGeometry(assets.states), [assets]);

  const uniforms = useMemo(() => ({
    uWin: { value: WIN_RAD }, uHeight: { value: textures.height }, uExag: { value: 0 },
    uColor: { value: colour }, uMask: { value: textures.mask }, uIds: { value: textures.ids },
    uRain: { value: textures.none }, uSun: { value: SUN }, uAlpha: { value: 0 },
    uTexel: { value: 1 / assets.heights.size }, uRainOn: { value: 0 }, uMaskDim: { value: 0 }, uHoverId: { value: 0 },
  }), [textures, colour, assets]);
  const lineUniforms = useMemo(() => ({ uWin: uniforms.uWin, uHeight: uniforms.uHeight, uExag: uniforms.uExag }), [uniforms]);

  const terrainMaterial = useMemo(() => new THREE.ShaderMaterial({
    uniforms, vertexShader: terrainVertex, fragmentShader: terrainFragment, transparent: true, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
  }), [uniforms]);
  useEffect(() => () => terrainMaterial.dispose(), [terrainMaterial]);

  const bordersReveal = useRef(0);
  const outlineReveal = useRef(0);

  useEffect(() => {
    uniforms.uRain.value = rainTexture ?? textures.none;
    invalidate();
  }, [rainTexture, uniforms, textures, invalidate]);

  useEffect(() => {
    uniforms.uHoverId.value = hover?.id ?? 0;
    invalidate();
  }, [hover, uniforms, invalidate]);

  useFrame(() => {
    const j = state.current;
    uniforms.uExag.value = j.exag;
    uniforms.uAlpha.value = j.patchAlpha;
    uniforms.uMaskDim.value = j.maskDim;
    uniforms.uRainOn.value = j.rain * j.layerOn.rain;
    bordersReveal.current = j.borders * j.layerOn.borders;
    outlineReveal.current = Math.max(j.borders, 0) * j.layerOn.borders;
    if (patch.current) patch.current.visible = j.patchAlpha > 0.002;
  });

  return (
    <>
      <mesh ref={patch} geometry={geometry} material={terrainMaterial} frustumCulled={false} renderOrder={1} />
      <Border name="states" geometry={borders.states} uniforms={lineUniforms} color="#d9d3c3" opacity={0.42} revealRef={bordersReveal} />
      <Border name="outline" geometry={borders.outline} uniforms={lineUniforms} color="#f4efe4" opacity={0.95} revealRef={outlineReveal} />
      {markers.stations.map((m) => <Marker key={m.id} marker={m} assets={assets} state={state} kind="stations" />)}
      {markers.warnings.map((m) => <Marker key={m.id} marker={m} assets={assets} state={state} kind="warnings" />)}
      <RegionLabels assets={assets} state={state} />
    </>
  );
}

// ---------------------------------------------------------------------------
// World: owns the globe group, the camera path and the render loop
// ---------------------------------------------------------------------------
function World({ progress, layout, tier, active, layers, rain, markers, onHover, onReady, hover }) {
  const { camera, size, invalidate } = useThree();
  const dpr = useThree((st) => st.viewport.dpr);
  const group = useRef();
  const materials = useRef({});
  const [assets, setAssets] = useState(null);
  const state = useRef({ ...journeyAt(0, layout), layerOn: { rain: 1, borders: 1, stations: 1, warnings: 1, regions: 1 } });
  const clock = useRef({ autoYaw: 0, time: 0, lastP: -1, lastChange: 0, lastRender: 0 });
  const hitSphere = useRef();
  const lastHoverId = useRef(0);

  // India data loads once the globe is up; the first frames never wait for it.
  useEffect(() => {
    let alive = true;
    loadIndiaAssets().then((a) => { if (alive) setAssets(a); }).catch((err) => console.warn("[home] India terrain unavailable:", err?.message || err));
    return () => { alive = false; };
  }, []);

  useEffect(() => { state.current.layerOn = { rain: layers.rain ? 1 : 0, borders: layers.borders ? 1 : 0, stations: layers.stations ? 1 : 0, warnings: layers.warnings ? 1 : 0, regions: layers.regions ? 1 : 0 }; invalidate(); }, [layers, invalidate]);

  // Render pacing: ~30 fps while the globe idles, full rate while scrolling,
  // and nothing at all once the India view is settled (events re-trigger a frame).
  useEffect(() => {
    if (!active) return undefined;
    let raf;
    const loop = (now) => {
      const c = clock.current;
      const p = progress.current;
      if (p !== c.lastP) { c.lastP = p; c.lastChange = now; }
      const settled = p > 0.9 && now - c.lastChange > 800;
      const idle = now - c.lastChange > 120;
      if (!settled && (!idle || now - c.lastRender > 33)) { c.lastRender = now; invalidate(); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [active, invalidate, progress]);

  // Resizing the canvas (window resize, or a dpr change) clears its drawing buffer;
  // always repaint afterwards, even when the render loop has gone quiet.
  useEffect(() => { invalidate(); }, [size.width, size.height, dpr, invalidate]);

  const alignYaw = -Math.PI / 2 - INDIA_FOCUS.lon * DEG;
  const alignPitch = INDIA_FOCUS.lat * DEG;
  const tmp = useMemo(() => ({ target: new THREE.Vector3(), pos: new THREE.Vector3() }), []);

  useFrame((s, delta) => {
    const c = clock.current;
    const dt = Math.min(delta, 0.1);
    c.time += dt;
    const p = progress.current;
    const j = journeyAt(p, layout);
    Object.assign(state.current, j);

    // Slow eastward spin while the globe is the subject; India starts 20 degrees west of centre and crosses.
    c.autoYaw += dt * 0.012;
    const spinYaw = alignYaw - 20 * DEG + c.autoYaw;
    const yaw = lerpAngle(spinYaw, alignYaw, j.align);
    const pitch = lerp(0.38, alignPitch, j.align);
    group.current.rotation.set(pitch, yaw, 0, "XYZ");

    // Camera: orbit target blends from the globe's centre to India's surface; swings south for the oblique view.
    tmp.target.set(0, -0.02 * j.targetBlend, j.targetBlend * 1.0);
    tmp.pos.set(0, -Math.sin(j.tilt) * j.distance, Math.cos(j.tilt) * j.distance).add(tmp.target);
    camera.position.copy(tmp.pos);
    camera.up.set(0, 1, 0);
    camera.lookAt(tmp.target);
    camera.setViewOffset(size.width, size.height, -j.shiftX * size.width, -j.shiftY * size.height, size.width, size.height);

    const m = materials.current;
    if (m.globe) { m.globe.uniforms.uTime.value = c.time; m.globe.uniforms.uCloudAmount.value = j.cloud; }
    if (m.cloud) { m.cloud.uniforms.uTime.value = c.time; m.cloud.uniforms.uCloudAmount.value = j.cloud; }
    if (m.atmosphere) m.atmosphere.uniforms.uStrength.value = j.atmosphere;
  });

  useEffect(() => () => { camera.clearViewOffset(); }, [camera]);

  const segments = tier === "full" ? [128, 80] : [80, 56];

  function handleMove(e) {
    if (progress.current < 0.85 || !assets) return;
    const local = group.current.worldToLocal(e.point.clone());
    const { lon, lat } = xyzToLonLat(local.x, local.y, local.z);
    const { u, v } = windowUV(lon, lat);
    const id = stateIdAt(assets.ids, u, v);
    if (id !== lastHoverId.current) {
      lastHoverId.current = id;
      onHover(id ? { id, name: assets.states.states[id - 1]?.name } : null);
    }
  }

  return (
    <>
      <Stars count={tier === "full" ? 1800 : 900} />
      <group ref={group}>
        <Suspense fallback={null}>
          <Globe materialRefs={materials} segments={segments} />
          <Ready onReady={onReady} />
        </Suspense>
        {assets && (
          <Suspense fallback={null}>
            <IndiaLayer assets={assets} state={state} rain={rain} markers={markers} hover={hover} tier={tier} />
          </Suspense>
        )}
        <mesh ref={hitSphere} visible={false} onPointerMove={handleMove} onPointerLeave={() => { lastHoverId.current = 0; onHover(null); }}>
          <sphereGeometry args={[1.0, 64, 40]} />
          <meshBasicMaterial />
        </mesh>
      </group>
    </>
  );
}

function Ready({ onReady }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    invalidate();
    const id = requestAnimationFrame(() => onReady?.());
    return () => cancelAnimationFrame(id);
  }, [onReady, invalidate]);
  return null;
}

export default function EarthScene({ progress, layout, tier, active, layers, rain, markers, onHover, onReady, hover }) {
  const [dprMax, setDprMax] = useState(tier === "full" ? 1.75 : 1.25);
  return (
    <Canvas
      frameloop="demand"
      dpr={[1, dprMax]}
      camera={{ fov: 32, near: 0.03, far: 200, position: [0, 0, 4.0] }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      style={{ position: "absolute", inset: 0 }}
      onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
      aria-hidden="true"
    >
      {/* The scene deliberately idles at ~30 fps, so "slow" means below 20 fps, not below 50. */}
      <PerformanceMonitor bounds={() => [20, 45]} onDecline={() => setDprMax((d) => Math.max(1, d - 0.4))} flipflops={2} />
      <World progress={progress} layout={layout} tier={tier} active={active} layers={layers} rain={rain} markers={markers} onHover={onHover} onReady={onReady} hover={hover} />
    </Canvas>
  );
}
