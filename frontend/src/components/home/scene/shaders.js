// GLSL for the home-page Earth. Kept as plain strings so the scene stays one
// small dependency-free module.
//
// Lighting model, shared by the globe and the India terrain patch so the two
// match exactly when one fades into the other: a single directional sun
// (world space), a soft terminator, a faint blue ambient on the night side,
// city lights only where it is dark, and a thin atmospheric rim. Textures are
// sRGB-decoded by the GPU, shading is linear, and the output chunk converts
// back for display.

// Angular distance (radians) from the India focus, in the globe's local frame,
// used to keep the DECORATIVE cloud layer thin over India. Clouds on the globe
// are procedural art, not weather data; see scripts/build-geo-assets.py.
const indiaClear = /* glsl */ `
uniform vec3 uIndiaDir;
float indiaClear(vec3 dirL) {
  float c = clamp(dot(normalize(dirL), uIndiaDir), -1.0, 1.0);
  float ang = acos(c);
  return 1.0 - 0.9 * exp(-pow(ang / 0.42, 2.0));
}
`;

export const globeVertex = /* glsl */ `
varying vec2 vUv;
varying vec3 vNormalW;
varying vec3 vPosW;
varying vec3 vDirL;
void main() {
  vUv = uv;
  vDirL = normalize(position);
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vPosW = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

export const globeFragment = /* glsl */ `
uniform sampler2D uDay;
uniform sampler2D uNight;
uniform sampler2D uClouds;
uniform vec3 uSun;
uniform float uCloudAmount;
uniform float uTime;
varying vec2 vUv;
varying vec3 vNormalW;
varying vec3 vPosW;
varying vec3 vDirL;
${indiaClear}
void main() {
  vec3 N = normalize(vNormalW);
  vec3 V = normalize(cameraPosition - vPosW);
  float ndl = dot(N, uSun);
  float day = smoothstep(-0.10, 0.40, ndl);

  vec3 albedo = texture2D(uDay, vUv).rgb;
  float clear = indiaClear(vDirL);
  vec2 drift = vec2(uTime * 0.00035, 0.0);
  float cloud = texture2D(uClouds, vUv + drift).r * uCloudAmount * clear;
  float shadow = texture2D(uClouds, vUv + drift + vec2(0.0035, -0.002)).r * uCloudAmount * clear;

  vec3 sunColour = vec3(1.0, 0.97, 0.92);
  vec3 lit = albedo * (vec3(0.018, 0.028, 0.050) + day * sunColour * 1.05);
  lit *= 1.0 - 0.38 * shadow * day;

  // Sun glint on open water (water = blue-dominant pixels of the day map).
  float ocean = smoothstep(0.02, 0.16, albedo.b - max(albedo.r, albedo.g));
  vec3 H = normalize(uSun + V);
  float spec = pow(max(dot(N, H), 0.0), 90.0) * ocean * day * (1.0 - 0.85 * cloud);
  lit += vec3(0.75, 0.85, 1.0) * spec * 0.45;

  // City lights on the night side only.
  float lights = texture2D(uNight, vUv).r;
  lit += vec3(1.0, 0.74, 0.42) * pow(lights, 1.15) * (1.0 - day) * 0.9 * (1.0 - cloud);

  // Thin scattering rim.
  float fres = pow(1.0 - max(dot(N, V), 0.0), 3.2);
  lit += vec3(0.24, 0.45, 0.85) * fres * (0.10 + 0.90 * day) * 0.75;

  gl_FragColor = vec4(lit, 1.0);
  #include <colorspace_fragment>
}
`;

export const cloudFragment = /* glsl */ `
uniform sampler2D uClouds;
uniform vec3 uSun;
uniform float uCloudAmount;
uniform float uTime;
varying vec2 vUv;
varying vec3 vNormalW;
varying vec3 vPosW;
varying vec3 vDirL;
${indiaClear}
void main() {
  vec3 N = normalize(vNormalW);
  float day = smoothstep(-0.18, 0.45, dot(N, uSun));
  float c = texture2D(uClouds, vUv + vec2(uTime * 0.00035, 0.0)).r * uCloudAmount * indiaClear(vDirL);
  vec3 col = mix(vec3(0.020, 0.030, 0.050), vec3(0.96, 0.97, 1.0), day);
  col *= 0.78 + 0.22 * smoothstep(0.2, 1.0, c);
  float alpha = c * (0.22 + 0.78 * day) * 0.80;
  gl_FragColor = vec4(col, alpha);
  #include <colorspace_fragment>
}
`;

export const atmosphereFragment = /* glsl */ `
uniform vec3 uSun;
uniform float uStrength;
varying vec3 vNormalW;
varying vec3 vPosW;
void main() {
  vec3 N = normalize(vNormalW);
  vec3 V = normalize(cameraPosition - vPosW);
  // Back faces of a slightly larger sphere: visible only as a thin halo around the limb.
  float rim = pow(clamp(-dot(N, V) * 2.6, 0.0, 1.0), 2.2);
  float sunSide = smoothstep(-0.35, 0.55, dot(N, uSun));
  vec3 col = vec3(0.30, 0.55, 0.98);
  gl_FragColor = vec4(col, rim * (0.16 + 0.84 * sunSide) * uStrength);
  #include <colorspace_fragment>
}
`;

// ---------------------------------------------------------------------------
// India terrain patch: a lon/lat grid on the sphere, displaced in the vertex
// shader by the real elevation grid, shaded from slopes computed per pixel.
// ---------------------------------------------------------------------------
const terrainBasis = /* glsl */ `
// Window in radians: x = west, y = east, z = south, w = north.
uniform vec4 uWin;
uniform sampler2D uHeight;
uniform float uExag;
`;

export const terrainVertex = /* glsl */ `
${terrainBasis}
varying vec2 vUv;
varying vec3 vE;
varying vec3 vNo;
varying vec3 vU;
varying vec3 vPosW;
void main() {
  vUv = uv;
  float lon = mix(uWin.x, uWin.y, uv.x);
  float lat = mix(uWin.z, uWin.w, uv.y);
  float cl = cos(lat), sl = sin(lat), co = cos(lon), so = sin(lon);
  vec3 dir = vec3(cl * co, sl, -cl * so);
  vec3 east = vec3(-so, 0.0, -co);
  vec3 north = vec3(-sl * co, cl, sl * so);
  float h = texture2D(uHeight, uv).r;
  vec3 pos = dir * (1.0004 + h * uExag / 6371000.0);
  vec4 wp = modelMatrix * vec4(pos, 1.0);
  vPosW = wp.xyz;
  mat3 m = mat3(modelMatrix);
  vU = normalize(m * dir);
  vE = normalize(m * east);
  vNo = normalize(m * north);
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

export const terrainFragment = /* glsl */ `
${terrainBasis}
uniform sampler2D uColor;
uniform sampler2D uMask;
uniform sampler2D uIds;
uniform sampler2D uRain;
uniform vec3 uSun;
uniform float uAlpha;
uniform float uTexel;
uniform float uRainOn;
uniform float uMaskDim;
uniform float uHoverId;
varying vec2 vUv;
varying vec3 vE;
varying vec3 vNo;
varying vec3 vU;
varying vec3 vPosW;
void main() {
  float t = uTexel;
  float hE = texture2D(uHeight, vUv + vec2(t, 0.0)).r;
  float hW = texture2D(uHeight, vUv - vec2(t, 0.0)).r;
  float hN = texture2D(uHeight, vUv + vec2(0.0, t)).r;
  float hS = texture2D(uHeight, vUv - vec2(0.0, t)).r;
  float lat = mix(uWin.z, uWin.w, vUv.y);
  float dsx = (uWin.y - uWin.x) * t * max(cos(lat), 0.05);
  float dsy = (uWin.w - uWin.z) * t;
  float k = uExag / 6371000.0;
  float dhdx = (hE - hW) * k / (2.0 * dsx);
  float dhdy = (hN - hS) * k / (2.0 * dsy);
  vec3 n = normalize(vU - dhdx * vE - dhdy * vNo);

  float up = dot(vU, uSun);
  float day = smoothstep(-0.10, 0.40, up);
  float relief = clamp(max(dot(n, uSun), 0.0) / max(max(up, 0.0), 0.2), 0.0, 1.8);

  vec3 albedo = texture2D(uColor, vUv).rgb;
  float inside = texture2D(uMask, vUv).r;
  float grey = dot(albedo, vec3(0.299, 0.587, 0.114));
  vec3 outside = mix(albedo, vec3(grey), 0.55) * 0.5;
  albedo = mix(albedo, mix(outside, albedo, inside), uMaskDim);

  vec3 col = albedo * (vec3(0.018, 0.028, 0.050) + day * vec3(1.0, 0.97, 0.92) * (0.68 + 0.80 * mix(1.0, relief, 1.0)));

  float id = floor(texture2D(uIds, vUv).r * 255.0 + 0.5);
  float hovered = step(abs(id - uHoverId), 0.5) * step(0.5, uHoverId);
  col += hovered * vec3(0.055, 0.065, 0.085) * (0.4 + day);

  vec4 rain = texture2D(uRain, vUv);
  col = mix(col, rain.rgb, rain.a * uRainOn);

  float edge = smoothstep(0.0, 0.05, min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y)));
  gl_FragColor = vec4(col, uAlpha * edge);
  #include <colorspace_fragment>
}
`;

// Boundary lines follow the terrain: lon/lat per vertex, lifted by the same height field.
export const lineVertex = /* glsl */ `
${terrainBasis}
attribute vec2 aLonLat;
void main() {
  float lon = radians(aLonLat.x);
  float lat = radians(aLonLat.y);
  vec2 uv = vec2((lon - uWin.x) / (uWin.y - uWin.x), (lat - uWin.z) / (uWin.w - uWin.z));
  float h = texture2D(uHeight, clamp(uv, 0.0, 1.0)).r;
  float cl = cos(lat);
  vec3 dir = vec3(cl * cos(lon), sin(lat), -cl * sin(lon));
  vec3 pos = dir * (1.0016 + h * uExag / 6371000.0);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

export const lineFragment = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
void main() {
  gl_FragColor = vec4(uColor, uOpacity);
  #include <colorspace_fragment>
}
`;
