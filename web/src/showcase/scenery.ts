// Scenery for the stage: sky dome (space nebula / ocean water column), bright
// stars with diffraction spikes, a planet with an atmosphere, the engraved
// instrument disc, the machined pedestal and the final vignette pass.
// Everything here is procedural (shaders), so it costs no downloads.

import * as THREE from "three";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";

const NOISE = /* glsl */ `
  float h3(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
  float n3(vec3 x) {
    vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(h3(i), h3(i + vec3(1,0,0)), f.x), mix(h3(i + vec3(0,1,0)), h3(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(h3(i + vec3(0,0,1)), h3(i + vec3(1,0,1)), f.x), mix(h3(i + vec3(0,1,1)), h3(i + vec3(1,1,1)), f.x), f.y), f.z);
  }
  float fbm(vec3 p) { float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++) { s += a * n3(p); p *= 2.03; a *= 0.5; } return s; }
`;

export interface SkyUniforms {
  uMode: { value: number }; // 0 space, 1 ocean
  uTop: { value: THREE.Color };
  uHorizon: { value: THREE.Color };
  uDeep: { value: THREE.Color };
  uNebA: { value: THREE.Color };
  uNebB: { value: THREE.Color };
  uTime: { value: number };
}

/** A unit sphere seen from inside; placed and scaled around the camera. */
export function makeSkyDome(): { mesh: THREE.Mesh; uniforms: SkyUniforms } {
  const uniforms: SkyUniforms = {
    uMode: { value: 0 },
    uTop: { value: new THREE.Color(0x000000) },
    uHorizon: { value: new THREE.Color(0x000000) },
    uDeep: { value: new THREE.Color(0x000000) },
    uNebA: { value: new THREE.Color(0x5b2a86) },
    uNebB: { value: new THREE.Color(0x15507a) },
    uTime: { value: 0 },
  };
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
    fog: false,
    uniforms: uniforms as unknown as Record<string, THREE.IUniform>,
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform float uMode, uTime; uniform vec3 uTop, uHorizon, uDeep, uNebA, uNebB;
      varying vec3 vDir;
      ${NOISE}
      void main() {
        vec3 d = normalize(vDir);
        if (uMode < 0.5) {
          // deep space: near-black with two faint nebula clouds and a dusty band
          float n1 = fbm(d * 2.2 + vec3(0.0, 0.0, 3.1));
          float n2 = fbm(d * 3.7 + vec3(5.2, 1.3, 0.0));
          float cloudA = smoothstep(0.52, 0.85, n1) * smoothstep(-0.2, 0.6, d.x + 0.3 * d.z);
          float cloudB = smoothstep(0.55, 0.9, n2) * smoothstep(-0.5, 0.4, -d.x + d.z);
          float band = exp(-pow((d.z - 0.15 * d.x) * 4.0, 2.0)) * (0.4 + 0.6 * fbm(d * 6.0));
          vec3 col = vec3(0.006, 0.007, 0.014);
          col += uNebA * cloudA * 0.18 + uNebB * cloudB * 0.1 + vec3(0.07, 0.065, 0.08) * band * 0.22;
          gl_FragColor = vec4(col, 1.0);
        } else {
          // water column: light from above, falling off to the deep
          float up = d.z;
          vec3 col = mix(uHorizon, uTop, smoothstep(0.0, 0.8, up));
          col = mix(col, uDeep, smoothstep(0.0, -0.6, up));
          float rays = pow(max(0.0, sin(d.x * 18.0 + fbm(d * 3.0 + uTime * 0.05) * 3.0)), 6.0) * smoothstep(0.05, 0.7, up);
          col += uTop * rays * 0.12;
          gl_FragColor = vec4(col, 1.0);
        }
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), mat);
  mesh.renderOrder = -100;
  mesh.frustumCulled = false;
  return { mesh, uniforms };
}

/** A few bright stars with four-point diffraction spikes. */
export function makeBrightStars(n: number, uniforms: { uTime: { value: number }; uPx: { value: number } }): THREE.Points {
  const pos = new Float32Array(n * 3);
  const seed = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.random() * 1.6 - 0.6; // mostly above the horizon
    const phi = Math.random() * Math.PI * 2;
    const r = Math.sqrt(1 - u * u);
    pos.set([r * Math.cos(phi), r * Math.sin(phi), u], i * 3);
    seed[i] = Math.random();
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("seed", new THREE.BufferAttribute(seed, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms,
    vertexShader: /* glsl */ `
      uniform float uTime, uPx; attribute float seed; varying float vS; varying float vA;
      void main() {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uPx * (14.0 + seed * 22.0);
        vS = seed;
        vA = 0.75 + 0.25 * sin(uTime * (0.4 + seed) + seed * 30.0);
      }`,
    fragmentShader: /* glsl */ `
      varying float vS; varying float vA;
      void main() {
        vec2 p = gl_PointCoord - 0.5;
        float core = exp(-dot(p, p) * 900.0);
        float halo = exp(-dot(p, p) * 60.0) * 0.18;
        float spikes = (exp(-abs(p.x) * 160.0) * exp(-abs(p.y) * 6.0) + exp(-abs(p.y) * 160.0) * exp(-abs(p.x) * 6.0)) * 0.55;
        vec3 tint = mix(vec3(1.0, 0.86, 0.72), vec3(0.75, 0.85, 1.0), vS);
        float a = (core * 1.6 + halo + spikes) * vA;
        gl_FragColor = vec4(tint * a, a);
      }`,
  });
  const p = new THREE.Points(g, mat);
  p.frustumCulled = false;
  return p;
}

/** A banded planet lit by the sun, with a fresnel atmosphere. */
export function makePlanet(): { group: THREE.Group; uniforms: { uSun: { value: THREE.Vector3 } } } {
  const uniforms = { uSun: { value: new THREE.Vector3(0.3, -0.5, 0.8).normalize() } };
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(1, 96, 64),
    new THREE.ShaderMaterial({
      fog: false,
      uniforms,
      vertexShader: /* glsl */ `
        varying vec3 vN; varying vec3 vP; varying vec3 vV;
        void main() {
          vN = normalize(mat3(modelMatrix) * normal);
          vP = position;
          vec4 w = modelMatrix * vec4(position, 1.0);
          vV = normalize(cameraPosition - w.xyz);
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uSun; varying vec3 vN; varying vec3 vP; varying vec3 vV;
        ${NOISE}
        void main() {
          // soft latitude bands broken by turbulence (an ice giant)
          float lat = vP.z + 0.08 * fbm(vP * 4.0);
          float bands = 0.5 + 0.5 * sin(lat * 22.0 + fbm(vP * 2.0) * 3.0);
          vec3 a = vec3(0.32, 0.44, 0.58), b = vec3(0.62, 0.70, 0.78), c = vec3(0.86, 0.80, 0.70);
          vec3 albedo = mix(a, b, bands);
          albedo = mix(albedo, c, smoothstep(0.75, 1.0, fbm(vP * 6.0)) * 0.35);
          float lambert = max(0.0, dot(normalize(vN), uSun));
          float term = smoothstep(-0.08, 0.25, dot(normalize(vN), uSun));
          vec3 col = albedo * (0.004 + 0.75 * lambert) * term;
          float rim = pow(1.0 - max(0.0, dot(normalize(vN), vV)), 3.0);
          col += vec3(0.35, 0.6, 1.0) * rim * term * 0.5;
          gl_FragColor = vec4(col, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }),
  );
  const atmo = new THREE.Mesh(
    new THREE.SphereGeometry(1.035, 96, 64),
    new THREE.ShaderMaterial({
      fog: false,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms,
      vertexShader: /* glsl */ `
        varying vec3 vN; varying vec3 vV;
        void main() {
          vN = normalize(mat3(modelMatrix) * normal);
          vec4 w = modelMatrix * vec4(position, 1.0);
          vV = normalize(cameraPosition - w.xyz);
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uSun; varying vec3 vN; varying vec3 vV;
        void main() {
          float f = pow(1.0 - abs(dot(normalize(vN), vV)), 2.2);
          float lit = smoothstep(-0.3, 0.4, dot(-normalize(vN), uSun));
          vec3 c = vec3(0.35, 0.62, 1.0) * f * lit * 1.1;
          gl_FragColor = vec4(c, f * lit);
        }`,
    }),
  );
  const group = new THREE.Group();
  group.add(body, atmo);
  return { group, uniforms };
}

/**
 * The floor in space: a dark machined disc with engraved range rings (one
 * per scale step, every tenth deeper) and angle ticks, rimmed by a thin light.
 */
export function makeInstrumentDisc(L: number): THREE.Group {
  const R = L * 1.7;
  const step = 10 ** Math.floor(Math.log10(L / 3));
  // dark polished stone that dissolves into space toward its edge
  const mat = new THREE.MeshPhysicalMaterial({ color: 0x06070a, metalness: 0.2, roughness: 0.75, clearcoat: 0.3, clearcoatRoughness: 0.5, transparent: true });
  const u = { uStep: { value: step }, uR: { value: R }, uCenter: { value: new THREE.Vector2(L * 0.5, 0) } };
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec2 vDisc;")
      .replace("#include <project_vertex>", "#include <project_vertex>\nvDisc = (modelMatrix * vec4(transformed, 1.0)).xy;");
    sh.fragmentShader = sh.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
        varying vec2 vDisc; uniform float uStep, uR; uniform vec2 uCenter;`,
      )
      .replace(
        "#include <opaque_fragment>",
        `vec2 q = vDisc - uCenter;
        float r = length(q);
        float ang = atan(q.y, q.x);
        float fr = r / uStep;
        float ring = 1.0 - smoothstep(0.0, fwidth(fr) * 1.2, abs(fract(fr - 0.5) - 0.5));
        float major = 1.0 - smoothstep(0.0, fwidth(fr / 10.0) * 1.2, abs(fract(fr / 10.0 - 0.5) - 0.5));
        float fa = ang / (3.14159265 / 12.0);
        float tick = (1.0 - smoothstep(0.0, fwidth(fa) * 1.2, abs(fract(fa - 0.5) - 0.5))) * smoothstep(uR * 0.86, uR * 0.9, r);
        float engr = max(ring * 0.3, max(major * 0.8, tick));
        outgoingLight += vec3(0.42, 0.5, 0.65) * engr * 0.11;
        diffuseColor.a *= 1.0 - smoothstep(uR * 0.35, uR, r);
        #include <opaque_fragment>`,
      );
  };
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(R, R * 1.01, L * 0.018, 128, 1), mat);
  disc.rotation.x = Math.PI / 2;
  disc.position.set(L * 0.5, 0, -L * 0.009);
  disc.receiveShadow = true;
  const g = new THREE.Group();
  g.add(disc);
  return g;
}

/**
 * Machined graphite pedestal: a stepped foot, a slim column and a collar that
 * clamps the base link, with a thin light ring around the collar.
 */
export function makePedestal(mount: [number, number, number], bs: number, glow: number): THREE.Group {
  const metal = new THREE.MeshPhysicalMaterial({ color: 0x24262c, metalness: 0.85, roughness: 0.3, clearcoat: 0.5, clearcoatRoughness: 0.2 });
  const h = mount[2];
  const prof = [
    [0, 0], [bs * 0.95, 0], [bs * 0.98, bs * 0.03], [bs * 0.98, bs * 0.1], [bs * 0.92, bs * 0.13],
    [bs * 0.4, bs * 0.16], [bs * 0.24, bs * 0.3], [bs * 0.2, h - bs * 0.6], [bs * 0.26, h - bs * 0.48], [0, h - bs * 0.48],
  ].map(([r, z]) => new THREE.Vector2(r, z));
  const foot = new THREE.Mesh(new THREE.LatheGeometry(prof, 64), metal);
  foot.rotation.x = Math.PI / 2;
  foot.position.set(mount[0] - bs * 0.55, mount[1], 0);
  // collar along the chain axis (+X) holding the base link
  const collarProf = [
    [0, -bs * 0.5], [bs * 0.62, -bs * 0.5], [bs * 0.66, -bs * 0.46], [bs * 0.66, -bs * 0.08], [bs * 0.58, 0], [0, 0],
  ].map(([r, z]) => new THREE.Vector2(r, z));
  const collar = new THREE.Mesh(new THREE.LatheGeometry(collarProf, 64), metal);
  collar.rotation.z = -Math.PI / 2;
  collar.position.set(mount[0], mount[1], mount[2]);
  const ringMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0x8cc8ff).multiplyScalar(glow), toneMapped: false });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(bs * 0.63, bs * 0.012, 8, 96), ringMat);
  ring.rotation.y = Math.PI / 2;
  ring.position.set(mount[0] - bs * 0.06, mount[1], mount[2]);
  const arm = new THREE.Mesh(new THREE.CylinderGeometry(bs * 0.16, bs * 0.16, bs * 0.6, 32), metal);
  arm.rotation.z = Math.PI / 2;
  arm.position.set(mount[0] - bs * 0.55 + bs * 0.05, mount[1], mount[2]);
  const g = new THREE.Group();
  for (const m of [foot, collar, arm]) {
    m.castShadow = true;
    m.receiveShadow = true;
  }
  g.add(foot, collar, arm, ring);
  return g;
}

/** Final pass: a soft vignette (after tone mapping). */
export function makeVignettePass(): ShaderPass {
  return new ShaderPass({
    uniforms: { tDiffuse: { value: null }, uAmount: { value: 0.55 } },
    vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tDiffuse; uniform float uAmount; varying vec2 vUv;
      void main() {
        vec4 c = texture2D(tDiffuse, vUv);
        vec2 q = vUv - 0.5;
        float v = smoothstep(0.85, 0.25, length(q * vec2(1.15, 1.0)));
        c.rgb *= mix(1.0 - uAmount, 1.0, v);
        gl_FragColor = c;
      }`,
  });
}
