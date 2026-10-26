// Full-bleed three.js stage: the robot under water. A sand seabed with
// moving caustics, depth fog sized to the robot, drifting marine snow, a
// nacre (mother-of-pearl) robot, tendons, tip trail, stroboscopic ghosts,
// the scene object and scale references, plus pointer picking.
//
// Performance: frames are rendered on demand (simulation frame, camera
// motion, interaction) instead of continuously; line buffers are updated in
// place; three quality tiers trade shadows, resolution and ambient effects
// for speed, and "auto" steps down when the frame rate drops.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { makeBrightStars, makeInstrumentDisc, makePedestal, makePlanet, makeSkyDome, makeVignettePass } from "./scenery";
import type { LinkMesh } from "../core/linkMesh";
import type { ObjectKind } from "./physics";
import type { Frame, WorkerMeta } from "./protocol";

/** Tendon colours: coral, sea glass, kelp, sun, anemone, lagoon, urchin, slate. */
export const TENDON_HEX = [0xffa45c, 0x7cc8ff, 0xb6e36a, 0xffd36e, 0xd29bff, 0x5fe0d0, 0xff8fc0, 0xc8ccd4];

/** Glazed porcelain body; joints glow amber as they bend. */
const NACRE = new THREE.Color(0xe9e5dd);
const CORAL = new THREE.Color(0xff9a4a);

export type Quality = "high" | "medium" | "low";
/** Where the robot is shown: deep space (default) or under water. */
export type Environment = "space" | "ocean";
export type PickHit = { kind: "link"; index: number; point: THREE.Vector3 } | { kind: "object"; point: THREE.Vector3 };

/** Water colours per theme: [fog / horizon, deep]. */
const WATER = { dark: 0x0a3646, light: 0x9fd3d3 };

export function initialQuality(): Quality {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const coarse = matchMedia("(pointer: coarse)").matches;
  if ((nav.deviceMemory ?? 8) <= 2 || (navigator.hardwareConcurrency ?? 8) <= 2) return "low";
  if (coarse || (nav.deviceMemory ?? 8) <= 4 || (navigator.hardwareConcurrency ?? 8) <= 4) return "medium";
  return "high";
}

export class Stage {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(32, 1, 0.001, 1000);
  readonly controls: OrbitControls;
  private robot = new THREE.Group();
  links: THREE.Mesh[] = [];
  private linkMats: THREE.MeshStandardMaterial[] = [];
  private linkGeo: THREE.BufferGeometry | null = null;
  private edgeGeo: THREE.BufferGeometry | null = null;
  private tendons: Line2[] = [];
  private trail: Line2;
  private trailPts: number[] = [];
  private trailSpeed: number[] = [];
  private trailBox: number[] = [];
  private ghosts = new THREE.Group();
  private refs = new THREE.Group();
  private refsBox = new THREE.Box3();
  private object: THREE.Mesh | null = null;
  private objectKind: ObjectKind = "none";
  private mount = new THREE.Group();
  private seabed: THREE.Mesh;
  private seabedMat: THREE.MeshLambertMaterial;
  private seabedUniforms = {
    uTime: { value: 0 },
    uScale: { value: 0.1 },
    uGrid: { value: 0.01 },
    uSand: { value: new THREE.Color(0xd8c7a4) },
    uCaustic: { value: new THREE.Color(0xbff5ee) },
    uCausticAmp: { value: 0.35 },
    uGridAmp: { value: 0.18 },
  };
  private snow: THREE.Points | null = null;
  private snowUniforms = { uTime: { value: 0 }, uSize: { value: 2 }, uBox: { value: new THREE.Vector3(1, 1, 1) }, uOrigin: { value: new THREE.Vector3() } };
  private grabLine: Line2;
  private composer: EffectComposer | null = null;
  private bloom: UnrealBloomPass | null = null;
  private sky = makeSkyDome();
  private brightStars: THREE.Points | null = null;
  private planet = makePlanet();
  private disc: THREE.Group | null = null;
  env: Environment = "space";
  private hemi: THREE.HemisphereLight;
  private space = new THREE.Group();
  private stars: THREE.Points | null = null;
  private galaxy: THREE.Points | null = null;
  private shadowPlane: THREE.Mesh;
  private spiralB = 0.1764;
  private spaceUniforms = { uTime: { value: 0 }, uPx: { value: 1 } };
  private sun: THREE.DirectionalLight;
  private meta: WorkerMeta | null = null;
  private autoFrame = true;
  private safe = { left: 0, top: 0, right: 0, bottom: 0 };
  private viewBox: { minX: number; maxX: number; minZ: number; maxZ: number } | null = null;
  private observer: ResizeObserver;
  readonly labels: HTMLElement;
  private labelItems: { el: HTMLElement; pos: THREE.Vector3 }[] = [];
  private dirty = true;
  private dark = matchMedia("(prefers-color-scheme: dark)").matches;
  private time = 0;
  quality: Quality = initialQuality();
  autoQuality = true;
  private slowWindows = 0;
  private frameStats = { slow: 0, fast: 0 };
  heat = true;
  showEdges = false;
  showTrail = true;
  showTendons = true;
  /** rendered frames per second (0 when idle) */
  fps = 60;
  renders = 0;
  onQuality: ((q: Quality) => void) | null = null;
  /** called after each rendered frame with metres per pixel at the focus */
  onRender: ((metresPerPixel: number) => void) | null = null;

  constructor(readonly host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x000000, 1);
    host.appendChild(this.renderer.domElement);
    this.labels = document.createElement("div");
    this.labels.className = "stage-labels";
    host.appendChild(this.labels);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    this.scene.environmentIntensity = 0.5;

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.25, -1.3, 0.5);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.addEventListener("start", () => (this.autoFrame = false));
    this.controls.addEventListener("change", () => this.invalidate());

    // light from the surface: a cool sky fill and a warm key sun
    this.hemi = new THREE.HemisphereLight(0xcdeff0, 0x3b2f22, 0.9);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff3dd, 2.4);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.sun, this.sun.target);
    const rim = new THREE.DirectionalLight(0x7fd1c7, 0.8);
    rim.position.set(-1, 1.5, 0.6);
    this.scene.add(rim);

    this.seabedMat = this.makeSeabedMaterial();
    this.seabed = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.seabedMat);
    this.seabed.receiveShadow = true;
    this.shadowPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ opacity: 0.35 }));
    this.shadowPlane.receiveShadow = true;
    this.space.add(this.shadowPlane);
    this.scene.add(this.sky.mesh, this.seabed, this.space, this.robot, this.ghosts, this.refs, this.mount);
    this.shadowPlane.visible = false;
    this.brightStars = makeBrightStars(46, this.spaceUniforms);
    this.space.add(this.brightStars, this.planet.group);

    this.trail = new Line2(new LineGeometry(), new LineMaterial({ linewidth: 2.5, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
    this.trail.frustumCulled = false;
    this.trail.visible = false;
    this.scene.add(this.trail);
    this.grabLine = new Line2(new LineGeometry(), new LineMaterial({ color: 0xffc145, linewidth: 2, depthTest: false }));
    this.grabLine.renderOrder = 20;
    this.grabLine.visible = false;
    this.grabLine.frustumCulled = false;
    this.grabLine.geometry.setPositions([0, 0, 0, 0, 0, 0]);
    this.scene.add(this.grabLine);

    this.applyTheme();
    this.applyQuality();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();

    let last = performance.now();
    let lastAmbient = 0;
    let frames = 0;
    let fpsT = last;
    const loop = () => {
      requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      this.time += dt;
      const cam = this.updateCamera(dt);
      const damping = this.controls.update();
      // ambient water motion at ≤30 fps, only on the high tier
      let ambient = false;
      if (this.quality === "high" && now - lastAmbient > 33) {
        lastAmbient = now;
        ambient = true;
      }
      if (this.dirty || cam || damping || ambient) {
        // judge the device only on back-to-back frames: long gaps mean the
        // tab was idle or throttled, not that the GPU is slow
        const gap = dt * 1000;
        if (gap < 250) {
          if (gap > 26) this.frameStats.slow++;
          else this.frameStats.fast++;
        }
        this.dirty = false;
        this.updateFog();
        this.placeSky();
        this.seabedUniforms.uTime.value = this.time;
        this.snowUniforms.uTime.value = this.time;
        this.spaceUniforms.uTime.value = this.time;
        this.draw();
        this.placeLabels();
        if (this.onRender) {
          const d = this.camera.position.distanceTo(this.controls.target);
          const h = Math.max(1, this.host.clientHeight);
          this.onRender((2 * d * Math.tan((this.camera.fov * Math.PI) / 360)) / h);
        }
        this.renders++;
        frames++;
      }
      if (now - fpsT >= 1000) {
        this.fps = (frames * 1000) / (now - fpsT);
        frames = 0;
        fpsT = now;
        this.watchFps();
      }
    };
    loop();
  }

  invalidate(): void {
    this.dirty = true;
  }

  /** Render one frame (through the post-processing chain when enabled). */
  draw(): void {
    this.sky.uniforms.uTime.value = this.time;
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  }

  /**
   * HDR chain: multisampled half-float render → selective bloom (only the
   * glowing joints, tendons, rings and stars exceed the threshold) → tone
   * mapping and sRGB (OutputPass) → vignette.
   */
  private setupComposer(): void {
    this.composer?.dispose();
    this.composer = null;
    this.bloom = null;
    if (this.quality === "low") return;
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight);
    const rt = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples: this.quality === "high" ? 4 : 2 });
    const c = new EffectComposer(this.renderer, rt);
    c.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w / 2, h / 2), 0.42, 0.5, 1.0);
    c.addPass(this.bloom);
    c.addPass(new OutputPass());
    c.addPass(makeVignettePass());
    this.composer = c;
  }

  // ---------------------------------------------------------------- quality
  setQuality(q: Quality, auto = false): void {
    this.autoQuality = auto;
    if (q === this.quality) return;
    this.quality = q;
    this.applyQuality();
    this.onQuality?.(q);
  }

  private applyQuality(): void {
    const q = this.quality;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, q === "high" ? 2 : q === "medium" ? 1.5 : 1));
    this.renderer.shadowMap.enabled = q !== "low";
    this.sun.castShadow = q !== "low";
    const size = q === "high" ? 2048 : 1024;
    if (this.sun.shadow.mapSize.x !== size) {
      this.sun.shadow.mapSize.set(size, size);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null as never;
    }
    this.renderer.shadowMap.type = q === "high" ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
    if (this.snow) this.snow.visible = q === "high" && this.env === "ocean";
    this.setupComposer();
    this.seabedUniforms.uCausticAmp.value = q === "low" ? 0 : this.dark ? 0.16 : 0.14;
    // material swap: iridescent nacre only on the high tier
    if (this.meta && this.linkGeo) this.rebuildLinkMaterials();
    this.seabedMat.needsUpdate = true;
    this.resize();
  }

  /**
   * Step the quality down when most frames take longer than ~26 ms (under
   * ~38 fps) for three seconds in a row while the scene is animating.
   */
  private watchFps(): void {
    const { slow, fast } = this.frameStats;
    this.frameStats = { slow: 0, fast: 0 };
    if (!this.autoQuality || document.visibilityState !== "visible") return;
    if (slow + fast < 20) return; // not animating enough to judge
    this.slowWindows = slow > 0.6 * (slow + fast) ? this.slowWindows + 1 : 0;
    if (this.slowWindows >= 3 && this.quality !== "low") {
      this.slowWindows = 0;
      this.quality = this.quality === "high" ? "medium" : "low";
      this.applyQuality();
      this.onQuality?.(this.quality);
    }
  }

  resize(): void {
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    if (this.composer) {
      this.composer.setPixelRatio(this.renderer.getPixelRatio());
      this.composer.setSize(w, h);
    }
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.applyViewOffset();
    this.camera.updateProjectionMatrix();
    const res = new THREE.Vector2(w, h);
    for (const l of [...this.tendons, this.trail, this.grabLine]) (l.material as LineMaterial).resolution.copy(res);
    this.snowUniforms.uSize.value = this.renderer.getPixelRatio() * 2.2;
    this.spaceUniforms.uPx.value = this.renderer.getPixelRatio();
    this.invalidate();
  }

  // ---------------------------------------------------------------- theme
  setTheme(dark: boolean): void {
    this.dark = dark;
    this.applyTheme();
  }

  setEnvironment(env: Environment): void {
    if (env === this.env) return;
    this.env = env;
    if (this.meta) this.buildEnvironment(this.meta);
    else this.applyTheme();
  }

  /** Spiral growth rate of the background galaxy (the robot's own b). */
  setSpiralB(b: number): void {
    if (Math.abs(b - this.spiralB) < 1e-6) return;
    this.spiralB = b;
    if (this.meta && this.env === "space") this.makeGalaxy(this.meta.lengthM);
  }

  private applyTheme(): void {
    const space = this.env === "space";
    this.space.visible = space;
    this.seabed.visible = !space;
    if (this.snow) this.snow.visible = !space && this.quality === "high";
    if (this.disc) this.disc.visible = space;
    this.planet.group.visible = space;
    if (space) {
      // no air, no water: hard sunlight, a dim blue bounce, no fog
      this.sky.uniforms.uMode.value = 0;
      if (this.bloom) this.bloom.threshold = 1.0;
      this.scene.fog = null;
      this.hemi.color.set(0x8fa8ff);
      this.hemi.groundColor.set(0x0b0d18);
      this.hemi.intensity = 0.55;
      this.sun.color.set(0xfff6e8);
      this.sun.intensity = 2.3;
      // a studio environment would light the dark floor grey; keep only a hint
      this.scene.environmentIntensity = 0.12;
      this.invalidate();
      return;
    }
    this.hemi.color.set(0xcdeff0);
    this.hemi.groundColor.set(0x3b2f22);
    this.hemi.intensity = 0.9;
    this.sun.color.set(0xfff3dd);
    this.sun.intensity = 2.4;
    this.scene.environmentIntensity = 0.5;
    const water = new THREE.Color(this.dark ? WATER.dark : WATER.light);
    const sk = this.sky.uniforms;
    sk.uMode.value = 1;
    sk.uTop.value.set(this.dark ? 0x2a7f88 : 0xe4f6f1);
    sk.uHorizon.value.copy(water);
    sk.uDeep.value.set(this.dark ? 0x02090d : 0xcfc2a2);
    if (this.bloom) this.bloom.threshold = this.dark ? 0.95 : 1.25;
    // linear fog that follows the camera distance (see updateFog): the robot
    // stays crisp, the far seabed melts into the water
    this.scene.fog = new THREE.Fog(water, 1, 10);
    this.updateFog();
    // the sand stays a backdrop: dim in the deep, pale in the shallows
    this.seabedUniforms.uSand.value.set(this.dark ? 0x3a3a30 : 0xd9cba9);
    this.seabedUniforms.uCaustic.value.set(this.dark ? 0x6fd8c8 : 0xffffff);
    this.seabedUniforms.uCausticAmp.value = this.quality === "low" ? 0 : this.dark ? 0.16 : 0.14;
    this.seabedUniforms.uGridAmp.value = this.dark ? 0.16 : 0.12;
    this.invalidate();
  }

  private updateFog(): void {
    const fog = this.scene.fog as THREE.Fog | null;
    if (!fog) return;
    const d = this.camera.position.distanceTo(this.controls.target) || 1;
    fog.near = d * 1.15;
    fog.far = d * (this.dark ? 3.2 : 4.2);
  }

  private makeSeabedMaterial(): THREE.MeshLambertMaterial {
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const u = this.seabedUniforms;
    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, u);
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vSeaPos;")
        .replace("#include <project_vertex>", "#include <project_vertex>\nvSeaPos = (modelMatrix * vec4(transformed, 1.0)).xyz;");
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          `#include <common>
          varying vec3 vSeaPos;
          uniform float uTime, uScale, uGrid, uCausticAmp, uGridAmp;
          uniform vec3 uSand, uCaustic;
          float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
          // two drifting layers of distorted ridges: cheap caustics
          float caustic(vec2 p, float t) {
            vec2 q = p;
            float c = 0.0;
            for (int i = 0; i < 3; i++) {
              q += vec2(sin(q.y * 1.7 + t * 0.6), cos(q.x * 1.3 - t * 0.5)) * 0.45;
              c += abs(sin(q.x * 2.1) * sin(q.y * 2.3));
            }
            return pow(clamp(1.0 - c / 3.0, 0.0, 1.0), 6.0);
          }
          float gridLines(vec2 p, float s) {
            vec2 g = abs(fract(p / s - 0.5) - 0.5) / fwidth(p / s);
            return 1.0 - min(min(g.x, g.y), 1.0);
          }`,
        )
        .replace(
          "#include <color_fragment>",
          `#include <color_fragment>
          vec2 sp = vSeaPos.xy / uScale;
          float grain = hash(floor(vSeaPos.xy / (uScale * 0.012))) * 0.06;
          float ripple = 0.04 * sin(sp.x * 9.0 + sin(sp.y * 3.0) * 2.0);
          diffuseColor.rgb = uSand * (0.94 + grain + ripple);`,
        )
        .replace(
          "#include <opaque_fragment>",
          `outgoingLight += uCaustic * caustic(sp * 1.6, uTime) * uCausticAmp;
          float gl = max(gridLines(vSeaPos.xy, uGrid) * 0.35, gridLines(vSeaPos.xy, uGrid * 10.0));
          outgoingLight = mix(outgoingLight, outgoingLight * 0.55, gl * uGridAmp * 2.0);
          #include <opaque_fragment>`,
        );
    };
    return mat;
  }

  // ---------------------------------------------------------------- space
  private disposePoints(p: THREE.Points | THREE.Mesh | null): void {
    if (!p) return;
    p.geometry.dispose();
    (p.material as THREE.Material).dispose();
    p.parent?.remove(p);
  }

  /** Background stars on a far sphere, gently twinkling on the high tier. */
  private makeStars(_L: number): void {
    this.disposePoints(this.stars);
    const n = 2600;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    const R = 1; // unit sky sphere, placed and scaled around the camera (see placeSky)
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      // uniform on the sphere
      const u = Math.random() * 2 - 1;
      const phi = Math.random() * Math.PI * 2;
      const r = Math.sqrt(1 - u * u);
      pos.set([R * r * Math.cos(phi), R * r * Math.sin(phi), R * u], i * 3);
      const temp = Math.random();
      c.set(temp < 0.15 ? 0xffc9a0 : temp < 0.35 ? 0xa8c4ff : 0xffffff);
      col.set([c.r, c.g, c.b], i * 3);
      seed[i] = Math.random();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setAttribute("seed", new THREE.BufferAttribute(seed, 1));
    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: this.spaceUniforms,
      vertexShader: `
        uniform float uTime, uPx;
        attribute float seed; attribute vec3 color;
        varying vec3 vC; varying float vA;
        void main() {
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          float big = step(0.97, seed);
          gl_PointSize = uPx * (1.1 + seed * 1.6 + big * 2.0);
          vA = (0.35 + 0.65 * seed) * (0.75 + 0.25 * sin(uTime * (0.6 + seed * 2.0) + seed * 40.0));
          vC = color;
        }`,
      fragmentShader: `
        varying vec3 vC; varying float vA;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          gl_FragColor = vec4(vC, vA * smoothstep(0.5, 0.0, d));
        }`,
    });
    this.stars = new THREE.Points(g, mat);
    this.stars.frustumCulled = false;
    this.space.add(this.stars);
  }

  /**
   * A two-armed spiral galaxy behind the robot, its arms following
   * r = a·e^(bθ) with the robot's own b: the same law at 10²¹ times the size.
   */
  private makeGalaxy(_L: number): void {
    this.disposePoints(this.galaxy);
    const n = 9000;
    const b = Math.max(0.05, this.spiralB);
    const turns = 3.2 * Math.PI;
    const Rg = 1; // unit size, scaled with the sky (see placeSky)
    const a = Rg / Math.exp(b * turns);
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const core = new THREE.Color(0xffdcae);
    const arm = new THREE.Color(0x9fbfff);
    const hii = new THREE.Color(0xff8fc0);
    const c = new THREE.Color();
    const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;
    for (let i = 0; i < n; i++) {
      let x: number;
      let y: number;
      let z: number;
      if (i < n * 0.22) {
        // bulge
        const r = Math.abs(gauss()) * Rg * 0.16;
        const t = Math.random() * Math.PI * 2;
        x = r * Math.cos(t);
        y = r * Math.sin(t);
        z = gauss() * Rg * 0.04;
        c.copy(core);
      } else {
        const th = Math.pow(Math.random(), 0.8) * turns;
        const r = a * Math.exp(b * th);
        const k = i % 2 ? Math.PI : 0;
        const spread = r * 0.16 + Rg * 0.015;
        x = r * Math.cos(th + k) + gauss() * spread;
        y = r * Math.sin(th + k) + gauss() * spread;
        z = gauss() * Rg * 0.015;
        const f = Math.min(1, r / Rg);
        c.copy(core).lerp(arm, Math.pow(f, 0.6));
        if (Math.random() < 0.04) c.copy(hii);
      }
      pos.set([x, y, z], i * 3);
      col.set([c.r, c.g, c.b], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: this.spaceUniforms,
      vertexShader: `
        uniform float uTime, uPx;
        attribute vec3 color; varying vec3 vC;
        void main() {
          // slow differential rotation: inner stars turn faster
          float r = length(position.xy);
          float ang = uTime * 0.01 / (0.3 + r);
          vec2 p = mat2(cos(ang), -sin(ang), sin(ang), cos(ang)) * position.xy;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, position.z, 1.0);
          gl_PointSize = uPx * 2.2;
          vC = color;
        }`,
      fragmentShader: `
        varying vec3 vC;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          gl_FragColor = vec4(vC * 0.8, smoothstep(0.5, 0.0, d));
        }`,
    });
    this.galaxy = new THREE.Points(g, mat);
    this.galaxy.frustumCulled = false;
    // tilted toward the viewer; placed in the sky by placeSky()
    this.galaxy.rotation.set(-1.05, 0.25, 0.4);
    this.space.add(this.galaxy);
  }

  /**
   * Sky objects move with the camera but keep their direction, like a sky
   * dome: the galaxy always hangs behind the robot, up and to the right.
   */
  private placeSky(): void {
    const d = this.camera.position.distanceTo(this.controls.target) || 1;
    const R = d * 20;
    if (this.camera.far < R * 1.5) {
      this.camera.far = R * 1.5;
      this.camera.updateProjectionMatrix();
    }
    this.sky.mesh.position.copy(this.camera.position);
    this.sky.mesh.scale.setScalar(R * 1.1);
    if (this.stars) {
      this.stars.position.copy(this.camera.position);
      this.stars.scale.setScalar(R);
    }
    if (this.brightStars) {
      this.brightStars.position.copy(this.camera.position);
      this.brightStars.scale.setScalar(R);
    }
    // a planet low on the right, its lit limb arcing behind the robot
    // lit from behind and above-left: mostly night side with a thin bright crescent
    this.planet.group.position.copy(this.camera.position).addScaledVector(skyDir(1.25, -1.45), R * 0.75);
    this.planet.group.scale.setScalar(R * 0.24);
    const toCam = this.camera.position.clone().sub(this.planet.group.position).normalize();
    this.planet.uniforms.uSun.value.copy(toCam).multiplyScalar(-0.9).add(new THREE.Vector3(-0.35, 0, 0.6).normalize().multiplyScalar(0.45)).normalize();
    if (this.galaxy) {
      // upper right of the default view; fixed in the world, so it drifts
      // naturally when the camera is orbited
      this.galaxy.position.copy(this.camera.position).addScaledVector(skyDir(0.62, 0.42), R * 0.8);
      this.galaxy.scale.setScalar(R * 0.16);
    }
  }

  private makeSnow(L: number): void {
    if (this.snow) {
      this.snow.geometry.dispose();
      (this.snow.material as THREE.Material).dispose();
      this.scene.remove(this.snow);
    }
    const n = 420;
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      pos.set([Math.random(), Math.random(), Math.random()], i * 3);
      seed[i] = Math.random();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("seed", new THREE.BufferAttribute(seed, 1));
    this.snowUniforms.uBox.value.set(L * 3, L * 2, L * 1.6);
    this.snowUniforms.uOrigin.value.set(-L * 0.9, -L * 1.2, 0);
    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: this.snowUniforms,
      vertexShader: `
        uniform float uTime, uSize; uniform vec3 uBox, uOrigin;
        attribute float seed;
        varying float vA;
        void main() {
          vec3 p = position;
          p.z = fract(p.z + uTime * (0.006 + seed * 0.01));
          p.x = fract(p.x + sin(uTime * 0.2 + seed * 6.28) * 0.01);
          vec4 mv = modelViewMatrix * vec4(uOrigin + p * uBox, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = uSize * (0.6 + seed);
          vA = 0.25 + 0.35 * seed;
        }`,
      fragmentShader: `
        varying float vA;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          gl_FragColor = vec4(0.85, 0.95, 0.95, vA * smoothstep(0.5, 0.0, d));
        }`,
    });
    this.snow = new THREE.Points(g, mat);
    this.snow.frustumCulled = false;
    this.snow.visible = this.quality === "high";
    this.scene.add(this.snow);
  }

  // ---------------------------------------------------------------- robot
  setRobot(meta: WorkerMeta, mesh: LinkMesh): void {
    const lengthChanged = !this.meta || Math.abs(this.meta.lengthM - meta.lengthM) > 1e-9;
    this.meta = meta;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(mesh.vertices, 3));
    geo.setIndex(new THREE.BufferAttribute(mesh.triangles, 1));
    // smooth across the bevels, crisp at real edges
    const smooth = toCreasedNormals(geo, Math.PI / 5);
    geo.dispose();
    this.linkGeo?.dispose();
    this.edgeGeo?.dispose();
    this.linkGeo = smooth;
    this.edgeGeo = new THREE.EdgesGeometry(smooth, 28);
    this.rebuildLinkMaterials();
    for (const t of this.tendons) {
      t.geometry.dispose();
      (t.material as LineMaterial).dispose();
      this.scene.remove(t);
    }
    this.tendons = Array.from({ length: meta.nu }, (_, i) => {
      // HDR colour so the cables glow through the bloom pass
      const l = new Line2(new LineGeometry(), new LineMaterial({ color: new THREE.Color(TENDON_HEX[i % TENDON_HEX.length]).multiplyScalar(2.2), linewidth: 1.6, transparent: true, opacity: 0.95 }));
      l.frustumCulled = false;
      (l.material as LineMaterial).resolution.set(this.host.clientWidth, this.host.clientHeight);
      l.userData.n = 0;
      this.scene.add(l);
      return l;
    });
    if (lengthChanged) this.buildEnvironment(meta);
    this.clearTrail();
    this.invalidate();
  }

  private rebuildLinkMaterials(): void {
    const meta = this.meta!;
    const geo = this.linkGeo!;
    for (const m of this.linkMats) m.dispose();
    // keep the current pose: a quality switch on a resting robot gets no new
    // frame, and new links would otherwise sit at the origin
    const prev = this.links.map((l) => [l.position.clone(), l.quaternion.clone()] as const);
    this.robot.clear();
    this.links = [];
    this.linkMats = [];
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x5b4a3a, transparent: true, opacity: 0.28 });
    const high = this.quality === "high";
    meta.linkScale.forEach((s) => {
      const mat = high
        ? new THREE.MeshPhysicalMaterial({
            color: NACRE,
            roughness: 0.36,
            metalness: 0,
            clearcoat: 0.55,
            clearcoatRoughness: 0.18,
            sheen: 0.35,
            sheenRoughness: 0.5,
            sheenColor: new THREE.Color(0xffffff),
            specularIntensity: 0.6,
            side: THREE.DoubleSide,
          })
        : new THREE.MeshStandardMaterial({ color: NACRE, roughness: 0.4, metalness: 0, side: THREE.DoubleSide });
      const link = new THREE.Mesh(geo, mat);
      link.castShadow = this.quality !== "low";
      link.receiveShadow = this.quality !== "low";
      link.scale.setScalar(0.001 * s);
      const e = new THREE.LineSegments(this.edgeGeo!, edgeMat);
      e.visible = this.showEdges && this.quality !== "low";
      link.add(e);
      link.userData.index = this.links.length;
      const was = prev.length === meta.linkScale.length ? prev[this.links.length] : null;
      if (was) {
        link.position.copy(was[0]);
        link.quaternion.copy(was[1]);
      }
      this.robot.add(link);
      this.links.push(link);
      this.linkMats.push(mat);
    });
  }

  private buildEnvironment(meta: WorkerMeta): void {
    const L = meta.lengthM;
    this.seabed.scale.setScalar(L * 60);
    this.seabed.position.set(L * 0.5, 0, 0);
    this.seabedUniforms.uScale.value = L * 0.5;
    this.seabedUniforms.uGrid.value = 10 ** Math.floor(Math.log10(L / 4));
    this.makeSnow(L);
    this.makeStars(L);
    this.makeGalaxy(L);
    this.placeSky();
    if (this.disc) {
      this.disc.traverse((o) => {
        (o as THREE.Mesh).geometry?.dispose();
        ((o as THREE.Mesh).material as THREE.Material | undefined)?.dispose();
      });
      this.space.remove(this.disc);
    }
    this.disc = makeInstrumentDisc(L);
    this.space.add(this.disc);

    this.mount.traverse((o) => {
      (o as THREE.Mesh).geometry?.dispose();
      ((o as THREE.Mesh).material as THREE.Material | undefined)?.dispose();
    });
    this.mount.clear();
    const bs = Math.max(L * 0.1, 0.004);
    this.mount.add(makePedestal(meta.mount, bs, this.env === "space" ? 3 : 1.6));

    this.sun.position.set(meta.mount[0] + L * 0.4, -L * 0.8, L * 2.4);
    this.sun.target.position.set(L * 0.5, 0, 0);
    const sc = this.sun.shadow.camera;
    sc.left = -L * 1.6;
    sc.right = L * 1.6;
    sc.top = L * 1.6;
    sc.bottom = -L * 1.6;
    sc.near = L * 0.1;
    sc.far = L * 6;
    sc.updateProjectionMatrix();
    this.applyTheme();
    this.viewBox = null;
  }

  /** Apply a simulation frame (or a replayed pose). */
  applyPoses(poses: Float32Array, bends?: Float32Array): void {
    const n = Math.min(this.links.length, poses.length / 7);
    for (let i = 0; i < n; i++) {
      const o = i * 7;
      const l = this.links[i];
      l.position.set(poses[o], poses[o + 1], poses[o + 2]);
      l.quaternion.set(poses[o + 4], poses[o + 5], poses[o + 6], poses[o + 3]);
    }
    this.colorize(bends);
    this.invalidate();
  }

  private colorize(bends?: Float32Array): void {
    const lim = Math.max(1, this.meta?.jointLimitDeg ?? 30);
    this.linkMats.forEach((m, i) => {
      if (!this.heat || !bends) {
        m.color.copy(NACRE);
        m.emissive.setRGB(0, 0, 0);
        return;
      }
      // bending warms the porcelain and lights it from within (blooms)
      const b = i === 0 ? 0 : Math.abs(bends[i - 1] ?? 0);
      const t = Math.min(1, b / lim);
      const k = t * t * (3 - 2 * t);
      m.color.copy(NACRE).lerp(CORAL, k * 0.3);
      m.emissive.copy(CORAL).multiplyScalar(k * 0.55);
    });
  }

  private lastPoses: Float32Array | null = null;
  applyFrame(f: Frame): void {
    if (f.poses.length) {
      // skip the redraw when nothing moved (robot at rest)
      const lp = this.lastPoses;
      let moved = !lp || lp.length !== f.poses.length;
      if (!moved && lp) {
        for (let i = 0; i < lp.length; i++) {
          if (Math.abs(lp[i] - f.poses[i]) > 1e-6) {
            moved = true;
            break;
          }
        }
      }
      this.lastPoses = f.poses;
      if (moved) this.applyPoses(f.poses, f.bends);
      f.tendons.forEach((pts, i) => {
        const line = this.tendons[i];
        if (!line) return;
        line.visible = this.showTendons && pts.length >= 6;
        if (line.visible && moved) setLinePoints(line, pts);
      });
    }
    const g = f.grab;
    if (g) {
      this.grabLine.visible = true;
      setLinePoints(this.grabLine, [...g.from, ...g.to]);
      this.invalidate();
    } else if (this.grabLine.visible) {
      this.grabLine.visible = false;
      this.invalidate();
    }
  }

  // ---------------------------------------------------------------- trail
  clearTrail(): void {
    this.trailBox = [];
    this.trailPts = [];
    this.trailSpeed = [];
    this.trail.visible = false;
    this.invalidate();
  }

  pushTrail(tip: [number, number, number], speed: number, vmax: number): void {
    if (!this.showTrail) {
      if (this.trail.visible) this.clearTrail();
      return;
    }
    const n = this.trailPts.length;
    if (n >= 3) {
      const d = Math.hypot(tip[0] - this.trailPts[n - 3], tip[1] - this.trailPts[n - 2], tip[2] - this.trailPts[n - 1]);
      if (d < (this.meta?.lengthM ?? 0.4) * 0.004) return;
    }
    this.trailPts.push(...tip);
    this.trailSpeed.push(speed);
    if (this.trailPts.length > 3 * 400) {
      this.trailPts.splice(0, 3);
      this.trailSpeed.shift();
    }
    this.setTrail(this.trailPts, this.trailSpeed, vmax);
  }

  setTrail(pts: ArrayLike<number>, speed: ArrayLike<number>, vmax: number): void {
    if (pts.length < 6) {
      this.trail.visible = false;
      return;
    }
    const colors: number[] = [];
    const slow = new THREE.Color(0x5fd3c4);
    const fast = new THREE.Color(0xffc145);
    const c = new THREE.Color();
    for (let i = 0; i < speed.length; i++) {
      c.copy(slow).lerp(fast, Math.min(1, speed[i] / Math.max(1e-6, vmax)));
      colors.push(c.r, c.g, c.b);
    }
    this.trailBox = Array.from(pts);
    setLinePoints(this.trail, this.trailBox, colors);
    this.trail.visible = true;
    this.invalidate();
  }

  // ---------------------------------------------------------------- ghosts
  setGhosts(list: Float32Array[]): void {
    this.ghosts.traverse((o) => ((o as THREE.Mesh).material as THREE.Material | undefined)?.dispose());
    this.ghosts.clear();
    this.invalidate();
    if (!this.linkGeo || !this.meta) return;
    list.forEach((poses, k) => {
      const t = list.length > 1 ? k / (list.length - 1) : 1;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x5fd3c4).lerp(new THREE.Color(0xff6b4a), t),
        transparent: true,
        opacity: 0.1 + 0.3 * t,
        depthWrite: false,
        roughness: 0.6,
        flatShading: true,
      });
      for (let i = 0; i < poses.length / 7; i++) {
        const o = i * 7;
        const g = new THREE.Mesh(this.linkGeo!, mat);
        g.scale.setScalar(0.001 * this.meta!.linkScale[i]);
        g.position.set(poses[o], poses[o + 1], poses[o + 2]);
        g.quaternion.set(poses[o + 4], poses[o + 5], poses[o + 6], poses[o + 3]);
        this.ghosts.add(g);
      }
    });
  }

  // ---------------------------------------------------------------- object
  setObject(kind: ObjectKind, sizeM: number, pos: [number, number, number], lengthM: number): void {
    if (this.object) {
      this.object.geometry.dispose();
      (this.object.material as THREE.Material).dispose();
      this.scene.remove(this.object);
      this.object = null;
    }
    this.objectKind = kind;
    this.invalidate();
    if (kind === "none") return;
    // a polished pebble / sea-glass look
    const mat = new THREE.MeshPhysicalMaterial({ color: 0x3fa7d6, roughness: 0.15, transmission: 0, clearcoat: 1, metalness: 0, sheen: 0.3 });
    const geo =
      kind === "cylinder"
        ? new THREE.CylinderGeometry(sizeM / 2, sizeM / 2, 0.6 * lengthM, 48)
        : kind === "sphere"
          ? new THREE.SphereGeometry(sizeM / 2, 48, 32)
          : new THREE.BoxGeometry(sizeM, sizeM, sizeM);
    this.object = new THREE.Mesh(geo, mat);
    this.object.castShadow = this.quality !== "low";
    this.object.receiveShadow = true;
    this.object.position.set(...pos);
    this.scene.add(this.object);
  }

  moveObject(pos: [number, number, number]): void {
    this.object?.position.set(...pos);
    this.invalidate();
  }

  // ---------------------------------------------------------------- scale references
  setReferences(kinds: Set<"coin" | "hand" | "human">, lengthM: number): void {
    this.refs.traverse((o) => {
      (o as THREE.Mesh).geometry?.dispose();
      ((o as THREE.Mesh).material as THREE.Material | undefined)?.dispose();
    });
    this.refs.clear();
    this.labelItems.forEach((l) => l.el.remove());
    this.labelItems = [];
    const y = 0;
    let x = lengthM * 1.2;
    const add = (obj: THREE.Object3D, width: number, height: number, label: string) => {
      obj.position.x = x + width / 2;
      obj.position.y = y;
      obj.traverse((o) => {
        if ((o as THREE.Mesh).isMesh) {
          o.castShadow = true;
          o.receiveShadow = true;
        }
      });
      this.refs.add(obj);
      this.label(label, new THREE.Vector3(x + width / 2, y, height * 1.08 + lengthM * 0.02));
      x += width + Math.max(0.01, lengthM * 0.15);
    };
    const skin = new THREE.MeshStandardMaterial({ color: 0xd9a98a, roughness: 0.7 });
    const slate = new THREE.MeshStandardMaterial({ color: 0x2f4f5a, roughness: 0.8 });
    if (kinds.has("coin")) {
      const coin = new THREE.Mesh(new THREE.CylinderGeometry(0.0121, 0.0121, 0.0019, 64), new THREE.MeshStandardMaterial({ color: 0xc9a24a, metalness: 0.9, roughness: 0.3 }));
      coin.position.z = 0.0121;
      const g = new THREE.Group();
      g.add(coin);
      add(g, 0.0242, 0.0242, "Coin · 24 mm");
    }
    if (kinds.has("hand")) {
      const g = new THREE.Group();
      const palm = new THREE.Mesh(new THREE.BoxGeometry(0.085, 0.022, 0.1), skin);
      palm.position.z = 0.05;
      g.add(palm);
      for (const [fx, fl] of [[-0.03, 0.075], [-0.01, 0.085], [0.01, 0.08], [0.03, 0.065]]) {
        const f = new THREE.Mesh(new THREE.CapsuleGeometry(0.009, fl - 0.018, 6, 12), skin);
        f.position.set(fx, 0, 0.1 + fl / 2);
        g.add(f);
      }
      const thumb = new THREE.Mesh(new THREE.CapsuleGeometry(0.01, 0.045, 6, 12), skin);
      thumb.position.set(-0.052, 0, 0.065);
      thumb.rotation.y = 0.6;
      g.add(thumb);
      add(g, 0.1, 0.19, "Hand · 19 cm");
    }
    if (kinds.has("human")) {
      const g = new THREE.Group();
      const sil = new THREE.Mesh(humanSilhouette(), slate);
      sil.rotation.x = Math.PI / 2;
      sil.position.y = 0.01;
      g.add(sil);
      add(g, 0.54, 1.75, this.env === "space" ? "Astronaut · 1.75 m" : "Diver · 1.75 m");
    }
    this.refsBox = this.refs.children.length ? new THREE.Box3().setFromObject(this.refs) : new THREE.Box3();
    this.invalidate();
  }

  label(text: string, pos: THREE.Vector3): HTMLElement {
    const el = document.createElement("div");
    el.className = "stage-label";
    el.textContent = text;
    this.labels.appendChild(el);
    this.labelItems.push({ el, pos });
    return el;
  }

  private placeLabels(): void {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    const v = new THREE.Vector3();
    for (const l of this.labelItems) {
      v.copy(l.pos).project(this.camera);
      const visible = v.z < 1 && Math.abs(v.x) < 1.2 && Math.abs(v.y) < 1.2;
      l.el.style.display = visible ? "" : "none";
      l.el.style.transform = `translate(${((v.x + 1) / 2) * w}px, ${((1 - v.y) / 2) * h}px) translate(-50%, -100%)`;
    }
  }

  // ---------------------------------------------------------------- camera
  frame(_extra = 0, force = false): void {
    if (force) this.autoFrame = true;
    this.invalidate();
  }

  private targetBox(): { minX: number; maxX: number; minZ: number; maxZ: number } | null {
    if (!this.meta) return null;
    const L = this.meta.lengthM;
    const m = L * 0.1; // links extend about half a base width past their origin
    const box = { minX: -0.14 * L, maxX: 0.2 * L, minZ: 0, maxZ: this.meta.mount[2] + 0.12 * L };
    const add = (x: number, z: number) => {
      if (x - m < box.minX) box.minX = x - m;
      if (x + m > box.maxX) box.maxX = x + m;
      if (z - m < box.minZ) box.minZ = z - m;
      if (z + m > box.maxZ) box.maxZ = z + m;
    };
    for (const l of this.links) add(l.position.x, l.position.z);
    for (const g of this.ghosts.children) add(g.position.x, g.position.z);
    if (this.trail.visible) for (let i = 0; i + 2 < this.trailBox.length; i += 3) add(this.trailBox[i], this.trailBox[i + 2]);
    if (!this.refsBox.isEmpty()) {
      box.minX = Math.min(box.minX, this.refsBox.min.x);
      box.maxX = Math.max(box.maxX, this.refsBox.max.x);
      box.maxZ = Math.max(box.maxZ, this.refsBox.max.z * 1.12);
    }
    return box;
  }

  setSafeArea(a: { left: number; top: number; right: number; bottom: number }): void {
    this.safe = a;
    this.applyViewOffset();
    this.invalidate();
  }

  private applyViewOffset(): void {
    const W = Math.max(1, this.host.clientWidth);
    const H = Math.max(1, this.host.clientHeight);
    const { left, top, right, bottom } = this.safe;
    const cx = left + (W - left - right) / 2;
    const cy = top + (H - top - bottom) / 2;
    this.camera.setViewOffset(W, H, W / 2 - cx, H / 2 - cy, W, H);
  }

  /** Returns true while the camera is still moving towards its target. */
  private updateCamera(dt: number): boolean {
    if (!this.autoFrame) return false;
    const target = this.targetBox();
    if (!target) return false;
    if (!this.viewBox) this.viewBox = { ...target };
    const vb = this.viewBox;
    const grow = 1 - Math.exp(-dt * 8);
    const shrink = 1 - Math.exp(-dt * 0.7);
    vb.minX += (target.minX - vb.minX) * (target.minX < vb.minX ? grow : shrink);
    vb.minZ += (target.minZ - vb.minZ) * (target.minZ < vb.minZ ? grow : shrink);
    vb.maxX += (target.maxX - vb.maxX) * (target.maxX > vb.maxX ? grow : shrink);
    vb.maxZ += (target.maxZ - vb.maxZ) * (target.maxZ > vb.maxZ ? grow : shrink);
    const center = new THREE.Vector3((vb.minX + vb.maxX) / 2, 0, (vb.minZ + vb.maxZ) / 2);
    const hw = ((vb.maxX - vb.minX) / 2) * 1.06;
    const hh = ((vb.maxZ - vb.minZ) / 2) * 1.1;
    const W = Math.max(1, this.host.clientWidth);
    const H = Math.max(1, this.host.clientHeight);
    const sw = Math.max(80, W - this.safe.left - this.safe.right) / W;
    const sh = Math.max(80, H - this.safe.top - this.safe.bottom) / H;
    const vfov = (this.camera.fov * Math.PI) / 180;
    const tv = Math.tan(vfov / 2) * sh;
    const th = Math.tan(vfov / 2) * this.camera.aspect * sw;
    const dir = new THREE.Vector3(0.16, -1, 0.26).normalize();
    const dist = Math.max(hw / th, (hh * 1.08) / tv) + Math.abs(hw * dir.x);
    const want = center.clone().addScaledVector(dir, dist);
    const k = 1 - Math.exp(-dt * 5);
    const moving = this.camera.position.distanceTo(want) > dist * 1e-3 || this.controls.target.distanceTo(center) > dist * 1e-3;
    if (!moving) return false;
    this.controls.target.lerp(center, k);
    this.camera.position.lerp(want, k);
    this.camera.near = dist / 200;
    this.camera.far = dist * 50;
    this.camera.updateProjectionMatrix();
    return true;
  }

  recenter(): void {
    this.autoFrame = true;
    this.invalidate();
  }

  // ---------------------------------------------------------------- picking
  private ray = new THREE.Raycaster();
  pick(clientX: number, clientY: number): PickHit | null {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(ndc, this.camera);
    const targets: THREE.Object3D[] = [...this.links];
    if (this.object) targets.push(this.object);
    const hit = this.ray.intersectObjects(targets, false)[0];
    if (!hit) return null;
    if (hit.object === this.object) return { kind: "object", point: hit.point.clone() };
    return { kind: "link", index: hit.object.userData.index as number, point: hit.point.clone() };
  }

  dragPoint(clientX: number, clientY: number, anchor: THREE.Vector3, normal?: THREE.Vector3): THREE.Vector3 | null {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(ndc, this.camera);
    const n = normal ?? this.camera.getWorldDirection(new THREE.Vector3());
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, anchor);
    return this.ray.ray.intersectPlane(plane, new THREE.Vector3());
  }

  setEdges(on: boolean): void {
    this.showEdges = on;
    for (const l of this.links) l.children.forEach((c) => (c.visible = on && this.quality !== "low"));
    this.invalidate();
  }

  /** PNG of the current view (rendered now, read before the buffer is presented). */
  screenshot(): Promise<Blob> {
    this.draw();
    return new Promise((res, rej) => this.renderer.domElement.toBlob((b) => (b ? res(b) : rej(new Error("toBlob"))), "image/png"));
  }

  get objectType(): ObjectKind {
    return this.objectKind;
  }
}

/** Update a fat line in place when the point count is unchanged. */
function setLinePoints(line: Line2, pts: ArrayLike<number>, colors?: number[]): void {
  const n = Math.floor(pts.length / 3);
  const geo = line.geometry as LineGeometry;
  const start = geo.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute | undefined;
  if (start && line.userData.n === n && !colors) {
    const arr = start.data.array as Float32Array;
    for (let i = 0; i < n - 1; i++) {
      arr[i * 6] = pts[i * 3];
      arr[i * 6 + 1] = pts[i * 3 + 1];
      arr[i * 6 + 2] = pts[i * 3 + 2];
      arr[i * 6 + 3] = pts[i * 3 + 3];
      arr[i * 6 + 4] = pts[i * 3 + 4];
      arr[i * 6 + 5] = pts[i * 3 + 5];
    }
    start.data.needsUpdate = true;
    return;
  }
  geo.dispose();
  const g = new LineGeometry();
  g.setPositions(Array.from(pts));
  if (colors) g.setColors(colors);
  line.geometry = g;
  line.userData.n = n;
}

/** 1.75 m human (diver) outline, metres, extruded 1.5 cm. */
function humanSilhouette(): THREE.ExtrudeGeometry {
  const right: [number, number][] = [
    [0.055, 1.52], [0.2, 1.46], [0.245, 1.38], [0.27, 0.86], [0.235, 0.83], [0.205, 1.25],
    [0.175, 1.0], [0.15, 0.92], [0.125, 0.08], [0.15, 0.0], [0.025, 0.0], [0.0, 0.8],
  ];
  const pts = [...right, ...right.slice(0, -1).reverse().map(([x, y]) => [-x, y] as [number, number])];
  const body = new THREE.Shape(pts.map(([x, y]) => new THREE.Vector2(x, y)));
  const head = new THREE.Shape();
  head.absarc(0, 1.645, 0.105, 0, Math.PI * 2, false);
  return new THREE.ExtrudeGeometry([body, head], { depth: 0.015, bevelEnabled: true, bevelSize: 0.004, bevelThickness: 0.004, bevelSegments: 2, curveSegments: 32 });
}

/**
 * A world direction that appears at (sx, sy) in the default framing, in
 * units of the half-field of view (1 = screen edge). The default camera looks
 * along -(0.16, -1, 0.26) with a 32° vertical field.
 */
function skyDir(sx: number, sy: number): THREE.Vector3 {
  const f = new THREE.Vector3(-0.16, 1, -0.26).normalize();
  const r = new THREE.Vector3().crossVectors(f, new THREE.Vector3(0, 0, 1)).normalize();
  const u = new THREE.Vector3().crossVectors(r, f).normalize();
  const tv = Math.tan((16 * Math.PI) / 180);
  return f.addScaledVector(r, sx * tv * 1.6).addScaledVector(u, sy * tv).normalize();
}
