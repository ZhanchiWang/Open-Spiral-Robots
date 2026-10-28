// Shared three.js viewport: Z-up camera, orbit controls, lights, resize
// handling and render-on-demand (both the CAD view and the MuJoCo view use it).

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly root = new THREE.Group();
  private needsRender = true;
  private raf = 0;
  private observer: ResizeObserver;
  onFrame: ((dt: number) => void) | null = null;
  private last = performance.now();
  fps = 0;

  constructor(readonly host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1e6);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1, -1.4, 0.9);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.addEventListener("change", () => this.invalidate());
    this.scene.add(this.root);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2, -3, 4);
    this.camera.add(key);
    this.scene.add(this.camera);
    this.setTheme(matchMedia("(prefers-color-scheme: dark)").matches);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
    const loop = () => {
      this.raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = (now - this.last) / 1000;
      this.last = now;
      this.fps = this.fps * 0.9 + (dt > 0 ? 1 / dt : 0) * 0.1;
      if (this.onFrame) {
        this.onFrame(dt);
        this.needsRender = true;
      }
      if (this.controls.update()) this.needsRender = true;
      if (this.needsRender && this.host.offsetParent !== null) {
        this.needsRender = false;
        this.renderer.render(this.scene, this.camera);
      }
    };
    loop();
  }

  setTheme(dark: boolean): void {
    this.scene.background = new THREE.Color(dark ? (document.documentElement.dataset.env === "space" ? 0x070914 : 0x061a22) : 0xf3eee2);
    this.invalidate();
  }

  invalidate(): void {
    this.needsRender = true;
  }

  resize(): void {
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  /** Frame a bounding box from the current viewing direction. */
  fit(box: THREE.Box3, keepDirection = true): void {
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getBoundingSphere(new THREE.Sphere()).radius || 1;
    const dir = keepDirection
      ? this.camera.position.clone().sub(this.controls.target).normalize()
      : new THREE.Vector3(0.55, -0.75, 0.45).normalize();
    if (!Number.isFinite(dir.x) || dir.lengthSq() < 1e-9) dir.set(0.55, -0.75, 0.45).normalize();
    const vfov = (this.camera.fov * Math.PI) / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect);
    const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.02;
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.camera.near = Math.max(dist / 1000, 1e-4);
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.invalidate();
  }

  /** PNG of the view; `light` forces the light background (for reports). */
  screenshot(light = false): Promise<Blob> {
    const bg = this.scene.background;
    if (light) this.scene.background = new THREE.Color(0xf3eee2);
    this.renderer.render(this.scene, this.camera);
    this.scene.background = bg;
    // toBlob snapshots the canvas now, so the normal background can come back
    const out = new Promise<Blob>((res, rej) => this.renderer.domElement.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), "image/png"));
    this.invalidate();
    return out;
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.observer.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}

export function disposeObject(obj: THREE.Object3D): void {
  obj.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
    else mat?.dispose();
  });
}

export function axesTriad(length: number): THREE.Group {
  const g = new THREE.Group();
  const colors = [0xe03030, 0x30b030, 0x3060e0];
  const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  dirs.forEach((d, i) => {
    g.add(new THREE.ArrowHelper(d, new THREE.Vector3(), length, colors[i], length * 0.18, length * 0.08));
  });
  return g;
}
