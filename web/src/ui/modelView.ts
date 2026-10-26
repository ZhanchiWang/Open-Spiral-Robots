// CAD model view: the desktop tool's VTK panel (body, cable-hole cutters,
// cone-cut regions, site points, export-frame axes) plus debug aids
// (B-rep edges, wireframe, section plane).

import * as THREE from "three";
import type { BuildResult, MeshData } from "../cad/protocol";
import { axesTriad, disposeObject, Viewport } from "./viewport";

export interface ModelOverlay {
  /** site points (x1, y1, x2, y2) on the base unit, straight pose only */
  sites: [number, number, number, number] | null;
  robotLength: number;
  baseSize: number;
}

export interface ModelViewOptions {
  edges: boolean;
  wireframe: boolean;
  holes: boolean;
  sites: boolean;
  axes: boolean;
  section: boolean;
  /** 0..1 position of the section plane along the robot (x) */
  sectionPos: number;
}

export class ModelView {
  readonly vp: Viewport;
  private group = new THREE.Group();
  private clip = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
  private first = true;
  private userMoved = false;
  private lastBuild: BuildResult | null = null;
  private lastOverlay: ModelOverlay | null = null;
  opts: ModelViewOptions = { edges: true, wireframe: false, holes: true, sites: true, axes: true, section: false, sectionPos: 0.5 };
  stats = { triangles: 0, edges: 0 };

  constructor(host: HTMLElement) {
    this.vp = new Viewport(host);
    this.vp.root.add(this.group);
    this.vp.controls.addEventListener("start", () => (this.userMoved = true));
    // keep the model framed when the panel is resized, until the user moves the camera
    new ResizeObserver(() => {
      if (!this.userMoved && this.lastBuild) this.resetView(false);
    }).observe(host);
  }

  private meshOf(m: MeshData, color: number, opacity: number, depthWrite = true): THREE.Mesh {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(m.vertices, 3));
    if (m.normals.length === m.vertices.length) g.setAttribute("normal", new THREE.BufferAttribute(m.normals, 3));
    else g.computeVertexNormals();
    g.setIndex(new THREE.BufferAttribute(m.triangles, 1));
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.55,
      metalness: 0.05,
      transparent: opacity < 1,
      opacity,
      side: THREE.DoubleSide,
      depthWrite,
      wireframe: this.opts.wireframe,
      clippingPlanes: this.opts.section ? [this.clip] : [],
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    this.stats.triangles += m.triangles.length / 3;
    return new THREE.Mesh(g, mat);
  }

  private edgesOf(m: MeshData, color: number): THREE.LineSegments | null {
    if (!m.edges || !m.edges.length) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(m.edges, 3));
    this.stats.edges += m.edges.length / 6;
    return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, clippingPlanes: this.opts.section ? [this.clip] : [] }));
  }

  show(build: BuildResult, overlay: ModelOverlay): void {
    this.lastBuild = build;
    this.lastOverlay = overlay;
    this.rebuild();
    if (this.first) {
      this.first = false;
      this.resetView();
    }
  }

  setOptions(o: Partial<ModelViewOptions>): void {
    this.opts = { ...this.opts, ...o };
    this.rebuild();
  }

  private rebuild(): void {
    disposeObject(this.group);
    this.group.clear();
    this.stats = { triangles: 0, edges: 0 };
    const build = this.lastBuild;
    const ov = this.lastOverlay;
    if (!build || !ov) return;
    this.clip.constant = ov.robotLength * this.opts.sectionPos;
    if (build.main) {
      this.group.add(this.meshOf(build.main, 0xe6dccb, 1));
      if (this.opts.edges) {
        const e = this.edgesOf(build.main, 0x5b4a3a);
        if (e) this.group.add(e);
      }
    }
    if (build.elastic) {
      this.group.add(this.meshOf(build.elastic, 0xff6b4a, 1));
      if (this.opts.edges) {
        const e = this.edgesOf(build.elastic, 0x9a4f10);
        if (e) this.group.add(e);
      }
    }
    if (this.opts.holes && build.holes) this.group.add(this.meshOf(build.holes, 0xff8c33, 0.5, false));
    const tints = [0xf28c0d, 0x8ccc0d];
    build.coneRegions.forEach((m, i) => {
      if (m) this.group.add(this.meshOf(m, tints[i % 2], 0.6, false));
    });
    if (this.opts.sites && ov.sites && !build.posed) {
      const [x1, y1, x2, y2] = ov.sites;
      const r = Math.max(0.2, 0.01 * Math.max(ov.baseSize, 1));
      const geo = new THREE.SphereGeometry(r, 24, 16);
      for (const [x, y, c] of [[x1, y1, 0xff3333], [x2, y2, 0x33cc33]] as const) {
        const s = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({ color: c }));
        s.position.set(x, y, 0);
        this.group.add(s);
      }
      geo.dispose();
    }
    if (this.opts.axes) {
      // Export frame: XML X = CAD Z, XML Y = CAD Y, XML Z = -CAD X, placed at
      // the base end face (same as the desktop tool's triad).
      const triad = axesTriad(Math.max(5, ov.baseSize * 0.35));
      triad.position.set(ov.robotLength, 0, 0);
      triad.rotation.set(0, -Math.PI / 2, 0);
      this.group.add(triad);
    }
    this.vp.invalidate();
  }

  resetView(manual = true): void {
    if (manual) this.userMoved = false;
    const box = new THREE.Box3().setFromObject(this.group);
    // Close to the desktop VTK view (looking down +Z), tilted a little so
    // Z-up orbiting stays well defined.
    this.vp.camera.position.set(0.1, -0.55, 1);
    this.vp.controls.target.set(0, 0, 0);
    this.vp.fit(box, true);
  }

  clear(): void {
    this.lastBuild = null;
    this.rebuild();
  }
}
