// Live MuJoCo session behind the showcase. The model is regenerated from the
// parameters (procedural link mesh + the same XML generator as the export),
// recompiled in a few milliseconds and the motion state is carried over, so
// the robot keeps moving while its shape and size change.

import type { MainModule, MjData, MjModel, MjVFS } from "@mujoco/mujoco";
import { baseUnitHeight, cableCount, sitePoints, type Design } from "../core/design";
import { meshToBinarySTL } from "../core/files";
import { baseLinkMesh, type LinkMesh } from "../core/linkMesh";
import type { Params } from "../core/params";
import { timestepFor } from "../core/scaling";
import { generateMujocoXml } from "../core/xml";
import { loadMujocoModule } from "../sim/loader";

export type ObjectKind = "none" | "cylinder" | "sphere" | "box";

export interface SceneObject {
  kind: ObjectKind;
  /** characteristic size as a fraction of the robot length */
  size: number;
  /** position in metres */
  pos: [number, number, number];
}

export interface PhysicsSpec {
  params: Params;
  design: Design;
  object: SceneObject;
}

export interface Meta {
  nq: number;
  nv: number;
  nu: number;
  nbody: number;
  /** body ids of link1..linkN (base -> tip) */
  links: number[];
  tipBody: number;
  /** tip end in the tip body's frame (m) */
  tipLocal: [number, number, number];
  hinge: boolean;
  jointQpos: number[];
  timestep: number;
  lengthM: number;
  /**
   * Similarity scale λ = L / 46.4 cm. The model is simulated in units of
   * length λ m, mass λ³ kg and time λ s: under the scaling laws in
   * core/scaling.ts a robot of any size is then numerically the reference
   * robot (only gravity changes, to g·λ), so millimetre and metre robots are
   * equally well conditioned. Positions and times are converted back on the
   * way out; velocities are the same in both unit systems.
   */
  lam: number;
  /** model timestep in simulation units */
  timestepU: number;
  mount: [number, number, number];
  totalMass: number;
  objectMocap: number;
  ctrlLo: number[];
  ctrlHi: number[];
  tendonAngles: number[];
}

/** Chain along +X, cable 1 (local +Y) pointing world +Z: curls in the X-Z plane. */
const ROOT_QUAT: [number, number, number, number] = [0.5, 0.5, 0.5, 0.5];

const REC_FIELDS = 6; // t, curl, tipSpeed, tipX, tipY, tipZ
const REC_CAP = 24000;

export class Recorder {
  nq = 0;
  private buf = new Float64Array(0);
  start = 0;
  count = 0;

  reset(nq: number): void {
    if (nq !== this.nq || !this.buf.length) {
      this.nq = nq;
      this.buf = new Float64Array(REC_CAP * (REC_FIELDS + nq));
    }
    this.start = 0;
    this.count = 0;
  }

  push(t: number, curl: number, speed: number, tip: ArrayLike<number>, qpos: ArrayLike<number>): void {
    const stride = REC_FIELDS + this.nq;
    const idx = (this.start + this.count) % REC_CAP;
    if (this.count < REC_CAP) this.count++;
    else this.start = (this.start + 1) % REC_CAP;
    const o = idx * stride;
    const b = this.buf;
    b[o] = t;
    b[o + 1] = curl;
    b[o + 2] = speed;
    b[o + 3] = tip[0];
    b[o + 4] = tip[1];
    b[o + 5] = tip[2];
    for (let i = 0; i < this.nq; i++) b[o + REC_FIELDS + i] = qpos[i];
  }

  private off(i: number): number {
    return ((this.start + i) % REC_CAP) * (REC_FIELDS + this.nq);
  }

  time(i: number): number {
    return this.buf[this.off(i)];
  }
  curl(i: number): number {
    return this.buf[this.off(i) + 1];
  }
  speed(i: number): number {
    return this.buf[this.off(i) + 2];
  }
  tip(i: number): [number, number, number] {
    const o = this.off(i);
    return [this.buf[o + 3], this.buf[o + 4], this.buf[o + 5]];
  }
  qpos(i: number): Float64Array {
    const o = this.off(i) + REC_FIELDS;
    return this.buf.subarray(o, o + this.nq);
  }
  /** index of the last sample with time <= t (-1 if none is kept) */
  find(t: number): number {
    let lo = 0;
    let hi = this.count - 1;
    if (hi < 0) return -1;
    // before the oldest kept sample of a wrapped buffer: overwritten, so not
    // available (an unwrapped recording starts at the reset, so t = 0 is fine)
    if (this.count === REC_CAP && t < this.time(0)) return -1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.time(mid) <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }
}

/** Contact / integration settings of the live model (tuned in tests/contact.test.ts). */
export interface Tuning {
  selfCollision: boolean;
  /** MuJoCo solref (timeconst, dampratio) in reference-size seconds */
  solref: [number, number];
  solimp: [number, number, number, number, number];
  /** contact margin of the links, as a fraction of the robot length */
  margin: number;
  /**
   * contact margin of the scene object (fraction of the robot length): fast
   * links are caught a step before they reach it instead of tunnelling.
   */
  objectMargin: number;
  /** angular segments of the N-cable link mesh (render and collision) */
  segments: number;
  /** timestep divisor relative to the 2 ms reference step */
  substeps: number;
  optionAttrs: string;
  optionFlags: string;
  /** joint armature (rotor inertia) in reference-size kg·m² */
  armature: number;
}

/**
 * Chosen with tests/contact.test.ts: stiff contacts on a half step plus a
 * little joint armature keep links out of grasped objects (≈1 mm worst case
 * during fast motion, 0 at rest; it was 20-30 mm with MuJoCo's defaults),
 * with the free curl, curl time and tip speed unchanged.
 */
export const DEFAULT_TUNING: Tuning = {
  selfCollision: true,
  solref: [0.002, 1],
  solimp: [0.99, 0.999, 0.0005, 0.5, 2],
  margin: 0,
  objectMargin: 0.004,
  segments: 24,
  substeps: 2,
  optionAttrs: "",
  optionFlags: "",
  // A touch of rotor inertia keeps the tiny tip link from chattering against
  // objects under full tendon pull (it caused the remaining ~5 mm clipping);
  // it changes the curl time and tip speed by under 1%.
  armature: 1e-6,
};

/** time (reference-size seconds) for a tendon to go from slack to full pull */
export const TENDON_RAMP = 0.35;

export class Physics {
  tuning: Tuning = { ...DEFAULT_TUNING };
  mj: MainModule | null = null;
  model: MjModel | null = null;
  data: MjData | null = null;
  /** scratch data for replay / ghost poses (never stepped) */
  view: MjData | null = null;
  private vfs: MjVFS | null = null;
  meta: Meta | null = null;
  linkMesh: LinkMesh | null = null;
  /** current (slewed) and requested tendon activation, 0..1 */
  activation: number[] = [];
  target: number[] = [];
  readonly rec = new Recorder();
  private prevTip: [number, number, number] | null = null;
  tipSpeed = 0;
  curl = 0;
  realtime = 0;
  stepCostUs = 0;
  compileMs = 0;
  private grab: { body: number; local: [number, number, number]; target: [number, number, number]; last: [number, number, number] | null } | null = null;
  onRebuild: (() => void) | null = null;

  async init(): Promise<void> {
    this.mj ??= await loadMujocoModule();
  }

  get ready(): boolean {
    return !!this.model && !!this.data;
  }

  /** Build or rebuild the model; keeps qpos/qvel/time when the layout matches. */
  rebuild(spec: PhysicsSpec): void {
    const mj = this.mj;
    if (!mj) throw new Error("MuJoCo not loaded");
    const t0 = performance.now();
    const { params: p, design: d, object } = spec;
    const tu = this.tuning;
    const L = d.length / 1000;
    const lam = L / 0.464;
    const mountU: [number, number, number] = [0, 0, (0.62 * L) / lam];
    const n = cableCount(p);
    const objectLines: string[] = [];
    if (object.kind !== "none") {
      const om = tu.objectMargin * 0.464;
      const s = Math.max(1e-4, (object.size * L) / lam);
      const mAttr = ` margin="${om.toPrecision(6)}"`;
      const pos = object.pos.map((v) => (v / lam).toFixed(6)).join(" ");
      const geom =
        object.kind === "cylinder"
          ? `<geom type="cylinder" size="${(s / 2).toFixed(6)} ${((0.3 * L) / lam).toFixed(6)}" quat="0.70710678 -0.70710678 0 0" rgba=".95 .78 .3 1"${mAttr}/>`
          : object.kind === "sphere"
            ? `<geom type="sphere" size="${(s / 2).toFixed(6)}" rgba=".95 .78 .3 1"${mAttr}/>`
            : `<geom type="box" size="${(s / 2).toFixed(6)} ${(s / 2).toFixed(6)} ${(s / 2).toFixed(6)}" rgba=".95 .78 .3 1"${mAttr}/>`;
      objectLines.push(`<body name="object" mocap="true" pos="${pos}">${geom}</body>`);
    }
    const unitH = baseUnitHeight(d) / lam;
    const scale = Math.exp(p.b * ((p.dtheta_deg * Math.PI) / 180));
    const sp = sitePoints(d, p.tip_hole_pos / 100, p.base_hole_pos / 100);
    const xml = generateMujocoXml({
      stlName: "link.stl",
      unitHeight: unitH,
      scale,
      numUnits: d.units,
      jointType: p.two_cable ? "hinge" : "ball",
      jointLimitDeg: p.dtheta_deg,
      robotLength: d.length / lam,
      sitePoints: sp ? (sp.map((v) => v / lam) as [number, number, number, number]) : null,
      cableMode: n,
      // stiffness: kg m² s⁻² -> λ³, damping: kg m² s⁻¹ -> λ⁴, force -> λ², g -> 1/λ
      jointStiffness: p.sim_stiffness / lam ** 3,
      jointDamping: p.sim_damping / lam ** 4,
      gravity: p.sim_gravity * lam,
      jointAngles: p.joint_angles,
      bendDirDeg: p.bend_dir_deg,
      tendonConfigs: (p.tendon_configs ?? []).map((c) => ({ ...c, gear: c.gear / lam ** 2 })),
      scene: {
        head: [
          `<default><geom solref="${tu.solref.join(" ")}" solimp="${tu.solimp.join(" ")}" margin="${(tu.margin * 0.464).toPrecision(6)}"/>${tu.armature ? `<joint armature="${tu.armature}"/>` : ""}</default>`,
        ],
        optionAttrs: tu.optionAttrs,
        optionFlags: tu.optionFlags,
        rootPos: mountU,
        rootQuat: ROOT_QUAT,
        timestep: timestepFor(d.length) / lam / tu.substeps,
        worldbody: objectLines,
        linkContact: tu.selfCollision ? [1, 1] : [2, 1],
      },
    });
    const mesh = baseLinkMesh(d, p, tu.segments);
    for (let i = 0; i < mesh.vertices.length; i++) mesh.vertices[i] /= lam;
    const stl = meshToBinarySTL([mesh]);

    const vfs = new mj.MjVFS();
    vfs.addBuffer("link.stl", stl);
    let model: MjModel;
    try {
      model = mj.MjModel.from_xml_string(xml, vfs);
    } catch (e) {
      vfs.delete();
      throw new Error(`MuJoCo compile failed: ${(e as Error).message ?? e}`);
    }
    const data = new mj.MjData(model);
    const view = new mj.MjData(model);

    // carry the motion over when the joint layout is unchanged
    const old = this.data;
    const oldMeta = this.meta;
    if (old && oldMeta && oldMeta.nq === model.nq && oldMeta.nv === model.nv) {
      (data.qpos as Float64Array).set(old.qpos as Float64Array);
      // angular velocities are per unit time: convert between time units
      const qv = data.qvel as Float64Array;
      qv.set(old.qvel as Float64Array);
      for (let i = 0; i < qv.length; i++) qv[i] *= lam / oldMeta.lam;
      data.time = (old.time * oldMeta.lam) / lam;
    }
    this.dispose();
    this.model = model;
    this.data = data;
    this.view = view;
    this.vfs = vfs;
    this.linkMesh = mesh;

    const links: number[] = [];
    for (let i = 1; i <= d.units; i++) {
      const b = model.body(`link${i}`);
      links.push(b.id);
      b.delete();
    }
    const jq = model.jnt_qposadr as Int32Array;
    const mass = model.body_mass as Float64Array;
    let total = 0;
    for (const b of links) total += mass[b] * lam ** 3;
    const objBody = object.kind !== "none" ? (() => {
      const b = model.body("object");
      const id = (model.body_mocapid as Int32Array)[b.id];
      b.delete();
      return id;
    })() : -1;
    const cr = model.actuator_ctrlrange as Float64Array;
    const tipH = (unitH * scale ** -(d.units - 1)) / 1000;
    this.meta = {
      nq: model.nq,
      nv: model.nv,
      nu: model.nu,
      nbody: model.nbody,
      links,
      tipBody: links[links.length - 1],
      tipLocal: [0, 0, tipH],
      hinge: p.two_cable,
      jointQpos: Array.from({ length: model.njnt }, (_, j) => jq[j]),
      timestep: model.opt.timestep * lam,
      timestepU: model.opt.timestep,
      lengthM: L,
      lam,
      mount: [mountU[0] * lam, mountU[1] * lam, mountU[2] * lam],
      totalMass: total,
      objectMocap: objBody,
      ctrlLo: Array.from({ length: model.nu }, (_, i) => cr[2 * i]),
      ctrlHi: Array.from({ length: model.nu }, (_, i) => cr[2 * i + 1]),
      tendonAngles: Array.from({ length: n }, (_, k) => 90 + (k * 360) / n),
    };
    if (this.activation.length !== model.nu) this.activation = new Array(model.nu).fill(0);
    if (this.target.length !== model.nu) this.target = [...this.activation];
    this.applyCtrl();
    mj.mj_forward(model, data);
    this.rec.reset(model.nq);
    this.prevTip = null;
    this.compileMs = performance.now() - t0;
    this.onRebuild?.();
  }

  dispose(): void {
    this.data?.delete();
    this.view?.delete();
    this.model?.delete();
    this.vfs?.delete();
    this.data = this.view = this.model = null;
    this.vfs = null;
  }

  /**
   * Requested activation 0..1 per tendon; a pull is the negative end of
   * ctrlrange. The tendon follows within TENDON_RAMP, like a real motor, which
   * also keeps the robot from whipping through objects.
   */
  setActivation(a: number[]): void {
    this.target = a.map((v) => Math.max(0, Math.min(1, v)));
  }

  /** True when nothing moves and nothing will: tendons settled, no grab. */
  isResting(): boolean {
    const d = this.data;
    const meta = this.meta;
    if (!d || !meta || this.grab) return false;
    for (let i = 0; i < meta.nu; i++) if ((this.activation[i] ?? 0) !== (this.target[i] ?? 0)) return false;
    const qv = d.qvel as Float64Array;
    for (let i = 0; i < qv.length; i++) if (Math.abs(qv[i]) > 2e-3) return false;
    return true;
  }

  /** Apply an activation instantly (the Speed mode's strike). */
  snapActivation(a: number[]): void {
    this.setActivation(a);
    this.activation = [...this.target];
    this.applyCtrl();
  }

  private slew(): void {
    const meta = this.meta!;
    const step = meta.timestepU / TENDON_RAMP;
    let changed = false;
    for (let i = 0; i < meta.nu; i++) {
      const a = this.activation[i] ?? 0;
      const t = this.target[i] ?? 0;
      if (a !== t) {
        this.activation[i] = t > a ? Math.min(t, a + step) : Math.max(t, a - step);
        changed = true;
      }
    }
    if (changed) this.applyCtrl();
  }

  private applyCtrl(): void {
    if (!this.data || !this.meta) return;
    const ctrl = this.data.ctrl as Float64Array;
    for (let i = 0; i < this.meta.nu; i++) ctrl[i] = (this.activation[i] ?? 0) * this.meta.ctrlLo[i];
  }

  /** Map a direction in the cross-section (x, y in the base frame, |v| <= 1) to tendon activations. */
  activationFromVector(x: number, y: number): number[] {
    const m = this.meta;
    if (!m) return [];
    const mag = Math.min(1, Math.hypot(x, y));
    if (mag < 1e-6) return new Array(m.nu).fill(0);
    const phi = Math.atan2(y, x);
    return m.tendonAngles.map((deg) => {
      const c = Math.cos(phi - (deg * Math.PI) / 180);
      return m.hinge ? (c > 0.5 ? mag : 0) : Math.max(0, c) * mag;
    });
  }

  setObjectPos(pos: [number, number, number]): void {
    if (!this.data || !this.meta || this.meta.objectMocap < 0) return;
    const l = this.meta.lam;
    (this.data.mocap_pos as Float64Array).set([pos[0] / l, pos[1] / l, pos[2] / l], 3 * this.meta.objectMocap);
  }

  /** Reset to the straight rest pose (the time restarts at 0). */
  resetPose(): void {
    if (!this.mj || !this.model || !this.data) return;
    const mocap = Float64Array.from(this.data.mocap_pos as Float64Array);
    this.mj.mj_resetData(this.model, this.data);
    (this.data.mocap_pos as Float64Array).set(mocap);
    this.applyCtrl();
    this.mj.mj_forward(this.model, this.data);
    this.rec.reset(this.model.nq);
    this.prevTip = null;
    this.tipSpeed = 0;
  }

  // ---------------------------------------------------------------- grab
  startGrab(body: number, pointM: [number, number, number]): void {
    if (!this.data || !this.meta) return;
    const l = this.meta.lam;
    const worldPoint: [number, number, number] = [pointM[0] / l, pointM[1] / l, pointM[2] / l];
    const xpos = this.data.xpos as Float64Array;
    const xmat = this.data.xmat as Float64Array;
    const r = [worldPoint[0] - xpos[3 * body], worldPoint[1] - xpos[3 * body + 1], worldPoint[2] - xpos[3 * body + 2]];
    const R = xmat.subarray(9 * body, 9 * body + 9);
    // local = R^T * r
    const local: [number, number, number] = [
      R[0] * r[0] + R[3] * r[1] + R[6] * r[2],
      R[1] * r[0] + R[4] * r[1] + R[7] * r[2],
      R[2] * r[0] + R[5] * r[1] + R[8] * r[2],
    ];
    this.grab = { body, local, target: [...worldPoint], last: null };
  }

  dragTo(p: [number, number, number]): void {
    const l = this.meta?.lam ?? 1;
    if (this.grab) this.grab.target = [p[0] / l, p[1] / l, p[2] / l];
  }

  endGrab(): void {
    this.grab = null;
    if (this.data) (this.data.xfrc_applied as Float64Array).fill(0);
  }

  grabPoint(): { from: [number, number, number]; to: [number, number, number] } | null {
    if (!this.grab || !this.data) return null;
    const l = this.meta!.lam;
    const f = this.worldOf(this.data, this.grab.body, this.grab.local);
    const t = this.grab.target;
    return { from: [f[0] * l, f[1] * l, f[2] * l], to: [t[0] * l, t[1] * l, t[2] * l] };
  }

  private worldOf(d: MjData, body: number, local: [number, number, number]): [number, number, number] {
    const xpos = d.xpos as Float64Array;
    const R = (d.xmat as Float64Array).subarray(9 * body, 9 * body + 9);
    return [
      xpos[3 * body] + R[0] * local[0] + R[1] * local[1] + R[2] * local[2],
      xpos[3 * body + 1] + R[3] * local[0] + R[4] * local[1] + R[5] * local[2],
      xpos[3 * body + 2] + R[6] * local[0] + R[7] * local[1] + R[8] * local[2],
    ];
  }

  private applyGrab(): void {
    const g = this.grab;
    const d = this.data;
    const m = this.meta;
    if (!g || !d || !m) return;
    const p = this.worldOf(d, g.body, g.local);
    // spring-damper on the grabbed point, tuned to the robot's own time scale
    // 3 Hz at the reference size; in similarity units that holds at any size
    const omega = 2 * Math.PI * 3;
    const mass = m.totalMass / m.lam ** 3;
    const v = g.last ? [0, 1, 2].map((i) => (p[i] - g.last![i]) / m.timestepU) : [0, 0, 0];
    g.last = p;
    const f = [0, 1, 2].map((i) => mass * (omega * omega * (g.target[i] - p[i]) - 2 * 0.7 * omega * v[i]));
    const xipos = d.xipos as Float64Array;
    const r = [p[0] - xipos[3 * g.body], p[1] - xipos[3 * g.body + 1], p[2] - xipos[3 * g.body + 2]];
    const xf = d.xfrc_applied as Float64Array;
    xf.fill(0);
    xf.set([f[0], f[1], f[2], r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]], 6 * g.body);
  }

  // ---------------------------------------------------------------- stepping
  /**
   * Advance by `simSeconds` of simulated time, spending at most `budgetMs`
   * of wall time. Returns the simulated time actually advanced.
   */
  advance(simSeconds: number, budgetMs: number): number {
    const mj = this.mj;
    const m = this.model;
    const d = this.data;
    const meta = this.meta;
    if (!mj || !m || !d || !meta) return 0;
    const ts = meta.timestep;
    const want = Math.max(1, Math.round(simSeconds / ts));
    const t0 = performance.now();
    let done = 0;
    let diverged = false;
    for (; done < want; done++) {
      this.slew();
      this.applyGrab();
      const before = d.time;
      mj.mj_step(m, d);
      // MuJoCo resets the data itself on a bad acceleration (time and ctrl go
      // back to 0); treat that like a blow-up instead of carrying on limp
      if (d.time < before) {
        diverged = true;
        break;
      }
      this.sample();
      if ((done & 15) === 15 && performance.now() - t0 > budgetMs) {
        done++;
        break;
      }
    }
    const spent = performance.now() - t0;
    this.stepCostUs = (spent * 1000) / Math.max(1, done);
    const qpos = d.qpos as Float64Array;
    if (diverged || !Number.isFinite(qpos[0])) {
      // release the cables so the reset pose is stable, then start over
      this.target = this.target.map(() => 0);
      this.activation = this.activation.map(() => 0);
      this.resetPose();
      throw new Error("Simulation diverged and was reset (cables released)");
    }
    return done * ts;
  }

  private sample(): void {
    const d = this.data!;
    const meta = this.meta!;
    const tip = this.worldOf(d, meta.tipBody, meta.tipLocal);
    const t = d.time * meta.lam;
    if (this.prevTip) {
      const dt = meta.timestepU;
      this.tipSpeed = Math.hypot(tip[0] - this.prevTip[0], tip[1] - this.prevTip[1], tip[2] - this.prevTip[2]) / dt;
    }
    this.prevTip = tip;
    this.curl = this.curlOf(d.qpos as Float64Array);
    const l = meta.lam;
    this.rec.push(t, this.curl, this.tipSpeed, [tip[0] * l, tip[1] * l, tip[2] * l], d.qpos as Float64Array);
  }

  /** Sum of joint bend angles in degrees (signed for hinges). */
  curlOf(qpos: ArrayLike<number>): number {
    const meta = this.meta!;
    let s = 0;
    for (const a of meta.jointQpos) {
      if (meta.hinge) s += qpos[a];
      else s += 2 * Math.acos(Math.min(1, Math.abs(qpos[a])));
    }
    return (s * 180) / Math.PI;
  }

  /** Per-joint bend (degrees, base -> tip) of a pose. */
  jointBends(qpos: ArrayLike<number>): number[] {
    const meta = this.meta!;
    return meta.jointQpos.map((a) =>
      meta.hinge ? (qpos[a] * 180) / Math.PI : (2 * Math.acos(Math.min(1, Math.abs(qpos[a]))) * 180) / Math.PI,
    );
  }

  /** Put a recorded (or any) pose into the scratch data and run kinematics. */
  poseView(qpos: ArrayLike<number>): MjData | null {
    if (!this.mj || !this.model || !this.view || !this.data) return null;
    (this.view.qpos as Float64Array).set(qpos);
    (this.view.mocap_pos as Float64Array).set(this.data.mocap_pos as Float64Array);
    this.mj.mj_kinematics(this.model, this.view);
    return this.view;
  }

  /** tip position in metres */
  tipOf(d: MjData): [number, number, number] {
    const p = this.worldOf(d, this.meta!.tipBody, this.meta!.tipLocal);
    const l = this.meta!.lam;
    return [p[0] * l, p[1] * l, p[2] * l];
  }

  /** tip speed in m/s (velocity units are the same in both systems) */
  get tipSpeedM(): number {
    return this.tipSpeed;
  }

  /** simulation time in seconds */
  timeOf(d: MjData): number {
    return d.time * (this.meta?.lam ?? 1);
  }

  /** link poses in metres: xyz + wxyz per link, base -> tip */
  posesOf(d: MjData): Float32Array {
    const meta = this.meta!;
    const l = meta.lam;
    const xpos = d.xpos as Float64Array;
    const xq = d.xquat as Float64Array;
    const out = new Float32Array(meta.links.length * 7);
    meta.links.forEach((b, i) => {
      out.set([xpos[3 * b] * l, xpos[3 * b + 1] * l, xpos[3 * b + 2] * l, xq[4 * b], xq[4 * b + 1], xq[4 * b + 2], xq[4 * b + 3]], i * 7);
    });
    return out;
  }

  /** tendon paths in metres */
  tendonPaths(d: MjData): Float32Array[] {
    const m = this.model!;
    const l = this.meta!.lam;
    const adr = d.ten_wrapadr as Int32Array;
    const num = d.ten_wrapnum as Int32Array;
    const obj = d.wrap_obj as Int32Array;
    const xp = d.wrap_xpos as Float64Array;
    const out: Float32Array[] = [];
    for (let t = 0; t < m.ntendon; t++) {
      const pts: number[] = [];
      for (let j = 0; j < num[t]; j++) {
        const k = adr[t] + j;
        if (obj[k] === -2) continue;
        pts.push(xp[3 * k] * l, xp[3 * k + 1] * l, xp[3 * k + 2] * l);
      }
      out.push(new Float32Array(pts));
    }
    return out;
  }
}
