/// <reference lib="webworker" />
// Physics worker: owns MuJoCo, rebuilds the model when the design changes,
// steps in (scaled) real time and streams link poses to the renderer.

import { resolve } from "../core/state";
import { Physics } from "./physics";
import type { Frame, FromWorker, ToWorker } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

const ph = new Physics();
let running = true;
let speed = 1;
let pendingBuild: Extract<ToWorker, { type: "build" }> | null = null;
let lastWall = performance.now();
let realtime = 0;
let epoch = 0;
// idle handling: once the robot has been at rest for a while, stop stepping
// and slow the loop down; any message wakes it up again
let restSince = 0;
let timer = 0;
const IDLE_AFTER_MS = 600;

const post = (m: FromWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);

function doBuild(msg: Extract<ToWorker, { type: "build" }>) {
  const { params, design } = resolve(msg.params);
  if (!design) throw new Error("No unit fits in θ max");
  ph.rebuild({ params, design, object: msg.object });
  if (msg.resetPose) ph.resetPose();
  ph.setObjectPos(msg.object.pos);
  const meta = ph.meta!;
  const step = Math.exp(params.b * ((params.dtheta_deg * Math.PI) / 180));
  post({
    type: "built",
    meta: {
      seq: msg.seq,
      units: design.units,
      hinge: meta.hinge,
      nu: meta.nu,
      lengthM: meta.lengthM,
      mount: meta.mount,
      totalMass: meta.totalMass,
      timestep: meta.timestep,
      compileMs: ph.compileMs,
      tendonAngles: meta.tendonAngles,
      linkScale: Array.from({ length: design.units }, (_, i) => step ** -i),
      jointLimitDeg: params.dtheta_deg,
    },
  });
}

function sendFrame() {
  const d = ph.data;
  if (!d || !ph.meta) return;
  const poses = ph.posesOf(d);
  const tendons = ph.tendonPaths(d);
  const bends = new Float32Array(ph.jointBends(d.qpos as Float64Array));
  const r = ph.rec;
  const frame: Frame = {
    type: "frame",
    epoch,
    time: ph.timeOf(d),
    poses,
    tendons,
    bends,
    curl: ph.curl,
    tipSpeed: ph.tipSpeedM,
    tip: ph.tipOf(d),
    realtime,
    stepUs: ph.stepCostUs,
    recFrom: r.count ? r.time(0) : 0,
    recTo: r.count ? r.time(r.count - 1) : 0,
    grab: ph.grabPoint(),
  };
  post(frame, [poses.buffer, bends.buffer, ...tendons.map((t) => t.buffer)]);
}

function tick() {
  const now = performance.now();
  const wall = Math.min(0.05, (now - lastWall) / 1000);
  lastWall = now;
  try {
    if (pendingBuild) {
      const b = pendingBuild;
      pendingBuild = null;
      doBuild(b);
    }
    const idle = restSince > 0 && now - restSince > IDLE_AFTER_MS;
    if (running && ph.ready && !idle) {
      const advanced = ph.advance(wall * speed, 9);
      realtime = wall > 0 ? advanced / wall : 0;
      if (ph.isResting()) restSince ||= now;
      else restSince = 0;
    } else {
      realtime = idle ? speed : 0;
    }
    sendFrame();
  } catch (e) {
    post({ type: "error", message: (e as Error).message ?? String(e) });
  }
  const idle = restSince > 0 && performance.now() - restSince > IDLE_AFTER_MS;
  timer = self.setTimeout(tick, idle ? 250 : Math.max(0, 14 - (performance.now() - now)));
}

function wake() {
  restSince = 0;
  clearTimeout(timer);
  lastWall = performance.now();
  timer = self.setTimeout(tick, 0);
}

self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  if (m.type !== "poses" && m.type !== "series" && restSince > 0) wake();
  try {
    switch (m.type) {
      case "build":
        pendingBuild = m;
        break;
      case "activation":
        ph.setActivation(m.a);
        break;
      case "run":
        running = m.running;
        speed = m.speed;
        break;
      case "reset":
        ph.resetPose();
        epoch++;
        break;
      case "strike": {
        ph.resetPose();
        epoch++;
        ph.snapActivation(ph.activationFromVector(0, m.a));
        running = true;
        break;
      }
      case "object":
        ph.setObjectPos(m.pos);
        break;
      case "grab":
        ph.startGrab(ph.meta!.links[m.body], m.point);
        break;
      case "drag":
        ph.dragTo(m.point);
        break;
      case "release":
        ph.endGrab();
        break;
      case "poses": {
        const r = ph.rec;
        const poses: Float32Array[] = [];
        const bends: Float32Array[] = [];
        for (const t of m.times) {
          const i = r.find(t);
          if (i < 0) continue;
          const v = ph.poseView(r.qpos(i));
          if (!v) continue;
          poses.push(ph.posesOf(v));
          bends.push(new Float32Array(ph.jointBends(r.qpos(i))));
        }
        post({ type: "poses", id: m.id, poses, bends });
        break;
      }
      case "series": {
        const r = ph.rec;
        const n = Math.max(2, m.n);
        const t = new Float64Array(n);
        const curl = new Float64Array(n);
        const sp = new Float64Array(n);
        const tip = new Float64Array(3 * n);
        let i0 = Math.max(0, r.find(m.tFrom));
        const i1 = Math.max(i0, r.find(m.tTo));
        for (let k = 0; k < n; k++) {
          const target = m.tFrom + ((m.tTo - m.tFrom) * k) / (n - 1);
          // max speed inside each bucket so peaks survive downsampling
          let best = -1;
          let idx = i0;
          while (idx <= i1 && r.time(idx) <= target) {
            best = Math.max(best, r.speed(idx));
            idx++;
          }
          const j = Math.min(i1, Math.max(0, idx - 1));
          t[k] = r.count ? r.time(j) : target;
          curl[k] = r.count ? r.curl(j) : 0;
          sp[k] = best >= 0 ? best : r.count ? r.speed(j) : 0;
          tip.set(r.count ? r.tip(j) : [0, 0, 0], 3 * k);
          i0 = idx;
        }
        post({ type: "series", id: m.id, t, curl, speed: sp, tip }, [t.buffer, curl.buffer, sp.buffer, tip.buffer]);
        break;
      }
    }
  } catch (e) {
    post({ type: "error", message: (e as Error).message ?? String(e) });
  }
};

ph.init()
  .then(() => {
    post({ type: "ready" });
    tick();
  })
  .catch((e) => post({ type: "error", message: `MuJoCo failed to load: ${(e as Error).message ?? e}` }));
