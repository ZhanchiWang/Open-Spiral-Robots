// Live showcase physics: procedural link mesh, hot rebuild, tendon mapping
// and the scaling law (curl time ∝ length).
import { beforeAll, describe, expect, it } from "vitest";
import loadMujoco from "@mujoco/mujoco";
import { baseLinkMesh, meshVolume, renderLinkMesh } from "../src/core/linkMesh";
import { defaultParams, type Params } from "../src/core/params";
import { scaleParams } from "../src/core/scaling";
import { resolve } from "../src/core/state";
import { Physics, type SceneObject } from "../src/showcase/physics";

const NONE: SceneObject = { kind: "none", size: 0.2, pos: [0, 0, 0] };
let mj: Awaited<ReturnType<typeof loadMujoco>>;
beforeAll(async () => {
  mj = await loadMujoco();
});

function physicsFor(p: Partial<Params>, scale = 1) {
  let params = resolve({ ...defaultParams(), ...p }).params;
  if (scale !== 1) params = resolve(scaleParams(params, scale)).params;
  const r = resolve(params);
  const ph = new Physics();
  ph.mj = mj;
  ph.rebuild({ params: r.params, design: r.design!, object: NONE });
  return { ph, design: r.design! };
}

/** time until the curl angle first reaches 90% of its peak */
function curlTime(ph: Physics, horizon: number): { t90: number; peak: number; vmax: number } {
  ph.snapActivation([1, 0]);
  let t = 0;
  while (t < horizon) t += ph.advance(horizon / 200, 1e9);
  const r = ph.rec;
  let peak = 0;
  let vmax = 0;
  for (let i = 0; i < r.count; i++) {
    peak = Math.max(peak, Math.abs(r.curl(i)));
    vmax = Math.max(vmax, r.speed(i));
  }
  for (let i = 0; i < r.count; i++) if (Math.abs(r.curl(i)) >= 0.9 * peak) return { t90: r.time(i), peak, vmax };
  return { t90: NaN, peak, vmax };
}

describe("procedural link mesh", () => {
  it("2-cable prism volume = outline area x thickness", () => {
    const { params, design } = resolve(defaultParams());
    const m = baseLinkMesh(design!, params);
    const last = design!.primary[design!.primary.length - 1];
    const area = (poly: number[][]) => Math.abs(poly.reduce((s, p, i) => s + p[0] * poly[(i + 1) % poly.length][1] - poly[(i + 1) % poly.length][0] * p[1], 0)) / 2;
    expect(meshVolume(m) / (2 * area(last) * params.extrusion)).toBeCloseTo(1, 6);
    // base face at z = 0, chain along +Z
    let zmin = Infinity;
    for (let i = 2; i < m.vertices.length; i += 3) zmin = Math.min(zmin, m.vertices[i]);
    expect(zmin).toBeCloseTo(0, 6);
  });
  it("N-cable solid of revolution has positive volume", () => {
    const { params, design } = resolve({ ...defaultParams(), two_cable: false });
    expect(meshVolume(baseLinkMesh(design!, params))).toBeGreaterThan(1000);
  });
  it("display mesh (chamfered, bevelled) stays within a few % of the collision mesh", () => {
    for (const two_cable of [true, false]) {
      const { params, design } = resolve({ ...defaultParams(), two_cable });
      const ratio = meshVolume(renderLinkMesh(design!, params)) / meshVolume(baseLinkMesh(design!, params));
      expect(ratio).toBeGreaterThan(0.85);
      expect(ratio).toBeLessThanOrEqual(1.0001);
    }
  });
});

describe("live physics", () => {
  it("compiles fast and tendon 1 curls the robot upward", () => {
    const { ph } = physicsFor({});
    console.log("compile ms", ph.compileMs.toFixed(1), "bodies", ph.meta!.nbody, "mass g", (ph.meta!.totalMass * 1e3).toFixed(1));
    expect(ph.compileMs).toBeLessThan(500);
    const z0 = ph.tipOf(ph.data!)[2];
    ph.setActivation([1, 0]);
    ph.advance(1.0, 1e9);
    const z1 = ph.tipOf(ph.data!)[2];
    expect(z1).toBeGreaterThan(z0 + 0.05); // pulled cable is on the +Z side
    expect(Math.abs(ph.curl)).toBeGreaterThan(90);
  });

  it("keeps the motion through a shape rebuild", () => {
    const { ph } = physicsFor({});
    ph.snapActivation([1, 0]);
    ph.advance(0.3, 1e9);
    const q = Float64Array.from(ph.data!.qpos as Float64Array);
    const r = resolve({ ...defaultParams(), p: 0.45 });
    ph.rebuild({ params: r.params, design: r.design!, object: NONE });
    expect(Array.from(ph.data!.qpos as Float64Array)).toEqual(Array.from(q));
    expect(ph.activation[0]).toBe(1);
  });

  it("smaller robots curl faster (t ∝ L)", () => {
    const res = [0.1, 1, 4].map((s) => {
      const { ph, design } = physicsFor({ theta_max_pi: 5 }, s);
      const c = curlTime(ph, 1.5 * s);
      return { L: design.length, ...c };
    });
    console.table(res);
    for (const r of res) expect(r.peak).toBeGreaterThan(60);
    const ratio = (i: number) => res[i].t90 / res[i].L;
    expect(ratio(0) / ratio(1)).toBeGreaterThan(0.8);
    expect(ratio(0) / ratio(1)).toBeLessThan(1.25);
    expect(ratio(2) / ratio(1)).toBeGreaterThan(0.8);
    expect(ratio(2) / ratio(1)).toBeLessThan(1.25);
  }, 60_000);

  it("compiles and runs at the slider's extremes (8 mm .. 2.5 m)", () => {
    for (const mm of [8, 2500]) {
      for (const two of [true, false]) {
        const base = resolve({ ...defaultParams(), two_cable: two }).design!.length;
        const { ph, design } = physicsFor({ two_cable: two }, mm / base);
        expect(design.length).toBeCloseTo(mm, 3);
        ph.setActivation([1, 0, 0]);
        ph.advance(0.3 * (mm / 464), 1e9);
        expect(Number.isFinite(ph.curl)).toBe(true);
        expect(Math.abs(ph.curl)).toBeGreaterThan(30);
      }
    }
  });

  it("N-cable direction mapping and grasping object compile", () => {
    const { ph } = physicsFor({ two_cable: false, num_cables: 4 });
    const a = ph.activationFromVector(0, 1);
    expect(a[0]).toBeCloseTo(1); // tendon 1 sits at +Y
    expect(a[2]).toBe(0);
    const r = resolve({ ...defaultParams(), two_cable: false, num_cables: 4 });
    ph.rebuild({ params: r.params, design: r.design!, object: { kind: "cylinder", size: 0.12, pos: [0.25, 0, 0.45] } });
    expect(ph.meta!.objectMocap).toBe(0);
    ph.setActivation(a);
    ph.advance(0.5, 1e9);
    expect(Number.isFinite(ph.curl)).toBe(true);
  });
});
