// Contact quality of the live model (the "no clipping" guarantee): how far
// links sink into a grasped object, during fast motion and at rest, and that
// the curled links do not overlap each other.
import { beforeAll, describe, expect, it } from "vitest";
import loadMujoco from "@mujoco/mujoco";
import { defaultParams, type Params } from "../src/core/params";
import { resolve } from "../src/core/state";
import { Physics, type ObjectKind, type SceneObject, type Tuning } from "../src/showcase/physics";
const TUNE: Partial<Tuning> = JSON.parse(process.env.TUNE ?? "{}");

let mj: Awaited<ReturnType<typeof loadMujoco>>;
beforeAll(async () => {
  mj = await loadMujoco();
});

/** Penetrations in mm: [worst during the motion, at the end]. */
function grasp(over: Partial<Params>, kind: ObjectKind, strike: boolean) {
  const r = resolve({ ...defaultParams(), ...over });
  const ph = new Physics();
  ph.mj = mj;
  ph.tuning = { ...ph.tuning, ...TUNE };
  const L = r.design!.length / 1000;
  const size = 0.16;
  const object: SceneObject = { kind, size, pos: [L * 0.62, 0, 0.62 * L + L * (0.1 + size / 2)] };
  ph.rebuild({ params: r.params, design: r.design!, object });
  ph.setObjectPos(object.pos);
  const a = ph.activationFromVector(0, 1);
  if (strike) ph.snapActivation(a);
  else ph.setActivation(a);
  const m = ph.model!;
  const d = ph.data!;
  const bodyOf = m.geom_bodyid as Int32Array;
  const links = ph.meta!.links;
  const lg: number[] = [];
  let og = -1;
  for (let g = 0; g < m.ngeom; g++) {
    if (links.includes(bodyOf[g])) lg.push(g);
    else if (bodyOf[g] > 0) og = g;
  }
  const ft = new Float64Array(6);
  const depth = () => {
    let o = 0;
    let s = 0;
    for (const g of lg) o = Math.min(o, mj.mj_geomDistance(m, d, g, og, 0.05, ft as never));
    for (let i = 0; i < lg.length; i++) for (let j = i + 2; j < lg.length; j++) s = Math.min(s, mj.mj_geomDistance(m, d, lg[i], lg[j], 0.02, ft as never));
    return [-o * ph.meta!.lam * 1000, -s * ph.meta!.lam * 1000];
  };
  let worst = 0;
  let selfWorst = 0;
  for (let k = 0; k < 40; k++) {
    ph.advance(0.05, 1e9);
    const [o, s] = depth();
    worst = Math.max(worst, o);
    selfWorst = Math.max(selfWorst, s);
  }
  return { worst, final: depth()[0], selfWorst, L: L * 1000 };
}

describe("contact quality", () => {
  for (const [label, over, kind] of [
    ["2-cable cylinder", {}, "cylinder"],
    ["2-cable box", {}, "box"],
    ["3-cable sphere", { two_cable: false }, "sphere"],
  ] as const) {
    for (const strike of [false, true]) {
      it(`${label}${strike ? " (strike)" : ""}`, () => {
        const r = grasp(over, kind, strike);
        console.log(label, strike ? "strike" : "ramp", r);
        // during fast motion: under 0.5% of the robot length (was 4-7%)
        expect(r.worst).toBeLessThan(0.005 * r.L);
        // at rest the drawn surfaces do not overlap
        expect(r.final).toBeLessThan(0.002 * r.L);
        expect(r.selfWorst).toBeLessThan(0.005 * r.L);
      }, 60_000);
    }
  }
});
