// CAD smoke test on the real OpenCascade build (slow: loads a 23 MB wasm).
import { beforeAll, describe, expect, it } from "vitest";
import opencascade from "replicad-opencascadejs";
import { makeCompound, measureVolume, setOC, type Shape3D } from "replicad";
import { baselinkSolid, buildParts, coneCutRegions, makeContext } from "../src/cad/cad";
import { defaultParams, type Params } from "../src/core/params";
import { resolve } from "../src/core/state";
import expected from "./fixtures/cad_python.json";

const vol = (s: Shape3D | null) => {
  if (!s) return 0;
  const m = s.mesh({ tolerance: 0.5, angularTolerance: 0.3 });
  let v = 0;
  const p = m.vertices;
  for (let i = 0; i < m.triangles.length; i += 3) {
    const [a, b, c] = [m.triangles[i] * 3, m.triangles[i + 1] * 3, m.triangles[i + 2] * 3];
    v += (p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c])) / 6;
  }
  return Math.abs(v);
};

function ctxFor(over: Partial<Params>) {
  const { params } = resolve({ ...defaultParams(), ...over });
  return makeContext(params, (m) => console.log("[cad]", m));
}

beforeAll(async () => {
  setOC((await opencascade()) as never);
}, 60_000);

const v = (s: Shape3D | null) => (s ? measureVolume(s) : 0);
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1e-9, Math.abs(b));
const alt = (_: number, i: number) => (i % 2 ? 8 : 12);

// Same cases as tests/fixtures/gen_cad_fixtures.py (CadQuery + DesignTool.py).
const CASES: Record<keyof typeof expected, Partial<Params> & { posed?: boolean }> = {
  "2cable": {},
  "3cable": { two_cable: false, theta_max_pi: 4 },
  "4cable_teardrop": { two_cable: false, theta_max_pi: 4, num_cables: 4, teardrop_holes: true },
  "2cable_posed": { theta_max_pi: 3.6, posed: true },
  "3cable_posed": { two_cable: false, theta_max_pi: 3.6, bend_dir_deg: 30, posed: true },
};

describe("CAD volumes match CadQuery", () => {
  for (const [name, cfg] of Object.entries(CASES)) {
    it(name, () => {
      const { posed, ...over } = cfg;
      let p = { ...over } as Partial<Params>;
      if (posed) {
        const base = resolve({ ...defaultParams(), ...over, pose_enabled: true }).params;
        p = { ...p, pose_enabled: true, joint_angles: base.joint_angles.map(alt) };
      }
      const t = performance.now();
      const parts = buildParts(ctxFor(p));
      const exp = expected[name as keyof typeof expected];
      console.log(name, Math.round(performance.now() - t), "ms", v(parts.main).toFixed(3), exp.main.toFixed(3), v(parts.elastic).toFixed(3), exp.elastic.toFixed(3));
      expect(rel(v(parts.main), exp.main)).toBeLessThan(1e-4);
      expect(rel(v(parts.elastic), exp.elastic)).toBeLessThan(1e-4);
    }, 120_000);
  }
});

describe("CAD exports", () => {
  it("baselink, STEP and cone regions", async () => {
    for (const two of [true, false]) {
      const s = baselinkSolid(ctxFor({ two_cable: two }));
      expect(vol(s)).toBeGreaterThan(10);
      // chain runs along +Z from the base face at z = 0
      expect(s!.boundingBox.bounds[0][2]).toBeGreaterThan(-1e-3);
    }
    const parts = buildParts(ctxFor({ theta_max_pi: 3 }));
    const step = await (makeCompound([parts.main!, parts.elastic!]) as Shape3D).blobSTEP().text();
    expect(step).toContain("ISO-10303-21");
    const [r1, r2] = coneCutRegions(ctxFor({ show_cone_cuts: true, theta_max_pi: 3 }));
    expect(vol(r1) + vol(r2)).toBeGreaterThan(0);
  }, 120_000);
});
