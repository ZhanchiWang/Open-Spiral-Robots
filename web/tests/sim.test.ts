// MuJoCo WASM: the generated XML (real design geometry, straight pose)
// compiles, the tendons produce joint torque and the chain curls.
import { describe, expect, it } from "vitest";
import loadMujoco from "@mujoco/mujoco";
import { baseUnitHeight, cableCount, sitePoints } from "../src/core/design";
import { meshToBinarySTL } from "../src/core/files";
import { defaultParams } from "../src/core/params";
import { resolve } from "../src/core/state";
import { generateMujocoXml } from "../src/core/xml";

/** A 20 x 20 x 14 mm box standing on z = 0 (stand-in for baselink.stl). */
function boxStl(): Uint8Array {
  const S = 20;
  const v = [0, 0, 0, S, 0, 0, S, S, 0, 0, S, 0, 0, 0, S, S, 0, S, S, S, S, 0, S, S].map((x, i) => (i % 3 === 2 ? x * 0.7 : x - S / 2));
  const t = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7];
  return meshToBinarySTL([{ vertices: v, triangles: t }]);
}

describe("MuJoCo", () => {
  for (const two of [true, false]) {
    it(`${two ? "2" : "3"}-cable straight chain curls when tendon 1 pulls`, async () => {
      const mj = await loadMujoco();
      const { params: p, design: d } = resolve({ ...defaultParams(), two_cable: two, theta_max_pi: 4 });
      const xml = generateMujocoXml({
        stlName: "baselink.stl",
        unitHeight: baseUnitHeight(d!),
        scale: Math.exp(p.b * ((p.dtheta_deg * Math.PI) / 180)),
        numUnits: d!.units,
        jointType: two ? "hinge" : "ball",
        jointLimitDeg: p.dtheta_deg,
        robotLength: d!.length,
        sitePoints: sitePoints(d!, p.tip_hole_pos / 100, p.base_hole_pos / 100),
        cableMode: cableCount(p),
        jointStiffness: p.sim_stiffness,
        jointDamping: p.sim_damping,
      });
      const vfs = new mj.MjVFS();
      vfs.addBuffer("baselink.stl", boxStl());
      const model = mj.MjModel.from_xml_string(xml, vfs);
      const data = new mj.MjData(model);
      expect(model.nu).toBe(cableCount(p));
      data.ctrl[0] = 1;
      mj.mj_forward(model, data);
      // tendon 1 must already produce joint torque in the straight pose
      const qfrc = Array.from(data.qfrc_actuator as Float64Array);
      expect(Math.max(...qfrc.map(Math.abs))).toBeGreaterThan(0.1);
      for (let i = 0; i < 500; i++) mj.mj_step(model, data);
      const q = Array.from(data.qpos as Float64Array);
      expect(q.every(Number.isFinite)).toBe(true);
      if (two) {
        // hinge angles grow toward the tip (spiral curl)
        expect(q[0]).toBeGreaterThan(0.01);
        expect(q[q.length - 1]).toBeGreaterThan(q[0]);
      } else {
        // first ball joint quaternion has rotated away from identity
        expect(Math.abs(q[0])).toBeLessThan(0.99999);
      }
      const act = model.actuator(0);
      expect(act.name).toBe("act1");
      act.delete();
      data.delete();
      model.delete();
      vfs.delete();
    }, 60_000);
  }
});
