// MuJoCo WASM: the generated XML (real design geometry, straight pose)
// compiles, the tendons produce joint torque and the chain curls.
import { describe, expect, it } from "vitest";
import loadMujoco from "@mujoco/mujoco";
import { baseUnitHeight, cableCount, sitePoints } from "../src/core/design";
import { meshToBinarySTL } from "../src/core/files";
import { baseLinkMesh } from "../src/core/linkMesh";
import { defaultParams, type Params } from "../src/core/params";
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

/** Compile the exported XML for a design with its real link mesh. */
async function modelFor(over: Partial<Params>) {
  const mj = await loadMujoco();
  const { params: p, design: d } = resolve({ ...defaultParams(), ...over });
  const xml = generateMujocoXml({
    stlName: "baselink.stl",
    unitHeight: baseUnitHeight(d!),
    scale: Math.exp(p.b * ((p.dtheta_deg * Math.PI) / 180)),
    numUnits: d!.units,
    jointType: p.two_cable ? "hinge" : "ball",
    jointLimitDeg: p.dtheta_deg,
    robotLength: d!.length,
    sitePoints: sitePoints(d!, p.tip_hole_pos / 100, p.base_hole_pos / 100),
    cableMode: cableCount(p),
    jointStiffness: p.sim_stiffness,
    jointDamping: p.sim_damping,
    jointAngles: p.joint_angles,
    bendDirDeg: p.bend_dir_deg,
    tendonConfigs: p.tendon_configs,
  });
  const vfs = new mj.MjVFS();
  vfs.addBuffer("baselink.stl", meshToBinarySTL([baseLinkMesh(d!, p, 24)]));
  const model = mj.MjModel.from_xml_string(xml, vfs);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  const done = () => (data.delete(), model.delete(), vfs.delete());
  return { mj, model, data, xml, p, done };
}

describe("MuJoCo export regressions", () => {
  it("a posed export bends the same way as the CAD (+angle -> tip toward +y)", async () => {
    const base = resolve({ ...defaultParams(), theta_max_pi: 3.6, pose_enabled: true }).params;
    const { model, data, done } = await modelFor({ ...base, joint_angles: base.joint_angles.map(() => 10) });
    const tip = model.nbody - 1;
    const y = (data.xpos as Float64Array)[3 * tip + 1];
    expect(y).toBeGreaterThan(0.01); // CAD (_apply_joint_pose) puts it at +34.5 mm
    done();
  }, 60_000);

  it("ball joints keep the ±Δθ joint limit", async () => {
    const base = resolve({ ...defaultParams(), two_cable: false, theta_max_pi: 4 }).params;
    const { mj, model, data, xml, p, done } = await modelFor({ ...base, tendon_configs: base.tendon_configs!.map((c) => ({ ...c, gear: 250 })) });
    expect(xml).toMatch(/<joint type="ball" range="0 /);
    data.ctrl[0] = -1;
    let worst = 0;
    const q = data.qpos as Float64Array;
    for (let i = 0; i < 1500; i++) {
      mj.mj_step(model, data);
      for (let j = 0; j + 3 < q.length; j += 4) worst = Math.max(worst, 2 * Math.acos(Math.min(1, Math.abs(q[j]))));
    }
    // soft limits may overshoot slightly; unlimited ball joints reached 32°+
    expect((worst * 180) / Math.PI).toBeLessThan(p.dtheta_deg + 1.5);
    done();
  }, 60_000);
});
