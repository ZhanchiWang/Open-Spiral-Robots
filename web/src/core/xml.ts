// MuJoCo XML generation: a port of design-tool/xml_generator.py
// (generate_mujoco_xml). tests/parity.test.ts compares the output with the
// Python generator.

import { fmtF, fmtG } from "./format";

export interface TendonConfigIn {
  gear?: number;
  ctrl_lo?: number;
  ctrl_hi?: number;
  joint?: number;
}

export interface MujocoXmlOptions {
  stlName: string;
  unitHeight: number;
  scale: number;
  numUnits: number;
  jointType: "hinge" | "ball" | string;
  jointLimitDeg?: number | null;
  robotLength?: number | null;
  sitePoints?: [number, number, number, number] | null;
  cableMode?: number;
  jointStiffness?: number;
  jointDamping?: number;
  gravity?: number;
  jointAngles?: number[] | null;
  bendDirDeg?: number;
  extrusionDepth?: number;
  tendonConfigs?: (TendonConfigIn | null | undefined)[] | null;
  /**
   * Web-only scene extras for the live showcase. Leave undefined to get the
   * desktop tool's exact output.
   */
  scene?: SceneExtras;
}

export interface SceneExtras {
  /** pose of link1 (the base) in the world, metres / wxyz */
  rootPos?: [number, number, number];
  rootQuat?: [number, number, number, number];
  timestep?: number;
  /** extra attributes for <option> (e.g. integrator="implicitfast") */
  optionAttrs?: string;
  /** <flag> attributes inside <option> (e.g. multiccd="enable") */
  optionFlags?: string;
  /** extra top-level lines right after <compiler> (e.g. <default>) */
  head?: string[];
  /** extra lines inside <asset> and <worldbody> */
  asset?: string[];
  worldbody?: string[];
  /** extra top-level elements (e.g. <contact>, <visual>) */
  extra?: string[];
  /** contype/conaffinity for the robot link geoms */
  linkContact?: [number, number];
}

type V3 = [number, number, number];
type Q = [number, number, number, number];

export function generateMujocoXml(o: MujocoXmlOptions): string {
  const numUnits = Math.max(1, Math.trunc(o.numUnits));
  let jointType = String(o.jointType).toLowerCase().trim();
  if (jointType !== "hinge" && jointType !== "ball") jointType = "hinge";
  const jointLimitDeg = o.jointLimitDeg ?? null;
  const robotLength = o.robotLength ?? null;
  const sitePoints = o.sitePoints ?? null;
  const cableMode = o.cableMode ?? 2;
  const gravity = o.gravity ?? 0;
  const bendDirDeg = o.bendDirDeg ?? 0;

  const scaleStep = 1 / Math.max(o.scale, 1e-9);
  const unitScale = 0.001;
  const nCables = Math.max(2, Math.trunc(cableMode));

  const tendPlans = Array.from({ length: nCables }, (_, k) => {
    const cfg = (o.tendonConfigs && k < o.tendonConfigs.length && o.tendonConfigs[k]) || {};
    let joint = Math.trunc(cfg.joint ?? numUnits - 1);
    joint = Math.max(0, Math.min(joint, numUnits - 1));
    const endLink = Math.max(1, Math.min(joint + 1, numUnits));
    return {
      gear: cfg.gear ?? 50,
      ctrl_lo: cfg.ctrl_lo ?? -1,
      ctrl_hi: cfg.ctrl_hi ?? 1,
      joint,
      endLink,
    };
  });

  let currentScale = 1;
  const meshes: string[] = [];
  for (let i = 0; i < numUnits; i++) {
    const s = fmtF(currentScale * unitScale, 9);
    meshes.push(`    <mesh name="baselink_${i + 1}" file="${o.stlName}" scale="${s} ${s} ${s}"/>`);
    currentScale *= scaleStep;
  }

  const jointXml = (stiffness: number, damping: number): string => {
    if (jointType === "hinge") {
      if (jointLimitDeg === null) {
        return `<joint type="hinge" axis="1 0 0" stiffness="${fmtF(stiffness, 6)}" damping="${fmtF(damping, 6)}"/>`;
      }
      const lim = (Math.abs(jointLimitDeg) * Math.PI) / 180;
      return (
        `<joint type="hinge" axis="1 0 0" range="${fmtF(-lim, 9)} ${fmtF(lim, 9)}" ` +
        `stiffness="${fmtF(stiffness, 6)}" damping="${fmtF(damping, 6)}"/>`
      );
    }
    return `<joint type="ball" stiffness="${fmtF(stiffness, 6)}" damping="${fmtF(damping, 6)}"/>`;
  };

  const buildSites = (scaleFactor: number, linkIndex: number): { xml: string; pos: V3 }[] => {
    if (sitePoints === null || robotLength === null) return [];
    const [x1, y1, x2, y2] = sitePoints;
    const z1 = robotLength - x1;
    const z2 = robotLength - x2;
    const n = Math.max(2, Math.trunc(cableMode));
    const sitesMm: V3[] = [];
    for (let k = 0; k < n; k++) {
      const ang = ((90 + (k * 360) / n) * Math.PI) / 180;
      let ca = Math.cos(ang);
      let sa = Math.sin(ang);
      if (Math.abs(ca) < 1e-12) ca = 0;
      if (Math.abs(sa) < 1e-12) sa = 0;
      sitesMm.push([y2 * ca, y2 * sa, z2]);
      sitesMm.push([y1 * ca, y1 * sa, z1]);
    }
    const sizeM = 0.001 * scaleFactor;
    return sitesMm.map(([sx, sy, sz], i) => {
      const px = sx * scaleFactor * unitScale;
      const py = sy * scaleFactor * unitScale;
      const pz = sz * scaleFactor * unitScale;
      // The Python code re-parses the formatted string for lengths; keep the
      // rounded values so tendon ranges match exactly.
      const fx = fmtF(px, 9);
      const fy = fmtF(py, 9);
      const fz = fmtF(pz, 9);
      const sz6 = fmtF(sizeM, 6);
      return {
        xml: `<site name="s${linkIndex}_${i + 1}" pos="${fx} ${fy} ${fz}" size="${sz6} ${sz6} ${sz6}"/>`,
        pos: [Number(fx), Number(fy), Number(fz)] as V3,
      };
    });
  };

  currentScale = 1;
  const lines: string[] = [];
  lines.push('<mujoco model="spiral_robot">');
  lines.push('  <compiler angle="radian"/>');
  const gvec = gravity === 0 ? "0 0 0" : `0 0 ${fmtG(-gravity, 9)}`;
  const sc = o.scene;
  if (sc?.head) lines.push(...sc.head.map((l) => `  ${l}`));
  const optHead = `  <option gravity="${gvec}"${sc?.timestep ? ` timestep="${fmtG(sc.timestep, 6)}"` : ""}${sc?.optionAttrs ? ` ${sc.optionAttrs}` : ""}`;
  lines.push(sc?.optionFlags ? `${optHead}><flag ${sc.optionFlags}/></option>` : `${optHead}/>`);
  lines.push("  <asset>");
  lines.push(...meshes);
  if (sc?.asset) lines.push(...sc.asset.map((l) => `    ${l}`));
  lines.push(
    '    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>',
  );
  lines.push(
    '    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>',
  );
  lines.push("  </asset>");
  lines.push("  <worldbody>");
  lines.push('    <geom name="ground" type="plane" size="10 10 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>');
  lines.push(
    '    <light pos="1 0.5 0.7" dir="-1 -0.5 -1" directional="true" castshadow="false" diffuse="1 1 1" specular="0.3 0.3 0.3"/>',
  );

  let stiffness = Number(o.jointStiffness ?? 0.04);
  let damping = Number(o.jointDamping ?? 0.02);

  const jointAnglesRad: number[] = (o.jointAngles ?? []).slice(0, numUnits - 1).map((a) => (a * Math.PI) / 180);
  while (jointAnglesRad.length < numUnits - 1) jointAnglesRad.push(0);

  let jointAxis: V3;
  if (jointType === "hinge") {
    jointAxis = [1, 0, 0];
  } else {
    const beta = (bendDirDeg * Math.PI) / 180;
    jointAxis = [Math.cos(beta), -Math.sin(beta), 0];
  }

  const quatStr = (angle: number): string => {
    if (Math.abs(angle) < 1e-9) return "";
    const half = angle / 2;
    const s = Math.sin(half);
    return ` quat="${fmtF(Math.cos(half), 9)} ${fmtF(jointAxis[0] * s, 9)} ${fmtF(jointAxis[1] * s, 9)} ${fmtF(jointAxis[2] * s, 9)}"`;
  };

  if (sc?.worldbody) lines.push(...sc.worldbody.map((l) => `    ${l}`));
  const rootPos = sc?.rootPos ? sc.rootPos.map((v) => fmtF(v, 9)).join(" ") : `0 0 ${fmtF(0, 9)}`;
  const rootQuat = sc?.rootQuat ? ` quat="${sc.rootQuat.map((v) => fmtF(v, 9)).join(" ")}"` : "";
  const geomContact = sc?.linkContact ? ` contype="${sc.linkContact[0]}" conaffinity="${sc.linkContact[1]}"` : "";
  lines.push(`    <body name="link1" pos="${rootPos}"${rootQuat}>`);
  lines.push(`      <geom type="mesh" mesh="baselink_1"${geomContact}/>`);
  for (const site of buildSites(1, 1)) lines.push(`      ${site.xml}`);

  for (let i = 1; i < numUnits; i++) {
    const zOffset = o.unitHeight * currentScale * unitScale;
    const linkIdx = i + 1;
    lines.push(`      <body name="link${linkIdx}" pos="0 0 ${fmtF(zOffset, 9)}"${quatStr(jointAnglesRad[i - 1])}>`);
    lines.push(`        ${jointXml(stiffness, damping)}`);
    lines.push(`        <geom type="mesh" mesh="baselink_${i + 1}"${geomContact}/>`);
    const childScale = currentScale * scaleStep;
    for (const site of buildSites(childScale, linkIdx)) lines.push(`        ${site.xml}`);
    currentScale *= scaleStep;
    stiffness *= scaleStep * scaleStep;
    damping *= scaleStep;
  }
  for (let i = 0; i < numUnits - 1; i++) lines.push("      </body>");
  lines.push("    </body>");
  lines.push("  </worldbody>");

  const tendonLengths = new Map<number, number>();
  if (numUnits > 0 && sitePoints !== null && robotLength !== null) {
    const localSites = new Map<number, V3[]>();
    let sc = 1;
    for (let link = 1; link <= numUnits; link++) {
      localSites.set(link, buildSites(sc, link).map((s) => s.pos));
      sc *= scaleStep;
    }
    const rotate = (q: Q, v: V3): V3 => {
      const [w, qx, qy, qz] = q;
      const [vx, vy, vz] = v;
      const tx = 2 * (qy * vz - qz * vy);
      const ty = 2 * (qz * vx - qx * vz);
      const tz = 2 * (qx * vy - qy * vx);
      return [vx + w * tx + (qy * tz - qz * ty), vy + w * ty + (qz * tx - qx * tz), vz + w * tz + (qx * ty - qy * tx)];
    };
    const qmul = (a: Q, b: Q): Q => {
      const [w1, x1, y1, z1] = a;
      const [w2, x2, y2, z2] = b;
      return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      ];
    };
    const bendQuat = (angle: number): Q => {
      const s = Math.sin(angle / 2);
      return [Math.cos(angle / 2), jointAxis[0] * s, jointAxis[1] * s, jointAxis[2] * s];
    };
    const worldSites = new Map<number, V3[]>();
    let worldPos: V3 = [0, 0, 0];
    let worldQuat: Q = [1, 0, 0, 0];
    sc = 1;
    const toWorld = (pts: V3[]) =>
      pts.map((p) => {
        const r = rotate(worldQuat, p);
        return [worldPos[0] + r[0], worldPos[1] + r[1], worldPos[2] + r[2]] as V3;
      });
    worldSites.set(1, toWorld(localSites.get(1) ?? []));
    for (let i = 1; i < numUnits; i++) {
      const zOffset = o.unitHeight * sc * unitScale;
      const tr = rotate(worldQuat, [0, 0, zOffset]);
      worldPos = [worldPos[0] + tr[0], worldPos[1] + tr[1], worldPos[2] + tr[2]];
      worldQuat = qmul(worldQuat, bendQuat(jointAnglesRad[i - 1]));
      worldSites.set(i + 1, toWorld(localSites.get(i + 1) ?? []));
      sc *= scaleStep;
    }
    for (let k = 0; k < nCables; k++) {
      const pts: V3[] = [];
      for (let i = 1; i <= tendPlans[k].endLink; i++) {
        const arr = worldSites.get(i) ?? [];
        for (const siteIdx of [2 * k + 1, 2 * k + 2]) {
          if (siteIdx - 1 >= 0 && siteIdx - 1 < arr.length) pts.push(arr[siteIdx - 1]);
        }
      }
      let total = 0;
      for (let i = 0; i + 1 < pts.length; i++) {
        total += Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1], pts[i + 1][2] - pts[i][2]);
      }
      tendonLengths.set(k, total);
    }
  }

  lines.push("  <tendon>");
  for (let k = 0; k < nCables; k++) {
    let tLen = tendonLengths.get(k) ?? 0;
    if (tLen <= 0 && robotLength !== null) tLen = robotLength * unitScale * 2;
    const lo = Math.max(0.001, tLen * 0.3);
    const hi = Math.max(lo + 0.001, tLen * 3);
    lines.push(
      `    <spatial name="tendon${k + 1}" width="0.0007" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="${fmtF(lo, 9)} ${fmtF(hi, 9)}">`,
    );
    for (let i = 1; i <= tendPlans[k].endLink; i++) {
      lines.push(`      <site site="s${i}_${2 * k + 1}"/>`);
      lines.push(`      <site site="s${i}_${2 * k + 2}"/>`);
    }
    lines.push("    </spatial>");
  }
  lines.push("  </tendon>");
  lines.push("  <actuator>");
  for (let k = 0; k < nCables; k++) {
    const p = tendPlans[k];
    lines.push(
      `    <motor name="act${k + 1}" tendon="tendon${k + 1}" ctrllimited="true" ` +
        `ctrlrange="${fmtG(p.ctrl_lo, 9)} ${fmtG(p.ctrl_hi, 9)}" gear="${fmtG(p.gear, 9)}"/>`,
    );
  }
  lines.push("  </actuator>");
  if (sc?.extra) lines.push(...sc.extra.map((l) => `  ${l}`));
  lines.push("</mujoco>");
  return lines.join("\n") + "\n";
}
