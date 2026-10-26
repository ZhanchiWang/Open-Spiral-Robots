// Parity with the desktop tool. Fixtures come from the Python code itself:
//   python3 web/tests/fixtures/gen_fixtures.py > web/tests/fixtures/python.json
import { describe, expect, it } from "vitest";
import { computeDesign, elasticUnitPolys, jointFillerDefaults } from "../src/core/design";
import { fmtG } from "../src/core/format";
import { defaultParams, parseParams, serializeParams } from "../src/core/params";
import { generateMujocoXml } from "../src/core/xml";
import fixtures from "./fixtures/python.json";

type Spins = Record<string, number | boolean>;
const CASE_SPINS: Record<string, Spins> = {
  default: {},
  fine: { a: 2.3, b: 0.12, dtheta_deg: 17, theta_max_pi: 7.3, p: 0.31, elastic_percent: 40 },
  coarse_no_elastic: { a: 12, b: 0.3, dtheta_deg: 45, theta_max_pi: 4.4, p: 0.12, elastic_enabled: false, extrusion: 100 },
  short: { theta_max_pi: 2.4, dtheta_deg: 10 },
};

const close = (a: number, b: number) => expect(a).toBeCloseTo(b, 9);
const closePoly = (a: number[][] | null, b: number[][] | null) => {
  if (a === null || b === null) {
    expect(a).toBe(b);
    return;
  }
  expect(a.length).toBe(b.length);
  a.forEach((p, i) => p.forEach((v, j) => close(v, b[i][j])));
};

describe("2D geometry matches DesignTool.py", () => {
  for (const fx of fixtures.designs) {
    it(fx.name, () => {
      const params = { ...defaultParams(), extrusion: 5, ...CASE_SPINS[fx.name] };
      const d = computeDesign(params)!;
      expect(d).not.toBeNull();
      expect(d.polar.unitCount).toBe(fx.units_polar);
      expect(d.polar.theta.length).toBe(fx.theta_len);
      expect(d.primary.length).toBe(fx.primary.length);
      d.primary.forEach((p, i) => closePoly(p, fx.primary[i]));
      d.mirror.forEach((p, i) => closePoly(p, fx.mirror[i]));
      d.polar.unitsMirror.forEach(([t, r], i) => {
        t.forEach((v, j) => close(v, fx.unitsMirror[i][0][j]));
        r.forEach((v, j) => close(v, fx.unitsMirror[i][1][j]));
      });
      closePoly(d.elastic, fx.elastic);
      closePoly([d.ray.start, d.ray.upper, d.ray.lower], fx.ray);
      close(d.baseSize, fx.baseSize);
      close(d.tipSize, fx.tipSize);
      close(d.taperDeg, fx.taperDeg);
      close(d.length, fx.length);
      const eu = elasticUnitPolys(d).polys;
      eu.forEach((p, i) => closePoly(p, fx.elasticUnits[i]));
      const [lo, hi] = d.extrusionRange;
      const ext = Math.max(lo, Math.min(hi, Number(params.extrusion)));
      const f = jointFillerDefaults(d, ext);
      f.widths.forEach((v, i) => close(v, fx.fillerWidths[i]));
      f.heights.forEach((v, i) => close(v, fx.fillerHeights[i]));
    });
  }
});

describe("MuJoCo XML matches xml_generator.py", () => {
  for (const fx of fixtures.xml) {
    it(fx.name, () => {
      const a = fx.args as Record<string, unknown>;
      const xml = generateMujocoXml({
        stlName: "baselink.stl",
        unitHeight: a.unit_height as number,
        scale: a.scale as number,
        numUnits: a.num_units as number,
        jointType: a.joint_type as string,
        jointLimitDeg: a.joint_limit_deg as number,
        robotLength: a.robot_length as number,
        sitePoints: a.site_points as [number, number, number, number],
        cableMode: a.cable_mode as number,
        jointStiffness: a.joint_stiffness as number,
        jointDamping: a.joint_damping as number,
        gravity: a.gravity as number,
        jointAngles: a.joint_angles as number[],
        bendDirDeg: a.bend_dir_deg as number,
        tendonConfigs: a.tendon_configs as never,
      });
      expect(xml).toBe(fx.xml);
    });
  }
});

describe("formatting and params", () => {
  it("fmtG follows Python's %g", () => {
    expect(fmtG(50, 9)).toBe("50");
    expect(fmtG(-1, 9)).toBe("-1");
    expect(fmtG(-9.81, 9)).toBe("-9.81");
    expect(fmtG(0.0001234, 9)).toBe("0.0001234");
    expect(fmtG(0.00001234, 9)).toBe("1.234e-05");
    expect(fmtG(1234567890123, 9)).toBe("1.23456789e+12");
  });
  it("round-trips last_params.json", () => {
    const p = { ...defaultParams(), a: 3.3, two_cable: false, joint_angles: [1, 2] };
    const back = parseParams(JSON.parse(serializeParams(p)));
    expect(back.a).toBe(3.3);
    expect(back.two_cable).toBe(false);
    expect(back.joint_angles).toEqual([1, 2]);
  });
  it("clamps bad input", () => {
    const p = parseParams({ a: 1e9, b: "x", dtheta_deg: 12.7, num_cables: 99, sim_stiffness: 168 });
    expect(p.a).toBe(1e4);
    expect(p.sim_stiffness).toBe(168); // scaled designs keep their physics
    expect(p.b).toBe(defaultParams().b);
    expect(p.dtheta_deg).toBe(13);
    expect(p.num_cables).toBe(8);
  });
});
