// 2D design geometry: a line-by-line port of the pure geometry in
// design-tool/DesignTool.py (_build_polar_units, _build_unfold_polygons,
// update_2d, _elastic_unit_polys, _joint_filler_defaults, cone ranges and the
// cable site points). tests/parity.test.ts checks it against the Python code.

import type { Params } from "./params";

export type Pt = [number, number];
export type Poly = Pt[];

export function polarToCart(theta: number, r: number): Pt {
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

export function cartToPolar(x: number, y: number): Pt {
  const r = Math.hypot(x, y);
  let theta = Math.atan2(y, x);
  if (theta < 0) theta += 2 * Math.PI;
  return [theta, r];
}

export function reflectPointAcrossLine(p: Pt, a: Pt, b: Pt): Pt {
  const [px, py] = p;
  const [ax, ay] = a;
  const vx = b[0] - ax;
  const vy = b[1] - ay;
  const denom = vx * vx + vy * vy;
  if (denom < 1e-12) return p;
  const t = ((px - ax) * vx + (py - ay) * vy) / denom;
  const projx = ax + t * vx;
  const projy = ay + t * vy;
  return [2 * projx - px, 2 * projy - py];
}

export function lineSegmentIntersection(a0: Pt, a1: Pt, b0: Pt, b1: Pt): Pt | null {
  const [ax, ay] = a0;
  const [cx, cy] = b0;
  const rx = a1[0] - ax;
  const ry = a1[1] - ay;
  const sx = b1[0] - cx;
  const sy = b1[1] - cy;
  const denom = rx * sy - ry * sx;
  if (Math.abs(denom) < 1e-12) return null;
  const t = ((cx - ax) * sy - (cy - ay) * sx) / denom;
  const u = ((cx - ax) * ry - (cy - ay) * rx) / denom;
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return [ax + t * rx, ay + t * ry];
  return null;
}

export function rodrigues(
  px: number, py: number, pz: number,
  ux: number, uy: number, uz: number,
  angleDeg: number,
): [number, number, number] {
  const th = (angleDeg * Math.PI) / 180;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const dot = ux * px + uy * py + uz * pz;
  return [
    px * c + (uy * pz - uz * py) * s + ux * dot * (1 - c),
    py * c + (uz * px - ux * pz) * s + uy * dot * (1 - c),
    pz * c + (ux * py - uy * px) * s + uz * dot * (1 - c),
  ];
}

export function rotatePointAboutAxis(
  p: [number, number, number],
  axisPt: [number, number, number],
  axisDir: [number, number, number],
  angleDeg: number,
): [number, number, number] {
  const r = rodrigues(
    p[0] - axisPt[0], p[1] - axisPt[1], p[2] - axisPt[2],
    axisDir[0], axisDir[1], axisDir[2], angleDeg,
  );
  return [r[0] + axisPt[0], r[1] + axisPt[1], r[2] + axisPt[2]];
}

export interface PolarUnits {
  theta: number[];
  r: number[];
  rc: number[];
  /** Each unit as (theta[], r[]) like the matplotlib fill() calls. */
  unitsPrimary: [number[], number[]][];
  unitsMirror: [number[], number[]][];
  unitCount: number;
  turns: number;
}

const cFactor = (b: number, p: number) => (1 - p) + p * Math.exp(2 * Math.PI * b);

export function buildPolarUnits(a: number, b: number, dthetaDeg: number, turns: number, p: number): PolarUnits {
  const thetaEnd = 2 * Math.PI * turns;
  const rcEnd = Math.max(0, thetaEnd - 2 * Math.PI);
  const dtheta = (Math.max(1, Math.trunc(dthetaDeg)) * Math.PI) / 180;
  const c = cFactor(b, p);
  const theta: number[] = [];
  const r: number[] = [];
  const rc: number[] = [];
  // Accumulate theta the same way as the Python loop so the sample count
  // (and therefore the unit count) matches bit for bit.
  let t = 0;
  while (t <= thetaEnd + 1e-12) {
    const rr = a * Math.exp(b * t);
    theta.push(t);
    r.push(rr);
    rc.push(c * rr);
    t += dtheta;
  }
  const unitsPrimary: [number[], number[]][] = [];
  const unitsMirror: [number[], number[]][] = [];
  let unitCount = 0;
  for (let i = 0; i < theta.length - 1; i++) {
    const t0 = theta[i];
    const t1 = theta[i + 1];
    if (t1 > rcEnd + 1e-12) break;
    const [r0, r1, rc0, rc1] = [r[i], r[i + 1], rc[i], rc[i + 1]];
    unitsPrimary.push([[t0, t1, t1, t0], [r0, r1, rc1, rc0]]);
    const p0 = polarToCart(t0, r0);
    const p1 = polarToCart(t1, r1);
    const q0 = polarToCart(t0, rc0);
    const q1 = polarToCart(t1, rc1);
    const [t0m, r0m] = cartToPolar(...reflectPointAcrossLine(p0, q0, q1));
    const [t1m, r1m] = cartToPolar(...reflectPointAcrossLine(p1, q0, q1));
    unitsMirror.push([[t0, t1, t1m, t0m], [rc0, rc1, r1m, r0m]]);
    unitCount += 1;
  }
  return { theta, r, rc, unitsPrimary, unitsMirror, unitCount, turns };
}

export function buildUnfoldPolygons(
  a: number, b: number, dthetaDeg: number, p: number, unitCount: number,
): { primary: Poly[]; mirror: Poly[] } {
  const dtheta = (Math.max(1, Math.trunc(dthetaDeg)) * Math.PI) / 180;
  const gamma = Math.exp(b * dtheta);
  const c = cFactor(b, p);
  const r0 = a;
  const r1 = a * Math.exp(b * dtheta);
  const p0 = polarToCart(0, r0);
  const p1 = polarToCart(dtheta, r1);
  const q0 = polarToCart(0, c * r0);
  const q1 = polarToCart(dtheta, c * r1);
  const angle = -Math.atan2(q1[1] - q0[1], q1[0] - q0[0]);
  const ca = Math.cos(angle);
  const sa = Math.sin(angle);
  const rot = ([x, y]: Pt): Pt => [x * ca - y * sa, x * sa + y * ca];
  const base = [rot(p0), rot(p1), rot(q1), rot(q0)];
  const n = Math.max(1, unitCount);
  const primary: Poly[] = [];
  const mirror: Poly[] = [];
  let currentX = 0;
  for (let k = 0; k < n; k++) {
    const scale = gamma ** k;
    const scaled = base.map(([x, y]) => [x * scale, y * scale] as Pt);
    const dx = currentX - scaled[3][0];
    const dy = -scaled[3][1];
    const placed = scaled.map(([x, y]) => [x + dx, y + dy] as Pt);
    primary.push(placed);
    mirror.push(placed.map(([x, y]) => [x, -y] as Pt));
    currentX = dx + scaled[2][0];
  }
  return { primary, mirror };
}

export interface Design {
  polar: PolarUnits;
  primary: Poly[];
  mirror: Poly[];
  elastic: Poly | null;
  elasticMirror: Poly | null;
  ray: { start: Pt; upper: Pt; lower: Pt };
  baseSize: number;
  tipSize: number;
  taperDeg: number;
  length: number;
  units: number;
  joints: number;
  /** base_size * 0.6, the thickness update_2d keeps for the sketch. */
  sketchThickness: number;
  extrusionRange: [number, number];
  defaultExtrusion: number;
  cone2Max: number;
}

const maxOf = (xs: number[]) => xs.reduce((m, v) => (v > m ? v : m), -Infinity);

/** update_2d() without the widget plumbing. Returns null if no unit exists. */
export function computeDesign(params: Params): Design | null {
  const { a, b, p } = params;
  const dthetaDeg = Math.trunc(params.dtheta_deg);
  const turns = Math.max(0.1, params.theta_max_pi / 2);
  const polar = buildPolarUnits(a, b, dthetaDeg, turns, p);
  const { primary, mirror } = buildUnfoldPolygons(a, b, dthetaDeg, p, polar.unitCount);
  if (!primary.length) return null;

  const baseSize = 2 * maxOf(primary[primary.length - 1].map((q) => q[1]));
  const tipSize = 2 * maxOf(primary[0].map((q) => q[1]));
  const eb = Math.exp(2 * Math.PI * b);
  const taper = 2 * Math.atan((b * (eb - 1)) / (Math.sqrt(b * b + 1) * (eb + 1)));
  const length = maxOf(primary[primary.length - 1].map((q) => q[0]));

  const minExt = Math.max(0, baseSize * 0.2);
  const maxExt = Math.max(minExt, baseSize);

  const c = cFactor(b, p);
  const lVtip = (c * a * Math.sqrt(b * b + 1)) / b;
  const elasticAngle = (params.elastic_percent / 100) * (taper * 0.5);
  const m = elasticAngle !== 0 ? Math.tan(elasticAngle) : 0;
  const leftEdge: [Pt, Pt] = [primary[0][0], primary[0][3]];
  const last = primary[primary.length - 1];
  const rightEdge: [Pt, Pt] = [last[1], last[2]];
  const maxPolyX = maxOf(primary.flatMap((poly) => poly.map((q) => q[0])));
  const rayLen = Math.max(10, maxPolyX + lVtip + 10);
  const start: Pt = [-lVtip, 0];
  const upper: Pt = [-lVtip + rayLen, m * rayLen];
  const lower: Pt = [-lVtip + rayLen, -m * rayLen];

  let elastic: Poly | null = null;
  let elasticMirror: Poly | null = null;
  if (params.elastic_enabled) {
    const ul = lineSegmentIntersection(start, upper, leftEdge[0], leftEdge[1]);
    const ur = lineSegmentIntersection(start, upper, rightEdge[0], rightEdge[1]);
    if (ul && ur) {
      elastic = [leftEdge[1], ul, ur, rightEdge[1]];
      elasticMirror = elastic.map(([x, y]) => [x, -y] as Pt);
    }
  }

  const taperDeg = (taper * 180) / Math.PI;
  return {
    polar,
    primary,
    mirror,
    elastic,
    elasticMirror,
    ray: { start, upper, lower },
    baseSize,
    tipSize,
    taperDeg,
    length,
    units: primary.length,
    joints: Math.max(0, primary.length - 1),
    sketchThickness: Math.max(0.1, baseSize * 0.6),
    extrusionRange: [minExt, maxExt],
    defaultExtrusion: Math.max(0, baseSize * 0.6),
    cone2Max: Math.max(0, 4 * taperDeg),
  };
}

/** _update_cone1_range: max ConeAngle1 for a given extrusion. */
export function cone1Max(design: Design, extrusion: number): number {
  if (design.length <= 1e-6) return 0;
  return Math.max(0, (2 * Math.atan((extrusion * 0.5) / design.length) * 180) / Math.PI);
}

/** Split the full-length elastic quad into per-unit quads (+ mirror). */
export function elasticUnitPolys(design: Design): { polys: (Poly | null)[]; mirrors: (Poly | null)[] } {
  const n = design.primary.length;
  const empty = () => new Array<Poly | null>(n).fill(null);
  if (!design.elastic) return { polys: empty(), mirrors: empty() };
  const [qa, qb, qc, qd] = design.elastic;
  const interp = (p: Pt, q: Pt, x: number) => {
    if (Math.abs(q[0] - p[0]) < 1e-12) return p[1];
    const t = (x - p[0]) / (q[0] - p[0]);
    return p[1] + t * (q[1] - p[1]);
  };
  const polys: (Poly | null)[] = [];
  const mirrors: (Poly | null)[] = [];
  for (let k = 0; k < n; k++) {
    const x0 = Math.max(design.primary[k][3][0], qa[0]);
    const x1 = Math.min(design.primary[k][2][0], qd[0]);
    if (x1 - x0 < 1e-9) {
      polys.push(null);
      mirrors.push(null);
      continue;
    }
    const quad: Poly = [
      [x0, interp(qa, qd, x0)],
      [x1, interp(qa, qd, x1)],
      [x1, interp(qb, qc, x1)],
      [x0, interp(qb, qc, x0)],
    ];
    polys.push(quad);
    mirrors.push(quad.map(([x, y]) => [x, -y] as Pt));
  }
  return { polys, mirrors };
}

/** Default (width, height) per joint, base -> tip (_joint_filler_defaults). */
export function jointFillerDefaults(design: Design, extrusion: number): { widths: number[]; heights: number[] } {
  const nJoints = design.joints;
  const thickness = Math.max(0.1, extrusion);
  const { polys } = elasticUnitPolys(design);
  const widths: number[] = [];
  const heights: number[] = [];
  for (let j = 0; j < nJoints; j++) {
    const tipJ = nJoints - 1 - j;
    const q = tipJ < polys.length ? polys[tipJ] : null;
    widths.push(q ? 2 * maxOf(q.map(([, y]) => Math.abs(y))) : thickness);
    heights.push(thickness);
  }
  return { widths, heights };
}

/**
 * Cable site points (x1, y1, x2, y2) on the base unit: where the straight
 * tip-hole -> base-hole line crosses the unit's two joint edges.
 */
export function sitePoints(design: Design, tipHolePosFrac: number, baseHolePosFrac: number): [number, number, number, number] | null {
  const lastPoly = design.primary[design.primary.length - 1];
  if (!lastPoly || lastPoly.length < 4) return null;
  const p0: Pt = [0, tipHolePosFrac * design.tipSize * 0.5];
  const p1: Pt = [design.length, baseHolePosFrac * design.baseSize * 0.5];
  let dx = p1[0] - p0[0];
  let dy = p1[1] - p0[1];
  if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) {
    dx = 1;
    dy = 0;
  }
  const len = Math.hypot(dx, dy);
  dx /= len;
  dy /= len;
  const L = 1e6;
  const la: Pt = [p0[0] - dx * L, p0[1] - dy * L];
  const lb: Pt = [p0[0] + dx * L, p0[1] + dy * L];
  const leftEdge: [Pt, Pt] = [lastPoly[3], lastPoly[0]];
  const rightEdge: [Pt, Pt] = [lastPoly[2], lastPoly[1]];
  const lh = lineSegmentIntersection(la, lb, leftEdge[0], leftEdge[1]);
  const rh = lineSegmentIntersection(la, lb, rightEdge[0], rightEdge[1]);
  if (lh && rh) return [lh[0], lh[1], rh[0], rh[1]];
  const x1 = leftEdge[0][0];
  const x2 = rightEdge[0][0];
  let y1: number;
  let y2: number;
  if (Math.abs(p1[0] - p0[0]) < 1e-9) {
    y1 = p0[1];
    y2 = p1[1];
  } else {
    const slope = (p1[1] - p0[1]) / (p1[0] - p0[0]);
    y1 = p0[1] + slope * (x1 - p0[0]);
    y2 = p0[1] + slope * (x2 - p0[0]);
  }
  return [x1, y1, x2, y2];
}

/** Axial height of the base unit (export_xml's unit_height). */
export function baseUnitHeight(design: Design): number {
  const last = design.primary[design.primary.length - 1];
  const onAxis = last.filter(([, y]) => Math.abs(y) < 1e-6);
  const h = onAxis.length >= 2 ? Math.abs(onAxis[1][0] - onAxis[0][0]) : Math.abs(last[2][0] - last[3][0]);
  return Math.max(1e-6, h);
}

export function cableAngles(params: Params): number[] {
  const n = params.two_cable ? 2 : Math.max(3, Math.trunc(params.num_cables));
  return Array.from({ length: n }, (_, k) => (k * 360) / n);
}

export function cableCount(params: Params): number {
  return params.two_cable ? 2 : Math.max(3, Math.trunc(params.num_cables));
}

export interface Issue {
  level: "warn" | "error";
  message: string;
}

/** Design-rule checks surfaced in the debug panel. */
export function checkDesign(params: Params, design: Design | null): Issue[] {
  const issues: Issue[] = [];
  if (!design) {
    issues.push({ level: "error", message: "No unit fits in θ max: increase θ max above 2π + Δθ." });
    return issues;
  }
  if (design.polar.unitCount === 0) {
    issues.push({ level: "warn", message: "θ max leaves no unit before the centre spiral ends; one unit is drawn anyway." });
  }
  const tipR = design.tipSize / 2;
  const tipHoleR = params.tip_hole_size / 2;
  if (tipHoleR >= tipR) {
    issues.push({ level: "warn", message: `Tip cable hole (Ø${params.tip_hole_size.toFixed(2)} mm) is as wide as the tip (Ø${design.tipSize.toFixed(2)} mm).` });
  }
  if (params.tip_hole_size > params.base_hole_size) {
    issues.push({ level: "warn", message: "Tip hole is larger than the base hole: the cable channel narrows toward the base." });
  }
  if (params.two_cable && Number.isFinite(params.extrusion)) {
    const [lo, hi] = design.extrusionRange;
    if (params.extrusion < lo - 1e-9 || params.extrusion > hi + 1e-9) {
      issues.push({ level: "warn", message: `Extrusion ${params.extrusion.toFixed(2)} mm is outside ${lo.toFixed(2)}–${hi.toFixed(2)} mm.` });
    }
  }
  if (params.elastic_enabled && !design.elastic) {
    issues.push({
      level: "warn",
      message: `Elastic layer ${params.elastic_percent.toFixed(0)}% is too wide for this outline and is left out: lower Elastic % (or raise p).`,
    });
  }
  if (params.p <= 1e-6) {
    issues.push({ level: "error", message: "p = 0 gives a zero-width tip and base: increase p." });
  }
  if (design.units > 60) {
    issues.push({ level: "warn", message: `${design.units} units: CAD booleans will be slow. Increase Δθ or reduce θ max.` });
  }
  return issues;
}
