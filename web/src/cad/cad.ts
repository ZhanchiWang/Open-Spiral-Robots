// 3D CAD on OpenCascade (replicad): a port of the CadQuery code in
// design-tool/DesignTool.py (_build_cad_parts, _build_frustum_solid,
// _build_cable_holes, _build_cone2_preview_solid, _build_cable3_cut_solid,
// _build_joint_fillers, _apply_joint_pose, _build_cone_cut_regions and the
// base-unit part of export_xml). Runs inside the CAD web worker.

import {
  draw,
  makeBox,
  makeCompound,
  makeCylinder,
  makeSphere,
  Plane,
  type AnyShape,
  type Shape3D,
} from "replicad";
import {
  cableAngles,
  computeDesign,
  elasticUnitPolys,
  rodrigues,
  rotatePointAboutAxis,
  type Design,
  type Poly,
  type Pt,
} from "../core/design";
import type { Params } from "../core/params";

type V3 = [number, number, number];
const DEG = 180 / Math.PI;

export interface CadContext {
  params: Params;
  design: Design;
  log: (msg: string) => void;
}

export function makeContext(params: Params, log: (msg: string) => void): CadContext {
  const design = computeDesign(params);
  if (!design) throw new Error("No unit in the current design (θ max too small).");
  return { params, design, log };
}

/** Drop repeated vertices so OCCT does not get zero-length edges. */
function cleanPoly(poly: Poly): Poly {
  const out: Poly = [];
  for (const p of poly) {
    const q = out[out.length - 1];
    if (!q || Math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-9) out.push(p);
  }
  while (out.length > 1 && Math.hypot(out[0][0] - out[out.length - 1][0], out[0][1] - out[out.length - 1][1]) <= 1e-9) out.pop();
  return out;
}

function polyDrawing(poly: Poly) {
  const pts = cleanPoly(poly);
  if (pts.length < 3) return null;
  let pen = draw(pts[0]);
  for (let i = 1; i < pts.length; i++) pen = pen.lineTo(pts[i]);
  return pen.close();
}

/** cq.Workplane("XY").polyline(poly).close().extrude(t / 2, both=True) */
function extrudeBoth(poly: Poly, thickness: number): Shape3D | null {
  const d = polyDrawing(poly);
  if (!d) return null;
  return d.sketchOnPlane("XY", -thickness / 2).extrude(thickness) as Shape3D;
}

/** cq.Workplane("XY").polyline(poly).close().revolve(360, (0,0,0), (1,0,0)) */
function revolveX(poly: Poly): Shape3D | null {
  const d = polyDrawing(poly);
  if (!d) return null;
  return d.sketchOnPlane("XY").revolve([1, 0, 0], { origin: [0, 0, 0] }) as Shape3D;
}

function centeredBox(sx: number, sy: number, sz: number): Shape3D {
  return makeBox([-sx / 2, -sy / 2, -sz / 2], [sx / 2, sy / 2, sz / 2]) as Shape3D;
}

/** _halfspace_box: big box filling the half-space `normal` points into. */
function halfspaceBox(origin: V3, normal: V3, extent: number): Shape3D {
  const angle = Math.atan2(normal[0], normal[2]) * DEG;
  return centeredBox(extent, extent, extent)
    .rotate(angle, [0, 0, 0], [0, 1, 0])
    .translate(origin[0] + (normal[0] * extent) / 2, origin[1] + (normal[1] * extent) / 2, origin[2] + (normal[2] * extent) / 2);
}

function compound(shapes: (AnyShape | null | undefined)[]): Shape3D | null {
  const list = shapes.filter((s): s is AnyShape => !!s && !s.isNull);
  if (!list.length) return null;
  return makeCompound(list) as Shape3D;
}

/** _build_frustum_solid: one cable-hole cutter along the p0 -> p1 centreline. */
export function frustumSolid(ctx: CadContext, angleOffset = 0): Shape3D | null {
  const { params, design } = ctx;
  if (design.length <= 1e-6) return null;
  const tipPos = params.tip_hole_pos / 100;
  const basePos = params.base_hole_pos / 100;
  let y0 = Math.max(0, Math.min(1, tipPos)) * (design.tipSize * 0.5);
  const y1 = Math.max(0, Math.min(1, basePos)) * (design.baseSize * 0.5);
  let tipR = params.tip_hole_size * 0.5;
  const baseRh = params.base_hole_size * 0.5;
  // For small Δθ the tip unit reaches behind x = 0 (its angle at q0 is over
  // 90°): start the cutter there, on the same line and taper, so the hole
  // stays open at the tip.
  const tipMinX = Math.min(0, ...design.primary[0].map((q) => q[0]));
  const ext = tipMinX < 0 ? -tipMinX + Math.max(params.tip_hole_size, 0.1) : 0;
  if (ext > 0) {
    y0 -= ((y1 - y0) * ext) / design.length;
    tipR = Math.max(1e-3, tipR - ((baseRh - tipR) * ext) / design.length);
  }
  const p0: V3 = [-ext, y0, 0];
  const p1: V3 = [design.length, y1, 0];
  const dx = p1[0] - p0[0];
  const dy = p1[1] - p0[1];
  const dz = p1[2] - p0[2];
  const lengthAxis = Math.hypot(dx, dy, dz);
  if (lengthAxis <= 1e-6) return null;

  let frustum: Shape3D;
  if (params.teardrop_holes) {
    const rot = (angleOffset * Math.PI) / 180;
    const cr = Math.cos(rot);
    const sr = Math.sin(rot);
    const td = (r: number): Pt[] => {
      const t = r / Math.SQRT2;
      const apex = r * Math.SQRT2;
      const pts: Pt[] = [[t, t], [0, -r], [-t, t], [0, apex]];
      return pts.map(([y, z]) => [y * cr + z * sr, -y * sr + z * cr]);
    };
    const sketch = (q: Pt[], offset: number) =>
      draw(q[0]).threePointsArcTo(q[2], q[1]).lineTo(q[3]).close().sketchOnPlane("YZ", offset);
    const q0 = td(Math.max(0.05, tipR));
    const q1 = td(Math.max(0.05, baseRh));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    frustum = (sketch(q0, 0) as any).loftWith(sketch(q1, lengthAxis), { ruled: true }) as Shape3D;
  } else {
    const profile: Poly = [
      [0, 0],
      [lengthAxis, 0],
      [lengthAxis, baseRh],
      [0, tipR],
    ];
    const s = revolveX(profile);
    if (!s) return null;
    frustum = s;
  }

  const vx = dx / lengthAxis;
  const vy = dy / lengthAxis;
  const vz = dz / lengthAxis;
  const angle = Math.acos(Math.max(-1, Math.min(1, vx))) * DEG;
  let axis: V3 = [0, 0, 0];
  if (Math.abs(angle) > 1e-6) {
    axis = [0, -vz, vy];
    const n = Math.hypot(...axis);
    if (n > 1e-9) {
      axis = [axis[0] / n, axis[1] / n, axis[2] / n];
      frustum = frustum.rotate(angle, [0, 0, 0], axis);
    } else {
      axis = [0, 0, 0];
    }
  }
  const endRot =
    Math.abs(angle) <= 1e-6 || (axis[0] === 0 && axis[1] === 0 && axis[2] === 0)
      ? ([lengthAxis, 0, 0] as V3)
      : rodrigues(lengthAxis, 0, 0, axis[0], axis[1], axis[2], angle);
  return frustum.translate(p1[0] - endRot[0], p1[1] - endRot[1], p1[2] - endRot[2]);
}

/** _build_cable_holes: compound of hole cutters rotated about +X. */
export function cableHoles(ctx: CadContext, angles: number[]): Shape3D | null {
  const shapes: Shape3D[] = [];
  for (const ang of angles) {
    const f = frustumSolid(ctx, ang);
    if (!f) continue;
    shapes.push(Math.abs(ang) < 1e-9 ? f : f.rotate(ang, [0, 0, 0], [1, 0, 0]));
  }
  return compound(shapes);
}

/** _build_cone2_preview_solid (2-cable base chamfer cutter). */
export function cone2Solid(ctx: CadContext): Shape3D | null {
  const { params, design } = ctx;
  if (!params.two_cable || design.length <= 1e-6) return null;
  const cone1 = params.cone_angle1;
  const cone2 = params.cone_angle2;
  if (Math.abs(cone2) <= 1e-6) return null;
  const thickness = Math.max(0.1, params.extrusion);
  const baseX = design.length;
  const alpha = -(cone1 * 0.5) / DEG;
  const point1 = thickness * 0.5 + Math.tan(alpha) * baseX;
  const p0: V3 = [0, 0, point1];
  const p1: V3 = [design.length, 0, thickness * 0.5];
  const v1: V3 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const v2: V3 = [0, design.baseSize, 0];
  const lenV1 = Math.hypot(...v1);
  const lenV2 = Math.max(1e-6, design.baseSize);
  if (lenV1 <= 1e-6) return null;
  const xdir: V3 = [v1[0] / lenV1, v1[1] / lenV1, v1[2] / lenV1];
  const nx = v1[1] * v2[2] - v1[2] * v2[1];
  const ny = v1[2] * v2[0] - v1[0] * v2[2];
  const nz = v1[0] * v2[1] - v1[1] * v2[0];
  const nl = Math.hypot(nx, ny, nz);
  if (nl <= 1e-9) return null;
  const plane = new Plane(p0, xdir, [nx / nl, ny / nl, nz / nl]);
  const rectDrawing = polyDrawing([[0, 0], [lenV1, 0], [lenV1, lenV2], [0, lenV2]]);
  if (!rectDrawing) return null;
  let rect = rectDrawing.sketchOnPlane(plane).extrude(design.baseSize * 0.5) as Shape3D;
  rect = rect.rotate(-cone2, p0, v1);
  const pairXY = rect.fuse(rect.clone().mirror("XY"));
  return pairXY.fuse(pairXY.clone().mirror("XZ"));
}

/** Cone1 half-space boxes at the base face (2-cable). */
function cone1Boxes(ctx: CadContext, baseX: number): Shape3D[] {
  const { params, design } = ctx;
  const cone1 = params.cone_angle1;
  if (design.length <= 1e-6 || cone1 <= 1e-6) return [];
  const thickness = Math.max(0.1, params.extrusion);
  const alpha = -(cone1 * 0.5) / DEG;
  const halfZ = thickness * 0.5;
  const extent = Math.max(design.length, design.baseSize, thickness) * 10;
  return [
    halfspaceBox([baseX, 0, halfZ], [Math.sin(alpha), 0, Math.cos(alpha)], extent),
    halfspaceBox([baseX, 0, -halfZ], [Math.sin(alpha), 0, -Math.cos(alpha)], extent),
  ];
}

/** _build_cable3_cut_solid: flats between adjacent cable holes (N-cable). */
export function cable3CutSolid(ctx: CadContext): Shape3D | null {
  const { params, design } = ctx;
  if (params.two_cable || !params.cable3_cut_enabled || design.length <= 1e-6) return null;
  const n = Math.max(3, Math.trunc(params.num_cables));
  const L = design.length;
  const tipR = Math.max(1e-6, design.tipSize * 0.5);
  const baseR = Math.max(1e-6, design.baseSize * 0.5);
  const a1 = Math.max(0, Math.min(0.5, params.cable3_cut_pos / 100));
  const a2 = Math.max(0, Math.min(0.2, params.cable3_cut_size / 100));
  if (a2 <= 1e-9) return null;
  const xStart = 2 * a1 * L;
  if (xStart >= L - 1e-6) return null;
  const cosStep = Math.cos(Math.PI / n);
  const depthFrac = a2 / 0.2;
  const bodyR = (x: number) => tipR + (baseR - tipR) * (x / L);
  const tipPos = Math.max(0, Math.min(1, params.tip_hole_pos / 100));
  const basePos = Math.max(0, Math.min(1, params.base_hole_pos / 100));
  const tipHoleR = params.tip_hole_size * 0.5;
  const baseHoleR = params.base_hole_size * 0.5;
  const holeExtent = params.teardrop_holes ? Math.SQRT2 : 1;
  const rho0 = tipPos * tipR;
  const rho1 = basePos * baseR;
  const holeMargin = 0.3;
  const apothem = (x: number) => {
    const want = bodyR(x) * (1 - depthFrac * (1 - cosStep));
    const safe = (rho0 + (rho1 - rho0) * (x / L)) * cosStep + (tipHoleR + (baseHoleR - tipHoleR) * (x / L)) * holeExtent + holeMargin;
    return Math.min(bodyR(x), Math.max(want, safe));
  };
  const h0 = apothem(xStart);
  const h1 = apothem(L);
  const slope = (h1 - h0) / (L - xStart);
  let nx = -slope;
  let ny = 1;
  const norm = Math.hypot(nx, ny);
  nx /= norm;
  ny /= norm;
  const extent = Math.max(L, baseR, tipR) * 10;
  const gamma = Math.atan2(-nx, ny) * DEG;
  const planeBox = centeredBox(extent, extent, extent)
    .rotate(gamma, [0, 0, 0], [0, 0, 1])
    .translate(xStart + (nx * extent) / 2, h0 + (ny * extent) / 2, 0);
  const xLen = L * 1.1 - xStart;
  const slab = centeredBox(xLen, extent, extent).translate(xStart + xLen / 2, 0, 0);
  const cutter0 = planeBox.intersect(slab);
  // With 5+ cables neighbouring flats overlap, and OCCT's cut with a compound
  // of overlapping tools leaves fragments: fuse the flats into one tool first.
  let tool: Shape3D | null = null;
  for (let k = 0; k < n; k++) {
    const phi = ((k + 0.5) * 360) / n;
    const flat = cutter0.clone().rotate(phi, [0, 0, 0], [1, 0, 0]);
    if (!tool) {
      tool = flat;
      continue;
    }
    const fused: Shape3D = tool.fuse(flat);
    tool.delete();
    flat.delete();
    tool = fused;
  }
  cutter0.delete();
  return tool;
}

/** _build_joint_fillers: knuckle fillers aligned with units (tip -> base). */
function jointFillers(ctx: CadContext, elasticPolys: (Poly | null)[]): (Shape3D | null)[] {
  const { params, design } = ctx;
  const n = design.primary.length;
  const fillers: (Shape3D | null)[] = new Array(n).fill(null);
  const nJoints = n - 1;
  for (let j = 0; j < nJoints; j++) {
    const eq = elasticPolys[j];
    const nq = elasticPolys[j + 1];
    if (!eq || !nq) continue;
    const xE = eq[1][0];
    if (Math.abs(nq[0][0] - xE) > 1e-6) continue;
    if (params.two_cable) {
      const idx = nJoints - 1 - j;
      const width = params.joint_filler_widths[idx] ?? 0;
      const height = params.joint_filler_heights[idx] ?? 0;
      if (width <= 1e-9 || height <= 1e-9) continue;
      const yP = design.primary[j][2][1];
      fillers[j] = makeCylinder(width / 2, height, [xE, yP, -height / 2], [0, 0, 1]) as Shape3D;
    } else {
      const r = Math.max(...eq.map(([, y]) => Math.abs(y)));
      if (r < 1e-9) continue;
      fillers[j] = (makeSphere(r) as Shape3D).translate(xE, 0, 0);
    }
  }
  return fillers;
}

/** _apply_joint_pose: bend per-unit solids at the joints (serial chain). */
function applyJointPose(
  ctx: CadContext,
  groups: (Shape3D | null)[][],
): (Shape3D | null)[][] {
  const { params, design } = ctx;
  const angles = [...params.joint_angles].reverse();
  const n = design.primary.length;
  if (n < 2 || angles.length !== n - 1 || angles.every((a) => Math.abs(a) < 1e-9)) return groups;
  let pts: V3[];
  let dirs: V3[];
  if (params.two_cable) {
    pts = design.primary.slice(0, n - 1).map((p) => [p[2][0], p[2][1], 0] as V3);
    dirs = pts.map(() => [0, 0, 1] as V3);
  } else {
    const beta = params.bend_dir_deg / DEG;
    pts = design.primary.slice(0, n - 1).map((p) => [p[2][0], 0, 0] as V3);
    dirs = pts.map(() => [0, -Math.sin(beta), Math.cos(beta)] as V3);
  }
  // Frozen axis of each joint at the moment it is applied (later joints move
  // with the tail, exactly like the gp_Trsf composition in the desktop tool).
  const axes: { ang: number; pt: V3; dir: V3 }[] = [];
  for (let j = 0; j < angles.length; j++) {
    const ang = angles[j];
    axes.push({ ang, pt: pts[j], dir: dirs[j] });
    if (Math.abs(ang) < 1e-9) continue;
    for (let jj = j + 1; jj < n - 1; jj++) {
      pts[jj] = rotatePointAboutAxis(pts[jj], pts[j], dirs[j], ang);
      dirs[jj] = rodrigues(dirs[jj][0], dirs[jj][1], dirs[jj][2], dirs[j][0], dirs[j][1], dirs[j][2], ang);
    }
  }
  const tf = (s: Shape3D | null, k: number) => {
    if (!s) return null;
    let out = s;
    for (let j = 0; j < k; j++) {
      const a = axes[j];
      if (Math.abs(a.ang) < 1e-9) continue;
      out = out.rotate(a.ang, a.pt, a.dir);
    }
    return out;
  };
  return groups.map((g) => g.map((s, k) => tf(s, k)));
}

function bboxOverlap(a: Shape3D, b: Shape3D, tol = 1e-6): boolean {
  const ba = a.boundingBox.bounds;
  const bb = b.boundingBox.bounds;
  for (let i = 0; i < 3; i++) {
    if (ba[0][i] > bb[1][i] + tol || ba[1][i] < bb[0][i] - tol) return false;
  }
  return true;
}

function cutPerUnit(ctx: CadContext, shapes: (Shape3D | null)[], cutters: (Shape3D | null)[]): (Shape3D | null)[] {
  const tools = cutters.filter((c): c is Shape3D => !!c && !c.isNull);
  if (!tools.length) return shapes;
  return shapes.map((s) => {
    if (!s) return null;
    let cur = s;
    for (const c of tools) {
      if (!bboxOverlap(cur, c)) continue;
      try {
        const r = cur.cut(c);
        if (r && !r.isNull) cur = r;
      } catch (e) {
        ctx.log(`unit cut failed: ${(e as Error).message ?? e}`);
      }
    }
    return cur;
  });
}

function cutGlobal(ctx: CadContext, shapes: (Shape3D | null)[], cutters: (Shape3D | null)[]): Shape3D | null {
  let comp: Shape3D | null = compound(shapes);
  if (!comp) return null;
  for (const c of cutters) {
    if (!c || c.isNull) continue;
    try {
      const r: Shape3D = comp.cut(c);
      if (r && !r.isNull) comp = r;
    } catch (e) {
      ctx.log(`global cut failed: ${(e as Error).message ?? e}`);
    }
  }
  return comp;
}

export interface CadParts {
  main: Shape3D | null;
  elastic: Shape3D | null;
  posed: boolean;
}

/** _build_cad_parts: the full robot as (main body, elastic layer). */
export function buildParts(ctx: CadContext): CadParts {
  const { params, design } = ctx;
  const n = design.primary.length;
  const { polys: ePolys, mirrors: eMirrors } = elasticUnitPolys(design);
  const posed = params.pose_enabled && params.joint_angles.some((a) => Math.abs(a) > 1e-9);

  let mainUnits: (Shape3D | null)[];
  let elasticUnits: (Shape3D | null)[];
  let cutters: (Shape3D | null)[];

  if (params.two_cable) {
    const t = Math.max(0.1, params.extrusion);
    const fusePair = (a: Poly | null, b: Poly | null) => {
      const sa = a ? extrudeBoth(a, t) : null;
      const sb = b ? extrudeBoth(b, t) : null;
      if (sa && sb) return sa.fuse(sb);
      return sa ?? sb;
    };
    mainUnits = design.primary.map((p, k) => fusePair(p, design.mirror[k] ?? null));
    elasticUnits = Array.from({ length: n }, (_, k) => (ePolys[k] ? fusePair(ePolys[k], eMirrors[k]) : null));
    const baseX = Math.max(...design.primary.flatMap((p) => p.map((q) => q[0])));
    cutters = [...cone1Boxes(ctx, baseX), cableHoles(ctx, [0, 180]), cone2Solid(ctx)];
  } else {
    mainUnits = design.primary.map((p) => revolveX(p));
    elasticUnits = ePolys.map((p) => (p ? revolveX(p) : null));
    cutters = [cableHoles(ctx, cableAngles(params)), cable3CutSolid(ctx)];
  }

  if (posed) {
    mainUnits = cutPerUnit(ctx, mainUnits, cutters);
    elasticUnits = cutPerUnit(ctx, elasticUnits, cutters);
    let fillers = jointFillers(ctx, ePolys);
    fillers = cutPerUnit(ctx, fillers, cutters);
    const [m, e, f] = applyJointPose(ctx, [mainUnits, elasticUnits, fillers]);
    return { main: compound(m), elastic: compound([...e, ...f]), posed };
  }
  return {
    main: compound([cutGlobal(ctx, mainUnits, cutters)]),
    elastic: compound([cutGlobal(ctx, elasticUnits, cutters)]),
    posed,
  };
}

/** Hole cutters for the orange preview overlay (straight pose only). */
export function holePreview(ctx: CadContext): Shape3D | null {
  return cableHoles(ctx, cableAngles(ctx.params));
}

/** _build_cone_cut_regions: material cone1 / cone2 remove (yellow preview). */
export function coneCutRegions(ctx: CadContext): [Shape3D | null, Shape3D | null] {
  const { params, design } = ctx;
  if (!params.two_cable || design.length <= 1e-6) return [null, null];
  const t = Math.max(0.1, params.extrusion);
  let body: Shape3D | null = null;
  design.primary.forEach((p, k) => {
    let u = extrudeBoth(p, t);
    const m = design.mirror[k] ? extrudeBoth(design.mirror[k], t) : null;
    if (u && m) u = u.fuse(m);
    if (!u) return;
    try {
      body = body ? body.fuse(u) : u;
    } catch {
      /* same as the desktop tool: skip pieces that fail to fuse */
    }
  });
  if (!body) return [null, null];
  const b = body as Shape3D;
  let r1: Shape3D | null = null;
  const baseX = Math.max(...design.primary.flatMap((p) => p.map((q) => q[0])));
  for (const box of cone1Boxes(ctx, baseX)) {
    try {
      const r = b.clone().intersect(box);
      if (r && !r.isNull) r1 = r1 ? r1.fuse(r) : r;
    } catch {
      /* ignore */
    }
  }
  let r2: Shape3D | null = null;
  const c2 = cone2Solid(ctx);
  if (c2) {
    try {
      const r = b.clone().intersect(c2);
      if (r && !r.isNull) r2 = r;
    } catch {
      /* ignore */
    }
  }
  return [r1, r2];
}

/**
 * export_xml's base link: the base (largest) unit only, without the elastic
 * layer, with the same cuts, moved so the chain runs along +Z from the base
 * face (translate -L along X, rotate +90° about Y).
 */
export function baselinkSolid(ctx: CadContext): Shape3D | null {
  const { params, design } = ctx;
  const last = design.primary[design.primary.length - 1];
  let solid: Shape3D | null;
  if (params.two_cable) {
    const t = Math.max(0.1, params.extrusion);
    const a = extrudeBoth(last, t);
    const mp = design.mirror[design.mirror.length - 1];
    const b = mp ? extrudeBoth(mp, t) : null;
    solid = a && b ? a.fuse(b) : a ?? b;
    if (!solid) return null;
    if (design.length > 1e-6 && params.cone_angle1 > 1e-6) {
      let baseX = design.length;
      try {
        baseX = solid.boundingBox.bounds[1][0];
      } catch {
        /* keep robot length */
      }
      for (const box of cone1Boxes(ctx, baseX)) solid = solid.cut(box);
    }
    const c2 = cone2Solid(ctx);
    if (c2) solid = solid.cut(c2);
  } else {
    solid = revolveX(last);
    if (!solid) return null;
    const holes = cableHoles(ctx, cableAngles(params));
    if (holes) solid = solid.cut(holes);
    const c3 = cable3CutSolid(ctx);
    if (c3) solid = solid.cut(c3);
  }
  return solid.translate(-design.length, 0, 0).rotate(90, [0, 0, 0], [0, 1, 0]);
}

/** STL deflection sized to the model (_stl_mesh_tolerance). */
export function stlTolerance(design: Design): number {
  return Math.max(0.05, Math.max(1, design.length) * 0.0003);
}
