// Procedural mesh of the base link, in the frame the MuJoCo XML expects
// (millimetres, chain along +Z from the base face at z = 0): the plain unit
// body without holes or cone cuts. It is what the live showcase simulates and
// draws; it takes well under a millisecond, so the model can be rebuilt on
// every slider move. The exact OpenCascade solid is only used for exports.

import type { Design, Pt } from "./design";
import type { Params } from "./params";

export interface LinkMesh {
  vertices: Float32Array;
  triangles: Uint32Array;
}

function convexHull(points: Pt[]): Pt[] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 1e-12) lower.pop();
    lower.push(p);
  }
  const upper: Pt[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 1e-12) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1)); // counter-clockwise
}

/** CAD (x along the chain, y width, z thickness) -> XML frame. */
const toXml = (L: number, x: number, y: number, z: number): [number, number, number] => [z, y, L - x];

export function baseLinkMesh(design: Design, params: Params, segments = 40): LinkMesh {
  const L = design.length;
  const last = design.primary[design.primary.length - 1];
  const verts: number[] = [];
  const tris: number[] = [];
  const add = (x: number, y: number, z: number) => {
    const [a, b, c] = toXml(L, x, y, z);
    verts.push(a, b, c);
    return verts.length / 3 - 1;
  };

  if (params.two_cable) {
    const mirror = design.mirror[design.mirror.length - 1];
    const outline = convexHull([...last, ...mirror]);
    const t = Math.max(0.1, params.extrusion) / 2;
    const n = outline.length;
    const top = outline.map(([x, y]) => add(x, y, t));
    const bot = outline.map(([x, y]) => add(x, y, -t));
    for (let i = 1; i + 1 < n; i++) {
      tris.push(top[0], top[i], top[i + 1]);
      tris.push(bot[0], bot[i + 1], bot[i]);
    }
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      tris.push(bot[i], bot[j], top[j], bot[i], top[j], top[i]);
    }
  } else {
    // revolve the unit profile (y >= 0) about the chain axis
    const prof = convexHull(last.map(([x, y]) => [x, Math.max(0, y)] as Pt));
    const rings = prof.map(([x, r]) =>
      Array.from({ length: segments }, (_, k) => {
        const phi = (2 * Math.PI * k) / segments;
        return add(x, r * Math.cos(phi), r * Math.sin(phi));
      }),
    );
    for (let i = 0; i < prof.length; i++) {
      const j = (i + 1) % prof.length;
      if (prof[i][1] < 1e-9 && prof[j][1] < 1e-9) continue;
      for (let k = 0; k < segments; k++) {
        const k2 = (k + 1) % segments;
        const [a, b, c, d] = [rings[i][k], rings[i][k2], rings[j][k2], rings[j][k]];
        if (prof[i][1] >= 1e-9) tris.push(a, b, c);
        if (prof[j][1] >= 1e-9) tris.push(a, c, d);
      }
    }
  }
  const mesh = { vertices: new Float32Array(verts), triangles: new Uint32Array(tris) };
  // outward-facing triangles (positive volume) for correct lighting
  if (meshVolume(mesh) < 0) {
    for (let i = 0; i < mesh.triangles.length; i += 3) {
      const t = mesh.triangles[i + 1];
      mesh.triangles[i + 1] = mesh.triangles[i + 2];
      mesh.triangles[i + 2] = t;
    }
  }
  return mesh;
}

/** Signed volume (mm³) of a closed triangle mesh. */
export function meshVolume(m: LinkMesh): number {
  const v = m.vertices;
  let vol = 0;
  for (let i = 0; i < m.triangles.length; i += 3) {
    const a = m.triangles[i] * 3, b = m.triangles[i + 1] * 3, c = m.triangles[i + 2] * 3;
    vol +=
      (v[a] * (v[b + 1] * v[c + 2] - v[b + 2] * v[c + 1]) -
        v[a + 1] * (v[b] * v[c + 2] - v[b + 2] * v[c]) +
        v[a + 2] * (v[b] * v[c + 1] - v[b + 1] * v[c])) / 6;
  }
  return vol;
}

/** Cut every polygon corner (except those on the chain axis) by `c`. */
function chamferCorners(poly: Pt[], c: number, keep: (p: Pt) => boolean = () => false): Pt[] {
  const n = poly.length;
  const out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const p = poly[i];
    if (keep(p)) {
      out.push(p);
      continue;
    }
    const a = poly[(i + n - 1) % n];
    const b = poly[(i + 1) % n];
    const la = Math.hypot(a[0] - p[0], a[1] - p[1]);
    const lb = Math.hypot(b[0] - p[0], b[1] - p[1]);
    const ka = Math.min(c, la * 0.4) / la;
    const kb = Math.min(c, lb * 0.4) / lb;
    out.push([p[0] + (a[0] - p[0]) * ka, p[1] + (a[1] - p[1]) * ka]);
    out.push([p[0] + (b[0] - p[0]) * kb, p[1] + (b[1] - p[1]) * kb]);
  }
  return out;
}

/** Offset a counter-clockwise convex polygon inwards by `d`. */
function inset(poly: Pt[], d: number): Pt[] {
  const n = poly.length;
  const lines = poly.map((p, i) => {
    const q = poly[(i + 1) % n];
    const dx = q[0] - p[0];
    const dy = q[1] - p[1];
    const l = Math.hypot(dx, dy) || 1;
    const nx = -dy / l;
    const ny = dx / l; // left normal = inward for CCW
    return { p: [p[0] + nx * d, p[1] + ny * d] as Pt, dir: [dx / l, dy / l] as Pt };
  });
  return lines.map((L1, i) => {
    const L0 = lines[(i + n - 1) % n];
    const cross = L0.dir[0] * L1.dir[1] - L0.dir[1] * L1.dir[0];
    if (Math.abs(cross) < 1e-9) return L1.p;
    const t = ((L1.p[0] - L0.p[0]) * L1.dir[1] - (L1.p[1] - L0.p[1]) * L1.dir[0]) / cross;
    return [L0.p[0] + L0.dir[0] * t, L0.p[1] + L0.dir[1] * t] as Pt;
  });
}

/**
 * Display mesh of the base link: the same body as baseLinkMesh with softened
 * edges (corner chamfers and a bevel on the 2-cable faces), so highlights run
 * along the edges like on a machined or moulded part. Only used for drawing.
 */
export function renderLinkMesh(design: Design, params: Params, segments = 48): LinkMesh {
  const L = design.length;
  const last = design.primary[design.primary.length - 1];
  const unitH = Math.abs(last[2][0] - last[3][0]);
  const verts: number[] = [];
  const tris: number[] = [];
  const add = (x: number, y: number, z: number) => {
    const [a, b, c] = toXml(L, x, y, z);
    verts.push(a, b, c);
    return verts.length / 3 - 1;
  };
  if (params.two_cable) {
    const mirror = design.mirror[design.mirror.length - 1];
    const t = Math.max(0.1, params.extrusion) / 2;
    const c = Math.max(1e-4, 0.1 * Math.min(unitH, 2 * t));
    const outer = chamferCorners(convexHull([...last, ...mirror]), c * 1.4);
    const inner = inset(outer, c);
    const ring = (poly: Pt[], z: number) => poly.map(([x, y]) => add(x, y, z));
    const top = ring(inner, t);
    const upper = ring(outer, t - c);
    const lower = ring(outer, -t + c);
    const bot = ring(inner, -t);
    const n = outer.length;
    for (let i = 1; i + 1 < n; i++) {
      tris.push(top[0], top[i], top[i + 1]);
      tris.push(bot[0], bot[i + 1], bot[i]);
    }
    const band = (a: number[], b: number[]) => {
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        tris.push(a[i], a[j], b[j], a[i], b[j], b[i]);
      }
    };
    band(upper, top);
    band(lower, upper);
    band(bot, lower);
  } else {
    const c = Math.max(1e-4, 0.12 * unitH);
    const hull = convexHull(last.map(([x, y]) => [x, Math.max(0, y)] as Pt));
    const prof = chamferCorners(hull, c, ([, r]) => r < 1e-9);
    const rings = prof.map(([x, r]) =>
      Array.from({ length: segments }, (_, k) => {
        const phi = (2 * Math.PI * k) / segments;
        return add(x, r * Math.cos(phi), r * Math.sin(phi));
      }),
    );
    for (let i = 0; i < prof.length; i++) {
      const j = (i + 1) % prof.length;
      if (prof[i][1] < 1e-9 && prof[j][1] < 1e-9) continue;
      for (let k = 0; k < segments; k++) {
        const k2 = (k + 1) % segments;
        const [a, b, cc, d] = [rings[i][k], rings[i][k2], rings[j][k2], rings[j][k]];
        if (prof[i][1] >= 1e-9) tris.push(a, b, cc);
        if (prof[j][1] >= 1e-9) tris.push(a, cc, d);
      }
    }
  }
  const mesh = { vertices: new Float32Array(verts), triangles: new Uint32Array(tris) };
  if (meshVolume(mesh) < 0) {
    for (let i = 0; i < mesh.triangles.length; i += 3) {
      const tmp = mesh.triangles[i + 1];
      mesh.triangles[i + 1] = mesh.triangles[i + 2];
      mesh.triangles[i + 2] = tmp;
    }
  }
  return mesh;
}
