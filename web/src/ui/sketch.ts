// 2D sketch plots (the desktop tool's matplotlib polar + "Unfolded
// (Cartesian)" axes), drawn as SVG so they export as vector files.

import type { Design, Poly, Pt } from "../core/design";

export interface Palette {
  bg: string;
  fg: string;
  muted: string;
  grid: string;
  spiral: string;
  center: string;
  primaryFill: string;
  primaryEdge: string;
  mirrorFill: string;
  mirrorEdge: string;
  elastic: string;
  ray: string;
  highlight: string;
}

/** "Tide" palette: ink spiral, coral centre line, sea-glass and sand units. */
export const LIGHT: Palette = {
  bg: "#fbf7ee",
  fg: "#10262c",
  muted: "#5b7174",
  grid: "#e2dccd",
  spiral: "#10262c",
  center: "#e0522e",
  primaryFill: "rgba(15,122,110,0.16)",
  primaryEdge: "#0f7a6e",
  mirrorFill: "rgba(201,138,0,0.14)",
  mirrorEdge: "#b98b2c",
  elastic: "#e0522e",
  ray: "#8a9a9c",
  highlight: "#e0522e",
};

export const DARK: Palette = {
  ...LIGHT,
  bg: "#071e27",
  fg: "#e8eee8",
  muted: "#8aa4a7",
  grid: "#16343f",
  spiral: "#e8eee8",
  center: "#ff6b4a",
  primaryFill: "rgba(95,211,196,0.16)",
  primaryEdge: "#5fd3c4",
  mirrorFill: "rgba(255,193,69,0.12)",
  mirrorEdge: "#d9a441",
  elastic: "#ff6b4a",
};

const NS = "http://www.w3.org/2000/svg";

function svgEl(tag: string, attrs: Record<string, string | number>, parent?: Element): SVGElement {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.appendChild(e);
  return e;
}

function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

const fmtTick = (v: number, step: number) => {
  const d = Math.max(0, -Math.floor(Math.log10(step)));
  return v.toFixed(d);
};

const pathOf = (pts: Pt[], close = true) =>
  pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join("") + (close ? "Z" : "");

/** A polar polygon with straight edges, as the desktop tool's plot draws it. */
function polarPolyPath(theta: number[], r: number[], toXY: (t: number, r: number) => Pt): string {
  return pathOf(theta.map((t, i) => toXY(t, r[i])));
}

export interface SketchOptions {
  width: number;
  height: number;
  palette: Palette;
  title?: string;
  highlightUnit?: number | null;
}

export function renderPolar(svg: SVGSVGElement, design: Design, o: SketchOptions): void {
  svg.replaceChildren();
  const { width: W, height: H, palette: P } = o;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", String(W));
  svg.setAttribute("height", String(H));
  svgEl("rect", { x: 0, y: 0, width: W, height: H, fill: P.bg }, svg);
  const top = o.title ? 26 : 8;
  if (o.title) svgEl("text", { x: W / 2, y: 18, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: P.fg }, svg).textContent = o.title;
  const R = Math.max(10, Math.min(W, H - top) / 2 - 26);
  const cx = W / 2;
  const cy = top + (H - top) / 2;
  const { polar } = design;
  const rmax = Math.max(...polar.r, 1e-9);
  const toXY = (t: number, r: number): Pt => [cx + (r / rmax) * R * Math.cos(t), cy - (r / rmax) * R * Math.sin(t)];

  const grid = svgEl("g", { stroke: P.grid, fill: "none", "stroke-width": 1 }, svg);
  for (const f of [0.5, 1]) svgEl("circle", { cx, cy, r: R * f }, grid);
  for (let k = 0; k < 8; k++) {
    const t = (k * Math.PI) / 4;
    const [x, y] = toXY(t, rmax);
    svgEl("line", { x1: cx, y1: cy, x2: x, y2: y }, grid);
    const [lx, ly] = [cx + (R + 14) * Math.cos(t), cy - (R + 14) * Math.sin(t)];
    svgEl("text", { x: lx, y: ly + 4, "text-anchor": "middle", "font-size": 10, fill: P.muted, stroke: "none" }, svg).textContent = `${k * 45}°`;
  }
  for (const f of [0.5, 1]) {
    svgEl("text", { x: cx + R * f * Math.cos(Math.PI / 8) + 2, y: cy - R * f * Math.sin(Math.PI / 8), "font-size": 9, fill: P.muted }, svg).textContent =
      (rmax * f).toPrecision(3);
  }

  const units = svgEl("g", { "stroke-width": 0.6 }, svg);
  polar.unitsPrimary.forEach(([t, r], i) => {
    svgEl("path", { d: polarPolyPath(t, r, toXY), fill: P.primaryFill, stroke: i === o.highlightUnit ? P.highlight : P.primaryEdge }, units);
  });
  polar.unitsMirror.forEach(([t, r], i) => {
    svgEl("path", { d: polarPolyPath(t, r, toXY), fill: P.mirrorFill, stroke: i === o.highlightUnit ? P.highlight : P.mirrorEdge }, units);
  });

  const spiral = polar.theta.map((t, i) => toXY(t, polar.r[i]));
  svgEl("path", { d: pathOf(spiral, false), fill: "none", stroke: P.spiral, "stroke-width": 2 }, svg);
  const rcEnd = Math.max(0, 2 * Math.PI * polar.turns - 2 * Math.PI);
  const center = polar.theta.filter((t) => t <= rcEnd + 1e-12).map((t, i) => toXY(t, polar.rc[i]));
  if (center.length > 1) svgEl("path", { d: pathOf(center, false), fill: "none", stroke: P.center, "stroke-width": 2 }, svg);

  const lg = svgEl("g", { "font-size": 10, fill: P.fg }, svg);
  svgEl("line", { x1: 8, y1: H - 22, x2: 26, y2: H - 22, stroke: P.spiral, "stroke-width": 2 }, lg);
  svgEl("text", { x: 30, y: H - 18 }, lg).textContent = "Original spiral";
  svgEl("line", { x1: 8, y1: H - 8, x2: 26, y2: H - 8, stroke: P.center, "stroke-width": 2 }, lg);
  svgEl("text", { x: 30, y: H - 4 }, lg).textContent = "Centre spiral";
}

export interface CartesianView {
  /** data -> px */
  toPx: (p: Pt) => Pt;
  /** px -> data */
  toData: (p: Pt) => Pt;
}

export function renderCartesian(svg: SVGSVGElement, design: Design, showRays: boolean, o: SketchOptions): CartesianView {
  svg.replaceChildren();
  const { width: W, height: H, palette: P } = o;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", String(W));
  svg.setAttribute("height", String(H));
  svgEl("rect", { x: 0, y: 0, width: W, height: H, fill: P.bg }, svg);
  const top = o.title ? 26 : 8;
  if (o.title) svgEl("text", { x: W / 2, y: 18, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: P.fg }, svg).textContent = o.title;
  const left = 46;
  const bottom = 34;
  const right = 10;

  const xs = design.primary.flatMap((p) => p.map((q) => q[0]));
  let minX = Math.min(...xs, design.ray.start[0]);
  let maxX = Math.max(...xs, design.ray.upper[0], design.ray.lower[0]);
  const padX = 0.05 * (maxX - minX + 1e-6);
  minX -= padX;
  maxX += padX;
  const yLim = Math.max(design.baseSize * 0.75, 1e-6);
  const pw = W - left - right;
  const ph = H - top - bottom;
  // equal aspect (matplotlib "equal", adjustable="box"): shrink the box
  const s = Math.min(pw / (maxX - minX), ph / (2 * yLim));
  const bw = (maxX - minX) * s;
  const bh = 2 * yLim * s;
  const ox = left + (pw - bw) / 2;
  const oy = top + (ph - bh) / 2;
  const toPx = ([x, y]: Pt): Pt => [ox + (x - minX) * s, oy + (yLim - y) * s];
  const toData = ([px, py]: Pt): Pt => [(px - ox) / s + minX, yLim - (py - oy) / s];

  const clipId = `clip${Math.random().toString(36).slice(2, 8)}`;
  const defs = svgEl("defs", {}, svg);
  const cp = svgEl("clipPath", { id: clipId }, defs);
  svgEl("rect", { x: ox, y: oy, width: bw, height: bh }, cp);

  const grid = svgEl("g", { stroke: P.grid, "stroke-width": 1 }, svg);
  const labels = svgEl("g", { "font-size": 10, fill: P.muted }, svg);
  const sx = niceStep(maxX - minX, Math.max(2, bw / 80));
  for (let x = Math.ceil(minX / sx) * sx; x <= maxX; x += sx) {
    const [px] = toPx([x, 0]);
    svgEl("line", { x1: px, y1: oy, x2: px, y2: oy + bh }, grid);
    svgEl("text", { x: px, y: oy + bh + 13, "text-anchor": "middle" }, labels).textContent = fmtTick(x, sx);
  }
  const sy = niceStep(2 * yLim, Math.max(2, bh / 40));
  for (let y = Math.ceil(-yLim / sy) * sy; y <= yLim; y += sy) {
    const [, py] = toPx([0, y]);
    svgEl("line", { x1: ox, y1: py, x2: ox + bw, y2: py }, grid);
    svgEl("text", { x: ox - 4, y: py + 3, "text-anchor": "end" }, labels).textContent = fmtTick(y, sy);
  }
  svgEl("rect", { x: ox, y: oy, width: bw, height: bh, fill: "none", stroke: P.muted, "stroke-width": 0.8 }, svg);
  svgEl("text", { x: ox + bw / 2, y: oy + bh + 28, "text-anchor": "middle", "font-size": 11, fill: P.fg }, svg).textContent = "x (mm)";
  const yl = svgEl("text", { x: 12, y: oy + bh / 2, "text-anchor": "middle", "font-size": 11, fill: P.fg, transform: `rotate(-90 12 ${oy + bh / 2})` }, svg);
  yl.textContent = "y (mm)";

  const body = svgEl("g", { "clip-path": `url(#${clipId})`, "stroke-width": 0.6 }, svg);
  const poly = (p: Poly) => pathOf(p.map(toPx));
  design.primary.forEach((p, i) =>
    svgEl("path", { d: poly(p), fill: P.primaryFill, stroke: i === o.highlightUnit ? P.highlight : P.primaryEdge, "data-unit": i }, body),
  );
  design.mirror.forEach((p, i) =>
    svgEl("path", { d: poly(p), fill: P.mirrorFill, stroke: i === o.highlightUnit ? P.highlight : P.mirrorEdge, "data-unit": i }, body),
  );
  for (const e of [design.elastic, design.elasticMirror]) {
    if (e) svgEl("path", { d: poly(e), fill: P.elastic, "fill-opacity": 0.28, stroke: P.elastic, "stroke-width": 0.9 }, body);
  }
  if (showRays) {
    const clip = (start: Pt, end: Pt, xMax: number): Pt => {
      if (end[0] <= xMax) return end;
      if (Math.abs(end[0] - start[0]) < 1e-9) return [xMax, start[1]];
      const t = (xMax - start[0]) / (end[0] - start[0]);
      return [start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])];
    };
    for (const end of [design.ray.upper, design.ray.lower]) {
      const e = clip(design.ray.start, end, design.length);
      const [a, b] = [toPx(design.ray.start), toPx(e)];
      svgEl("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: P.ray, "stroke-width": 0.6 }, body);
    }
    const v = toPx(design.ray.start);
    svgEl("circle", { cx: v[0], cy: v[1], r: 2.2, fill: P.ray }, body);
    svgEl("text", { x: v[0] + 3, y: v[1] - 5, "font-size": 9, fill: P.ray }, body).textContent = "vtip";
  }
  return { toPx, toData };
}

export function svgToString(svg: SVGSVGElement): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", NS);
  clone.setAttribute("font-family", "Helvetica, Arial, sans-serif");
  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}

export async function svgToPng(svg: SVGSVGElement, scale = 2): Promise<Blob> {
  const w = Number(svg.getAttribute("width"));
  const h = Number(svg.getAttribute("height"));
  const url = URL.createObjectURL(new Blob([svgToString(svg)], { type: "image/svg+xml" }));
  try {
    const img = new Image();
    await new Promise<void>((res, rej) => {
      img.onload = () => res();
      img.onerror = () => rej(new Error("SVG rasterisation failed"));
      img.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = w * scale;
    canvas.height = h * scale;
    const g = canvas.getContext("2d")!;
    g.scale(scale, scale);
    g.drawImage(img, 0, 0, w, h);
    return await new Promise<Blob>((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), "image/png"));
  } finally {
    URL.revokeObjectURL(url);
  }
}
