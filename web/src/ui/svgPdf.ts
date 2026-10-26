// Draws the sketch SVGs (rect, line, circle, path with M/L/H/V/Z, text,
// groups with rotate() transforms and rect clip paths) onto a PdfCanvas as
// vector graphics, and builds the PDF files the desktop tool's
// "Save 2D Sketch" writes.

import { PdfCanvas, textRuns, type PathStyle, type Rgba } from "../core/pdf";

const NAMED: Record<string, [number, number, number]> = { black: [0, 0, 0], white: [255, 255, 255] };

export function parseColor(v: string | null | undefined, opacity = 1): Rgba | null {
  if (!v || v === "none" || v === "transparent") return null;
  v = v.trim();
  let r = 0, g = 0, b = 0, a = 1;
  let m: RegExpMatchArray | null;
  if ((m = v.match(/^#([0-9a-f]{3})$/i))) {
    [r, g, b] = [...m[1]].map((c) => parseInt(c + c, 16));
  } else if ((m = v.match(/^#([0-9a-f]{6})$/i))) {
    r = parseInt(m[1].slice(0, 2), 16);
    g = parseInt(m[1].slice(2, 4), 16);
    b = parseInt(m[1].slice(4, 6), 16);
  } else if ((m = v.match(/^rgba?\(([^)]+)\)$/i))) {
    const parts = m[1].split(",").map((x) => Number(x.trim()));
    [r, g, b] = parts;
    a = parts[3] ?? 1;
  } else if (NAMED[v]) {
    [r, g, b] = NAMED[v];
  } else {
    return null;
  }
  return { r: r / 255, g: g / 255, b: b / 255, a: a * opacity };
}

interface Inherited {
  fill: string | null;
  stroke: string | null;
  strokeWidth: number;
  fillOpacity: number;
  fontSize: number;
  fontWeight: string;
  anchor: string;
}

let measureCtx: CanvasRenderingContext2D | null = null;
/** Text width in the PDF's Helvetica / Symbol metrics, measured with a canvas. */
export function textWidth(s: string, size: number, bold: boolean): number {
  measureCtx ??= document.createElement("canvas").getContext("2d");
  if (!measureCtx) return s.length * size * 0.55;
  let w = 0;
  for (const run of textRuns(s, bold)) {
    measureCtx.font = run.font === "F3" ? `${size}px Symbol, "Times New Roman", serif` : `${bold ? "bold " : ""}${size}px Helvetica, Arial, sans-serif`;
    w += measureCtx.measureText(run.text).width;
  }
  return w;
}

function parsePath(d: string): { pts: [number, number][]; closed: boolean }[] {
  const out: { pts: [number, number][]; closed: boolean }[] = [];
  const tokens = d.match(/[MLHVZmlhvz]|-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/g) ?? [];
  let cur: { pts: [number, number][]; closed: boolean } | null = null;
  let cmd = "";
  let x = 0, y = 0;
  let i = 0;
  const num = () => Number(tokens[i++]);
  while (i < tokens.length) {
    const t = tokens[i];
    if (/[A-Za-z]/.test(t)) {
      cmd = t;
      i++;
      if (cmd === "Z" || cmd === "z") {
        if (cur) cur.closed = true;
        cur = null;
        continue;
      }
    }
    switch (cmd) {
      case "M": case "m":
        x = cmd === "m" ? x + num() : num();
        y = cmd === "m" ? y + num() : num();
        cur = { pts: [[x, y]], closed: false };
        out.push(cur);
        cmd = cmd === "m" ? "l" : "L";
        break;
      case "L": case "l":
        x = cmd === "l" ? x + num() : num();
        y = cmd === "l" ? y + num() : num();
        cur?.pts.push([x, y]);
        break;
      case "H": case "h":
        x = cmd === "h" ? x + num() : num();
        cur?.pts.push([x, y]);
        break;
      case "V": case "v":
        y = cmd === "v" ? y + num() : num();
        cur?.pts.push([x, y]);
        break;
      default:
        i++;
    }
  }
  return out;
}

function circlePath(cx: number, cy: number, r: number) {
  const k = 0.5522847498 * r;
  return {
    pts: [
      [cx + r, cy], [cx + r, cy + k], [cx + k, cy + r], [cx, cy + r],
      [cx - k, cy + r], [cx - r, cy + k], [cx - r, cy],
      [cx - r, cy - k], [cx - k, cy - r], [cx, cy - r],
      [cx + k, cy - r], [cx + r, cy - k], [cx + r, cy],
    ] as [number, number][],
    closed: true,
    curves: true,
  };
}

/** Draw `svg` with its top-left corner at (x, y), scaled by `scale`. */
export function drawSvg(pdf: PdfCanvas, svg: SVGSVGElement, x: number, y: number, scale: number): void {
  const clips = new Map<string, [number, number, number, number]>();
  svg.querySelectorAll("clipPath").forEach((cp) => {
    const r = cp.querySelector("rect");
    if (r) clips.set(cp.id, ["x", "y", "width", "height"].map((a) => Number(r.getAttribute(a))) as [number, number, number, number]);
  });
  pdf.save();
  pdf.transform(scale, 0, 0, scale, x, y);
  const W = Number(svg.getAttribute("width"));
  const H = Number(svg.getAttribute("height"));
  pdf.clipRect(0, 0, W, H);
  const base: Inherited = { fill: "#000000", stroke: null, strokeWidth: 1, fillOpacity: 1, fontSize: 12, fontWeight: "400", anchor: "start" };
  for (const c of Array.from(svg.children)) walk(pdf, c, base, clips);
  pdf.restore();
}

function walk(pdf: PdfCanvas, node: Element, parent: Inherited, clips: Map<string, [number, number, number, number]>): void {
  const tag = node.tagName.toLowerCase();
  if (tag === "defs" || tag === "clippath" || tag === "title") return;
  const attr = (n: string) => node.getAttribute(n);
  const st: Inherited = {
    fill: attr("fill") ?? parent.fill,
    stroke: attr("stroke") ?? parent.stroke,
    strokeWidth: attr("stroke-width") !== null ? Number(attr("stroke-width")) : parent.strokeWidth,
    fillOpacity: attr("fill-opacity") !== null ? Number(attr("fill-opacity")) : parent.fillOpacity,
    fontSize: attr("font-size") !== null ? Number(attr("font-size")) : parent.fontSize,
    fontWeight: attr("font-weight") ?? parent.fontWeight,
    anchor: attr("text-anchor") ?? parent.anchor,
  };
  pdf.save();
  const tf = attr("transform");
  const rot = tf?.match(/rotate\(\s*(-?[\d.]+)(?:[\s,]+(-?[\d.]+)[\s,]+(-?[\d.]+))?\s*\)/);
  if (rot) {
    const a = (Number(rot[1]) * Math.PI) / 180;
    const cx = Number(rot[2] ?? 0);
    const cy = Number(rot[3] ?? 0);
    const c = Math.cos(a);
    const s = Math.sin(a);
    pdf.transform(c, s, -s, c, cx - c * cx + s * cy, cy - s * cx - c * cy);
  }
  const clip = attr("clip-path")?.match(/url\(#([^)]+)\)/);
  if (clip && clips.has(clip[1])) pdf.clipRect(...clips.get(clip[1])!);

  const style: PathStyle = {
    fill: parseColor(st.fill, st.fillOpacity),
    stroke: parseColor(st.stroke),
    lineWidth: st.strokeWidth,
  };
  const n = (a: string) => Number(attr(a) ?? 0);
  switch (tag) {
    case "g":
      for (const c of Array.from(node.children)) walk(pdf, c, st, clips);
      break;
    case "rect": {
      const [x, y, w, h] = [n("x"), n("y"), n("width"), n("height")];
      pdf.path([{ pts: [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], closed: true }], style);
      break;
    }
    case "line":
      pdf.path([{ pts: [[n("x1"), n("y1")], [n("x2"), n("y2")]], closed: false }], { stroke: style.stroke, lineWidth: st.strokeWidth });
      break;
    case "circle":
      pdf.path([circlePath(n("cx"), n("cy"), n("r"))], style);
      break;
    case "path":
      pdf.path(parsePath(attr("d") ?? ""), style);
      break;
    case "text": {
      const text = node.textContent ?? "";
      const bold = Number(st.fontWeight) >= 600 || st.fontWeight === "bold";
      const w = textWidth(text, st.fontSize, bold);
      let x = n("x");
      if (st.anchor === "middle") x -= w / 2;
      else if (st.anchor === "end") x -= w;
      const color = parseColor(st.fill, st.fillOpacity);
      if (color) pdf.text(x, n("y"), text, st.fontSize, color, bold);
      break;
    }
    default:
      break;
  }
  pdf.restore();
}

/** One SVG on a page of the same size (points = px). */
export function svgPage(svg: SVGSVGElement) {
  const W = Number(svg.getAttribute("width"));
  const H = Number(svg.getAttribute("height"));
  const pdf = new PdfCanvas(W, H);
  drawSvg(pdf, svg, 0, 0, 1);
  return pdf.page();
}

export async function blobToJpeg(png: Blob, quality = 0.92): Promise<{ jpeg: Uint8Array; width: number; height: number }> {
  const bmp = await createImageBitmap(png);
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  const g = c.getContext("2d")!;
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(bmp, 0, 0);
  const blob = await new Promise<Blob>((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("JPEG encode failed"))), "image/jpeg", quality));
  return { jpeg: new Uint8Array(await blob.arrayBuffer()), width: c.width, height: c.height };
}
