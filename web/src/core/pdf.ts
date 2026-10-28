// Minimal PDF writer (no dependencies): vector paths, text in the standard
// Helvetica / Helvetica-Bold / Symbol fonts, transparency and JPEG images.
// Used for the 2D sketch PDFs and the report (desktop tool: report_<ts>.pdf).

export interface PdfImage {
  /** JPEG bytes (embedded as DCTDecode) */
  jpeg: Uint8Array;
  width: number;
  height: number;
}

export interface PdfPageSpec {
  /** page size in points (1/72 in) */
  width: number;
  height: number;
  /** content stream; draw with PdfCanvas */
  content: string;
  images: PdfImage[];
}

const f = (v: number) => {
  const s = (Math.round(v * 1000) / 1000).toString();
  return s === "-0" ? "0" : s;
};

// WinAnsi code points for the non-ASCII characters the app writes.
const WIN_ANSI: Record<string, number> = {
  "°": 0xb0, "×": 0xd7, "µ": 0xb5, "²": 0xb2, "–": 0x96, "—": 0x97, "…": 0x85, "·": 0xb7, "±": 0xb1, "é": 0xe9,
};
// Greek letters in the Symbol font.
const SYMBOL: Record<string, string> = {
  "Δ": "D", "θ": "q", "π": "p", "α": "a", "β": "b", "γ": "g", "δ": "d", "φ": "f", "ρ": "r", "σ": "s", "τ": "t", "ω": "w", "μ": "m",
};

function pdfString(s: string): string {
  let out = "(";
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    let code: number;
    if (c >= 32 && c < 127) code = c;
    else if (WIN_ANSI[ch] !== undefined) code = WIN_ANSI[ch];
    else code = 63; // '?'
    const b = String.fromCharCode(code);
    if (b === "(" || b === ")" || b === "\\") out += "\\" + b;
    else if (code > 126) out += "\\" + code.toString(8).padStart(3, "0");
    else out += b;
  }
  return out + ")";
}

export type FontName = "F1" | "F2" | "F3"; // Helvetica, Helvetica-Bold, Symbol

/** Splits text into runs of the regular font and the Symbol font. */
export function textRuns(text: string, bold: boolean): { font: FontName; text: string }[] {
  const runs: { font: FontName; text: string }[] = [];
  for (const ch of text) {
    const sym = SYMBOL[ch];
    const font: FontName = sym ? "F3" : bold ? "F2" : "F1";
    const t = sym ?? ch;
    const last = runs[runs.length - 1];
    if (last && last.font === font) last.text += t;
    else runs.push({ font, text: t });
  }
  return runs;
}

/**
 * Drawing API in a top-left origin, y-down coordinate system (like SVG); the
 * page transform flips it into PDF space.
 */
export class PdfCanvas {
  private ops: string[] = [];
  readonly alphas = new Set<string>();
  readonly images: PdfImage[] = [];

  constructor(readonly width: number, readonly height: number) {
    this.ops.push(`1 0 0 -1 0 ${f(height)} cm`);
  }

  raw(op: string): void {
    this.ops.push(op);
  }

  save(): void {
    this.ops.push("q");
  }

  restore(): void {
    this.ops.push("Q");
  }

  transform(a: number, b: number, c: number, d: number, e: number, g: number): void {
    this.ops.push(`${f(a)} ${f(b)} ${f(c)} ${f(d)} ${f(e)} ${f(g)} cm`);
  }

  private alpha(fill: number, stroke: number): void {
    const key = `${f(fill)}_${f(stroke)}`;
    this.alphas.add(key);
    this.ops.push(`/GS${key.replace(/\./g, "p").replace(/-/g, "m")} gs`);
  }

  /** subpaths: arrays of points; closed adds "h" */
  path(subpaths: { pts: [number, number][]; closed: boolean; curves?: boolean }[], style: PathStyle): void {
    if (!style.fill && !style.stroke) return;
    this.save();
    this.alpha(style.fill?.a ?? 1, style.stroke?.a ?? 1);
    if (style.fill) this.ops.push(`${f(style.fill.r)} ${f(style.fill.g)} ${f(style.fill.b)} rg`);
    if (style.stroke) {
      this.ops.push(`${f(style.stroke.r)} ${f(style.stroke.g)} ${f(style.stroke.b)} RG ${f(style.lineWidth ?? 1)} w 1 j 1 J`);
      if (style.dash) this.ops.push(`[${style.dash.map(f).join(" ")}] 0 d`);
    }
    for (const sp of subpaths) {
      if (!sp.pts.length) continue;
      if (sp.curves) {
        // pts: start, then triples of bezier control points
        this.ops.push(`${f(sp.pts[0][0])} ${f(sp.pts[0][1])} m`);
        for (let i = 1; i + 2 < sp.pts.length; i += 3) {
          const [a, b, c] = [sp.pts[i], sp.pts[i + 1], sp.pts[i + 2]];
          this.ops.push(`${f(a[0])} ${f(a[1])} ${f(b[0])} ${f(b[1])} ${f(c[0])} ${f(c[1])} c`);
        }
      } else {
        sp.pts.forEach(([x, y], i) => this.ops.push(`${f(x)} ${f(y)} ${i ? "l" : "m"}`));
      }
      if (sp.closed) this.ops.push("h");
    }
    this.ops.push(style.fill && style.stroke ? "B" : style.fill ? "f" : "S");
    this.restore();
  }

  clipRect(x: number, y: number, w: number, h: number): void {
    this.ops.push(`${f(x)} ${f(y)} ${f(w)} ${f(h)} re W n`);
  }

  /** Text with its baseline at (x, y); width() must match for anchoring. */
  text(x: number, y: number, s: string, size: number, color: Rgba, bold = false): void {
    this.save();
    this.alpha(color.a, color.a);
    this.ops.push(`${f(color.r)} ${f(color.g)} ${f(color.b)} rg BT 1 0 0 -1 ${f(x)} ${f(y)} Tm`);
    for (const run of textRuns(s, bold)) this.ops.push(`/${run.font} ${f(size)} Tf ${pdfString(run.text)} Tj`);
    this.ops.push("ET");
    this.restore();
  }

  image(img: PdfImage, x: number, y: number, w: number, h: number): void {
    this.images.push(img);
    const name = `Im${this.images.length}`;
    // image space is y-up; undo our flip locally
    this.ops.push(`q ${f(w)} 0 0 ${f(-h)} ${f(x)} ${f(y + h)} cm /${name} Do Q`);
  }

  page(): PdfPageSpec & { alphas: Set<string> } {
    return { width: this.width, height: this.height, content: this.ops.join("\n"), images: this.images, alphas: this.alphas };
  }
}

export interface Rgba {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface PathStyle {
  fill?: Rgba | null;
  stroke?: Rgba | null;
  lineWidth?: number;
  dash?: number[];
}

export function buildPdf(pages: (PdfPageSpec & { alphas?: Set<string> })[], title = "OpenSpiRobs"): Uint8Array {
  const chunks: Uint8Array[] = [];
  let length = 0;
  const offsets: number[] = [];
  const push = (data: string | Uint8Array) => {
    const b = typeof data === "string" ? latin1(data) : data;
    chunks.push(b);
    length += b.length;
  };
  // object ids: 1 catalog, 2 pages, 3..5 fonts, 6 info, then per page
  let nextId = 7;
  const objects: { id: number; body: (string | Uint8Array)[] }[] = [];
  const pageIds: number[] = [];
  const alphaKeys = new Set<string>();
  pages.forEach((p) => p.alphas?.forEach((a) => alphaKeys.add(a)));
  const gsIds = new Map<string, number>();
  for (const key of alphaKeys) {
    const id = nextId++;
    gsIds.set(key, id);
    const [fa, sa] = key.split("_");
    objects.push({ id, body: [`<< /Type /ExtGState /ca ${fa} /CA ${sa} >>`] });
  }
  const gsDict = [...gsIds].map(([k, id]) => `/GS${k.replace(/\./g, "p").replace(/-/g, "m")} ${id} 0 R`).join(" ");

  for (const p of pages) {
    const pageId = nextId++;
    const contentId = nextId++;
    const imgIds = p.images.map(() => nextId++);
    pageIds.push(pageId);
    const xobj = imgIds.map((id, i) => `/Im${i + 1} ${id} 0 R`).join(" ");
    objects.push({
      id: pageId,
      body: [
        `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${f(p.width)} ${f(p.height)}] /Contents ${contentId} 0 R ` +
          `/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> /ExtGState << ${gsDict} >> /XObject << ${xobj} >> >> >>`,
      ],
    });
    const content = latin1(p.content);
    objects.push({ id: contentId, body: [`<< /Length ${content.length} >>\nstream\n`, content, "\nendstream"] });
    p.images.forEach((img, i) => {
      objects.push({
        id: imgIds[i],
        body: [
          `<< /Type /XObject /Subtype /Image /Width ${img.width} /Height ${img.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${img.jpeg.length} >>\nstream\n`,
          img.jpeg,
          "\nendstream",
        ],
      });
    });
  }
  objects.unshift(
    { id: 1, body: ["<< /Type /Catalog /Pages 2 0 R >>"] },
    { id: 2, body: [`<< /Type /Pages /Kids [${pageIds.map((i) => `${i} 0 R`).join(" ")}] /Count ${pageIds.length} >>`] },
    { id: 3, body: ["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"] },
    { id: 4, body: ["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"] },
    { id: 5, body: ["<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>"] },
    { id: 6, body: [`<< /Title ${pdfString(title)} /Producer (OpenSpiRobs web) >>`] },
  );
  objects.sort((a, b) => a.id - b.id);

  push("%PDF-1.4\n%\xe2\xe3\xcf\xd3\n");
  for (const o of objects) {
    offsets[o.id] = length;
    push(`${o.id} 0 obj\n`);
    o.body.forEach(push);
    push("\nendobj\n");
  }
  const xref = length;
  const size = nextId;
  let table = `xref\n0 ${size}\n0000000000 65535 f \n`;
  for (let i = 1; i < size; i++) table += `${String(offsets[i] ?? 0).padStart(10, "0")} 00000 ${offsets[i] !== undefined ? "n" : "f"} \n`;
  push(table);
  push(`trailer\n<< /Size ${size} /Root 1 0 R /Info 6 0 R >>\nstartxref\n${xref}\n%%EOF\n`);

  const out = new Uint8Array(length);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function latin1(s: string): Uint8Array {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    out[i] = c < 256 ? c : 63;
  }
  return out;
}

