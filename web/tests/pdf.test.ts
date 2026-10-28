import { describe, expect, it } from "vitest";
import { buildPdf, PdfCanvas, textRuns } from "../src/core/pdf";

const latin1 = (b: Uint8Array) => Array.from(b, (c) => String.fromCharCode(c)).join("");

describe("PDF writer", () => {
  it("writes a valid xref and the drawing operators", () => {
    const pdf = new PdfCanvas(200, 100);
    pdf.path([{ pts: [[10, 10], [50, 10], [50, 40]], closed: true }], { fill: { r: 0.6, g: 0.8, b: 0.9, a: 0.35 }, stroke: { r: 0, g: 0, b: 1, a: 1 }, lineWidth: 0.6 });
    pdf.text(10, 80, "Δθ=30° (deg)", 9, { r: 0, g: 0, b: 0, a: 1 });
    const bytes = buildPdf([pdf.page(), new PdfCanvas(100, 100).page()]);
    const s = latin1(bytes);
    expect(s.startsWith("%PDF-1.4")).toBe(true);
    expect(s.trimEnd().endsWith("%%EOF")).toBe(true);
    const startxref = Number(s.match(/startxref\n(\d+)/)![1]);
    expect(s.slice(startxref, startxref + 4)).toBe("xref");
    const entries = s.slice(startxref).split("\n").slice(2).filter((l) => / [nf] $/.test(l));
    entries.forEach((e, id) => {
      if (id === 0 || e.endsWith("f ")) return;
      const off = Number(e.slice(0, 10));
      expect(s.slice(off, off + `${id} 0 obj`.length)).toBe(`${id} 0 obj`);
    });
    expect(s).toContain("/Count 2");
    expect(s).toContain("/BaseFont /Symbol");
    expect(s).toContain("/ca 0.35");
    expect(s).toContain("(Dq) Tj"); // Greek through the Symbol font
    expect(s).toContain("(=30\\260 \\(deg\\)) Tj"); // WinAnsi degree sign, escaped parens
  });

  it("splits Greek letters into Symbol runs", () => {
    expect(textRuns("θmax=6π", true)).toEqual([
      { font: "F3", text: "q" },
      { font: "F2", text: "max=6" },
      { font: "F3", text: "p" },
    ]);
  });
});
