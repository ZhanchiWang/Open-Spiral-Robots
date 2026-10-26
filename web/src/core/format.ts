// Python-compatible number formatting, so generated XML matches the desktop
// tool's output character for character.

/** Python f"{x:.{digits}f}" (keeps the sign of negative zero). */
export function fmtF(x: number, digits: number): string {
  const s = x.toFixed(digits);
  if ((Object.is(x, -0) || x < 0) && !s.startsWith("-")) return "-" + s;
  return s;
}

/** Python f"{x:.{precision}g}". */
export function fmtG(x: number, precision: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const p = Math.max(1, precision);
  const exp = Number(x.toExponential(p - 1).split("e")[1]);
  let out: string;
  if (exp < -4 || exp >= p) {
    const [mant, e] = x.toExponential(p - 1).split("e");
    const m = mant.includes(".") ? mant.replace(/0+$/, "").replace(/\.$/, "") : mant;
    const en = Number(e);
    out = `${m}e${en < 0 ? "-" : "+"}${String(Math.abs(en)).padStart(2, "0")}`;
  } else {
    out = x.toFixed(Math.max(0, p - 1 - exp));
    if (out.includes(".")) out = out.replace(/0+$/, "").replace(/\.$/, "");
  }
  return out;
}
