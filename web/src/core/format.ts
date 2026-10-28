// Python-compatible number formatting, so generated XML matches the desktop
// tool's output character for character.

/**
 * x.toFixed(digits), but exact ties round half to even like Python (JS rounds
 * them away from zero: 0.0078125 -> "0.007813", Python "0.007812"). A tie
 * needs an exact binary fraction with at most digits + 1 decimals, which
 * toFixed(100) spells out exactly.
 */
function toFixedEven(x: number, digits: number): string {
  const s = x.toFixed(digits);
  const ax = Math.abs(x);
  if (!Number.isFinite(x) || ax >= 1e21 || digits >= 100) return s;
  const [ip, fp = ""] = ax.toFixed(100).split(".");
  const rest = fp.slice(digits);
  if (rest[0] !== "5" || !/^0*$/.test(rest.slice(1))) return s;
  const kept = digits > 0 ? `${ip}.${fp.slice(0, digits)}` : ip;
  const last = Number(kept[kept.length - 1]);
  if (last % 2 === 1) return s; // odd: rounding away from zero is also to even
  return x < 0 ? `-${kept}` : kept;
}

/** Python f"{x:.{digits}f}" (keeps the sign of negative zero). */
export function fmtF(x: number, digits: number): string {
  const s = toFixedEven(x, digits);
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
    out = toFixedEven(x, Math.max(0, p - 1 - exp));
    if (out.includes(".")) out = out.replace(/0+$/, "").replace(/\.$/, "");
  }
  return out;
}
