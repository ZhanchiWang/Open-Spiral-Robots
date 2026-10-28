// Parameter model of the design tool.
//
// Keys and units match design-tool/exports/last_params.json written by the
// Python DesignTool (percent values stay in percent), so a parameter file can be
// moved between the desktop tool and the web tool unchanged.

export interface TendonConfig {
  gear: number;
  ctrl_lo: number;
  ctrl_hi: number;
  /** Joint the cable connects to; num_joints means "All (full chain)". */
  joint: number;
}

export interface Params {
  a: number;
  b: number;
  dtheta_deg: number;
  theta_max_pi: number;
  p: number;
  elastic_percent: number;
  elastic_enabled: boolean;
  extrusion: number;
  cone_angle1: number;
  cone_angle2: number;
  show_cone_cuts: boolean;
  tip_hole_pos: number; // %
  tip_hole_size: number; // mm
  base_hole_pos: number; // %
  base_hole_size: number; // mm
  teardrop_holes: boolean;
  two_cable: boolean;
  num_cables: number;
  cable3_cut_enabled: boolean;
  cable3_cut_pos: number; // %
  cable3_cut_size: number; // %
  sim_stiffness: number;
  sim_damping: number;
  sim_gravity: number;
  pose_enabled: boolean;
  filler_auto: boolean;
  bend_dir_deg: number;
  /** Base -> tip (J1 = base joint). */
  joint_angles: number[];
  joint_filler_widths: number[];
  joint_filler_heights: number[];
  /** Web-only: per-tendon MuJoCo settings (the desktop tool asks in a dialog). */
  tendon_configs?: TendonConfig[];
}

export const DEFAULT_PARAMS: Readonly<Params> = Object.freeze({
  a: 4.95,
  b: 0.1764,
  dtheta_deg: 30,
  theta_max_pi: 6.0,
  p: 0.5,
  elastic_percent: 5.0,
  elastic_enabled: true,
  // NaN = "not initialised yet": the first geometry pass sets it to
  // base_size * 0.6, like the desktop tool's _extrusion_initialized flag.
  extrusion: Number.NaN,
  cone_angle1: 5.0,
  cone_angle2: 15.0,
  show_cone_cuts: false,
  tip_hole_pos: 50.0,
  tip_hole_size: 1.4,
  base_hole_pos: 90.0,
  base_hole_size: 3.0,
  teardrop_holes: false,
  two_cable: true,
  num_cables: 3,
  cable3_cut_enabled: true,
  cable3_cut_pos: 0.0,
  cable3_cut_size: 14.0,
  sim_stiffness: 5.0,
  sim_damping: 0.2,
  sim_gravity: 0.0,
  pose_enabled: false,
  filler_auto: true,
  bend_dir_deg: 0.0,
  joint_angles: [] as number[],
  joint_filler_widths: [] as number[],
  joint_filler_heights: [] as number[],
}) as Params;

export const DEFAULT_TENDON: Readonly<Omit<TendonConfig, "joint">> = Object.freeze({
  gear: 50.0,
  ctrl_lo: -1.0,
  ctrl_hi: 1.0,
});

export function defaultParams(): Params {
  return structuredClone({ ...DEFAULT_PARAMS, joint_angles: [], joint_filler_widths: [], joint_filler_heights: [] });
}

export interface Range {
  min: number;
  max: number;
  step: number;
  decimals: number;
}

/** Static control ranges (dynamic ones come from geometry, see design.ts). */
export const RANGES = {
  a: { min: 0.1, max: 100, step: 0.01, decimals: 4 },
  b: { min: 0.01, max: 0.35, step: 0.001, decimals: 4 },
  dtheta_deg: { min: 1, max: 60, step: 1, decimals: 0 },
  theta_max_pi: { min: 1, max: 12, step: 0.1, decimals: 1 },
  p: { min: 0, max: 0.5, step: 0.01, decimals: 2 },
  elastic_percent: { min: 0, max: 100, step: 1, decimals: 1 },
  tip_hole_pos: { min: 0, max: 100, step: 1, decimals: 0 },
  tip_hole_size: { min: 0.1, max: 10, step: 0.1, decimals: 2 },
  base_hole_pos: { min: 0, max: 100, step: 1, decimals: 0 },
  base_hole_size: { min: 0.1, max: 10, step: 0.1, decimals: 2 },
  num_cables: { min: 3, max: 8, step: 1, decimals: 0 },
  cable3_cut_pos: { min: 0, max: 50, step: 1, decimals: 1 },
  cable3_cut_size: { min: 0, max: 20, step: 0.1, decimals: 2 },
  sim_stiffness: { min: 0, max: 100, step: 0.1, decimals: 3 },
  sim_damping: { min: 0, max: 100, step: 0.1, decimals: 3 },
  sim_gravity: { min: 0, max: 30, step: 0.1, decimals: 2 },
  bend_dir_deg: { min: 0, max: 360, step: 1, decimals: 0 },
  joint_angle: { min: -90, max: 90, step: 0.5, decimals: 1 },
} satisfies Record<string, Range>;

/**
 * Validation limits for loaded data. Wider than the slider ranges: uniform
 * scaling (core/scaling.ts) legitimately moves a, hole sizes, stiffness and
 * damping far outside the ranges the engineering sliders show.
 */
export const LIMITS: Record<string, { min: number; max: number }> = {
  ...RANGES,
  a: { min: 1e-4, max: 1e4 },
  tip_hole_size: { min: 1e-5, max: 1e3 },
  base_hole_size: { min: 1e-5, max: 1e3 },
  sim_stiffness: { min: 0, max: 1e12 },
  sim_damping: { min: 0, max: 1e12 },
};

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

const NUMBER_KEYS = [
  "a", "b", "dtheta_deg", "theta_max_pi", "p", "elastic_percent", "extrusion",
  "cone_angle1", "cone_angle2", "tip_hole_pos", "tip_hole_size", "base_hole_pos",
  "base_hole_size", "num_cables", "cable3_cut_pos", "cable3_cut_size",
  "sim_stiffness", "sim_damping", "sim_gravity", "bend_dir_deg",
] as const;
const BOOL_KEYS = [
  "elastic_enabled", "show_cone_cuts", "teardrop_holes", "two_cable",
  "cable3_cut_enabled", "pose_enabled", "filler_auto",
] as const;
const ARRAY_KEYS = ["joint_angles", "joint_filler_widths", "joint_filler_heights"] as const;

/**
 * Merge an untrusted JSON object (a last_params.json, a share link, ...) onto
 * the defaults. Unknown keys are ignored, bad values fall back to defaults and
 * static ranges are enforced; geometry-dependent ranges are applied later.
 */
export function parseParams(raw: unknown, base: Params = defaultParams()): Params {
  const out = structuredClone(base);
  if (!raw || typeof raw !== "object") return out;
  const src = raw as Record<string, unknown>;
  for (const k of NUMBER_KEYS) {
    const v = Number(src[k]);
    if (src[k] !== undefined && src[k] !== null && Number.isFinite(v)) out[k] = v;
  }
  for (const k of BOOL_KEYS) {
    if (typeof src[k] === "boolean") out[k] = src[k] as boolean;
  }
  for (const k of ARRAY_KEYS) {
    const v = src[k];
    if (Array.isArray(v)) out[k] = v.map(Number).filter(Number.isFinite);
  }
  if (Array.isArray(src.tendon_configs)) {
    out.tendon_configs = (src.tendon_configs as unknown[]).map((c) => {
      const o = (c ?? {}) as Record<string, unknown>;
      const num = (x: unknown, d: number) => (Number.isFinite(Number(x)) ? Number(x) : d);
      return {
        gear: num(o.gear, DEFAULT_TENDON.gear),
        ctrl_lo: num(o.ctrl_lo, DEFAULT_TENDON.ctrl_lo),
        ctrl_hi: num(o.ctrl_hi, DEFAULT_TENDON.ctrl_hi),
        joint: Math.round(num(o.joint, 1e9)),
      };
    });
  }
  // like the desktop tool: int() truncates, and each spin box keeps its own
  // decimals. Only scale-free fields are rounded: lengths (a, hole sizes,
  // extrusion, fillers) and stiffness/damping of scaled designs go far below
  // the desktop ranges and must survive a save/load round trip.
  out.dtheta_deg = Math.trunc(out.dtheta_deg);
  out.num_cables = Math.trunc(out.num_cables);
  const roundTo = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;
  for (const k of ["b", "theta_max_pi", "p", "elastic_percent", "tip_hole_pos", "base_hole_pos", "cable3_cut_pos", "sim_gravity", "bend_dir_deg"] as const) {
    out[k] = roundTo(out[k], RANGES[k].decimals);
  }
  out.joint_angles = out.joint_angles.map((v) => roundTo(v, RANGES.joint_angle.decimals));
  for (const [k, r] of Object.entries(LIMITS)) {
    if (k in out && typeof (out as unknown as Record<string, unknown>)[k] === "number") {
      const rec = out as unknown as Record<string, number>;
      rec[k] = clamp(rec[k], r.min, r.max);
    }
  }
  return out;
}

/** Serialise in the desktop tool's last_params.json layout. */
export function serializeParams(p: Params): string {
  const out: Record<string, unknown> = {};
  for (const k of [
    "a", "b", "dtheta_deg", "theta_max_pi", "p", "elastic_percent", "elastic_enabled",
    "extrusion", "cone_angle1", "cone_angle2", "show_cone_cuts", "tip_hole_pos",
    "tip_hole_size", "base_hole_pos", "base_hole_size", "teardrop_holes", "two_cable",
    "num_cables", "cable3_cut_enabled", "cable3_cut_pos", "cable3_cut_size",
    "sim_stiffness", "sim_damping", "sim_gravity", "pose_enabled", "filler_auto",
    "bend_dir_deg", "joint_angles", "joint_filler_widths", "joint_filler_heights",
    "tendon_configs",
  ] as const) {
    if (p[k] !== undefined) out[k] = p[k];
  }
  return JSON.stringify(out, null, 2);
}
