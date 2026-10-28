// Uniform scaling of a design, with physics that scale like a real body made
// of the same material:
//   lengths ∝ s, mass ∝ s³, bending stiffness of the elastic core ∝ s³,
//   damping (constant damping ratio, inertia ∝ s⁵) ∝ s⁴, tendon force ∝ s².
// The natural frequency then goes as 1/s: a 10× smaller robot curls 10×
// faster, which is what the Scale mode shows.

import { DEFAULT_TENDON, type Params } from "./params";

export function scaleParams(p: Params, s: number): Params {
  const out = structuredClone(p);
  out.a = p.a * s;
  out.tip_hole_size = p.tip_hole_size * s;
  out.base_hole_size = p.base_hole_size * s;
  if (Number.isFinite(p.extrusion)) out.extrusion = p.extrusion * s;
  out.joint_filler_widths = p.joint_filler_widths.map((v) => v * s);
  out.joint_filler_heights = p.joint_filler_heights.map((v) => v * s);
  out.sim_stiffness = p.sim_stiffness * s ** 3;
  out.sim_damping = p.sim_damping * s ** 4;
  out.tendon_configs = (p.tendon_configs ?? []).map((c) => ({ ...c, gear: (c.gear ?? DEFAULT_TENDON.gear) * s * s }));
  return out;
}

/** Simulation step that keeps the same number of steps per oscillation. */
export function timestepFor(lengthMm: number, refLengthMm = 464, refStep = 0.002): number {
  return Math.min(refStep, Math.max(2e-6, refStep * (lengthMm / refLengthMm)));
}

/** Pretty length with an adaptive unit. */
export function formatLength(mm: number): string {
  if (mm >= 1000) return `${(mm / 1000).toFixed(mm >= 10000 ? 1 : 2)} m`;
  if (mm >= 10) return `${(mm / 10).toFixed(mm >= 100 ? 1 : 2)} cm`;
  return `${mm.toFixed(2)} mm`;
}

export function formatTime(s: number): string {
  if (!Number.isFinite(s)) return "—";
  if (s >= 1) return `${s.toFixed(2)} s`;
  if (s >= 1e-3) return `${(s * 1e3).toFixed(s >= 0.1 ? 0 : 1)} ms`;
  return `${(s * 1e6).toFixed(0)} µs`;
}

export function formatMass(kg: number): string {
  if (kg >= 1) return `${kg.toFixed(2)} kg`;
  if (kg >= 1e-3) return `${(kg * 1e3).toFixed(kg >= 0.1 ? 0 : 1)} g`;
  return `${(kg * 1e6).toFixed(0)} mg`;
}
