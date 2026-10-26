// Ties parameters to geometry: the range updates and joint-control rebuilds
// that the desktop tool does inside update_2d / _rebuild_joint_controls.

import { cableCount, computeDesign, cone1Max, jointFillerDefaults, type Design } from "./design";
import { clamp, DEFAULT_TENDON, type Params, type TendonConfig } from "./params";

export interface Resolved {
  params: Params;
  design: Design | null;
  cone1Max: number;
  fillerMax: number;
}

/**
 * Compute the design and bring geometry-dependent parameters into range.
 * Returns a new params object; the input is not modified.
 */
export function resolve(input: Params): Resolved {
  const params = structuredClone(input);
  const design = computeDesign(params);
  if (!design) return { params, design, cone1Max: 0, fillerMax: 1 };

  const [minExt, maxExt] = design.extrusionRange;
  if (!Number.isFinite(params.extrusion)) params.extrusion = design.defaultExtrusion;
  params.extrusion = clamp(params.extrusion, minExt, maxExt);

  const c1 = cone1Max(design, params.extrusion);
  params.cone_angle1 = clamp(params.cone_angle1, 0, c1);
  params.cone_angle2 = clamp(params.cone_angle2, 0, design.cone2Max);

  const n = design.joints;
  const fit = (arr: number[], fill: (j: number) => number) =>
    Array.from({ length: n }, (_, j) => (j < arr.length ? arr[j] : fill(j)));
  params.joint_angles = fit(params.joint_angles, () => 0).map((a) => clamp(a, -90, 90));
  if (!params.pose_enabled) params.joint_angles = params.joint_angles.map(() => 0);

  const defaults = jointFillerDefaults(design, params.extrusion);
  const fillerMax = Math.max(design.baseSize * 3, 1);
  if (params.filler_auto) {
    params.joint_filler_widths = defaults.widths;
    params.joint_filler_heights = defaults.heights;
  } else {
    params.joint_filler_widths = fit(params.joint_filler_widths, (j) => defaults.widths[j]).map((v, j) =>
      v > 0 ? clamp(v, 0, fillerMax) : defaults.widths[j],
    );
    params.joint_filler_heights = fit(params.joint_filler_heights, (j) => defaults.heights[j]).map((v, j) =>
      v > 0 ? clamp(v, 0, fillerMax) : defaults.heights[j],
    );
  }

  params.tendon_configs = resolveTendons(params.tendon_configs, cableCount(params), n);
  return { params, design, cone1Max: c1, fillerMax };
}

export function resolveTendons(cfgs: TendonConfig[] | undefined, nTendons: number, nJoints: number): TendonConfig[] {
  return Array.from({ length: nTendons }, (_, k) => {
    const c = cfgs?.[k];
    const joint = c && Number.isFinite(c.joint) && c.joint >= 1 && c.joint < nJoints ? Math.round(c.joint) : nJoints;
    return {
      gear: c ? c.gear : DEFAULT_TENDON.gear,
      ctrl_lo: c ? clamp(c.ctrl_lo, -100, 0) : DEFAULT_TENDON.ctrl_lo,
      ctrl_hi: c ? clamp(c.ctrl_hi, 0, 100) : DEFAULT_TENDON.ctrl_hi,
      joint,
    };
  });
}
