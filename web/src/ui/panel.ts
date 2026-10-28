// Right-hand parameter panel (2D, fabrication, simulation, joint pose).

import { RANGES, type Params, type Range } from "../core/params";
import type { Resolved } from "../core/state";
import type { Store } from "../app/store";
import { el, numberControl, section, toggleControl, button, type NumberControl, type ToggleControl } from "./controls";

type NumKey = { [K in keyof Params]-?: Params[K] extends number ? K : never }[keyof Params];
type BoolKey = { [K in keyof Params]-?: Params[K] extends boolean ? K : never }[keyof Params];

export function buildPanel(store: Store, onAction: (a: "save-sketch" | "save-pdf" | "tendons") => void): HTMLElement {
  const nums = new Map<NumKey, NumberControl>();
  const bools = new Map<BoolKey, ToggleControl>();

  const num = (key: NumKey, label: string, unit = "", r: Range = (RANGES as Record<string, Range>)[key], title?: string) => {
    const c = numberControl({
      label,
      unit,
      ...r,
      value: store.params[key] as number,
      title,
      onInput: (v) => store.set({ [key]: v } as Partial<Params>),
    });
    nums.set(key, c);
    return c.row;
  };
  const bool = (key: BoolKey, label: string, title?: string) => {
    const c = toggleControl(label, store.params[key] as boolean, (v) => store.set({ [key]: v } as Partial<Params>), title);
    bools.set(key, c);
    return c.row;
  };

  // --- 2D ---
  const s2d = section(
    "2D Parameters",
    num("a", "a", "(mm)", RANGES.a, "Spiral scale: r = a·e^(bθ)"),
    num("b", "b", "", RANGES.b, "Spiral growth rate"),
    num("dtheta_deg", "Δθ", "(deg)", RANGES.dtheta_deg, "Angular size of one unit"),
    num("theta_max_pi", "θ max", "(π)", RANGES.theta_max_pi, "Spiral end angle in multiples of π"),
    num("p", "p", "", RANGES.p, "Centre-spiral position between the two spiral turns"),
    el(
      "div",
      { class: "btn-row" },
      button("Save 2D Sketch", () => onAction("save-sketch"), "btn", "Zip with SVG, PNG and PDF of both sketches, the 3D view and a report"),
      button("Save PDF Report", () => onAction("save-pdf"), "btn", "One-page PDF: parameters, vector sketches and the 3D view"),
    ),
  );

  // --- Fabrication ---
  const cableMode = el("div", { class: "seg", role: "radiogroup", "aria-label": "Number of cables" });
  const two = el("button", { type: "button", class: "seg-btn", role: "radio" }, "2 cables");
  const many = el("button", { type: "button", class: "seg-btn", role: "radio" }, "N cables");
  two.addEventListener("click", () => store.set({ two_cable: true }));
  many.addEventListener("click", () => store.set({ two_cable: false }));
  cableMode.append(two, many);
  const nCables = num("num_cables", "N", "", RANGES.num_cables, "Number of cables (3–8)");

  const twoWrap = el(
    "div",
    { class: "sub" },
    num("extrusion", "Extrusion", "(mm)", { min: 0, max: 1, step: 0.1, decimals: 2 }, "Body thickness (Z) of the 2-cable robot"),
    num("cone_angle1", "ConeAngle1", "(deg)", { min: 0, max: 1, step: 0.5, decimals: 1 }, "Base-face wedge"),
    num("cone_angle2", "ConeAngle2", "(deg)", { min: 0, max: 1, step: 0.5, decimals: 1 }, "Base chamfer"),
    bool("show_cone_cuts", "Show cone cuts (yellow)", "Highlight the material ConeAngle1 / ConeAngle2 remove"),
  );
  const cutParams = el(
    "div",
    { class: "sub" },
    num("cable3_cut_pos", "Cut start", "(%)", RANGES.cable3_cut_pos, "Axial start of the flats (0 = full length, 50 = base half)"),
    num("cable3_cut_size", "Cut size", "(%)", RANGES.cable3_cut_size, "Cut depth (0 = none, 20 = inscribed N-gon)"),
  );
  const manyWrap = el("div", { class: "sub" }, bool("cable3_cut_enabled", "Enable edge cut", "Flats between adjacent cable holes"), cutParams);

  const sFab = section(
    "Fabrication Parameters",
    bool("elastic_enabled", "Elastic layer / axis"),
    num("elastic_percent", "Elastic", "(%)", RANGES.elastic_percent, "Elastic layer half-angle as % of half the taper angle"),
    num("tip_hole_pos", "Pos TipHole", "(%)", RANGES.tip_hole_pos),
    num("tip_hole_size", "Size TipHole", "(mm)", RANGES.tip_hole_size),
    num("base_hole_pos", "Pos BaseHole", "(%)", RANGES.base_hole_pos),
    num("base_hole_size", "Size BaseHole", "(mm)", RANGES.base_hole_size),
    bool("teardrop_holes", "Teardrop holes (+Z)", "Teardrop hole profile with its tip toward +Z, for printing without supports"),
    el("div", { class: "ctl-row tight" }, el("span", { class: "ctl-label" }, "Num of cables"), cableMode),
    nCables,
    twoWrap,
    manyWrap,
  );

  // --- Simulation ---
  const tendonBtn = button("Tendon configuration…", () => onAction("tendons"), "btn block", "Per-tendon ctrl range, gear and the joint each cable reaches");
  const sSim = section(
    "Simulation Parameters",
    num("sim_stiffness", "Stiffness", "", RANGES.sim_stiffness, "Base joint stiffness (scaled down toward the tip)"),
    num("sim_damping", "Damping", "", RANGES.sim_damping),
    num("sim_gravity", "Gravity", "(m/s²)", RANGES.sim_gravity),
    tendonBtn,
  );

  // --- Joint pose ---
  const jointSelect = el("select", { "aria-label": "Joint" });
  let currentJoint = 0;
  const jointAngle = numberControl({
    label: "Angle", unit: "(deg)", ...RANGES.joint_angle, value: 0,
    onInput: (v) => store.set((p) => { p.joint_angles[currentJoint] = v; }),
  });
  const fillerW = numberControl({
    label: "Width", unit: "(mm)", min: 0, max: 1, step: 0.1, decimals: 2, value: 0,
    onInput: (v) => store.set((p) => { p.filler_auto = false; p.joint_filler_widths[currentJoint] = v; }),
  });
  const fillerH = numberControl({
    label: "Height", unit: "(mm)", min: 0, max: 1, step: 0.1, decimals: 2, value: 0,
    onInput: (v) => store.set((p) => { p.filler_auto = false; p.joint_filler_heights[currentJoint] = v; }),
  });
  jointSelect.addEventListener("change", () => {
    currentJoint = Number(jointSelect.value);
    refresh(store.resolved);
  });
  const bendDir = el("div", {}, num("bend_dir_deg", "Bend dir", "(deg)", RANGES.bend_dir_deg, "Bending direction of the N-cable ball joints"));
  const allJoints = numberControl({
    label: "All joints", unit: "(deg)", ...RANGES.joint_angle, value: 0,
    title: "Set every joint to the same angle (uniform curl)",
    onInput: (v) => store.set((p) => { p.joint_angles = p.joint_angles.map(() => v); }),
  });
  const poseWrap = el(
    "div",
    { class: "sub" },
    bendDir,
    allJoints.row,
    bool("filler_auto", "Auto filler size", "Size each knuckle filler from the elastic core"),
    el("div", { class: "ctl-row tight" }, el("span", { class: "ctl-label" }, "Joint"), jointSelect),
    jointAngle.row,
    fillerW.row,
    fillerH.row,
    button("Reset Pose", () => store.set((p) => { p.joint_angles = p.joint_angles.map(() => 0); p.bend_dir_deg = 0; }), "btn block"),
  );
  const sPose = section("Joint Pose", bool("pose_enabled", "Enable joint pose", "Bend the CAD model and the exported MuJoCo chain"), poseWrap);

  const root = el("div", { class: "panel-scroll" }, s2d, sFab, sSim, sPose);

  const refresh = (res: Resolved) => {
    const p = res.params;
    const d = res.design;
    for (const [k, c] of nums) c.set(p[k] as number);
    for (const [k, c] of bools) c.set(p[k] as boolean);
    if (d) {
      nums.get("extrusion")!.setRange(d.extrusionRange[0], d.extrusionRange[1]);
      nums.get("cone_angle1")!.setRange(0, res.cone1Max);
      nums.get("cone_angle2")!.setRange(0, d.cone2Max);
    }
    for (const k of ["extrusion", "cone_angle1", "cone_angle2"] as const) nums.get(k)!.set(p[k]);
    two.setAttribute("aria-checked", String(p.two_cable));
    many.setAttribute("aria-checked", String(!p.two_cable));
    two.classList.toggle("on", p.two_cable);
    many.classList.toggle("on", !p.two_cable);
    nums.get("num_cables")!.setEnabled(!p.two_cable);
    twoWrap.hidden = !p.two_cable;
    manyWrap.hidden = p.two_cable;
    cutParams.hidden = !p.cable3_cut_enabled;
    nums.get("elastic_percent")!.setEnabled(p.elastic_enabled);
    poseWrap.hidden = !p.pose_enabled;
    bendDir.hidden = p.two_cable;

    const n = p.joint_angles.length;
    if (jointSelect.options.length !== n) {
      jointSelect.replaceChildren(...Array.from({ length: n }, (_, j) => el("option", { value: j }, `J${j + 1}${j === 0 ? " (base)" : j === n - 1 ? " (tip)" : ""}`)));
    }
    currentJoint = Math.min(currentJoint, Math.max(0, n - 1));
    jointSelect.value = String(currentJoint);
    const has = n > 0;
    jointAngle.setEnabled(has);
    fillerW.setEnabled(has);
    fillerH.setEnabled(has);
    if (has) {
      jointAngle.set(p.joint_angles[currentJoint]);
      fillerW.setRange(0, res.fillerMax);
      fillerH.setRange(0, res.fillerMax);
      fillerW.set(p.joint_filler_widths[currentJoint]);
      fillerH.set(p.joint_filler_heights[currentJoint]);
    }
    const uniform = n > 0 && p.joint_angles.every((a) => a === p.joint_angles[0]);
    allJoints.set(uniform ? p.joint_angles[0] : 0);
  };
  refresh(store.resolved);
  store.subscribe((r) => refresh(r));
  return root;
}
