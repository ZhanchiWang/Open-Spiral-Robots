// Per-tendon MuJoCo settings (the desktop tool's TendonConfigDialog), with
// the cross-section schematic of where each tendon sits.

import { DEFAULT_TENDON, type TendonConfig } from "../core/params";
import { el } from "./controls";

// same order as the showcase stage (coral, sea glass, kelp, sun, anemone, lagoon, urchin, slate)
export const TENDON_COLORS = ["#ff6b4a", "#5fd3c4", "#9bc53d", "#ffc145", "#c77dff", "#3fa7d6", "#f26ca7", "#93a1a8"];

function sectionSvg(n: number, current: number): SVGSVGElement {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 160 172");
  svg.setAttribute("class", "section-svg");
  const add = (tag: string, a: Record<string, string | number>, text?: string) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v));
    if (text) e.textContent = text;
    svg.appendChild(e);
    return e;
  };
  const cx = 80;
  const cy = 92;
  const r = 52;
  add("text", { x: 4, y: 12, "font-size": 10, class: "muted-fill" }, `Cross-section (${n} cables)`);
  add("circle", { cx, cy, r, class: "section-disc" });
  add("line", { x1: cx - r - 4, y1: cy, x2: cx + r + 4, y2: cy, class: "section-axis" });
  add("line", { x1: cx, y1: cy - r - 4, x2: cx, y2: cy + r + 4, class: "section-axis" });
  add("text", { x: cx + r + 4, y: cy + 3, "font-size": 9, class: "muted-fill" }, "X");
  add("text", { x: cx + 3, y: cy - r - 6, "font-size": 9, class: "muted-fill" }, "Y");
  for (let k = 0; k < n; k++) {
    const t = ((90 + (k * 360) / n) * Math.PI) / 180;
    const sx = cx + r * Math.cos(t);
    const sy = cy - r * Math.sin(t);
    if (k === current) add("circle", { cx: sx, cy: sy, r: 10, fill: "none", class: "section-ring" });
    add("circle", { cx: sx, cy: sy, r: 5, fill: TENDON_COLORS[k % TENDON_COLORS.length], stroke: "#202020" });
    add("text", { x: cx + (r + 14) * Math.cos(t) - 3, y: cy - (r + 14) * Math.sin(t) + 4, "font-size": k === current ? 11 : 9, "font-weight": k === current ? 700 : 400, class: "fg-fill" }, String(k + 1));
  }
  return svg;
}

/** Resolves with the new configs, or null when cancelled. */
export function editTendons(initial: TendonConfig[], nJoints: number, title = "Tendon configuration"): Promise<TendonConfig[] | null> {
  const cfgs = initial.map((c) => ({ ...c }));
  let current = 0;
  return new Promise((resolve) => {
    const dlg = el("dialog", { class: "dialog" });
    const select = el("select", { "aria-label": "Tendon" });
    cfgs.forEach((_, i) => select.append(el("option", { value: i }, `Tendon ${i + 1}`)));
    const num = (min: number, max: number, step: number) => el("input", { type: "number", min, max, step, class: "ctl-number wide" });
    const lo = num(-100, 0, 0.1);
    const hi = num(0, 100, 0.1);
    const gear = num(0, 10000, 1);
    const joint = el("select", { "aria-label": "Connect to joint" });
    joint.append(el("option", { value: nJoints }, "All (full chain)"));
    for (let j = 1; j < nJoints; j++) joint.append(el("option", { value: j }, `J${j}`));
    const sectionHost = el("div", { class: "section-host" });
    const load = () => {
      const c = cfgs[current];
      lo.value = String(c.ctrl_lo);
      hi.value = String(c.ctrl_hi);
      gear.value = String(c.gear);
      joint.value = String(c.joint >= nJoints ? nJoints : c.joint);
      sectionHost.replaceChildren(sectionSvg(cfgs.length, current));
    };
    const save = () => {
      const clampN = (v: string, a: number, b: number, d: number) => {
        const x = Number(v);
        return Number.isFinite(x) ? Math.max(a, Math.min(b, x)) : d;
      };
      cfgs[current] = {
        ctrl_lo: clampN(lo.value, -100, 0, DEFAULT_TENDON.ctrl_lo),
        ctrl_hi: clampN(hi.value, 0, 100, DEFAULT_TENDON.ctrl_hi),
        gear: clampN(gear.value, 0, 10000, DEFAULT_TENDON.gear),
        joint: Number(joint.value),
      };
    };
    select.addEventListener("change", () => {
      save();
      current = Number(select.value);
      load();
    });
    const grid = el(
      "div",
      { class: "form-grid" },
      el("label", {}, "Ctrl min"), lo,
      el("label", {}, "Ctrl max"), hi,
      el("label", {}, "Gear (N)"), gear,
      el("label", {}, "Connect to joint"), joint,
    );
    const close = (v: TendonConfig[] | null) => {
      dlg.close();
      dlg.remove();
      resolve(v);
    };
    const reset = el("button", { type: "button", class: "btn" }, "Reset defaults");
    reset.addEventListener("click", () => {
      cfgs[current] = { ...DEFAULT_TENDON, joint: nJoints };
      load();
    });
    const cancel = el("button", { type: "button", class: "btn" }, "Cancel");
    cancel.addEventListener("click", () => close(null));
    const ok = el("button", { type: "button", class: "btn primary" }, "OK");
    ok.addEventListener("click", () => {
      save();
      close(cfgs);
    });
    dlg.addEventListener("cancel", (e) => {
      e.preventDefault();
      close(null);
    });
    dlg.append(
      el("h2", {}, title),
      el("div", { class: "row" }, el("label", {}, "Tendon"), select),
      el("div", { class: "dialog-body" }, grid, sectionHost),
      el("div", { class: "dialog-actions" }, reset, el("span", { class: "spacer" }), cancel, ok),
    );
    document.body.append(dlg);
    load();
    dlg.showModal();
  });
}
