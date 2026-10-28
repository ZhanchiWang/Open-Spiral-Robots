import { Store, shareLink } from "../app/store";
import { CadClient } from "../cad/client";
import type { BuildResult, CadResponse } from "../cad/protocol";
import {
  baseUnitHeight,
  cableCount,
  checkDesign,
  sitePoints,
  type Design,
} from "../core/design";
import { download, makeZip, timestamp } from "../core/files";
import { defaultParams, parseParams, serializeParams } from "../core/params";
import { resolve, resolveTendons } from "../core/state";
import { generateMujocoXml } from "../core/xml";
import { buildPdf, PdfCanvas } from "../core/pdf";
import { blobToJpeg, drawSvg, svgPage, textWidth } from "../ui/svgPdf";
import { getLog, log as rawLog, logger } from "../debug/log";
import { buildDebugPanel } from "../debug/panel";
import { button, el } from "../ui/controls";
import { ModelView } from "../ui/modelView";
import { buildPanel } from "../ui/panel";
import { DARK, LIGHT, renderCartesian, renderPolar, svgToPng, svgToString, type CartesianView } from "../ui/sketch";
import { editTendons } from "../ui/tendonDialog";

const log = logger("app");
const cadLog = logger("cad");
const VERSION = "0.1.0";

export interface EngineeringHooks {
  onSimulate: () => void;
  onClose: () => void;
  perf: () => { fps: number; stepUs: number };
}

/**
 * The full design tool (2D sketch, exact OpenCascade CAD, exports, all
 * parameters, debug drawer), mounted inside the showcase's Engineering sheet.
 */
export function mountEngineering(app: HTMLElement, store: Store, hooks: EngineeringHooks) {
const cad = new CadClient();
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
// the space scene keeps the whole app dark, whatever the system theme
const isDark = () => darkQuery.matches || document.documentElement.dataset.env === "space";
let visible = false;

// ---------------------------------------------------------------- layout
const status = el("span", { class: "status", role: "status" }, "");
const busy = el("span", { class: "spinner", hidden: true, "aria-hidden": "true" });
const autoBuild = el("input", { type: "checkbox", checked: true });
const loadInput = el("input", { type: "file", accept: ".json,application/json", hidden: true });

const header = el(
  "header",
  { class: "topbar" },
  el("div", { class: "brand" }, el("strong", {}, "Engineering"), el("span", { class: "muted" }, " · exact CAD, exports, all parameters")),
  el(
    "div",
    { class: "actions" },
    button("Load…", () => loadInput.click(), "btn", "Load a parameter file (last_params.json from the desktop tool works too)"),
    button("Save Parameters", () => saveParams(), "btn green", "Download the parameters as JSON"),
    button("Share link", () => copyShareLink(), "btn", "Copy a URL that reproduces this design"),
    button("Reset Parameters", () => resetParams(), "btn"),
    loadInput,
  ),
  el("div", { class: "status-wrap" }, busy, status),
  button("✕", () => hooks.onClose(), "btn icon", "Close (Esc)"),
);

// 2D sketch column
const polarSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
const cartSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
const polarHost = el("div", { class: "plot" });
const cartHost = el("div", { class: "plot" });
polarHost.append(polarSvg);
cartHost.append(cartSvg);
const hover = el("div", { class: "hover mono" }, "");
const metrics = el("div", { class: "metrics" });
const sketchCol = el("section", { class: "col col-sketch", "aria-label": "2D Sketch" }, el("h2", {}, "2D Sketch"), polarHost, el("div", { class: "plot-wrap" }, cartHost, hover), metrics);

// 3D / simulation column
const modelHost = el("div", { class: "view-host" });
const modelOverlay = el("div", { class: "view-overlay" }, "Building the 3D model (OpenCascade WASM)…");
const modelView = new ModelView(modelHost);
const viewOpts = el("div", { class: "inline" });
const optBox = (label: string, key: keyof ModelView["opts"], title?: string) => {
  const i = el("input", { type: "checkbox" });
  i.checked = modelView.opts[key] as boolean;
  i.addEventListener("change", () => {
    modelView.setOptions({ [key]: i.checked });
    if (key === "holes" && i.checked) scheduleBuild(0);
    sectionSlider.hidden = !modelView.opts.section;
  });
  viewOpts.append(el("label", { class: "check", title }, i, label));
};
optBox("Edges", "edges", "B-rep edges from OpenCascade");
optBox("Wireframe", "wireframe");
optBox("Holes", "holes", "Cable-hole cutters (orange)");
optBox("Sites", "sites", "Cable site points used for the MuJoCo export");
optBox("Axes", "axes", "Export frame at the base face (X red, Y green, Z blue)");
optBox("Section", "section", "Cut the view with a plane across the robot");
const sectionSlider = el("input", { type: "range", min: 0, max: 1, step: 0.005, value: 0.5, hidden: true, "aria-label": "Section position", class: "section-slider" });
sectionSlider.addEventListener("input", () => modelView.setOptions({ sectionPos: Number(sectionSlider.value) }));
const buildState = el("span", { class: "badge" }, "3D: —");
const modelToolbar = el(
  "div",
  { class: "toolbar" },
  button("Generate 3D Model", () => scheduleBuild(0), "btn primary"),
  el("label", { class: "check", title: "Rebuild the 3D model automatically after each change" }, autoBuild, "Auto"),
  button("Reset 3D View", () => modelView.resetView(), "btn"),
  el("span", { class: "sep" }),
  button("Export STEP", () => exportCad("step"), "btn blue"),
  button("Export STL", () => exportCad("stl"), "btn blue"),
  button("Export XML", () => exportXml(), "btn blue", "MuJoCo robot.xml + baselink.stl (zip)"),
  el("span", { class: "sep" }),
  buildState,
);
const modelTab = el("div", { class: "model-tab" }, modelToolbar, el("div", { class: "toolbar sub" }, viewOpts, sectionSlider), el("div", { class: "view-wrap" }, modelHost, modelOverlay));

const centerCol = el("section", { class: "col col-3d" }, el("h2", {}, "3D Model · OpenCascade"), modelTab);

// parameter column
const panel = buildPanel(store, (a) => (a === "save-sketch" ? void saveSketch() : a === "save-pdf" ? void savePdfReport() : void configureTendons()));
const paramCol = el("section", { class: "col col-params", "aria-label": "Parameters" }, el("h2", {}, "Parameters"), panel);

const debug = buildDebugPanel();
app.append(header, el("main", { class: "layout" }, sketchCol, centerCol, paramCol), debug.root);

// ---------------------------------------------------------------- 2D sketch
let cartView: CartesianView | null = null;
let highlight: number | null = null;
function drawSketch() {
  const res = store.resolved;
  const d = res.design;
  const palette = isDark() ? DARK : LIGHT;
  if (!d) {
    polarSvg.replaceChildren();
    cartSvg.replaceChildren();
    metrics.textContent = "No unit fits — increase θ max.";
    return;
  }
  const pw = polarHost.clientWidth || 400;
  const ph = polarHost.clientHeight || 360;
  renderPolar(polarSvg, d, { width: pw, height: ph, palette, title: "SpiRobs Design", highlightUnit: highlight });
  const cw = cartHost.clientWidth || 400;
  const ch = cartHost.clientHeight || 260;
  cartView = renderCartesian(cartSvg, d, res.params.elastic_enabled, { width: cw, height: ch, palette, title: "Unfolded (Cartesian)", highlightUnit: highlight });
  const m: [string, string][] = [
    ["Taper Angle", `${d.taperDeg.toFixed(2)}°`],
    ["Tip Size", `${d.tipSize.toFixed(2)} mm`],
    ["Num of Units", String(d.units)],
    ["Robot Length", `${d.length.toFixed(2)} mm`],
    ["Base Size", `${d.baseSize.toFixed(2)} mm`],
  ];
  metrics.replaceChildren(...m.map(([k, v]) => el("div", { class: "metric" }, el("span", { class: "k" }, k), el("span", { class: "v" }, v))));
}
let sketchQueued = false;
function queueSketch() {
  if (sketchQueued) return;
  sketchQueued = true;
  requestAnimationFrame(() => {
    sketchQueued = false;
    drawSketch();
  });
}
new ResizeObserver(queueSketch).observe(polarHost);
new ResizeObserver(queueSketch).observe(cartHost);
darkQuery.addEventListener("change", () => {
  queueSketch();
  modelView.vp.setTheme(isDark());
});
cartSvg.addEventListener("mousemove", (ev) => {
  const d = store.resolved.design;
  if (!cartView || !d) return;
  const r = cartSvg.getBoundingClientRect();
  const [x, y] = cartView.toData([ev.clientX - r.left, ev.clientY - r.top]);
  // units are ordered tip -> base; joints J1 (base) .. Jn (tip)
  const k = d.primary.findIndex((p) => x >= p[3][0] && x <= p[2][0]);
  const unitText = k >= 0 ? ` · unit ${k + 1}/${d.units} from tip${k < d.units - 1 ? ` · joint to base side J${d.units - 1 - k}` : ""}` : "";
  hover.textContent = `x ${x.toFixed(2)}  y ${y.toFixed(2)} mm${unitText}`;
  const h = k >= 0 ? k : null;
  if (h !== highlight) {
    highlight = h;
    queueSketch();
  }
});
cartSvg.addEventListener("mouseleave", () => {
  hover.textContent = "";
  if (highlight !== null) {
    highlight = null;
    queueSketch();
  }
});

// ---------------------------------------------------------------- 3D build
let lastCad: CadResponse | null = null;
let buildTimer = 0;
let builtKey = "";
let triangles = 0;
let edges = 0;

cad.onBusy = (b, what) => {
  busy.hidden = !b;
  if (b) status.textContent = what === "build" ? "Building 3D model…" : `Running ${what}…`;
};
cad.onResponse = (r) => {
  lastCad = r;
  for (const l of r.logs) cadLog.debug(l);
  const t = r.timings;
  cadLog[r.ok ? "info" : "error"](`${r.type} ${r.ok ? "ok" : "failed"} in ${t.total?.toFixed(0)} ms${r.error ? `: ${r.error}` : ""} (${Object.entries(t).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${v.toFixed(0)}`).join(", ")})`);
  status.textContent = r.ok ? `${r.type} done in ${(t.total / 1000).toFixed(2)} s` : `${r.type} failed`;
};

function overlayFor(d: Design) {
  const p = store.params;
  return { sites: sitePoints(d, p.tip_hole_pos / 100, p.base_hole_pos / 100), robotLength: d.length, baseSize: d.baseSize };
}

function scheduleBuild(delay = 400) {
  clearTimeout(buildTimer);
  buildTimer = window.setTimeout(() => void build(), delay);
}

async function build(): Promise<BuildResult | null> {
  const res = store.resolved;
  if (!res.design) {
    buildState.textContent = "3D: no geometry";
    return null;
  }
  const key = JSON.stringify(res.params) + modelView.opts.holes;
  buildState.textContent = "3D: building…";
  buildState.className = "badge busy";
  try {
    const r = await cad.request("build", res.params, { holes: modelView.opts.holes });
    const b = r.build!;
    modelView.show(b, overlayFor(res.design));
    modelOverlay.hidden = true;
    triangles = modelView.stats.triangles;
    edges = modelView.stats.edges;
    builtKey = key;
    const stale = key !== JSON.stringify(store.params) + modelView.opts.holes;
    buildState.textContent = stale ? "3D: outdated" : `3D: ${(r.timings.total / 1000).toFixed(2)} s`;
    buildState.className = stale ? "badge warn" : "badge ok";
    return b;
  } catch (e) {
    if ((e as { superseded?: boolean }).superseded) return null;
    buildState.textContent = "3D: error";
    buildState.className = "badge err";
    modelOverlay.textContent = `3D build failed: ${(e as Error).message}`;
    modelOverlay.hidden = false;
    debug.open("Log");
    return null;
  }
}

// ---------------------------------------------------------------- exports
const ts = () => timestamp();

function saveParams() {
  download("last_params.json", serializeParams(store.params), "application/json");
  log.info("parameters saved");
}

loadInput.addEventListener("change", async () => {
  const f = loadInput.files?.[0];
  loadInput.value = "";
  if (!f) return;
  try {
    store.replace(parseParams(JSON.parse(await f.text())));
    log.info(`loaded parameters from ${f.name}`);
  } catch (e) {
    log.error(`could not read ${f.name}: ${(e as Error).message}`);
  }
});

function resetParams() {
  store.replace(defaultParams());
  log.info("parameters reset to defaults");
}

async function copyShareLink() {
  const url = shareLink(store.params);
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    status.textContent = "Share link copied";
  } catch {
    status.textContent = "Share link is in the address bar";
  }
}

async function exportCad(kind: "step" | "stl") {
  try {
    const r = await cad.request(kind === "step" ? "export-step" : "export-stl", store.params);
    download(`spi_rob_${ts()}.${kind}`, r.bytes!, kind === "step" ? "application/step" : "model/stl");
  } catch (e) {
    log.error(`export ${kind} failed: ${(e as Error).message}`);
    debug.open("Log");
  }
}

async function configureTendons(): Promise<boolean> {
  const res = store.resolved;
  if (!res.design) return false;
  const n = cableCount(res.params);
  const cfgs = await editTendons(resolveTendons(res.params.tendon_configs, n, res.design.joints), res.design.joints);
  if (!cfgs) return false;
  store.set({ tendon_configs: cfgs });
  return true;
}

/** Build robot.xml + baselink.stl exactly as the desktop tool's Export XML. */
async function buildXmlBundle(): Promise<{ xml: string; stl: Uint8Array }> {
  const res = resolve(store.params);
  const d = res.design;
  const p = res.params;
  if (!d) throw new Error("No geometry to export.");
  const r = await cad.request("baselink", p);
  const xml = generateMujocoXml({
    stlName: "baselink.stl",
    unitHeight: baseUnitHeight(d),
    scale: Math.exp(p.b * ((p.dtheta_deg * Math.PI) / 180)),
    numUnits: d.units,
    jointType: p.two_cable ? "hinge" : "ball",
    jointLimitDeg: p.dtheta_deg,
    robotLength: d.length,
    sitePoints: sitePoints(d, p.tip_hole_pos / 100, p.base_hole_pos / 100),
    cableMode: cableCount(p),
    jointStiffness: p.sim_stiffness,
    jointDamping: p.sim_damping,
    gravity: p.sim_gravity,
    jointAngles: p.joint_angles,
    bendDirDeg: p.bend_dir_deg,
    extrusionDepth: p.two_cable ? p.extrusion : 0,
    tendonConfigs: p.tendon_configs,
  });
  return { xml, stl: r.bytes! };
}

async function exportXml() {
  if (!(await configureTendons())) return;
  try {
    const { xml, stl } = await buildXmlBundle();
    const stamp = ts();
    download(`xml_${stamp}.zip`, makeZip([{ name: `xml_${stamp}/robot.xml`, data: xml }, { name: `xml_${stamp}/baselink.stl`, data: stl }]), "application/zip");
    log.info(`exported xml_${stamp}.zip (${(stl.length / 1024).toFixed(0)} KB STL)`);
  } catch (e) {
    log.error(`XML export failed: ${(e as Error).message}`);
    debug.open("Log");
  }
}


async function saveSketch() {
  const stamp = ts();
  const zip = await sketchZip(stamp);
  if (zip) download(`sketch_${stamp}.zip`, zip, "application/zip");
}

async function sketchZip(stamp = ts()): Promise<Uint8Array | null> {
  const d = store.resolved.design;
  if (!d) return null;
  const W = 640;
  const polar = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const cart = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  renderPolar(polar, d, { width: W, height: 560, palette: LIGHT, title: "SpiRobs Design" });
  renderCartesian(cart, d, store.params.elastic_enabled, { width: W, height: 420, palette: LIGHT, title: "Unfolded (Cartesian)" });
  const files: { name: string; data: Uint8Array | string }[] = [
    { name: `polar_${stamp}.svg`, data: svgToString(polar) },
    { name: `cartesian_${stamp}.svg`, data: svgToString(cart) },
  ];
  // svgToPng needs the elements in the document for font metrics
  const holder = el("div", { style: "position:absolute;left:-9999px;top:0" });
  holder.append(polar, cart);
  document.body.append(holder);
  try {
    files.push({ name: `polar_${stamp}.png`, data: new Uint8Array(await (await svgToPng(polar)).arrayBuffer()) });
    files.push({ name: `cartesian_${stamp}.png`, data: new Uint8Array(await (await svgToPng(cart)).arrayBuffer()) });
    files.push({ name: `polar_${stamp}.pdf`, data: buildPdf([svgPage(polar)], "SpiRobs Design (polar)") });
    files.push({ name: `cartesian_${stamp}.pdf`, data: buildPdf([svgPage(cart)], "Unfolded (Cartesian)") });
    let shot = "";
    let png: Blob | null = null;
    if (builtKey) {
      const shotPng = await modelView.vp.screenshot(true);
      png = shotPng;
      files.push({ name: `preview_3d_${stamp}.png`, data: new Uint8Array(await shotPng.arrayBuffer()) });
      shot = await new Promise<string>((res) => {
        const fr = new FileReader();
        fr.onload = () => res(String(fr.result));
        fr.readAsDataURL(shotPng);
      });
    }
    files.push({ name: `report_${stamp}.svg`, data: reportSvg(polar, cart, shot, d) });
    files.push({ name: `report_${stamp}.pdf`, data: await reportPdf(d, png) });
  } finally {
    holder.remove();
  }
  return makeZip(files);
}

async function savePdfReport() {
  const d = store.resolved.design;
  if (!d) return;
  const png = builtKey ? await modelView.vp.screenshot(true) : null;
  download(`report_${ts()}.pdf`, await reportPdf(d, png), "application/pdf");
}

/**
 * The desktop tool's report_<ts>.pdf: an 11 x 6 in page with a parameter
 * header, the polar and unfolded sketches as vector graphics and the 3D view
 * as an image (same grid: polar top-left, unfolded bottom, 3D on the right).
 */
async function reportPdf(d: Design, shot: Blob | null): Promise<Uint8Array> {
  const p = store.params;
  const W = 792;
  const H = 432;
  const pdf = new PdfCanvas(W, H);
  const fg = { r: 0.2, g: 0.2, b: 0.2, a: 1 };
  const muted = { r: 0.33, g: 0.33, b: 0.33, a: 1 };
  pdf.text(12, 22, "OpenSpiRob Design", 14, fg, true);
  pdf.text(12, 36, `a=${p.a.toFixed(4)} mm  b=${p.b.toFixed(4)}  Δθ=${p.dtheta_deg}°  θmax=${p.theta_max_pi.toFixed(1)}π  p=${p.p.toFixed(2)}`, 9, muted);
  pdf.text(12, 48, `Taper=${d.taperDeg.toFixed(2)}°  Tip=${d.tipSize.toFixed(2)} mm  Base=${d.baseSize.toFixed(2)} mm  Length=${d.length.toFixed(2)} mm  Units=${d.units}`, 9, muted);
  const holder = el("div", { style: "position:absolute;left:-9999px;top:0" });
  document.body.append(holder);
  try {
    // render at 2x the target box, then draw at half scale
    const polar = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const cart = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    holder.append(polar, cart);
    renderPolar(polar, d, { width: 560, height: 370, palette: LIGHT });
    renderCartesian(cart, d, p.elastic_enabled, { width: 1120, height: 380, palette: LIGHT, title: "Unfolded (Cartesian)" });
    drawSvg(pdf, polar, 10, 56, 0.5);
    drawSvg(pdf, cart, 6, 240, 0.5);
  } finally {
    holder.remove();
  }
  pdf.text(560 + 116 - textWidth("3D Model", 11, true) / 2, 70, "3D Model", 11, fg, true);
  if (shot) {
    const img = await blobToJpeg(shot);
    const boxW = 224;
    const boxH = 340;
    const s = Math.min(boxW / img.width, boxH / img.height);
    const w = img.width * s;
    const h = img.height * s;
    pdf.image(img, 564 + (boxW - w) / 2, 80 + (boxH - h) / 2, w, h);
  } else {
    pdf.text(600, 240, "(generate the 3D model first)", 9, muted);
  }
  return buildPdf([pdf.page()], "OpenSpiRob Design report");
}

function reportSvg(polar: SVGSVGElement, cart: SVGSVGElement, shot: string, d: Design): string {
  const p = store.params;
  const inner = (s: SVGSVGElement, x: number, y: number) => {
    const c = s.cloneNode(true) as SVGSVGElement;
    c.setAttribute("x", String(x));
    c.setAttribute("y", String(y));
    return new XMLSerializer().serializeToString(c);
  };
  const line1 = `a=${p.a.toFixed(4)} mm  b=${p.b.toFixed(4)}  Δθ=${p.dtheta_deg}°  θmax=${p.theta_max_pi.toFixed(1)}π  p=${p.p.toFixed(2)}`;
  const line2 = `Taper=${d.taperDeg.toFixed(2)}°  Tip=${d.tipSize.toFixed(2)} mm  Base=${d.baseSize.toFixed(2)} mm  Length=${d.length.toFixed(2)} mm  Units=${d.units}`;
  const W = 1320;
  const H = 1080;
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Helvetica, Arial, sans-serif">
<rect width="${W}" height="${H}" fill="#fff"/>
<text x="20" y="34" font-size="22" font-weight="700" fill="#333">OpenSpiRob Design</text>
<text x="20" y="60" font-size="13" fill="#555">${line1}</text>
<text x="20" y="80" font-size="13" fill="#555">${line2}</text>
${inner(polar, 10, 100)}
${inner(cart, 10, 660)}
${shot ? `<text x="680" y="120" font-size="14" font-weight="600" fill="#333">3D Model</text><image x="670" y="130" width="640" height="560" preserveAspectRatio="xMidYMid meet" href="${shot}"/>` : ""}
</svg>`;
}

// ---------------------------------------------------------------- reactions
store.subscribe((res, changed) => {
  queueSketch();
  debug.updateState(res);
  const key = JSON.stringify(res.params) + modelView.opts.holes;
  if (key !== builtKey) {
    buildState.textContent = "3D: outdated";
    buildState.className = "badge warn";
    if (autoBuild.checked && visible) scheduleBuild();
  }
  if ([...changed].some((k) => ["pose_enabled", "two_cable", "num_cables"].includes(k))) log.debug(`changed: ${[...changed].join(", ")}`);
});
autoBuild.addEventListener("change", () => autoBuild.checked && scheduleBuild(0));

setInterval(() => {
  debug.updatePerf({
    cad: lastCad?.timings ?? null,
    cadLogs: lastCad?.logs ?? [],
    triangles,
    edges,
    fps3d: modelView.vp.fps,
    fpsSim: hooks.perf().fps,
    simStepUs: hooks.perf().stepUs,
  });
}, 1000);

// ---------------------------------------------------------------- console API
const api = {
  setVisible(v: boolean) {
    visible = v;
    if (v) {
      modelView.vp.setTheme(isDark());
      modelView.vp.resize();
      queueSketch();
      if (JSON.stringify(store.params) + modelView.opts.holes !== builtKey) scheduleBuild(0);
    }
  },
  debug,
  saveParams,
  copyShareLink,
  configureTendons,
  version: VERSION,
  params: () => structuredClone(store.params),
  set: (p: Record<string, unknown>) => store.set(parseParams(p, store.params)),
  design: () => store.resolved.design,
  checks: () => checkDesign(store.params, store.resolved.design),
  build: async () => {
    await build();
    return lastCad?.timings;
  },
  exportXml: () => buildXmlBundle(),
  exportStep: async () => (await cad.request("export-step", store.params)).bytes,
  exportStl: async () => (await cad.request("export-stl", store.params)).bytes,
  sketchZip: () => sketchZip(),
  reportPdf: async () => (store.resolved.design ? reportPdf(store.resolved.design, builtKey ? await modelView.vp.screenshot(true) : null) : null),
  simulate: () => hooks.onSimulate(),
  modelView,
  log: () => getLog(),
  reset: () => resetParams(),
  store,
};

// ---------------------------------------------------------------- start
drawSketch();
debug.updateState(store.resolved);
rawLog("info", "app", `OpenSpiRobs web ${VERSION}`);
for (const i of checkDesign(store.params, store.resolved.design)) log.warn(i.message);
return api;
}
