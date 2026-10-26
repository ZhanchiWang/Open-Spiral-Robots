import "@fontsource/instrument-serif/400.css";
import "@fontsource/instrument-serif/400-italic.css";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./ui/style.css";
import "./showcase/showcase.css";
import { Vector3 } from "three";
import { Store, shareLink } from "./app/store";
import { cableCount, checkDesign } from "./core/design";
import { renderLinkMesh } from "./core/linkMesh";
import { defaultParams, parseParams, RANGES, type Params } from "./core/params";
import { formatLength, formatMass, formatTime, scaleParams } from "./core/scaling";
import { resolve } from "./core/state";
import { getLog, logger } from "./debug/log";
import { button, el, numberControl, type NumberControl } from "./ui/controls";
import { icon, installRangePainter } from "./ui/icons";
import { PhysicsClient } from "./showcase/client";
import { getLang, setLang, t, type Key } from "./showcase/i18n";
import { DEFAULT_TUNING, type ObjectKind, type SceneObject } from "./showcase/physics";
import type { Frame, WorkerMeta } from "./showcase/protocol";
import { initialQuality, Stage, type Environment, type Quality } from "./showcase/stage";
import { joystick, logSlider, timeline } from "./showcase/widgets";

const log = logger("show");
type Mode = "flex" | "speed" | "scale";

// ---------------------------------------------------------------- state
const { params: initialParams } = Store.loadInitial();
const store = new Store(initialParams);
const hashArgs = new URLSearchParams(location.hash.slice(1));
const ui = {
  mode: (["flex", "speed", "scale"].includes(hashArgs.get("m") ?? "") ? hashArgs.get("m") : "flex") as Mode,
  object: { kind: "none", size: 0.16, pos: [0, 0, 0] } as SceneObject,
  refs: new Set<"coin" | "hand" | "human">(),
  running: true,
  speed: 1,
  replayT: null as number | null,
  measures: [] as { L: number; t90: number; vmax: number }[],
};

const app = document.getElementById("app")!;
app.className = "show";
let env: Environment = (() => {
  const h = hashArgs.get("e");
  if (h === "space" || h === "ocean") return h;
  try {
    const v = localStorage.getItem("openspirobs.env");
    if (v === "space" || v === "ocean") return v;
  } catch {
    /* default below */
  }
  return "space";
})();
document.documentElement.dataset.env = env;
const stageHost = el("div", { class: "stage" });
const stage = new Stage(stageHost);
stage.setEnvironment(env);
stage.setSpiralB(store.params.b);
const phys = new PhysicsClient();
const loading = el("div", { class: "stage-loading" }, nautilus("loader"), el("span", {}, t("loading")));

// ---------------------------------------------------------------- top bar
const tabs = el("div", { class: "modes", role: "tablist" });
const modeBtn = (m: Mode, k: Key) => {
  const n = { flex: "01", speed: "02", scale: "03" }[m];
  const b = button("", () => setMode(m), "mode-tab");
  b.append(el("span", { class: "mode-num" }, n), el("span", { class: "mode-lbl" }, t(k)));
  b.setAttribute("role", "tab");
  b.dataset.mode = m;
  tabs.append(b);
};
modeBtn("flex", "modeFlex");
modeBtn("speed", "modeSpeed");
modeBtn("scale", "modeScale");
const toast = el("div", { class: "toast", role: "status" });
function say(msg: string) {
  toast.textContent = msg;
  toast.classList.add("on");
  setTimeout(() => toast.classList.remove("on"), 1800);
}
// the live law of the current design, typeset under the wordmark
const equation = el("div", { class: "equation", "aria-live": "off" });
const iconBtn = (name: Parameters<typeof icon>[0], label: string, onClick: () => void, cls = "chip") => {
  const b = button("", onClick, cls, label);
  b.setAttribute("aria-label", label);
  b.append(icon(name), el("span", { class: "chip-label" }, label));
  return b;
};
const langBtn = iconBtn("globe", getLang() === "zh" ? "EN" : "中文", () => {
  setLang(getLang() === "zh" ? "en" : "zh");
  location.reload();
});
const topbar = el(
  "header",
  { class: "topbar-show" },
  el("div", { class: "brand-show" }, nautilus("logo"), el("div", { class: "brand-text" }, el("div", { class: "wordmark" }, "OpenSpiRobs"), equation)),
  tabs,
  el(
    "div",
    { class: "top-actions" },
    langBtn,
    envBtn(),
    iconBtn("share", t("share"), () => void share(), "chip desktop-only"),
    iconBtn("sliders", t("engineering"), () => openEngineering(true), "chip accent"),
  ),
);

// ---------------------------------------------------------------- shape panel
const cablesSeg = el("div", { class: "seg-show" });
for (const n of [2, 3, 4, 6]) {
  const b = button(String(n), () => store.set({ two_cable: n === 2, num_cables: n === 2 ? store.params.num_cables : n }), "seg-item");
  b.dataset.n = String(n);
  cablesSeg.append(b);
}
const lengthReadout = el("span", { class: "value-big" });
const lengthSlider = logSlider(8, 2500, 464, (v) => (v >= 1000 ? `${v / 1000} m` : v >= 10 ? `${v / 10} cm` : `${v} mm`), [10, 100, 1000], (v) => setLength(v));
const ctl = (key: "dtheta_deg" | "theta_max_pi" | "b" | "p", label: string, unit = ""): NumberControl =>
  numberControl({ label, unit, ...RANGES[key], value: store.params[key], onInput: (v) => store.set({ [key]: v } as Partial<Params>) });
const cDtheta = ctl("dtheta_deg", t("dtheta"), "°");
const cTurns = ctl("theta_max_pi", t("turns"), "π");
const cB = ctl("b", t("taper"));
const cP = ctl("p", t("centre"));
const gravity = el("input", { type: "checkbox", role: "switch" });
gravity.addEventListener("change", () => store.set({ sim_gravity: gravity.checked ? 9.81 : 0 }));
const unitsInfo = el("span", { class: "muted mono" });
const shapePanel = el(
  "section",
  { class: "cpanel shape-col", "aria-label": t("shape") },
  el("h3", {}, t("shape")),
  el("div", { class: "len-row" }, el("span", { class: "muted" }, t("length")), lengthReadout),
  lengthSlider.root,
  el("div", { class: "shape-grid" }, cDtheta.row, cTurns.row, cB.row, cP.row),
  el(
    "div",
    { class: "row-split" },
    el("span", { class: "muted" }, t("cables")),
    cablesSeg,
    el("label", { class: "toggle-row" }, gravity, el("span", { class: "switch" }), el("span", {}, t("gravity"))),
  ),
  el("div", { class: "row-split" }, unitsInfo,
  button(t("more"), () => openEngineering(true), "link-btn")),
);

function setLength(mm: number) {
  const d = store.resolved.design;
  if (!d) return;
  const f = mm / d.length;
  if (Math.abs(f - 1) < 1e-4) return;
  store.replace(resolve(scaleParams(store.params, f)).params);
}

// ---------------------------------------------------------------- HUD
const hud = el("aside", { class: "readouts", "aria-label": "Metrics" });
const hudItems = new Map<string, HTMLElement>();
for (const [k, label] of [
  ["curl", t("curl")],
  ["speed", t("tipSpeed")],
  ["peak", t("peakSpeed")],
  ["mass", t("mass")],
  ["rt", t("realtime")],
] as const) {
  const v = el("span", { class: "hud-v" }, "—");
  hudItems.set(k, v);
  hud.append(el("div", { class: "hud-row", "data-k": k }, el("span", { class: "hud-k" }, label), v));
}
let peakSpeed = 0;

// ---------------------------------------------------------------- dock: flex
const joy = joystick((x, y) => {
  stopAuto();
  setDirection(x, y);
});
function setDirection(x: number, y: number) {
  const meta = phys.meta;
  if (!meta) return;
  const ang = meta.tendonAngles;
  const mag = Math.min(1, Math.hypot(x, y));
  const phi = Math.atan2(y, x);
  const a = ang.map((deg) => {
    const c = Math.cos(phi - (deg * Math.PI) / 180);
    return meta.hinge ? (c > 0.5 ? mag : 0) : Math.max(0, c) * mag;
  });
  phys.send({ type: "activation", a });
}
let autoTimer = 0;
function stopAuto() {
  cancelAnimationFrame(autoTimer);
  autoTimer = 0;
}
function autoMotion(kind: "wave" | "circle") {
  stopAuto();
  const t0 = performance.now();
  const period = 2.4 * Math.max(0.25, Math.min(3, (phys.meta?.lengthM ?? 0.46) / 0.464));
  const loop = () => {
    const s = (performance.now() - t0) / 1000 / period;
    // circle: partial pull so the body sweeps a cone; wave: curl up and down
    const circle = kind === "circle" && !phys.meta?.hinge;
    const x = circle ? 0.6 * Math.cos(2 * Math.PI * s) : 0;
    const y = circle ? 0.6 * Math.sin(2 * Math.PI * s) : 0.85 * Math.sin(2 * Math.PI * s);
    joy.set(x, y);
    setDirection(x, y);
    autoTimer = requestAnimationFrame(loop);
  };
  loop();
}
const objChips = el("div", { class: "seg-show" });
for (const [kind, k] of [["none", "none"], ["cylinder", "cylinder"], ["sphere", "sphere"], ["box", "box"]] as const) {
  const b = button(t(k), () => setObject(kind), "seg-item");
  b.dataset.kind = kind;
  objChips.append(b);
}
const objSize = el("input", { type: "range", min: 0.04, max: 0.4, step: 0.005, value: ui.object.size, "aria-label": t("size") });
objSize.addEventListener("input", () => {
  ui.object.size = Number(objSize.value);
  rebuild();
});
const graspBtn = button(t("grasp"), () => toggleGrasp(), "btn-show primary");
let grasping = false;
function toggleGrasp() {
  grasping = !grasping;
  stopAuto();
  graspBtn.textContent = grasping ? t("release") : t("grasp");
  const meta = phys.meta;
  if (!meta) return;
  // pull toward the object (it sits on the +Z side of the chain by default)
  const dir = ui.object.kind !== "none" && ui.object.pos[2] < meta.mount[2] ? -1 : 1;
  const y = grasping ? dir : 0;
  joy.set(0, y);
  setDirection(0, y);
}
const heatChk = checkbox(t("heat"), true, (v) => (stage.heat = v));
const flexDock = el(
  "section",
  { class: "cpanel mode-col", "data-mode": "flex" },
  el("div", { class: "dock-col" }, el("h3", {}, t("steer")), joy.root, el("p", { class: "hint" }, t("steerHint"))),
  el(
    "div",
    { class: "dock-col grow" },
    el(
      "div",
      { class: "btn-group" },
      button(t("curlBtn"), () => (stopAuto(), joy.set(0, 1), setDirection(0, 1)), "btn-show"),
      button(t("relaxBtn"), () => (stopAuto(), joy.set(0, 0), setDirection(0, 0)), "btn-show"),
      button(t("waveBtn"), () => autoMotion("wave"), "btn-show"),
      button(t("circleBtn"), () => autoMotion("circle"), "btn-show circle-only"),
    ),
    el("h3", {}, t("object")),
    objChips,
    el("div", { class: "row-split" }, el("span", { class: "muted" }, t("size")), objSize),
    el("div", { class: "row-split" }, graspBtn, heatChk),
    el("p", { class: "hint" }, t("dragHint")),
  ),
);

function setObject(kind: ObjectKind) {
  const meta = phys.meta;
  const L = meta?.lengthM ?? 0.464;
  ui.object.kind = kind;
  // just above the chain, ~60% along it: the robot wraps upward around it
  ui.object.pos = [L * 0.62, 0, (meta?.mount[2] ?? 0.62 * L) + L * (0.1 + ui.object.size / 2)];
  grasping = false;
  graspBtn.textContent = t("grasp");
  // the object appears where the straight chain is free: start from a relaxed,
  // straight pose so a curled body never spawns inside it
  stopAuto();
  joy.set(0, 0);
  setDirection(0, 0);
  rebuild(true);
  syncChips();
}

// ---------------------------------------------------------------- dock: speed
const force = el("input", { type: "range", min: 0.2, max: 1, step: 0.05, value: 1, "aria-label": t("force") });
const results = el("div", { class: "results" });
const ghostChk = checkbox(t("ghosts"), true, () => showStrikeArtifacts());
const trailChk = checkbox(t("trail"), true, (v) => {
  stage.showTrail = v;
  showStrikeArtifacts();
});
const slowSel = el("select", { "aria-label": t("slowFactor") });
for (const f of [1, 4, 10, 30, 100]) slowSel.append(el("option", { value: f, selected: f === 10 }, `1/${f}×`));
const strikeBtn = button("", () => void strike(), "btn-show strike");
strikeBtn.append(icon("bolt", 18), el("span", {}, t("strike")), el("kbd", {}, "Space"));
addEventListener("keydown", (e: KeyboardEvent) => {
  const tag = (e.target as HTMLElement).tagName;
  if (e.code === "Space" && ui.mode === "speed" && tag !== "INPUT" && tag !== "SELECT" && tag !== "BUTTON" && !strikeBtn.disabled) {
    e.preventDefault();
    void strike();
  }
});
const speedDock = el(
  "section",
  { class: "cpanel mode-col", "data-mode": "speed" },
  el("div", { class: "dock-col" }, strikeBtn, el("div", { class: "row-split" }, el("span", { class: "muted" }, t("force")), force), el("p", { class: "hint" }, t("strikeHint"))),
  el("div", { class: "dock-col grow" }, results, el("div", { class: "row-split" }, ghostChk, trailChk), el("div", { class: "row-split" }, iconLabel(button(t("slowmo"), () => void replayStrike(), "btn-show"), "replay"), slowSel)),
);
let lastStrike: { tEnd: number; series: Awaited<ReturnType<PhysicsClient["series"]>>; t90: number; vmax: number; peak: number; L: number } | null = null;

async function strike(): Promise<void> {
  const meta = phys.meta;
  if (!meta) return;
  stopAuto();
  stopReplay();
  stage.setGhosts([]);
  stage.clearTrail();
  joy.set(0, Number(force.value));
  setRun(true, 1);
  const epoch = (phys.frame?.epoch ?? 0) + 1;
  phys.send({ type: "strike", a: Number(force.value) });
  timelineW.clear();
  peakSpeed = 0;
  strikeBtn.disabled = true;
  results.replaceChildren(el("span", { class: "muted" }, t("measuring")));
  // the curl time scales with length: ~0.6 s for the 46 cm default
  const horizon = 0.9 * (meta.lengthM / 0.464) + 0.05;
  const t0 = performance.now();
  await new Promise<void>((res) => {
    const check = () => {
      const f = phys.frame;
      if ((f && f.epoch >= epoch && f.recTo >= horizon) || performance.now() - t0 > 30000) res();
      else setTimeout(check, 50);
    };
    check();
  });
  const series = await phys.series(0, horizon, 400);
  let peak = 0;
  let vmax = 0;
  for (let i = 0; i < series.t.length; i++) {
    peak = Math.max(peak, Math.abs(series.curl[i]));
    vmax = Math.max(vmax, series.speed[i]);
  }
  let t90 = NaN;
  for (let i = 0; i < series.t.length; i++) {
    if (Math.abs(series.curl[i]) >= 0.9 * peak) {
      t90 = series.t[i];
      break;
    }
  }
  lastStrike = { tEnd: horizon, series, t90, vmax, peak, L: meta.lengthM * 1000 };
  ui.measures = ui.measures.filter((m) => Math.abs(Math.log(m.L / lastStrike!.L)) > 0.02);
  ui.measures.push({ L: lastStrike.L, t90, vmax });
  strikeBtn.disabled = false;
  renderResults();
  showStrikeArtifacts();
  drawLaw();
}

function renderResults() {
  const s = lastStrike;
  if (!s) {
    results.replaceChildren(el("span", { class: "muted" }, t("strikeHint")));
    return;
  }
  const item = (k: string, v: string) => el("div", { class: "result" }, el("span", { class: "result-v" }, v), el("span", { class: "result-k" }, k));
  results.replaceChildren(item(t("t90"), formatTime(s.t90)), item(t("peakSpeed"), `${s.vmax.toFixed(2)} m/s`), item(t("peakCurl"), `${s.peak.toFixed(0)}°`));
}

async function showStrikeArtifacts() {
  const s = lastStrike;
  if (!s) return;
  if (stage.showTrail) {
    const n = s.series.t.length;
    const end = Math.min(n, Math.max(2, s.series.t.findIndex((x) => x > Math.min(s.tEnd, (s.t90 || s.tEnd) * 1.6))) || n);
    stage.setTrail(s.series.tip.subarray(0, 3 * end), s.series.speed.subarray(0, end), s.vmax);
  } else stage.clearTrail();
  const ghosts = (ghostChk.querySelector("input") as HTMLInputElement).checked;
  if (ghosts && Number.isFinite(s.t90)) {
    const times = Array.from({ length: 9 }, (_, i) => (i / 8) * s.t90 * 1.25);
    const { poses } = await phys.poses(times);
    stage.setGhosts(poses);
  } else stage.setGhosts([]);
}

let replayRaf = 0;
function stopReplay() {
  cancelAnimationFrame(replayRaf);
  replayRaf = 0;
}
async function replayStrike() {
  const s = lastStrike;
  if (!s) return;
  stopReplay();
  setRun(false, ui.speed);
  const slow = Number(slowSel.value);
  const dur = s.tEnd * slow;
  const start = performance.now();
  let busyReq = false;
  const loop = async () => {
    const tt = Math.min(s.tEnd, ((performance.now() - start) / 1000 / dur) * s.tEnd);
    if (!busyReq) {
      busyReq = true;
      const { poses, bends } = await phys.poses([tt]);
      busyReq = false;
      if (poses[0]) stage.applyPoses(poses[0], bends[0]);
      ui.replayT = tt;
      timelineW.draw(tt);
    }
    if (tt < s.tEnd) replayRaf = requestAnimationFrame(() => void loop());
  };
  void loop();
}

// ---------------------------------------------------------------- dock: scale
const bigLen = el("div", { class: "scale-big" });
const scaleSlider = logSlider(8, 2500, 464, (v) => (v >= 1000 ? `${v / 1000} m` : v >= 10 ? `${v / 10} cm` : `${v} mm`), [10, 100, 1000], (v) => setLength(v));
const refChips = el("div", { class: "seg-show multi" });
for (const [k, label] of [["coin", "coin"], ["hand", "hand"], ["human", "human"]] as const) {
  const b = button(t(label), () => {
    if (ui.refs.has(k)) ui.refs.delete(k);
    else ui.refs.add(k);
    refsTouched = true;
    applyRefs();
  }, "seg-item");
  b.dataset.ref = k;
  refChips.append(b);
}
let refsTouched = false;
const scaleStats = el("div", { class: "results" });
const lawCanvas = el("canvas", { class: "law", "aria-label": t("law") });
const scaleDock = el(
  "section",
  { class: "cpanel mode-col", "data-mode": "scale" },
  el("div", { class: "dock-col grow" }, el("h3", {}, t("scaleTitle")), bigLen, scaleSlider.root, el("p", { class: "hint" }, t("scaleHint")), el("div", { class: "row-split" }, el("span", { class: "muted" }, t("refs")), refChips)),
  el("div", { class: "dock-col" }, scaleStats, el("h3", {}, `${t("law")} · ${t("lawText")}`), lawCanvas, button(t("measureHere"), () => void strike(), "btn-show")),
);

function applyRefs() {
  const L = phys.meta?.lengthM ?? 0.464;
  if (!refsTouched) {
    ui.refs.clear();
    if (L < 0.12) ui.refs.add("coin");
    if (L >= 0.04 && L < 0.9) ui.refs.add("hand");
    if (L >= 0.3) ui.refs.add("human");
  }
  const show = ui.mode === "scale" ? ui.refs : new Set<"coin" | "hand" | "human">();
  stage.setReferences(show, L);
  const extra = (show.has("coin") ? 0.03 : 0) + (show.has("hand") ? 0.12 : 0) + (show.has("human") ? 0.6 : 0);
  stage.frame(extra ? extra + L * 0.2 : 0);
  stage.recenter();
  for (const b of refChips.querySelectorAll<HTMLElement>("[data-ref]")) b.classList.toggle("on", ui.refs.has(b.dataset.ref as "coin"));
}

function drawLaw() {
  const c = lawCanvas;
  const dpr = devicePixelRatio || 1;
  const W = (c.width = Math.max(1, c.clientWidth * dpr));
  const H = (c.height = Math.max(1, c.clientHeight * dpr));
  const g = c.getContext("2d")!;
  g.clearRect(0, 0, W, H);
  const muted = getComputedStyle(c).getPropertyValue("--muted").trim() || "#888";
  const lx = (L: number) => ((Math.log10(L) - 0.8) / (3.5 - 0.8)) * (W - 30 * dpr) + 24 * dpr;
  const ly = (tt: number) => H - 14 * dpr - ((Math.log10(tt) + 3.2) / 4) * (H - 24 * dpr);
  g.strokeStyle = muted;
  g.lineWidth = dpr;
  g.font = `${9 * dpr}px system-ui`;
  g.fillStyle = muted;
  for (const L of [10, 100, 1000]) {
    g.fillText(L >= 1000 ? "1 m" : L >= 100 ? "10 cm" : "1 cm", lx(L) - 10 * dpr, H - 2 * dpr);
  }
  for (const tt of [0.001, 0.01, 0.1, 1]) g.fillText(formatTime(tt), 0, ly(tt) + 3 * dpr);
  const ref = ui.measures[ui.measures.length - 1];
  if (ref && Number.isFinite(ref.t90)) {
    // t ∝ L through the latest measurement
    g.setLineDash([4 * dpr, 4 * dpr]);
    g.beginPath();
    g.moveTo(lx(8), ly((ref.t90 * 8) / ref.L));
    g.lineTo(lx(2500), ly((ref.t90 * 2500) / ref.L));
    g.stroke();
    g.setLineDash([]);
  }
  g.fillStyle = "#ff8a3d";
  for (const m of ui.measures) {
    if (!Number.isFinite(m.t90)) continue;
    g.beginPath();
    g.arc(lx(m.L), ly(m.t90), 3.5 * dpr, 0, Math.PI * 2);
    g.fill();
  }
  const cur = phys.meta ? phys.meta.lengthM * 1000 : 464;
  g.strokeStyle = "#3ea6ff";
  g.beginPath();
  g.moveTo(lx(cur), 6 * dpr);
  g.lineTo(lx(cur), H - 12 * dpr);
  g.stroke();
}

// ---------------------------------------------------------------- timeline
const timelineW = timeline({ curl: t("curl"), speed: t("tipSpeed") });
const playBtn = button("", () => setRun(!ui.running, ui.speed), "icon-btn", t("pause"));
playBtn.append(icon("pause"));
const liveBtn = button("", () => goLive(), "chip small live-chip");
liveBtn.append(icon("live", 10), el("span", {}, t("live")));
const speedSel = el("select", { "aria-label": t("timeScale") });
for (const f of [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2]) speedSel.append(el("option", { value: f, selected: f === 1 }, `${f}×`));
speedSel.addEventListener("change", () => setRun(ui.running, Number(speedSel.value)));
const timeLabel = el("span", { class: "mono muted tl-time" });
const timelineBar = el(
  "footer",
  { class: "timeline-row" },
  playBtn,
  iconOnly(button("", () => {
    phys.send({ type: "reset" });
    timelineW.clear();
    stage.clearTrail();
    peakSpeed = 0;
  }, "icon-btn", t("resetPose")), "reset"),
  speedSel,
  timelineW.root,
  timeLabel,
  liveBtn,
);
timelineW.onScrub = async (tt) => {
  stopReplay();
  setRun(false, ui.speed);
  ui.replayT = tt;
  const { poses, bends } = await phys.poses([tt]);
  if (poses[0]) stage.applyPoses(poses[0], bends[0]);
  timelineW.draw(tt);
};
function setRun(running: boolean, speed: number) {
  ui.running = running;
  ui.speed = speed;
  if (running) {
    ui.replayT = null;
    stopReplay();
  }
  playBtn.replaceChildren(icon(running ? "pause" : "play"));
  playBtn.title = running ? t("pause") : t("play");
  liveBtn.classList.toggle("on", running);
  phys.send({ type: "run", running, speed });
}
function goLive() {
  stopReplay();
  setRun(true, ui.speed);
}

// ---------------------------------------------------------------- engineering sheet
const engHost = el("div", { class: "eng-sheet", hidden: true, role: "dialog", "aria-label": t("engineering") });
type Eng = ReturnType<typeof import("./engineering/engineering").mountEngineering>;
let engPromise: Promise<Eng> | null = null;
/** The engineering tool (sketch, OpenCascade CAD, PDF, ...) loads on first use. */
function ensureEngineering(): Promise<Eng> {
  engPromise ??= import("./engineering/engineering").then(({ mountEngineering }) =>
    mountEngineering(engHost, store, {
      onSimulate: () => openEngineering(false),
      onClose: () => openEngineering(false),
      perf: () => ({ fps: stage.fps, stepUs: phys.frame?.stepUs ?? 0 }),
    }),
  );
  return engPromise;
}
async function openEngineering(open: boolean) {
  if (open && !engPromise) say(t("loading"));
  const eng = open || engPromise ? await ensureEngineering() : null;
  engHost.hidden = !open;
  document.body.classList.toggle("eng-open", open);
  eng?.setVisible(open);
  if (!open) stage.resize();
}
addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Escape" && !engHost.hidden) openEngineering(false);
});

// ---------------------------------------------------------------- layout
// ---------------------------------------------------------------- console
// one bottom console: metrics (always visible), the mode and shape controls
// (collapsible) and the timeline
const qualitySel = el("select", { "aria-label": t("quality"), title: t("quality") });
for (const [v, k] of [["auto", "auto"], ["high", "high"], ["medium", "medium"], ["low", "low"]] as const) qualitySel.append(el("option", { value: v }, t(k)));
qualitySel.value = "auto";
qualitySel.addEventListener("change", () => {
  const v = qualitySel.value;
  if (v === "auto") stage.setQuality(initialQuality(), true);
  else stage.setQuality(v as Quality, false);
  syncQuality();
});
function syncQuality() {
  app.classList.toggle("q-low", stage.quality === "low");
  app.classList.toggle("q-high", stage.quality === "high");
  const auto = qualitySel.options[0];
  auto.textContent = stage.autoQuality ? `${t("auto")} · ${t(stage.quality)}` : t("auto");
}
stage.onQuality = () => syncQuality();
syncQuality();
let collapsed = (() => {
  try {
    const v = localStorage.getItem("openspirobs.collapsed");
    if (v !== null) return v === "1";
  } catch {
    /* default below */
  }
  return matchMedia("(max-width: 700px)").matches;
})();
const collapseBtn = button("", () => setCollapsed(!collapsed), "collapse-btn");
const chapter = el("div", { class: "chapter" });
const consoleEl = el(
  "section",
  { class: "console", "aria-label": "Controls" },
  el("div", { class: "console-head" }, chapter, el("span", { class: "spacer" }), el("label", { class: "select-wrap", title: t("quality") }, el("span", { class: "select-k" }, t("quality")), qualitySel), collapseBtn),
  el("div", { class: "console-body", id: "console-body" }, flexDock, speedDock, scaleDock, shapePanel),
  timelineBar,
);
collapseBtn.setAttribute("aria-controls", "console-body");
function setCollapsed(c: boolean) {
  collapsed = c;
  consoleEl.classList.toggle("collapsed", c);
  collapseBtn.replaceChildren(icon(c ? "chevronUp" : "chevronDown"), el("span", {}, c ? t("expand") : t("collapse")));
  collapseBtn.setAttribute("aria-expanded", String(!c));
  try {
    localStorage.setItem("openspirobs.collapsed", c ? "1" : "0");
  } catch {
    /* ignore */
  }
  requestAnimationFrame(updateSafeArea);
}
addEventListener("keydown", (e: KeyboardEvent) => {
  const tag = (e.target as HTMLElement).tagName;
  if (e.key.toLowerCase() === "h" && !e.metaKey && !e.ctrlKey && tag !== "INPUT" && tag !== "SELECT" && tag !== "TEXTAREA") setCollapsed(!collapsed);
});

// cinematic frame marks around the free area, and a live scale bar
const frameEl = el("div", { class: "frame-marks", "aria-hidden": "true" }, el("i", { class: "tl" }), el("i", { class: "tr" }), el("i", { class: "bl" }), el("i", { class: "br" }));
const scaleBarLine = el("span", { class: "sb-line" });
const scaleBarText = el("span", { class: "sb-text" });
const scaleBar = el("div", { class: "scale-bar", "aria-hidden": "true" }, scaleBarLine, scaleBarText);
frameEl.append(scaleBar);
stage.onRender = (mpp: number) => {
  // a 1-2-5 length near 110 px
  const target = mpp * 110;
  const p10 = 10 ** Math.floor(Math.log10(target));
  const nice = [1, 2, 5, 10].map((k) => k * p10).reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
  scaleBarLine.style.width = `${(nice / mpp).toFixed(1)}px`;
  scaleBarText.textContent = formatLength(nice * 1000);
};
app.append(stageHost, frameEl, loading, topbar, hud, consoleEl, toast, engHost);
setCollapsed(collapsed);

function setMode(m: Mode) {
  ui.mode = m;
  app.dataset.mode = m;
  const meta = { flex: ["01", "modeFlex", "descFlex"], speed: ["02", "modeSpeed", "descSpeed"], scale: ["03", "modeScale", "descScale"] }[m] as [string, Key, Key];
  chapter.replaceChildren(el("span", { class: "chapter-num" }, meta[0]), el("span", { class: "chapter-title" }, t(meta[1])), el("span", { class: "chapter-desc" }, t(meta[2])));
  for (const b of tabs.querySelectorAll<HTMLElement>(".mode-tab")) b.setAttribute("aria-selected", String(b.dataset.mode === m));
  stopReplay();
  if (m !== "speed") {
    stage.setGhosts([]);
    if (m !== "flex") stage.clearTrail();
  } else {
    renderResults();
    showStrikeArtifacts();
  }
  if (m !== "flex") stopAuto();
  applyRefs();
  requestAnimationFrame(updateSafeArea);
  if (m === "scale") requestAnimationFrame(drawLaw);
  const h = new URLSearchParams(location.hash.slice(1));
  h.set("m", m);
  history.replaceState(null, "", `#${h.toString()}`);
}

/** Keep the robot framed in the part of the screen the panels leave free. */
function updateSafeArea() {
  const W = app.clientWidth;
  const H = app.clientHeight;
  const wide = W > 900;
  const top = topbar.getBoundingClientRect().bottom + 8;
  const bottom = Math.max(0, H - consoleEl.getBoundingClientRect().top + 8);
  const right = wide ? W - hud.getBoundingClientRect().left + 12 : 0;
  stage.setSafeArea({ left: wide ? 24 : 0, right, top, bottom });
  Object.assign(frameEl.style, { top: `${top}px`, bottom: `${bottom}px`, left: `${wide ? 24 : 10}px`, right: `${wide ? right : 10}px` });
}
new ResizeObserver(updateSafeArea).observe(app);
new ResizeObserver(updateSafeArea).observe(consoleEl);

function syncChips() {
  const p = store.params;
  const n = cableCount(p);
  for (const b of cablesSeg.querySelectorAll<HTMLElement>("[data-n]")) b.classList.toggle("on", Number(b.dataset.n) === n);
  for (const b of objChips.querySelectorAll<HTMLElement>("[data-kind]")) b.classList.toggle("on", b.dataset.kind === ui.object.kind);
  app.classList.toggle("hinge", p.two_cable);
}

function syncShape() {
  const p = store.params;
  const d = store.resolved.design;
  cDtheta.set(p.dtheta_deg);
  cTurns.set(p.theta_max_pi);
  cB.set(p.b);
  cP.set(p.p);
  gravity.checked = p.sim_gravity > 0;
  if (d) {
    lengthReadout.textContent = formatLength(d.length);
    bigLen.textContent = formatLength(d.length);
    lengthSlider.set(d.length);
    scaleSlider.set(d.length);
    unitsInfo.textContent = `${d.units} ${t("units")}`;
    equation.innerHTML = `<span class="eq-r">r</span> = ${p.a.toFixed(2)}·e<sup>${p.b.toFixed(3)}θ</sup><span class="eq-sep"></span>θ ∈ [0, ${p.theta_max_pi.toFixed(1)}π]<span class="eq-sep"></span>${d.units} ${t("units")} · ${formatLength(d.length)}`;
  }
  syncChips();
}

// ---------------------------------------------------------------- physics wiring
const pendingMesh = new Map<number, ReturnType<typeof renderLinkMesh>>();
let rebuildTimer = 0;
function rebuild(resetPose = false) {
  clearTimeout(rebuildTimer);
  rebuildTimer = window.setTimeout(() => {
    const r = store.resolved;
    if (!r.design) return;
    const seq = phys.build(r.params, ui.object, resetPose);
    // the display mesh: same body as the collision mesh, with softened edges
    pendingMesh.set(seq, renderLinkMesh(r.design, r.params, Math.max(48, DEFAULT_TUNING.segments)));
  }, 16);
}

phys.onReady = () => {
  loading.hidden = true;
  setTimeout(() => loading.remove(), 700);
  rebuild(true);
};
phys.onBuilt = (meta: WorkerMeta) => {
  const mesh = pendingMesh.get(meta.seq);
  for (const k of [...pendingMesh.keys()]) if (k <= meta.seq) pendingMesh.delete(k);
  if (mesh) stage.setRobot(meta, mesh);
  // a new model starts a new recording: a replayed or scrubbed time no longer
  // exists, so show the live pose again (the new links would sit at the origin)
  if (ui.replayT !== null) {
    stopReplay();
    ui.replayT = null;
  }
  stage.setObject(ui.object.kind, ui.object.size * meta.lengthM, ui.object.pos, meta.lengthM);
  joy.setCables(meta.tendonAngles, meta.hinge);
  hudItems.get("mass")!.textContent = formatMass(meta.totalMass);
  applyRefs();
  log.debug(`physics model: ${meta.units} links, compile ${meta.compileMs.toFixed(0)} ms, dt ${meta.timestep * 1e6 | 0} µs`);
};
phys.onError = (m) => {
  log.error(m);
  say(m);
};

let lastHud = 0;
phys.onFrame = (f: Frame) => {
  if (ui.replayT === null) {
    stage.applyFrame(f);
    if (ui.mode === "flex" || (ui.mode === "speed" && !lastStrike)) stage.pushTrail(f.tip, f.tipSpeed, Math.max(peakSpeed, 0.5));
  } else {
    stage.applyFrame({ ...f, poses: new Float32Array(0), tendons: [], grab: null });
  }
  timelineW.push(f.time, Math.abs(f.curl), f.tipSpeed);
  peakSpeed = Math.max(peakSpeed, f.tipSpeed);
  const now = performance.now();
  if (now - lastHud > 120) {
    lastHud = now;
    hudItems.get("curl")!.textContent = `${Math.abs(f.curl).toFixed(0)}°`;
    hudItems.get("speed")!.textContent = `${f.tipSpeed.toFixed(2)} m/s`;
    hudItems.get("peak")!.textContent = `${peakSpeed.toFixed(2)} m/s`;
    hudItems.get("rt")!.textContent = ui.running ? `${f.realtime.toFixed(f.realtime < 0.1 ? 3 : 2)}×` : "—";
    timeLabel.textContent = formatTime(ui.replayT ?? f.time);
    timelineW.draw(ui.replayT);
    if (ui.mode === "scale") {
      const meta = phys.meta;
      if (meta) {
        const ref = ui.measures[ui.measures.length - 1];
        const pred = ref && Number.isFinite(ref.t90) ? (ref.t90 * meta.lengthM * 1000) / ref.L : NaN;
        scaleStats.replaceChildren(
          ...[
            [t("mass"), formatMass(meta.totalMass)],
            [t("t90"), Number.isFinite(pred) ? `≈ ${formatTime(pred)}` : "—"],
            ["dt", formatTime(meta.timestep)],
          ].map(([k, v]) => el("div", { class: "result" }, el("span", { class: "result-v" }, v), el("span", { class: "result-k" }, k))),
        );
      }
    }
  }
};

// pointer: grab the robot / move the object
let drag: { kind: "link" | "object"; anchor: Vector3 } | null = null;
stageHost.addEventListener("pointerdown", (ev) => {
  if (ev.button !== 0 || ui.replayT !== null) return;
  const hit = stage.pick(ev.clientX, ev.clientY);
  if (!hit) return;
  ev.stopPropagation();
  stage.controls.enabled = false;
  stageHost.setPointerCapture(ev.pointerId);
  drag = { kind: hit.kind, anchor: hit.point };
  if (hit.kind === "link") phys.send({ type: "grab", body: hit.index, point: hit.point.toArray() as [number, number, number] });
}, { capture: true });
stageHost.addEventListener("pointermove", (ev) => {
  if (!drag) {
    stageHost.style.cursor = stage.pick(ev.clientX, ev.clientY) ? "grab" : "";
    return;
  }
  stageHost.style.cursor = "grabbing";
  // the object moves in the curl plane (y = const); the robot in the view plane
  const p = drag.kind === "object" ? stage.dragPoint(ev.clientX, ev.clientY, drag.anchor, new Vector3(0, 1, 0)) : stage.dragPoint(ev.clientX, ev.clientY, drag.anchor);
  if (!p) return;
  if (drag.kind === "link") phys.send({ type: "drag", point: p.toArray() as [number, number, number] });
  else {
    const L = phys.meta?.lengthM ?? 0.5;
    ui.object.pos = [p.x, ui.object.pos[1], Math.max(ui.object.size * L * 0.5, p.z)];
    stage.moveObject(ui.object.pos);
    phys.send({ type: "object", pos: ui.object.pos });
  }
});
const endDrag = () => {
  if (!drag) return;
  if (drag.kind === "link") phys.send({ type: "release" });
  drag = null;
  stage.controls.enabled = true;
  stageHost.style.cursor = "";
};
stageHost.addEventListener("pointerup", endDrag);
stageHost.addEventListener("pointercancel", endDrag);

// ---------------------------------------------------------------- reactions
store.subscribe(() => {
  syncShape();
  rebuild();
  stage.setSpiralB(store.params.b);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => stage.setTheme(e.matches));

async function share() {
  const url = `${shareLink(store.params)}&m=${ui.mode}&e=${env}`;
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    say(t("copied"));
  } catch {
    say(url);
  }
}

/** Space / ocean switch: shows where you would go. */
function envBtn(): HTMLButtonElement {
  const b = button("", () => {
    env = env === "space" ? "ocean" : "space";
    document.documentElement.dataset.env = env;
    stage.setEnvironment(env);
    applyRefs();
    try {
      localStorage.setItem("openspirobs.env", env);
    } catch {
      /* ignore */
    }
    sync();
  }, "chip env-chip");
  const sync = () => {
    b.replaceChildren(icon(env === "space" ? "wave" : "star"), el("span", { class: "chip-label" }, env === "space" ? t("ocean") : t("space")));
    b.title = env === "space" ? t("toOcean") : t("toSpace");
    for (const c of refChips.querySelectorAll<HTMLElement>('[data-ref="human"]')) c.textContent = env === "space" ? t("astronaut") : t("human");
  };
  queueMicrotask(sync);
  return b;
}

function iconOnly(b: HTMLButtonElement, name: Parameters<typeof icon>[0]): HTMLButtonElement {
  b.append(icon(name));
  b.setAttribute("aria-label", b.title);
  return b;
}

function iconLabel(b: HTMLButtonElement, name: Parameters<typeof icon>[0]): HTMLButtonElement {
  b.prepend(icon(name));
  return b;
}

function checkbox(label: string, value: boolean, onChange: (v: boolean) => void): HTMLElement {
  const i = el("input", { type: "checkbox" });
  i.checked = value;
  i.addEventListener("change", () => onChange(i.checked));
  return el("label", { class: "check" }, i, label);
}

// ---------------------------------------------------------------- console API
const lazy =
  <K extends keyof Eng>(k: K) =>
  async (...args: unknown[]) => {
    const e = await ensureEngineering();
    return (e[k] as (...a: unknown[]) => unknown)(...args);
  };
const api = {
  engineering: () => ensureEngineering(),
  exportXml: lazy("exportXml"),
  exportStep: lazy("exportStep"),
  exportStl: lazy("exportStl"),
  sketchZip: lazy("sketchZip"),
  reportPdf: lazy("reportPdf"),
  build: lazy("build"),
  params: () => structuredClone(store.params),
  set: (p: Record<string, unknown>) => store.set(parseParams(p, store.params)),
  design: () => store.resolved.design,
  checks: () => checkDesign(store.params, store.resolved.design),
  log: () => getLog(),
  reset: () => store.replace(defaultParams()),
  store,
  stage,
  physics: phys,
  mode: (m?: Mode) => (m ? setMode(m) : ui.mode),
  strike: () => strike(),
  lastStrike: () => lastStrike,
  setLength: (mm: number) => setLength(mm),
  setDirection: (x: number, y: number) => (joy.set(x, y), setDirection(x, y)),
  object: (kind: ObjectKind) => setObject(kind),
  ui,
};
(window as unknown as { osr: typeof api }).osr = api;

syncShape();
setMode(ui.mode);
renderResults();
installRangePainter();

/**
 * The mark: a nautilus drawn from the robot's own law r = a·e^(bθ), with the
 * chamber walls (septa) every 30° — the same spiral the robot is cut from.
 */
function nautilus(cls: string): SVGSVGElement {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "-50 -50 100 100");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  const b = 0.1764;
  const tMax = 3.2 * Math.PI;
  const a = 46 / Math.exp(b * tMax);
  const pt = (th: number, k = 1): [number, number] => {
    const r = a * Math.exp(b * th) * k;
    return [r * Math.cos(th + 0.6), -r * Math.sin(th + 0.6)];
  };
  let d = "";
  for (let th = 0; th <= tMax + 1e-9; th += 0.05) {
    const [x, y] = pt(th);
    d += `${d ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }
  // the inner whorl the shell closes onto, one turn back
  let septa = "";
  const inner = Math.exp(-2 * Math.PI * b);
  for (let th = 2 * Math.PI; th <= tMax; th += Math.PI / 6) {
    const [x1, y1] = pt(th);
    const [x2, y2] = pt(th, inner);
    septa += `M${x1.toFixed(2)},${y1.toFixed(2)}Q${((x1 + x2) / 2 * 0.92).toFixed(2)},${((y1 + y2) / 2 * 0.92).toFixed(2)} ${x2.toFixed(2)},${y2.toFixed(2)}`;
  }
  const shell = document.createElementNS(NS, "path");
  shell.setAttribute("d", d);
  shell.setAttribute("class", "nautilus-shell");
  const walls = document.createElementNS(NS, "path");
  walls.setAttribute("d", septa);
  walls.setAttribute("class", "nautilus-septa");
  svg.append(walls, shell);
  return svg;
}
