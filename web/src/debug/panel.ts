// Debug drawer: log, design-rule checks, live state, performance counters
// and the window.osr console API reference.

import { checkDesign } from "../core/design";
import { defaultParams, serializeParams } from "../core/params";
import type { Resolved } from "../core/state";
import { button, el } from "../ui/controls";
import { clearLog, formatEntry, getLog, onLog, type Level, type LogEntry } from "./log";

export interface PerfInfo {
  cad: Record<string, number> | null;
  cadLogs: string[];
  triangles: number;
  edges: number;
  fps3d: number;
  fpsSim: number;
  simStepUs: number;
}

export interface DebugPanel {
  root: HTMLElement;
  updateState(res: Resolved): void;
  updatePerf(p: PerfInfo): void;
  open(tab?: string): void;
}

export function buildDebugPanel(): DebugPanel {
  const tabs = ["Log", "Checks", "State", "Perf", "Console"] as const;
  type Tab = (typeof tabs)[number];
  const bodies = new Map<Tab, HTMLElement>();
  const tabBtns = new Map<Tab, HTMLButtonElement>();
  const counts = el("span", { class: "badge", hidden: true });
  const checksBadge = el("span", { class: "badge warn", hidden: true });

  // --- Log ---
  const levelSel = el("select", { "aria-label": "Minimum level" });
  for (const l of ["debug", "info", "warn", "error"]) levelSel.append(el("option", { value: l, selected: l === "info" }, l));
  const filter = el("input", { type: "search", placeholder: "filter…", "aria-label": "Filter log" });
  const pre = el("pre", { class: "log" });
  const rank: Record<Level, number> = { debug: 0, info: 1, warn: 2, error: 3 };
  let errors = 0;
  const lineFor = (e: LogEntry) => el("div", { class: `log-${e.level}` }, formatEntry(e));
  const visible = (e: LogEntry) => rank[e.level] >= rank[levelSel.value as Level] && (!filter.value || formatEntry(e).toLowerCase().includes(filter.value.toLowerCase()));
  const renderLog = () => {
    pre.replaceChildren(...getLog().filter(visible).map(lineFor));
    pre.scrollTop = pre.scrollHeight;
  };
  levelSel.addEventListener("change", renderLog);
  filter.addEventListener("input", renderLog);
  onLog((e) => {
    if (e.level === "error") {
      errors++;
      counts.textContent = String(errors);
      counts.hidden = false;
    }
    if (!visible(e)) return;
    const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
    pre.append(lineFor(e));
    if (pre.childElementCount > 1000) pre.firstElementChild?.remove();
    if (stick) pre.scrollTop = pre.scrollHeight;
  });
  const copyLog = button("Copy", () => navigator.clipboard?.writeText(getLog().map(formatEntry).join("\n")), "btn small");
  const clear = button("Clear", () => {
    clearLog();
    errors = 0;
    counts.hidden = true;
    renderLog();
  }, "btn small");
  bodies.set("Log", el("div", { class: "dbg-body" }, el("div", { class: "toolbar" }, levelSel, filter, copyLog, clear), pre));

  // --- Checks ---
  const checks = el("ul", { class: "checks" });
  bodies.set("Checks", el("div", { class: "dbg-body" }, checks));

  // --- State ---
  const statePre = el("pre", { class: "state" });
  const diffPre = el("pre", { class: "state" });
  let stateJson = "";
  bodies.set(
    "State",
    el(
      "div",
      { class: "dbg-body two-col" },
      el("div", {}, el("div", { class: "toolbar" }, el("strong", {}, "Resolved parameters + derived"), button("Copy", () => navigator.clipboard?.writeText(stateJson), "btn small")), statePre),
      el("div", {}, el("div", { class: "toolbar" }, el("strong", {}, "Changed from defaults")), diffPre),
    ),
  );

  // --- Perf ---
  const perf = el("div", { class: "readout" });
  const cadLog = el("pre", { class: "log small" });
  bodies.set("Perf", el("div", { class: "dbg-body two-col" }, perf, el("div", {}, el("strong", {}, "Last CAD worker log"), cadLog)));

  // --- Console ---
  bodies.set(
    "Console",
    el(
      "div",
      { class: "dbg-body" },
      el("p", {}, "Open the browser DevTools console; the app is exposed as ", el("code", {}, "window.osr"), ":"),
      el(
        "pre",
        { class: "state" },
        [
          "osr.params()                 // current (resolved) parameters",
          "osr.set({ a: 5, b: 0.2 })    // change parameters (same as the panel)",
          "osr.design()                 // 2D geometry: polygons, sizes, rays …",
          "osr.checks()                 // design-rule warnings",
          "await osr.build()            // run the CAD worker, returns timings",
          "await osr.exportXml()        // { xml, stl } exactly as downloaded",
          "await osr.exportStep()       // STEP bytes (Uint8Array)",
          "await osr.exportStl()        // binary STL bytes",
          "await osr.sketchZip()        // the Save 2D Sketch zip",
          "await osr.reportPdf()        // the one-page PDF report (bytes)",
          "await osr.simulate()         // load current design into MuJoCo",
          "osr.sim.setCtrl(0, 0.8)      // drive actuator 0",
          "osr.sim.readout()            // time, tendon lengths, joint angles …",
          "osr.log()                    // the debug log entries",
          "osr.reset()                  // default parameters",
        ].join("\n"),
      ),
      el("p", {}, "Append ", el("code", {}, "?debug"), " to the URL to open this drawer and log at debug level on start."),
    ),
  );

  const header = el("div", { class: "dbg-tabs", role: "tablist" });
  for (const t of tabs) {
    const b = button(t, () => select(t), "tab");
    b.setAttribute("role", "tab");
    if (t === "Log") b.append(counts);
    if (t === "Checks") b.append(checksBadge);
    tabBtns.set(t, b);
    header.append(b);
  }
  const content = el("div", { class: "dbg-content" });
  const select = (t: Tab) => {
    tabBtns.forEach((b, k) => b.setAttribute("aria-selected", String(k === t)));
    content.replaceChildren(bodies.get(t)!);
    if (t === "Log") renderLog();
  };
  select("Log");

  const details = el("details", { class: "debug-drawer" }, el("summary", {}, "Debug"), header, content);
  if (new URLSearchParams(location.search).has("debug")) {
    details.open = true;
    levelSel.value = "debug";
  }

  return {
    root: details,
    updateState(res) {
      const d = res.design;
      const derived = d
        ? {
            units: d.units,
            joints: d.joints,
            taper_deg: +d.taperDeg.toFixed(4),
            tip_size_mm: +d.tipSize.toFixed(4),
            base_size_mm: +d.baseSize.toFixed(4),
            length_mm: +d.length.toFixed(4),
            extrusion_range_mm: d.extrusionRange.map((v) => +v.toFixed(4)),
            cone1_max_deg: +res.cone1Max.toFixed(4),
            cone2_max_deg: +d.cone2Max.toFixed(4),
            vtip: d.ray.start.map((v) => +v.toFixed(4)),
            elastic_quad: d.elastic?.map((p) => p.map((v) => +v.toFixed(4))) ?? null,
          }
        : null;
      stateJson = JSON.stringify({ params: JSON.parse(serializeParams(res.params)), derived }, null, 2);
      statePre.textContent = stateJson;
      const def = defaultParams() as unknown as Record<string, unknown>;
      const cur = res.params as unknown as Record<string, unknown>;
      const diff = Object.keys(cur)
        .filter((k) => !Array.isArray(cur[k]) && k !== "tendon_configs" && JSON.stringify(cur[k]) !== JSON.stringify(def[k]))
        .map((k) => `${k}: ${JSON.stringify(def[k])} → ${JSON.stringify(cur[k])}`);
      diffPre.textContent = diff.join("\n") || "(none)";
      const issues = checkDesign(res.params, d);
      checks.replaceChildren(
        ...(issues.length ? issues.map((i) => el("li", { class: `check-${i.level}` }, i.message)) : [el("li", { class: "check-ok" }, "No design-rule issues.")]),
      );
      checksBadge.textContent = String(issues.length);
      checksBadge.hidden = issues.length === 0;
    },
    updatePerf(p) {
      const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
      const rows: [string, string][] = [
        ["CAD total", p.cad ? `${p.cad.total?.toFixed(0)} ms` : "—"],
        ...Object.entries(p.cad ?? {})
          .filter(([k]) => k !== "total")
          .map(([k, v]) => [`  ${k}`, `${v.toFixed(0)} ms`] as [string, string]),
        ["triangles", p.triangles.toLocaleString()],
        ["B-rep edge segments", p.edges.toLocaleString()],
        ["3D view fps", p.fps3d.toFixed(0)],
        ["sim view fps", p.fpsSim.toFixed(0)],
        ["mj_step", p.simStepUs ? `${p.simStepUs.toFixed(1)} µs` : "—"],
        ["JS heap", mem ? `${(mem.usedJSHeapSize / 1048576).toFixed(0)} MB` : "n/a"],
      ];
      perf.replaceChildren(...rows.flatMap(([k, v]) => [el("span", { class: "k" }, k), el("span", { class: "v mono" }, v)]));
      cadLog.textContent = p.cadLogs.join("\n") || "(empty)";
    },
    open(tab) {
      details.open = true;
      if (tab && (tabs as readonly string[]).includes(tab)) select(tab as Tab);
    },
  };
}

