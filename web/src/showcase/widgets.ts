// Showcase widgets: tendon joystick, log-scale slider, rolling timeline chart.

import { el } from "../ui/controls";
import { paintRange } from "../ui/icons";
import { TENDON_HEX } from "./stage";

const hex = (c: number) => `#${c.toString(16).padStart(6, "0")}`;

export interface Joystick {
  root: HTMLElement;
  set(x: number, y: number): void;
  setCables(angles: number[], hinge: boolean): void;
  readonly value: [number, number];
}

/** Cross-section pad: tendon markers around the rim, knob = pull direction. */
export function joystick(onChange: (x: number, y: number) => void): Joystick {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "-60 -60 120 120");
  svg.setAttribute("class", "joy");
  svg.setAttribute("role", "slider");
  svg.setAttribute("tabindex", "0");
  svg.setAttribute("aria-label", "Tendon direction");
  const mk = (tag: string, a: Record<string, string | number>, parent: Element = svg) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v));
    parent.appendChild(e);
    return e;
  };
  mk("circle", { r: 50, class: "joy-disc" });
  mk("circle", { r: 25, class: "joy-ring" });
  const track = mk("line", { x1: 0, y1: -50, x2: 0, y2: 50, class: "joy-track" });
  const markers = mk("g", {});
  const stem = mk("line", { x1: 0, y1: 0, x2: 0, y2: 0, class: "joy-stem" });
  const knob = mk("circle", { r: 9, class: "joy-knob" });
  let value: [number, number] = [0, 0];
  let hinge = true;

  const draw = () => {
    const [x, y] = value;
    knob.setAttribute("cx", String(x * 50));
    knob.setAttribute("cy", String(-y * 50));
    stem.setAttribute("x2", String(x * 50));
    stem.setAttribute("y2", String(-y * 50));
    svg.setAttribute("aria-valuetext", `${Math.round(Math.hypot(x, y) * 100)}%`);
  };
  const set = (x: number, y: number, emit: boolean) => {
    if (hinge) x = 0;
    const m = Math.hypot(x, y);
    if (m > 1) {
      x /= m;
      y /= m;
    }
    value = [x, y];
    draw();
    if (emit) onChange(x, y);
  };
  const fromEvent = (ev: PointerEvent) => {
    const r = svg.getBoundingClientRect();
    const x = ((ev.clientX - r.left) / r.width) * 2 - 1;
    const y = -(((ev.clientY - r.top) / r.height) * 2 - 1);
    set(x * 1.2, y * 1.2, true);
  };
  svg.addEventListener("pointerdown", (ev) => {
    svg.setPointerCapture(ev.pointerId);
    fromEvent(ev);
  });
  svg.addEventListener("pointermove", (ev) => {
    if (svg.hasPointerCapture(ev.pointerId)) fromEvent(ev);
  });
  svg.addEventListener("dblclick", () => set(0, 0, true));
  svg.addEventListener("keydown", (ev) => {
    const d = 0.1;
    const [x, y] = value;
    if (ev.key === "ArrowUp") set(x, y + d, true);
    else if (ev.key === "ArrowDown") set(x, y - d, true);
    else if (ev.key === "ArrowLeft") set(x - d, y, true);
    else if (ev.key === "ArrowRight") set(x + d, y, true);
    else if (ev.key === "Home" || ev.key === "0") set(0, 0, true);
    else return;
    ev.preventDefault();
  });
  draw();
  return {
    root: svg as unknown as HTMLElement,
    get value() {
      return value;
    },
    set: (x, y) => set(x, y, false),
    setCables(angles, h) {
      hinge = h;
      track.setAttribute("visibility", h ? "visible" : "hidden");
      markers.replaceChildren();
      angles.forEach((deg, i) => {
        const a = (deg * Math.PI) / 180;
        mk("circle", { cx: 50 * Math.cos(a), cy: -50 * Math.sin(a), r: 5.5, fill: hex(TENDON_HEX[i % TENDON_HEX.length]), class: "joy-cable" }, markers);
      });
      set(value[0], value[1], false);
    },
  };
}

export interface LogSlider {
  root: HTMLElement;
  set(v: number): void;
}

/** Log-scale slider with decade ticks. */
export function logSlider(min: number, max: number, value: number, format: (v: number) => string, ticks: number[], onInput: (v: number) => void): LogSlider {
  const toPos = (v: number) => (Math.log(v / min) / Math.log(max / min)) * 1000;
  const fromPos = (p: number) => min * (max / min) ** (p / 1000);
  const input = el("input", { type: "range", min: 0, max: 1000, step: 1, class: "log-range", "aria-label": "Length" });
  input.value = String(toPos(value));
  paintRange(input);
  const tickRow = el("div", { class: "log-ticks" });
  for (const t of ticks) {
    const s = el("span", { style: `left:${toPos(t) / 10}%` }, format(t));
    tickRow.append(s);
  }
  input.addEventListener("input", () => onInput(fromPos(Number(input.value))));
  return {
    root: el("div", { class: "log-slider" }, input, tickRow),
    set(v) {
      if (document.activeElement !== input) input.value = String(toPos(v));
      paintRange(input);
    },
  };
}

export interface Timeline {
  root: HTMLElement;
  push(t: number, curl: number, speed: number): void;
  clear(): void;
  draw(cursor?: number | null): void;
  onScrub: ((t: number) => void) | null;
  readonly span: [number, number];
}

/** Rolling chart of curl (deg) and tip speed (m/s) with a scrub cursor. */
export function timeline(labels: { curl: string; speed: string }): Timeline {
  const canvas = el("canvas", { class: "tl-canvas", "aria-label": "Timeline" });
  const N = 900;
  const ts: number[] = [];
  const cs: number[] = [];
  const vs: number[] = [];
  let cursor: number | null = null;
  const api: Timeline = {
    root: canvas,
    onScrub: null,
    get span(): [number, number] {
      return ts.length ? [ts[0], ts[ts.length - 1]] : [0, 0];
    },
    push(t, c, v) {
      if (ts.length && t < ts[ts.length - 1]) api.clear();
      ts.push(t);
      cs.push(c);
      vs.push(v);
      if (ts.length > N) {
        ts.shift();
        cs.shift();
        vs.shift();
      }
    },
    clear() {
      ts.length = cs.length = vs.length = 0;
    },
    draw(cur) {
      if (cur !== undefined) cursor = cur;
      const dpr = devicePixelRatio || 1;
      const W = (canvas.width = Math.max(1, canvas.clientWidth * dpr));
      const H = (canvas.height = Math.max(1, canvas.clientHeight * dpr));
      const g = canvas.getContext("2d")!;
      g.clearRect(0, 0, W, H);
      const style = getComputedStyle(canvas);
      const muted = style.getPropertyValue("--muted").trim() || "#888";
      g.font = `${10 * dpr}px system-ui`;
      if (ts.length < 2) return;
      const t0 = ts[0];
      const t1 = ts[ts.length - 1];
      const x = (t: number) => ((t - t0) / Math.max(1e-9, t1 - t0)) * (W - 8 * dpr) + 4 * dpr;
      // floors keep numerical noise at rest from filling the chart
      const series = (vals: number[], color: string, fill: boolean, floor: number) => {
        let hi = floor;
        for (const v of vals) hi = Math.max(hi, Math.abs(v));
        const y = (v: number) => H - 4 * dpr - (Math.abs(v) / hi) * (H - 18 * dpr);
        g.beginPath();
        vals.forEach((v, i) => (i ? g.lineTo(x(ts[i]), y(v)) : g.moveTo(x(ts[i]), y(v))));
        g.strokeStyle = color;
        g.lineWidth = 1.6 * dpr;
        g.stroke();
        if (fill) {
          g.lineTo(x(ts[ts.length - 1]), H);
          g.lineTo(x(ts[0]), H);
          g.closePath();
          const grad = g.createLinearGradient(0, 0, 0, H);
          grad.addColorStop(0, color + "55");
          grad.addColorStop(1, color + "00");
          g.fillStyle = grad;
          g.fill();
        }
        return hi;
      };
      const cMax = series(cs, "#ff8a3d", true, 30);
      const vMax = series(vs, "#3ea6ff", false, 0.25);
      g.fillStyle = "#ff8a3d";
      g.fillText(`${labels.curl} ${cMax.toFixed(0)}°`, 6 * dpr, 12 * dpr);
      g.fillStyle = "#3ea6ff";
      g.fillText(`${labels.speed} ${vMax.toFixed(2)} m/s`, 110 * dpr, 12 * dpr);
      g.fillStyle = muted;
      const span = t1 - t0;
      const lab = span >= 1 ? `${span.toFixed(1)} s` : `${(span * 1000).toFixed(0)} ms`;
      g.fillText(lab, W - g.measureText(lab).width - 6 * dpr, 12 * dpr);
      if (cursor !== null && cursor >= t0 && cursor <= t1) {
        g.strokeStyle = muted;
        g.lineWidth = 1 * dpr;
        g.beginPath();
        g.moveTo(x(cursor), 0);
        g.lineTo(x(cursor), H);
        g.stroke();
      }
    },
  };
  const scrubAt = (ev: PointerEvent) => {
    if (ts.length < 2) return;
    const r = canvas.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    api.onScrub?.(ts[0] + f * (ts[ts.length - 1] - ts[0]));
  };
  canvas.addEventListener("pointerdown", (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    scrubAt(ev);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (canvas.hasPointerCapture(ev.pointerId)) scrubAt(ev);
  });
  return api;
}
