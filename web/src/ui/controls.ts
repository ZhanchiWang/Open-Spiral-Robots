// Tiny DOM helpers for the parameter panel: label + slider + number box rows
// (the desktop tool's _add_double_control), toggles and selects.

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number | boolean | undefined> = {},
  ...children: (Node | string | null | undefined | false)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    if (k === "class") node.className = String(v);
    else if (k === "text") node.textContent = String(v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(c));
  }
  return node;
}

export interface NumberControl {
  row: HTMLElement;
  set(value: number): void;
  setRange(min: number, max: number): void;
  setEnabled(on: boolean): void;
  readonly value: number;
}

export interface NumberControlOptions {
  label: string;
  unit?: string;
  min: number;
  max: number;
  step: number;
  decimals: number;
  value: number;
  title?: string;
  /** called on every slider move (live) and on committed number edits */
  onInput: (v: number) => void;
}

const round = (v: number, d: number) => Number(v.toFixed(d));

export function numberControl(o: NumberControlOptions): NumberControl {
  let min = o.min;
  let max = o.max;
  let value = o.value;
  const id = `c${Math.random().toString(36).slice(2, 9)}`;
  const label = el("label", { for: id, class: "ctl-label", title: o.title }, o.label, o.unit ? el("span", { class: "unit" }, ` ${o.unit}`) : null);
  const slider = el("input", { type: "range", class: "ctl-slider", "aria-label": o.label });
  const box = el("input", { type: "number", id, class: "ctl-number", inputmode: "decimal" });
  const row = el("div", { class: "ctl-row" }, label, slider, box);

  const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(o.decimals) : "");
  const paint = () => {
    const p = max > min ? ((Math.min(max, Math.max(min, value)) - min) / (max - min)) * 100 : 0;
    slider.style.setProperty("--p", `${p}%`);
  };
  const sync = () => {
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(o.step);
    slider.value = String(value);
    box.min = String(round(min, o.decimals));
    box.max = String(round(max, o.decimals));
    box.step = String(o.step);
    if (document.activeElement !== box) box.value = fmt(value);
    paint();
  };
  sync();

  slider.addEventListener("input", () => {
    value = Number(slider.value);
    box.value = fmt(value);
    paint();
    o.onInput(value);
  });
  const commit = () => {
    const v = Number(box.value);
    if (box.value.trim() === "" || !Number.isFinite(v)) {
      box.value = fmt(value);
      return;
    }
    value = Math.max(min, Math.min(max, v));
    if (o.decimals === 0) value = Math.round(value);
    box.value = fmt(value);
    slider.value = String(value);
    o.onInput(value);
  };
  box.addEventListener("change", commit);
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      commit();
      box.select();
    }
  });
  box.addEventListener("blur", () => (box.value = fmt(value)));

  return {
    row,
    get value() {
      return value;
    },
    set(v: number) {
      value = v;
      sync();
    },
    setRange(a: number, b: number) {
      min = a;
      max = Math.max(a, b);
      sync();
    },
    setEnabled(on: boolean) {
      slider.disabled = !on;
      box.disabled = !on;
      row.classList.toggle("disabled", !on);
    },
  };
}

export interface ToggleControl {
  row: HTMLElement;
  set(v: boolean): void;
  readonly value: boolean;
}

export function toggleControl(label: string, value: boolean, onChange: (v: boolean) => void, title?: string): ToggleControl {
  const input = el("input", { type: "checkbox", role: "switch" });
  input.checked = value;
  const row = el("label", { class: "toggle-row", title }, input, el("span", { class: "switch", "aria-hidden": "true" }), el("span", {}, label));
  input.addEventListener("change", () => onChange(input.checked));
  return {
    row,
    get value() {
      return input.checked;
    },
    set(v: boolean) {
      input.checked = v;
    },
  };
}

export function section(title: string, ...children: (Node | null | false)[]): HTMLElement {
  return el("section", { class: "panel-section" }, el("h3", {}, title), ...children);
}

export function button(text: string, onClick: () => void, cls = "btn", title?: string): HTMLButtonElement {
  const b = el("button", { type: "button", class: cls, title }, text);
  b.addEventListener("click", onClick);
  return b;
}
