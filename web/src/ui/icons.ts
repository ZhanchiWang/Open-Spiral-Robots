// 16×16 line icons (1.5 px strokes, round joins), drawn for this app.

const PATHS: Record<string, string> = {
  play: '<path d="M5 3.5v9l7-4.5z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M5.5 3.5v9M10.5 3.5v9"/>',
  reset: '<path d="M3.5 8a4.5 4.5 0 1 0 1.4-3.3"/><path d="M3.5 2.8v2.6h2.6"/>',
  chevronUp: '<path d="M4 10l4-4 4 4"/>',
  chevronDown: '<path d="M4 6l4 4 4-4"/>',
  share: '<path d="M6 9.5 10 12M10 4 6 6.5"/><circle cx="4.5" cy="8" r="1.8"/><circle cx="11.5" cy="3.5" r="1.8"/><circle cx="11.5" cy="12.5" r="1.8"/>',
  sliders: '<path d="M3 4.5h6M12 4.5h1M3 11.5h1M7 11.5h6"/><circle cx="10.5" cy="4.5" r="1.5"/><circle cx="5.5" cy="11.5" r="1.5"/>',
  star: '<path d="M8 1.8l1.2 4.9 4.9 1.3-4.9 1.2L8 14.2 6.8 9.2 1.9 8l4.9-1.3z" fill="currentColor" stroke="none"/>',
  wave: '<path d="M1.5 6c1.6-1.6 3.2-1.6 4.8 0s3.2 1.6 4.8 0 2.4-1.2 3.4-.6M1.5 10.5c1.6-1.6 3.2-1.6 4.8 0s3.2 1.6 4.8 0 2.4-1.2 3.4-.6"/>',
  globe: '<circle cx="8" cy="8" r="6"/><path d="M2 8h12M8 2c2 2.2 2 9.8 0 12M8 2c-2 2.2-2 9.8 0 12"/>',
  bolt: '<path d="M9 1.5 3.5 9H8l-1 5.5L12.5 7H8z" fill="currentColor" stroke="none"/>',
  replay: '<path d="M2.5 8a5.5 5.5 0 1 0 1.7-4"/><path d="M2.6 2.5v3h3"/><path d="M7 6v4l3-2z" fill="currentColor" stroke="none"/>',
  target: '<circle cx="8" cy="8" r="5.5"/><circle cx="8" cy="8" r="1.5" fill="currentColor"/>',
  close: '<path d="M4 4l8 8M12 4l-8 8"/>',
  live: '<circle cx="8" cy="8" r="3" fill="currentColor" stroke="none"/>',
};

export function icon(name: keyof typeof PATHS, size = 16): SVGSVGElement {
  const wrap = document.createElement("span");
  wrap.innerHTML = `<svg class="ico" width="${size}" height="${size}" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name]}</svg>`;
  return wrap.firstElementChild as SVGSVGElement;
}

/** Keep a range input's filled track in sync (CSS reads --p). */
export function paintRange(input: HTMLInputElement): void {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const v = Number(input.value);
  const p = max > min ? ((v - min) / (max - min)) * 100 : 0;
  input.style.setProperty("--p", `${Math.max(0, Math.min(100, p))}%`);
}

/** Paint every range input now and whenever one changes. */
export function installRangePainter(root: ParentNode = document): void {
  root.querySelectorAll<HTMLInputElement>('input[type="range"]').forEach(paintRange);
  document.addEventListener("input", (e) => {
    const t = e.target as HTMLInputElement;
    if (t?.type === "range") paintRange(t);
  });
}
