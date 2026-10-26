// App-wide log used by the debug panel. Also mirrors to the browser console
// so DevTools filtering keeps working.

export type Level = "debug" | "info" | "warn" | "error";

export interface LogEntry {
  t: number;
  level: Level;
  source: string;
  msg: string;
}

const MAX = 1000;
const entries: LogEntry[] = [];
const listeners = new Set<(e: LogEntry) => void>();
const t0 = performance.now();

export function log(level: Level, source: string, msg: string): void {
  const e = { t: performance.now() - t0, level, source, msg };
  entries.push(e);
  if (entries.length > MAX) entries.splice(0, entries.length - MAX);
  const c = level === "debug" ? console.debug : level === "info" ? console.info : level === "warn" ? console.warn : console.error;
  c(`[osr:${source}] ${msg}`);
  listeners.forEach((f) => f(e));
}

export const logger = (source: string) => ({
  debug: (m: string) => log("debug", source, m),
  info: (m: string) => log("info", source, m),
  warn: (m: string) => log("warn", source, m),
  error: (m: string) => log("error", source, m),
});

export function getLog(): readonly LogEntry[] {
  return entries;
}

export function clearLog(): void {
  entries.length = 0;
}

export function onLog(f: (e: LogEntry) => void): () => void {
  listeners.add(f);
  return () => listeners.delete(f);
}

export function formatEntry(e: LogEntry): string {
  return `${(e.t / 1000).toFixed(3).padStart(9)}s ${e.level.toUpperCase().padEnd(5)} ${e.source.padEnd(6)} ${e.msg}`;
}

window.addEventListener("error", (ev) => log("error", "window", `${ev.message} (${ev.filename}:${ev.lineno})`));
window.addEventListener("unhandledrejection", (ev) => {
  const r = ev.reason as { superseded?: boolean; message?: string } | undefined;
  if (r?.superseded) return;
  log("error", "promise", r?.message ?? String(ev.reason));
});
