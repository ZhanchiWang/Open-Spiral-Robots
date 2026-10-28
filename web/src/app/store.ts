// Single source of truth for the parameters. Every change re-resolves the
// geometry-dependent ranges and notifies listeners.

import { defaultParams, parseParams, serializeParams, type Params } from "../core/params";
import { resolve, type Resolved } from "../core/state";
import { logger } from "../debug/log";

const STORAGE_KEY = "openspirobs.params.v1";
const log = logger("params");

export class Store {
  private res: Resolved;
  private listeners = new Set<(r: Resolved, changed: Set<string>) => void>();
  private saveTimer = 0;

  constructor(initial: Params) {
    this.res = resolve(initial);
  }

  get resolved(): Resolved {
    return this.res;
  }

  get params(): Params {
    return this.res.params;
  }

  subscribe(f: (r: Resolved, changed: Set<string>) => void): () => void {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }

  /** Apply a partial update (or a mutator) and re-resolve. */
  set(update: Partial<Params> | ((p: Params) => void)): void {
    const next = structuredClone(this.res.params);
    if (typeof update === "function") update(next);
    else Object.assign(next, update);
    const before = this.res.params;
    this.res = resolve(next);
    const changed = new Set<string>();
    for (const k of Object.keys(this.res.params) as (keyof Params)[]) {
      if (JSON.stringify(before[k]) !== JSON.stringify(this.res.params[k])) changed.add(k);
    }
    if (!changed.size) return;
    this.listeners.forEach((f) => f(this.res, changed));
    clearTimeout(this.saveTimer);
    this.saveTimer = window.setTimeout(() => this.persist(), 300);
  }

  replace(p: Params): void {
    this.res = resolve(p);
    const all = new Set(Object.keys(p));
    this.listeners.forEach((f) => f(this.res, all));
    this.persist();
  }

  persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, serializeParams(this.res.params));
    } catch {
      /* storage may be unavailable (private mode); the app works without it */
    }
  }

  static loadInitial(): { params: Params; source: string } {
    const hash = new URLSearchParams(location.hash.slice(1)).get("p");
    if (hash) {
      try {
        const json = new TextDecoder().decode(Uint8Array.from(atob(hash.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0)));
        return { params: parseParams(JSON.parse(json)), source: "share link" };
      } catch (e) {
        log.warn(`ignoring malformed share link: ${(e as Error).message}`);
      }
    }
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return { params: parseParams(JSON.parse(saved)), source: "saved session" };
    } catch {
      /* fall back to defaults */
    }
    return { params: defaultParams(), source: "defaults" };
  }
}

export function shareLink(p: Params): string {
  const bytes = new TextEncoder().encode(JSON.stringify(JSON.parse(serializeParams(p))));
  let bin = "";
  bytes.forEach((b) => (bin += String.fromCharCode(b)));
  const b64 = btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  return `${location.origin}${location.pathname}#p=${b64}`;
}
