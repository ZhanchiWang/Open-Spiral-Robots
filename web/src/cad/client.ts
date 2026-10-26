// Main-thread side of the CAD worker. Builds are coalesced: while one runs,
// only the newest pending build request is kept (OCCT cannot be interrupted).

import type { Params } from "../core/params";
import type { CadRequest, CadRequestType, CadResponse } from "./protocol";

type Pending = { resolve: (r: CadResponse) => void; reject: (e: Error) => void };

export class CadClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private queuedBuild: { req: CadRequest; p: Pending } | null = null;
  private buildRunning = false;
  onResponse: ((r: CadResponse) => void) | null = null;
  onBusy: ((busy: boolean, what: string) => void) | null = null;

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<CadResponse>) => {
      const r = ev.data;
      this.onResponse?.(r);
      const p = this.pending.get(r.id);
      this.pending.delete(r.id);
      if (r.type === "build") {
        this.buildRunning = false;
        this.flushBuild();
      }
      if (this.pending.size === 0 && !this.buildRunning) this.onBusy?.(false, "");
      if (!p) return;
      if (r.ok) p.resolve(r);
      else p.reject(Object.assign(new Error(r.error ?? "CAD error"), { response: r }));
    };
    this.worker.onerror = (ev) => {
      const err = new Error(`CAD worker crashed: ${ev.message}`);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
      this.buildRunning = false;
      this.onBusy?.(false, "");
    };
  }

  private post(req: CadRequest, p: Pending) {
    this.pending.set(req.id, p);
    this.onBusy?.(true, req.type);
    this.worker.postMessage(req);
  }

  private flushBuild() {
    if (this.buildRunning || !this.queuedBuild) return;
    const { req, p } = this.queuedBuild;
    this.queuedBuild = null;
    this.buildRunning = true;
    this.post(req, p);
  }

  request(type: CadRequestType, params: Params, options?: CadRequest["options"]): Promise<CadResponse> {
    const req: CadRequest = { id: this.nextId++, type, params: structuredClone(params), options };
    return new Promise((resolve, reject) => {
      const p = { resolve, reject };
      if (type === "build") {
        this.queuedBuild?.p.reject(Object.assign(new Error("superseded"), { superseded: true }));
        this.queuedBuild = { req, p };
        this.flushBuild();
      } else {
        this.post(req, p);
      }
    });
  }
}
