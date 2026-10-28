// Main-thread handle on the physics worker.

import type { Params } from "../core/params";
import type { SceneObject } from "./physics";
import type { Frame, FromWorker, ToWorker, WorkerMeta } from "./protocol";

export class PhysicsClient {
  private worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  private nextId = 1;
  private waiting = new Map<number, (m: FromWorker) => void>();
  meta: WorkerMeta | null = null;
  frame: Frame | null = null;
  ready = false;
  onFrame: ((f: Frame) => void) | null = null;
  onBuilt: ((m: WorkerMeta) => void) | null = null;
  onError: ((msg: string) => void) | null = null;
  onReady: (() => void) | null = null;

  constructor() {
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => {
      const m = ev.data;
      switch (m.type) {
        case "ready":
          this.ready = true;
          this.onReady?.();
          break;
        case "built":
          this.meta = m.meta;
          this.onBuilt?.(m.meta);
          break;
        case "frame":
          this.frame = m;
          this.onFrame?.(m);
          break;
        case "error":
          this.onError?.(m.message);
          break;
        default: {
          const f = this.waiting.get(m.id);
          this.waiting.delete(m.id);
          f?.(m);
        }
      }
    };
    this.worker.onerror = (e) => this.onError?.(`physics worker: ${e.message}`);
  }

  send(m: ToWorker): void {
    this.worker.postMessage(m);
  }

  private seq = 0;
  /** Returns the sequence number echoed back in WorkerMeta.seq. */
  build(params: Params, object: SceneObject, resetPose = false): number {
    const seq = ++this.seq;
    this.send({ type: "build", seq, params: structuredClone(params), object, resetPose });
    return seq;
  }

  poses(times: number[]): Promise<{ poses: Float32Array[]; bends: Float32Array[] }> {
    const id = this.nextId++;
    return new Promise((res) => {
      this.waiting.set(id, (m) => res(m as { poses: Float32Array[]; bends: Float32Array[] }));
      this.send({ type: "poses", id, times });
    });
  }

  series(tFrom: number, tTo: number, n: number): Promise<{ t: Float64Array; curl: Float64Array; speed: Float64Array; tip: Float64Array }> {
    const id = this.nextId++;
    return new Promise((res) => {
      this.waiting.set(id, (m) => res(m as { t: Float64Array; curl: Float64Array; speed: Float64Array; tip: Float64Array }));
      this.send({ type: "series", id, tFrom, tTo, n });
    });
  }
}
