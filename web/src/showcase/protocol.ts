import type { Params } from "../core/params";
import type { SceneObject } from "./physics";

export interface WorkerMeta {
  seq: number;
  units: number;
  hinge: boolean;
  nu: number;
  lengthM: number;
  mount: [number, number, number];
  totalMass: number;
  timestep: number;
  compileMs: number;
  tendonAngles: number[];
  /** per-link uniform scale of the shared link mesh (base = 1) */
  linkScale: number[];
  jointLimitDeg: number;
}

export type ToWorker =
  | { type: "build"; seq: number; params: Params; object: SceneObject; resetPose?: boolean }
  | { type: "activation"; a: number[] }
  | { type: "run"; running: boolean; speed: number }
  | { type: "reset" }
  | { type: "strike"; a: number }
  | { type: "object"; pos: [number, number, number] }
  | { type: "grab"; body: number; point: [number, number, number] }
  | { type: "drag"; point: [number, number, number] }
  | { type: "release" }
  | { type: "poses"; id: number; times: number[] }
  | { type: "series"; id: number; tFrom: number; tTo: number; n: number };

export interface Frame {
  type: "frame";
  /** increments on every reset / strike, so stale frames can be told apart */
  epoch: number;
  time: number;
  /** link poses: xyz + wxyz per link (base -> tip) */
  poses: Float32Array;
  /** per tendon: flat xyz list of the path */
  tendons: Float32Array[];
  bends: Float32Array;
  curl: number;
  tipSpeed: number;
  tip: [number, number, number];
  realtime: number;
  stepUs: number;
  recFrom: number;
  recTo: number;
  grab: { from: [number, number, number]; to: [number, number, number] } | null;
}

export type FromWorker =
  | { type: "ready" }
  | { type: "built"; meta: WorkerMeta }
  | Frame
  | { type: "poses"; id: number; poses: Float32Array[]; bends: Float32Array[] }
  | { type: "series"; id: number; t: Float64Array; curl: Float64Array; speed: Float64Array; tip: Float64Array }
  | { type: "error"; message: string };
