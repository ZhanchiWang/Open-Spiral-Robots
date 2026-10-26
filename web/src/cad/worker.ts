/// <reference lib="webworker" />
// CAD web worker: loads OpenCascade (replicad) once and answers build/export
// requests. Every message gets one reply with timings and the worker's log so
// the debug panel can show what OCCT did.

import opencascade from "replicad-opencascadejs";
import wasmUrl from "replicad-opencascadejs/wasm?url";
import { setOC, type Shape3D } from "replicad";
import { meshToBinarySTL } from "../core/files";
import type { Params } from "../core/params";
import {
  baselinkSolid,
  buildParts,
  coneCutRegions,
  holePreview,
  makeContext,
  stlTolerance,
  type CadParts,
} from "./cad";
import type { CadRequest, CadResponse, MeshData } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

let ready: Promise<void> | null = null;
function init(): Promise<void> {
  ready ??= opencascade({ locateFile: () => wasmUrl }).then((oc) => {
    setOC(oc as never);
  });
  return ready;
}

let cache: { key: string; parts: CadParts } | null = null;

function meshOf(shape: Shape3D | null, tolerance: number, withEdges: boolean): MeshData | null {
  if (!shape || shape.isNull) return null;
  const m = shape.mesh({ tolerance, angularTolerance: 0.2 });
  const out: MeshData = {
    vertices: new Float32Array(m.vertices),
    normals: new Float32Array(m.normals),
    triangles: new Uint32Array(m.triangles),
    edges: null,
  };
  if (withEdges) {
    try {
      out.edges = new Float32Array(shape.meshEdges({ tolerance, angularTolerance: 0.2 }).lines);
    } catch {
      out.edges = null;
    }
  }
  return out;
}

function transfer(m: MeshData | null, list: Transferable[]) {
  if (!m) return;
  list.push(m.vertices.buffer, m.normals.buffer, m.triangles.buffer);
  if (m.edges) list.push(m.edges.buffer);
}

function getParts(params: Params, log: (s: string) => void, timings: Record<string, number>): CadParts {
  const key = JSON.stringify(params);
  if (cache?.key === key) {
    log("reusing cached solids");
    return cache.parts;
  }
  const t0 = performance.now();
  const parts = buildParts(makeContext(params, log));
  timings.solids = performance.now() - t0;
  cache = { key, parts };
  return parts;
}

self.onmessage = async (ev: MessageEvent<CadRequest>) => {
  const req = ev.data;
  const logs: string[] = [];
  const log = (s: string) => logs.push(s);
  const timings: Record<string, number> = {};
  const started = performance.now();
  const reply = (body: Partial<CadResponse>, transferList: Transferable[] = []) => {
    timings.total = performance.now() - started;
    self.postMessage({ id: req.id, type: req.type, ok: true, logs, timings, ...body } as CadResponse, transferList);
  };
  try {
    const t0 = performance.now();
    await init();
    timings.init = performance.now() - t0;
    const ctx = makeContext(req.params, log);
    const tol = Math.max(0.02, ctx.design.length * 0.001);

    if (req.type === "build") {
      const parts = getParts(req.params, log, timings);
      const t1 = performance.now();
      const result = {
        main: meshOf(parts.main, tol, true),
        elastic: meshOf(parts.elastic, tol, true),
        holes: null as MeshData | null,
        coneRegions: [] as (MeshData | null)[],
        posed: parts.posed,
      };
      if (!parts.posed && req.options?.holes) result.holes = meshOf(holePreview(ctx), tol, false);
      if (req.params.show_cone_cuts && req.params.two_cable) {
        const t2 = performance.now();
        result.coneRegions = coneCutRegions(ctx).map((s) => meshOf(s, tol, false));
        timings.coneRegions = performance.now() - t2;
      }
      timings.mesh = performance.now() - t1;
      const list: Transferable[] = [];
      transfer(result.main, list);
      transfer(result.elastic, list);
      transfer(result.holes, list);
      result.coneRegions.forEach((m) => transfer(m, list));
      reply({ build: result }, list);
      return;
    }

    if (req.type === "export-step" || req.type === "export-stl") {
      const parts = getParts(req.params, log, timings);
      const solids = [parts.main, parts.elastic].filter((s): s is Shape3D => !!s && !s.isNull);
      if (!solids.length) throw new Error("Nothing to export.");
      const t1 = performance.now();
      let bytes: Uint8Array;
      if (req.type === "export-step") {
        const { makeCompound } = await import("replicad");
        // makeCompound frees its inputs: hand it clones so the cache stays valid
        const comp = makeCompound(solids.map((s) => s.clone())) as Shape3D;
        bytes = new Uint8Array(await comp.blobSTEP().arrayBuffer());
      } else {
        const t = stlTolerance(ctx.design);
        bytes = meshToBinarySTL(solids.map((s) => s.mesh({ tolerance: t, angularTolerance: 0.1 })));
      }
      timings.export = performance.now() - t1;
      reply({ bytes }, [bytes.buffer]);
      return;
    }

    if (req.type === "baselink") {
      const t1 = performance.now();
      const solid = baselinkSolid(ctx);
      if (!solid) throw new Error("Could not build the base link.");
      timings.solids = performance.now() - t1;
      const t2 = performance.now();
      const bytes = meshToBinarySTL([solid.mesh({ tolerance: stlTolerance(ctx.design), angularTolerance: 0.1 })]);
      timings.export = performance.now() - t2;
      reply({ bytes }, [bytes.buffer]);
      return;
    }
    throw new Error(`unknown request ${(req as { type: string }).type}`);
  } catch (e) {
    timings.total = performance.now() - started;
    const err = e instanceof Error ? e.message : typeof e === "number" ? `OCCT exception #${e}` : String(e);
    self.postMessage({ id: req.id, type: req.type, ok: false, error: err, logs, timings } as CadResponse);
  }
};
