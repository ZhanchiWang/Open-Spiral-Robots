import type { Params } from "../core/params";

export interface MeshData {
  vertices: Float32Array;
  normals: Float32Array;
  triangles: Uint32Array;
  /** line-segment pairs (xyz xyz) of the B-rep edges */
  edges: Float32Array | null;
}

export interface BuildResult {
  main: MeshData | null;
  elastic: MeshData | null;
  holes: MeshData | null;
  coneRegions: (MeshData | null)[];
  posed: boolean;
}

export type CadRequestType = "build" | "export-step" | "export-stl" | "baselink";

export interface CadRequest {
  id: number;
  type: CadRequestType;
  params: Params;
  options?: { holes?: boolean };
}

export interface CadResponse {
  id: number;
  type: CadRequestType;
  ok: boolean;
  error?: string;
  logs: string[];
  timings: Record<string, number>;
  build?: BuildResult;
  bytes?: Uint8Array;
}
