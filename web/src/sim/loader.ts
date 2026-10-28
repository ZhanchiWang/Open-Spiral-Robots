// Lazy loader for the official MuJoCo WASM build (single-threaded).
import type { MainModule } from "@mujoco/mujoco";
import wasmUrl from "@mujoco/mujoco/mujoco.wasm?url";

let modulePromise: Promise<MainModule> | null = null;

export function loadMujocoModule(): Promise<MainModule> {
  modulePromise ??= import("@mujoco/mujoco").then((m) => m.default({ locateFile: () => wasmUrl }) as Promise<MainModule>);
  return modulePromise;
}
