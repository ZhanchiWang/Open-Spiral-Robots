/// <reference types="vitest/config" />
import { defineConfig } from "vite";

// base "./" keeps every asset path relative, so the same build works from
// GitHub Pages (/Open-Spiral-Robots/), a sub-folder, or `npx openspirobs-web`.
export default defineConfig({
  base: "./",
  worker: { format: "es" },
  optimizeDeps: {
    exclude: ["replicad-opencascadejs", "@mujoco/mujoco"],
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 4096,
  },
  test: {
    environment: "node",
  },
});
