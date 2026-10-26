#!/usr/bin/env node
// Serve the built web app: `npx openspirobs-web [--port 5180] [--host 127.0.0.1]`.
// No dependencies; sends the COOP/COEP headers that SharedArrayBuffer
// (multi-threaded WASM) needs, and the right MIME type for .wasm.
import { createServer } from "node:http";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : process.env[name.toUpperCase()] ?? def;
};
if (args.includes("--help") || args.includes("-h")) {
  console.log("usage: openspirobs-web [--port 5180] [--host 127.0.0.1]");
  process.exit(0);
}
const port = Number(opt("port", 5180));
const host = opt("host", "127.0.0.1");
const root = resolve(fileURLToPath(new URL("../dist", import.meta.url)));
if (!existsSync(join(root, "index.html"))) {
  console.error(`No build found in ${root}. Run "npm run build" first.`);
  process.exit(1);
}
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".map": "application/json",
};

const COMPRESSIBLE = new Set([".html", ".js", ".mjs", ".css", ".json", ".wasm", ".svg", ".map"]);
const gzCache = new Map();

createServer((req, res) => {
  let path = decodeURIComponent(new URL(req.url ?? "/", "http://x").pathname);
  if (path.endsWith("/")) path += "index.html";
  const file = normalize(join(root, path));
  if (file !== root && !file.startsWith(root + sep)) {
    res.writeHead(403).end();
    return;
  }
  if (!existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("not found");
    return;
  }
  const type = TYPES[extname(file)] ?? "application/octet-stream";
  const headers = {
    "Content-Type": type,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cache-Control": path.includes("/assets/") ? "public, max-age=31536000, immutable" : "no-cache",
    Vary: "Accept-Encoding",
  };
  // the WebAssembly and JS bundles shrink 3-4x with gzip: compress once, keep in memory
  if (COMPRESSIBLE.has(extname(file)) && /\bgzip\b/.test(String(req.headers["accept-encoding"] ?? ""))) {
    let gz = gzCache.get(file);
    if (!gz) {
      gz = gzipSync(readFileSync(file), { level: 6 });
      gzCache.set(file, gz);
    }
    res.writeHead(200, { ...headers, "Content-Encoding": "gzip", "Content-Length": gz.length });
    res.end(gz);
    return;
  }
  res.writeHead(200, headers);
  createReadStream(file).pipe(res);
}).listen(port, host, () => {
  console.log(`OpenSpiRobs web: http://${host === "0.0.0.0" ? "localhost" : host}:${port}/`);
});
