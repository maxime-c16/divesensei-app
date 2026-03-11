import fs from "node:fs";
import path from "node:path";

const projectRoot = process.cwd();
const entryPath = path.join(projectRoot, "dist", "server", "entry.mjs");

if (!fs.existsSync(entryPath)) {
  throw new Error(`Missing Astro server entry: ${entryPath}`);
}

const original = fs.readFileSync(entryPath, "utf-8");
const next = original
  .replace(/"client":\s*"file:[^"]+",/, '"client": new URL("../client/", import.meta.url).toString(),')
  .replace(/"server":\s*"file:[^"]+",/, '"server": new URL("./", import.meta.url).toString(),')
  .replace(/"port":\s*5173,/, '"port": Number(process.env.DIVESENSEI_PORT ?? 5173),');

if (next === original) {
  throw new Error("Astro server entry patch did not apply. Check dist/server/entry.mjs format.");
}

fs.writeFileSync(entryPath, next);
console.log(`Patched ${entryPath} for packaged Electron runtime.`);
