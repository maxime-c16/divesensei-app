import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { waitForUrl } from "./shared.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const devUrl = process.env.ELECTRON_START_URL ?? "http://127.0.0.1:5173/";

let astroProcess = null;
let electronProcess = null;

function localBin(name) {
  return path.join(projectRoot, "node_modules", ".bin", name);
}

async function ensureAstroServer() {
  try {
    await waitForUrl(devUrl, 1200);
    return;
  } catch {
    astroProcess = spawn(localBin("astro"), ["dev", "--host", "127.0.0.1", "--port", "5173"], {
      cwd: projectRoot,
      stdio: "inherit",
    });
    await waitForUrl(devUrl, 30000);
  }
}

function shutdown(code = 0) {
  if (electronProcess && !electronProcess.killed) electronProcess.kill("SIGTERM");
  if (astroProcess && !astroProcess.killed) astroProcess.kill("SIGTERM");
  process.exit(code);
}

async function main() {
  await ensureAstroServer();
  electronProcess = spawn(localBin("electron"), ["./electron/main.mjs"], {
    cwd: projectRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      ELECTRON_START_URL: devUrl,
    },
  });
  electronProcess.on("exit", (code) => shutdown(code ?? 0));
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => shutdown(0));
}

main().catch((error) => {
  console.error(error);
  shutdown(1);
});
