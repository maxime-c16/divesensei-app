import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { waitForUrl } from "./shared.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const serverEntry = path.join(projectRoot, "dist", "server", "entry.mjs");
const appUrl = process.env.ELECTRON_START_URL ?? "http://127.0.0.1:5173/";

let serverProcess = null;
let electronProcess = null;

function localBin(name) {
  return path.join(projectRoot, "node_modules", ".bin", name);
}

function shutdown(code = 0) {
  if (electronProcess && !electronProcess.killed) electronProcess.kill("SIGTERM");
  if (serverProcess && !serverProcess.killed) serverProcess.kill("SIGTERM");
  process.exit(code);
}

async function main() {
  serverProcess = spawn(process.execPath, [serverEntry], {
    cwd: projectRoot,
    stdio: "inherit",
  });

  await waitForUrl(appUrl, 30000);

  electronProcess = spawn(localBin("electron"), ["./electron/main.mjs"], {
    cwd: projectRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      ELECTRON_START_URL: appUrl,
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
