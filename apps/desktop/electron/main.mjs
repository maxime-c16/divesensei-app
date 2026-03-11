import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { app, BrowserWindow, ipcMain, shell } from "electron";
import { requestJson } from "./shared.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const preloadPath = path.join(__dirname, "preload.mjs");
const packagedPort = process.env.DIVESENSEI_PORT ?? "35817";
let desktopStartUrl = process.env.ELECTRON_START_URL ?? `http://127.0.0.1:${packagedPort}/`;

function resolveServerEntryPath() {
  const appRoot = app.getAppPath();
  const candidates = [
    path.join(appRoot, "dist", "server", "entry.mjs"),
    path.join(process.resourcesPath, "app", "dist", "server", "entry.mjs"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? candidates[0];
}

async function ensureBundledServer() {
  if (process.env.ELECTRON_START_URL) return;
  process.env.DIVESENSEI_PORT = packagedPort;
  const serverEntryPath = resolveServerEntryPath();
  const serverEntry = await import(pathToFileURL(serverEntryPath).href);
  if (typeof serverEntry.startServer !== "function") {
    throw new Error(`Bundled server entry is missing startServer(): ${serverEntryPath}`);
  }
  await serverEntry.startServer();
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1520,
    height: 980,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#08131d",
    autoHideMenuBar: true,
    title: "DiveSensei",
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  window.loadURL(desktopStartUrl);
}

function registerIpcHandlers() {
  ipcMain.handle("media:listSources", async () => {
    const payload = await requestJson(desktopStartUrl, "/api/catalog");
    const seen = new Map();
    for (const session of payload.sessions ?? []) {
      if (!seen.has(session.mediaSourceId)) {
        seen.set(session.mediaSourceId, {
          id: session.mediaSourceId,
          sourcePath: session.sourceVideoPath,
          sourceName: session.sourceName,
          availabilityStatus: session.sourceAvailability,
          lastSeenAt: session.lastOpenedAt ?? session.updatedAt,
        });
      }
    }
    return Array.from(seen.values());
  });

  ipcMain.handle("media:relinkSource", async (_event, input) => {
    await requestJson(desktopStartUrl, "/api/catalog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "relink",
        mediaSourceId: input.mediaSourceId,
        nextPath: input.nextPath,
      }),
    });
  });

  ipcMain.handle("analysis:startRun", async (_event, input) => {
    return requestJson(desktopStartUrl, "/api/analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videoPath: input.sourcePath, profile: input.profile }),
    });
  });

  ipcMain.handle("analysis:getRun", async (_event, input) => {
    try {
      return requestJson(desktopStartUrl, `/api/analysis?job=${encodeURIComponent(input.runId)}`);
    } catch {
      return null;
    }
  });

  ipcMain.handle("review:saveDecision", async (_event, input) => {
    return requestJson(desktopStartUrl, "/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  });

  ipcMain.handle("review:listDecisions", async (_event, input) => {
    return requestJson(desktopStartUrl, `/api/review?analysisRunId=${encodeURIComponent(input.analysisRunId)}`);
  });

  ipcMain.handle("files:openPath", async (_event, input) => {
    await shell.openPath(input.targetPath);
  });

  ipcMain.handle("files:revealPath", async (_event, input) => {
    shell.showItemInFolder(input.targetPath);
  });
}

app.whenReady().then(() => {
  ensureBundledServer()
    .then(() => {
      registerIpcHandlers();
      createWindow();
    })
    .catch((error) => {
      console.error(error);
      app.quit();
    });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
