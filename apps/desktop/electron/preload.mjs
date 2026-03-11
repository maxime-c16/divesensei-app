import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("divesenseiDesktop", {
  media: {
    listSources: () => ipcRenderer.invoke("media:listSources"),
    relinkSource: (mediaSourceId, nextPath) => ipcRenderer.invoke("media:relinkSource", { mediaSourceId, nextPath }),
  },
  analysis: {
    startRun: (input) => ipcRenderer.invoke("analysis:startRun", input),
    getRun: (runId) => ipcRenderer.invoke("analysis:getRun", { runId }),
  },
  review: {
    saveDecision: (input) => ipcRenderer.invoke("review:saveDecision", input),
    listDecisions: (analysisRunId) => ipcRenderer.invoke("review:listDecisions", { analysisRunId }),
  },
  files: {
    openPath: (targetPath) => ipcRenderer.invoke("files:openPath", { targetPath }),
    revealPath: (targetPath) => ipcRenderer.invoke("files:revealPath", { targetPath }),
  },
});
