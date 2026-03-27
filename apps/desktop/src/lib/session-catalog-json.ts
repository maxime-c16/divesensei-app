import fs from "node:fs";
import path from "node:path";
import { ensureRuntimeDirs, sessionCatalogPath } from "@/lib/runtime-config";
import {
  type CatalogBackend,
  type CatalogSessionRecord,
  type CatalogState,
  type StoredAnalysisRun,
  type StoredMediaSource,
  type StoredReviewDecision,
  cloneEmptyState,
  includeInBootstrap,
  readManifest,
  removeGeneratedTargets,
  sortRuns,
  statFile,
  updateManifestSourceVideoPath,
} from "@/lib/session-catalog-core";

function loadCatalogState(): CatalogState {
  ensureRuntimeDirs();
  if (!fs.existsSync(sessionCatalogPath)) {
    return cloneEmptyState();
  }
  try {
    const raw = JSON.parse(fs.readFileSync(sessionCatalogPath, "utf-8")) as Partial<CatalogState> & {
      sessions?: Array<{ manifestPath?: string }>;
    };
    if (Array.isArray(raw.mediaSources) && Array.isArray(raw.analysisRuns) && Array.isArray(raw.reviewDecisions)) {
      return {
        mediaSources: raw.mediaSources,
        analysisRuns: raw.analysisRuns,
        reviewDecisions: raw.reviewDecisions,
      };
    }
  } catch {
    return cloneEmptyState();
  }
  return cloneEmptyState();
}

function saveCatalogState(state: CatalogState): void {
  ensureRuntimeDirs();
  fs.writeFileSync(sessionCatalogPath, JSON.stringify(state, null, 2));
}

function refreshAvailabilityForSource(source: StoredMediaSource): void {
  const now = new Date().toISOString();
  const fileExists = fs.existsSync(source.sourcePath);
  const stat = statFile(source.sourcePath);
  source.fileSizeBytes = stat.size;
  source.modifiedTimeMs = stat.mtimeMs;
  source.availabilityStatus = fileExists ? "available" : "missing";
  source.updatedAt = now;
  source.lastSeenAt = fileExists ? now : null;
}

function registerSessionManifestInState(state: CatalogState, manifestPath: string): void {
  const manifest = readManifest(manifestPath);
  if (!manifest) return;

  const now = new Date().toISOString();
  const sourcePath = manifest.session.source_video_path;
  const sourceStat = statFile(sourcePath);
  const mediaSourceId = sourcePath;
  const existingSource = state.mediaSources.find((entry) => entry.id === mediaSourceId);
  const nextSource: StoredMediaSource = {
    id: mediaSourceId,
    sourcePath,
    sourceName: path.basename(sourcePath),
    fileSizeBytes: sourceStat.size,
    modifiedTimeMs: sourceStat.mtimeMs,
    profileHint: manifest.session.profile,
    availabilityStatus: fs.existsSync(sourcePath) ? "available" : "missing",
    createdAt: existingSource?.createdAt ?? now,
    updatedAt: now,
    lastSeenAt: fs.existsSync(sourcePath) ? now : null,
  };
  if (existingSource) {
    Object.assign(existingSource, nextSource);
  } else {
    state.mediaSources.push(nextSource);
  }

  const existingRun = state.analysisRuns.find((entry) => entry.id === manifest.session.id);
  const nextRun: StoredAnalysisRun = {
    id: manifest.session.id,
    mediaSourceId,
    title: manifest.session.session_name ?? manifest.session.title,
    profile: manifest.session.profile,
    manifestPath,
    outputDir: manifest.session.output_dir,
    status: manifest.session.status,
    candidateCount: manifest.session.candidate_count,
    extractedCount: manifest.session.extracted_count,
    startedAt: manifest.session.created_at ?? manifest.generated_at,
    finishedAt: manifest.session.updated_at ?? manifest.generated_at,
    lastOpenedAt: existingRun?.lastOpenedAt ?? null,
    createdAt: existingRun?.createdAt ?? manifest.session.created_at ?? now,
    updatedAt: manifest.session.updated_at ?? now,
  };
  if (existingRun) {
    Object.assign(existingRun, nextRun);
  } else {
    state.analysisRuns.push(nextRun);
  }
}

function reconcileState(state: CatalogState): void {
  const liveRunIds = new Set<string>();
  const liveMediaSourceIds = new Set<string>();
  for (const run of state.analysisRuns) {
    if (!fs.existsSync(run.manifestPath)) {
      continue;
    }
    const manifest = readManifest(run.manifestPath);
    if (!manifest) {
      continue;
    }
    registerSessionManifestInState(state, run.manifestPath);
    liveRunIds.add(run.id);
    liveMediaSourceIds.add(run.mediaSourceId);
  }
  state.analysisRuns = state.analysisRuns.filter((entry) => liveRunIds.has(entry.id));
  state.reviewDecisions = state.reviewDecisions.filter((entry) => liveRunIds.has(entry.analysisRunId));
  state.mediaSources = state.mediaSources.filter((entry) => liveMediaSourceIds.has(entry.id));
}

function toCatalogSessionRecord(state: CatalogState, run: StoredAnalysisRun): CatalogSessionRecord | null {
  const source = state.mediaSources.find((entry) => entry.id === run.mediaSourceId);
  if (!source) return null;
  return {
    sessionId: run.id,
    mediaSourceId: source.id,
    title: run.title,
    sessionName: run.title,
    profile: run.profile,
    manifestPath: run.manifestPath,
    outputDir: run.outputDir,
    status: run.status,
    sourceVideoPath: source.sourcePath,
    sourceAvailability: source.availabilityStatus,
    sourceName: source.sourceName,
    fileSizeBytes: source.fileSizeBytes,
    modifiedTimeMs: source.modifiedTimeMs,
    candidateCount: run.candidateCount,
    extractedCount: run.extractedCount,
    createdAt: run.createdAt,
    updatedAt: run.updatedAt,
    lastOpenedAt: run.lastOpenedAt,
  };
}

export function loadLegacyCatalogState(): CatalogState {
  const state = loadCatalogState();
  reconcileState(state);
  return state;
}

export const jsonCatalogBackend: CatalogBackend = {
  kind: "json",

  registerSessionManifest(manifestPath: string) {
    const state = loadCatalogState();
    registerSessionManifestInState(state, manifestPath);
    saveCatalogState(state);
  },

  listCatalogSessions() {
    const state = loadCatalogState();
    reconcileState(state);
    saveCatalogState(state);
    return sortRuns(state.analysisRuns)
      .map((run) => toCatalogSessionRecord(state, run))
      .filter((entry): entry is CatalogSessionRecord => entry !== null);
  },

  saveReviewDecision(analysisRunId, detectionId, label, notes = "") {
    const state = loadCatalogState();
    reconcileState(state);
    const run = state.analysisRuns.find((entry) => entry.id === analysisRunId);
    if (!run) {
      throw new Error("Analysis run not found.");
    }

    const now = new Date().toISOString();
    const id = `${analysisRunId}:${detectionId}`;
    const existing = state.reviewDecisions.find((entry) => entry.analysisRunId === analysisRunId && entry.detectionId === detectionId);
    const decision: StoredReviewDecision = {
      id,
      analysisRunId,
      detectionId,
      label,
      notes,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };
    if (existing) {
      Object.assign(existing, decision);
    } else {
      state.reviewDecisions.push(decision);
    }
    saveCatalogState(state);
    return decision;
  },

  listReviewDecisions(analysisRunId) {
    const state = loadCatalogState();
    return state.reviewDecisions
      .filter((entry) => entry.analysisRunId === analysisRunId)
      .sort((a, b) => {
        if (a.updatedAt === b.updatedAt) return a.detectionId.localeCompare(b.detectionId);
        return b.updatedAt.localeCompare(a.updatedAt);
      });
  },

  markSessionOpened(analysisRunId) {
    const state = loadCatalogState();
    const run = state.analysisRuns.find((entry) => entry.id === analysisRunId);
    if (!run) return;
    const now = new Date().toISOString();
    run.lastOpenedAt = now;
    run.updatedAt = now;
    saveCatalogState(state);
  },

  renameSessionRun(analysisRunId, nextName) {
    const trimmed = nextName.trim();
    if (!trimmed) {
      throw new Error("Session name cannot be empty.");
    }
    const state = loadCatalogState();
    const run = state.analysisRuns.find((entry) => entry.id === analysisRunId);
    if (!run) {
      throw new Error("Analysis run not found.");
    }
    const now = new Date().toISOString();
    run.title = trimmed;
    run.updatedAt = now;
    const manifest = readManifest(run.manifestPath);
    if (manifest) {
      manifest.session.title = trimmed;
      manifest.session.session_name = trimmed;
      manifest.session.updated_at = now;
      fs.writeFileSync(run.manifestPath, JSON.stringify(manifest, null, 2));
    }
    registerSessionManifestInState(state, run.manifestPath);
    saveCatalogState(state);
  },

  getManifestPathForAnalysisRun(analysisRunId) {
    const state = loadCatalogState();
    return state.analysisRuns.find((entry) => entry.id === analysisRunId)?.manifestPath ?? null;
  },

  deleteSessionRun(analysisRunId) {
    const state = loadCatalogState();
    const run = state.analysisRuns.find((entry) => entry.id === analysisRunId);
    if (!run) return false;
    removeGeneratedTargets(analysisRunId, run);
    state.analysisRuns = state.analysisRuns.filter((entry) => entry.id !== analysisRunId);
    state.reviewDecisions = state.reviewDecisions.filter((entry) => entry.analysisRunId !== analysisRunId);
    const stillReferenced = state.analysisRuns.some((entry) => entry.mediaSourceId === run.mediaSourceId);
    if (!stillReferenced) {
      state.mediaSources = state.mediaSources.filter((entry) => entry.id !== run.mediaSourceId);
    }
    saveCatalogState(state);
    return true;
  },

  refreshCatalogAvailability() {
    const state = loadCatalogState();
    for (const source of state.mediaSources) {
      refreshAvailabilityForSource(source);
    }
    saveCatalogState(state);
    return this.listCatalogSessions();
  },

  relinkCatalogSource(mediaSourceId, nextPath) {
    const resolvedPath = path.resolve(nextPath);
    if (!fs.existsSync(resolvedPath)) {
      throw new Error("Target path does not exist.");
    }

    const state = loadCatalogState();
    const source = state.mediaSources.find((entry) => entry.id === mediaSourceId);
    if (!source) {
      throw new Error("Media source not found.");
    }

    const now = new Date().toISOString();
    const stat = statFile(resolvedPath);
    source.sourcePath = resolvedPath;
    source.sourceName = path.basename(resolvedPath);
    source.fileSizeBytes = stat.size;
    source.modifiedTimeMs = stat.mtimeMs;
    source.availabilityStatus = "available";
    source.updatedAt = now;
    source.lastSeenAt = now;

    for (const run of state.analysisRuns.filter((entry) => entry.mediaSourceId === mediaSourceId)) {
      run.updatedAt = now;
      updateManifestSourceVideoPath(run.manifestPath, resolvedPath);
    }

    saveCatalogState(state);
  },

  resolveCatalogManifestPaths(discoveredManifestPaths) {
    const state = loadCatalogState();
    reconcileState(state);
    const existingCatalogPaths = sortRuns(state.analysisRuns)
      .map((entry) => entry.manifestPath)
      .filter((manifestPath) => fs.existsSync(manifestPath));

    if (existingCatalogPaths.length > 0) {
      saveCatalogState(state);
      return existingCatalogPaths;
    }

    for (const manifestPath of discoveredManifestPaths.filter(includeInBootstrap)) {
      registerSessionManifestInState(state, manifestPath);
    }
    saveCatalogState(state);

    return sortRuns(state.analysisRuns)
      .map((entry) => entry.manifestPath)
      .filter((manifestPath) => fs.existsSync(manifestPath));
  },
};
