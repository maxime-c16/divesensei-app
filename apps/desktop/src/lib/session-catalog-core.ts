import fs from "node:fs";
import path from "node:path";
import type { ReviewDecision, SessionManifest } from "@/types/ui";
import { outputsRoot, repoRoot } from "@/lib/runtime-config";

export interface CatalogSessionRecord {
  sessionId: string;
  mediaSourceId: string;
  title: string;
  sessionName?: string;
  profile: string;
  manifestPath: string;
  outputDir: string;
  status: string;
  sourceVideoPath: string;
  sourceAvailability: "available" | "missing" | "relink-required";
  sourceName: string;
  fileSizeBytes: number | null;
  modifiedTimeMs: number | null;
  candidateCount: number;
  extractedCount: number;
  createdAt: string;
  updatedAt: string;
  lastOpenedAt: string | null;
}

export interface StoredMediaSource {
  id: string;
  sourcePath: string;
  sourceName: string;
  fileSizeBytes: number | null;
  modifiedTimeMs: number | null;
  profileHint?: string;
  availabilityStatus: "available" | "missing" | "relink-required";
  createdAt: string;
  updatedAt: string;
  lastSeenAt: string | null;
}

export interface StoredAnalysisRun {
  id: string;
  mediaSourceId: string;
  title: string;
  profile: string;
  manifestPath: string;
  outputDir: string;
  status: string;
  candidateCount: number;
  extractedCount: number;
  startedAt: string | null;
  finishedAt: string | null;
  lastOpenedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface StoredReviewDecision extends ReviewDecision {}

export interface CatalogState {
  mediaSources: StoredMediaSource[];
  analysisRuns: StoredAnalysisRun[];
  reviewDecisions: StoredReviewDecision[];
}

export const EMPTY_STATE: CatalogState = {
  mediaSources: [],
  analysisRuns: [],
  reviewDecisions: [],
};

export interface CatalogBackend {
  readonly kind: "sqlite" | "json";
  registerSessionManifest(manifestPath: string): void;
  listCatalogSessions(): CatalogSessionRecord[];
  saveReviewDecision(
    analysisRunId: string,
    detectionId: string,
    label: ReviewDecision["label"],
    notes?: string,
  ): ReviewDecision;
  listReviewDecisions(analysisRunId: string): ReviewDecision[];
  markSessionOpened(analysisRunId: string): void;
  renameSessionRun(analysisRunId: string, nextName: string): void;
  getManifestPathForAnalysisRun(analysisRunId: string): string | null;
  deleteSessionRun(analysisRunId: string): boolean;
  refreshCatalogAvailability(): CatalogSessionRecord[];
  relinkCatalogSource(mediaSourceId: string, nextPath: string): void;
  resolveCatalogManifestPaths(discoveredManifestPaths: string[]): string[];
}

export function cloneEmptyState(): CatalogState {
  return {
    mediaSources: [],
    analysisRuns: [],
    reviewDecisions: [],
  };
}

export function readManifest(manifestPath: string): SessionManifest | null {
  try {
    return JSON.parse(fs.readFileSync(manifestPath, "utf-8")) as SessionManifest;
  } catch {
    return null;
  }
}

export function statFile(filePath: string): { size: number | null; mtimeMs: number | null } {
  try {
    const stat = fs.statSync(filePath);
    return { size: stat.size, mtimeMs: stat.mtimeMs };
  } catch {
    return { size: null, mtimeMs: null };
  }
}

export function includeInBootstrap(manifestPath: string): boolean {
  const dirName = path.basename(path.dirname(manifestPath));
  return dirName.startsWith(".tmp_ui_run_") || manifestPath.startsWith(outputsRoot) || manifestPath.startsWith(repoRoot);
}

export function sortRuns<T extends { updatedAt: string; lastOpenedAt: string | null }>(runs: T[]): T[] {
  return runs.slice().sort((a, b) => {
    const aStamp = a.lastOpenedAt ?? a.updatedAt;
    const bStamp = b.lastOpenedAt ?? b.updatedAt;
    if (aStamp === bStamp) return b.updatedAt.localeCompare(a.updatedAt);
    return bStamp.localeCompare(aStamp);
  });
}

export function cleanupTargetsForRun(run: { outputDir: string; manifestPath: string }): string[] {
  const targets = [run.outputDir];
  if (run.manifestPath && !run.manifestPath.startsWith(`${run.outputDir}${path.sep}`)) {
    targets.push(run.manifestPath);
  }
  return targets;
}

export function removeGeneratedTargets(analysisRunId: string, run: { outputDir: string; manifestPath: string }): void {
  for (const target of cleanupTargetsForRun(run)) {
    if (!target || !fs.existsSync(target)) continue;
    try {
      fs.rmSync(target, { recursive: true, force: true, maxRetries: 2 });
    } catch (error) {
      throw new Error(`Could not remove generated files for ${analysisRunId}: ${error instanceof Error ? error.message : "unknown cleanup error"}`);
    }
  }
}

export function updateManifestSourceVideoPath(manifestPath: string, nextPath: string): void {
  const manifest = readManifest(manifestPath);
  if (!manifest) return;
  manifest.session.source_video_path = nextPath;
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

export function updateManifestSessionName(manifestPath: string, nextName: string, updatedAt: string): void {
  const manifest = readManifest(manifestPath);
  if (!manifest) return;
  manifest.session.title = nextName;
  manifest.session.session_name = nextName;
  manifest.session.updated_at = updatedAt;
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}
