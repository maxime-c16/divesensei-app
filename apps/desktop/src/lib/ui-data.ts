import fs from "node:fs";
import path from "node:path";
import { getManifestPathForAnalysisRun, listCatalogSessions, resolveCatalogManifestPaths } from "@/lib/session-catalog";
import type { DebugLogEntry, LibraryIndex, SessionManifest, UiDataBundle } from "@/types/ui";
import { outputsRoot, repoRoot } from "@/lib/runtime-config";

function readJsonFile<T>(filePath: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

function fileMtimeMs(filePath: string): number {
  try {
    return fs.statSync(filePath).mtimeMs;
  } catch {
    return 0;
  }
}

function safePreview(filePath: string): string {
  try {
    return fs.readFileSync(filePath, "utf-8").slice(0, 1400);
  } catch {
    return "Artifact unavailable in current workspace.";
  }
}

function readJsonlFile<T>(filePath: string): T[] {
  try {
    return fs.readFileSync(filePath, "utf-8").split("\n").filter(Boolean).map((line) => JSON.parse(line) as T);
  } catch {
    return [];
  }
}

function readLogs(filePath: string): DebugLogEntry[] {
  try {
    return fs.readFileSync(filePath, "utf-8").split("\n").filter(Boolean).slice(-10).map((line, index) => {
      const parsed = JSON.parse(line) as Record<string, unknown>;
      return {
        timestamp: String(parsed.ts ?? parsed.timestamp ?? `2026-03-11T11:30:${10 + index}Z`),
        level: String(parsed.level ?? "INFO").toUpperCase() as DebugLogEntry["level"],
        message: String(parsed.message ?? parsed.msg ?? parsed.event ?? "Pipeline event"),
        stage: String(parsed.stage ?? parsed.component ?? parsed.event ?? "pipeline"),
      };
    });
  } catch {
    return [
      {
        timestamp: "2026-03-11T11:30:21Z",
        level: "INFO",
        message: "No structured log file found; using fallback messages.",
        stage: "bootstrap",
      },
    ];
  }
}

function discoverManifestPaths(root: string): string[] {
  const candidates: string[] = [];
  const roots = [outputsRoot, root];
  for (const scanRoot of roots) {
    if (!fs.existsSync(scanRoot)) continue;
    const entries = fs.readdirSync(scanRoot, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const manifestPath = path.join(scanRoot, entry.name, "ui_session_manifest.json");
      if (fs.existsSync(manifestPath)) {
        candidates.push(manifestPath);
      }
    }
  }
  return Array.from(new Set(candidates));
}

function buildLibraryFromCatalog(): LibraryIndex {
  const discoveredManifestPaths = discoverManifestPaths(repoRoot);
  resolveCatalogManifestPaths(discoveredManifestPaths);
  const sessions = listCatalogSessions().map((session) => {
    const manifest = readJsonFile<SessionManifest>(session.manifestPath);
    return {
      id: session.sessionId,
      title: session.title,
      session_name: session.sessionName ?? session.title,
      mode: manifest?.session.mode ?? "standard",
      profile: session.profile,
      status: session.status,
      created_at: session.createdAt,
      updated_at: session.updatedAt,
      source_availability: session.sourceAvailability,
      candidate_count: session.candidateCount,
      extracted_count: session.extractedCount,
      source_video_path: session.sourceVideoPath,
      output_dir: session.outputDir,
      manifest_path: session.manifestPath,
      timestamp_range: manifest?.session.timestamp_range ?? { first: 0, last: 0 },
      telemetry: manifest?.session.telemetry ?? {
        detector_seconds: 0,
        extract_seconds: 0,
        total_runtime_seconds: 0,
        peak_rss_kb: 0,
      },
    };
  });

  return {
    schema_version: "1.0.0",
    kind: "divesensei.ui-library",
    generated_at: new Date().toISOString(),
    session_count: sessions.length,
    sessions,
  };
}

function buildEmptyLibrary(): LibraryIndex {
  return {
    schema_version: "1.0.0",
    kind: "divesensei.ui-library",
    generated_at: new Date().toISOString(),
    session_count: 0,
    sessions: [],
  };
}

const fallbackManifest: SessionManifest = {
  schema_version: "1.0.0",
  kind: "divesensei.ui-session",
  generated_at: "2026-03-11T11:31:00Z",
  session: {
    id: "",
    title: "No session selected",
    session_name: "No session selected",
    profile: "long-session",
    source_video_path: "",
    output_dir: "",
    status: "complete",
    created_at: "2026-03-11T11:31:00Z",
    updated_at: "2026-03-11T11:31:00Z",
    candidate_count: 0,
    extracted_count: 0,
    manifest_path: "",
    timestamp_range: { first: 0, last: 0 },
    telemetry: {
      detector_seconds: 0,
      extract_seconds: 0,
      total_runtime_seconds: 0,
      peak_rss_kb: 0,
    },
  },
  artifacts: {
    session_pipeline_report: "",
    session_debug_summary: "",
    session_pipeline_log: "",
    detections_csv: "",
  },
  detections: [],
};

function pickPrimaryManifest(manifests: SessionManifest[], selectedSessionId?: string): SessionManifest {
  if (selectedSessionId) {
    const matched = manifests.find((manifest) =>
      manifest.session.id === selectedSessionId ||
      path.basename(manifest.session.output_dir) === selectedSessionId ||
      manifest.session.title === selectedSessionId
    );
    if (matched) return matched;
  }
  return manifests.slice().sort((a, b) => fileMtimeMs(path.join(b.session.output_dir, "ui_session_manifest.json")) - fileMtimeMs(path.join(a.session.output_dir, "ui_session_manifest.json")))[0];
}

export function getUiData(selectedSessionId?: string): UiDataBundle {
  const library = buildLibraryFromCatalog();
  const selectedManifestPath = selectedSessionId
    ? getManifestPathForAnalysisRun(selectedSessionId)
    : library.sessions[0]?.manifest_path ?? null;
  const discoveredManifestPaths = discoverManifestPaths(repoRoot);
  const fallbackManifestPaths = selectedManifestPath
    ? [selectedManifestPath]
    : discoveredManifestPaths;
  const discoveredManifests = fallbackManifestPaths
    .map((manifestPath) => readJsonFile<SessionManifest>(manifestPath))
    .filter((manifest): manifest is SessionManifest => manifest !== null);
  const manifestPool = discoveredManifests.length > 0
    ? discoveredManifests
    : discoveredManifestPaths
        .map((manifestPath) => readJsonFile<SessionManifest>(manifestPath))
        .filter((item): item is SessionManifest => item !== null);
  const manifest = manifestPool.length > 0 ? pickPrimaryManifest(manifestPool, selectedSessionId) : fallbackManifest;
  const eventReviewSupportPath = manifest.artifacts?.event_review_support;
  const eventReviewSupportSummaryPath = manifest.artifacts?.event_review_support_summary;
  const eventReviewSupport = eventReviewSupportPath ? readJsonlFile<Record<string, unknown>>(eventReviewSupportPath) : [];
  const eventReviewSupportSummary = eventReviewSupportSummaryPath ? readJsonFile<Record<string, unknown>>(eventReviewSupportSummaryPath) ?? null : null;
  return {
    library,
    manifest,
    selectedSessionId: manifest.session.id,
    logs: readLogs(manifest.artifacts.session_pipeline_log ?? ""),
    artifactsPreview: {
      session_pipeline_report: safePreview(manifest.artifacts.session_pipeline_report ?? ""),
      session_debug_summary: safePreview(manifest.artifacts.session_debug_summary ?? ""),
      event_review_support: eventReviewSupportPath ? safePreview(eventReviewSupportPath) : "",
      event_review_support_summary: eventReviewSupportSummaryPath ? safePreview(eventReviewSupportSummaryPath) : "",
    },
    eventReviewSupport,
    eventReviewSupportSummary,
  };
}

export function formatSeconds(value: number): string {
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(3).padStart(6, "0")}`;
}

export function mediaUrl(filePath: string | null | undefined): string {
  if (!filePath) return "";
  return `/api/media?path=${encodeURIComponent(filePath)}`;
}

export function sessionStatusLabel(status: string): string {
  if (status === "evaluation_ready") return "Evaluation ready";
  if (status === "evaluation_proxy_error") return "Evaluation ready";
  if (status === "ready_proxy_pending") return "Ready for review";
  if (status === "complete_proxy_error") return "Needs attention";
  if (status === "complete_with_errors") return "Needs attention";
  if (status === "complete") return "Ready";
  return status.replaceAll("_", " ");
}

export function sessionIsReviewReady(status: string): boolean {
  return ["ready_proxy_pending", "complete", "complete_with_errors", "complete_proxy_error", "evaluation_ready", "evaluation_proxy_error"].includes(status);
}

export function resolveClipMediaPath(filePath: string | null | undefined): string | null {
  if (!filePath) return null;
  const proxyPath = path.join(path.dirname(filePath), "web", path.basename(filePath));
  return fs.existsSync(proxyPath) ? proxyPath : filePath;
}

export function resolveSourceMediaPath(filePath: string, outputDir: string): string {
  const proxyPath = path.join(outputDir, "web", "session_source_review.mp4");
  return fs.existsSync(proxyPath) ? proxyPath : filePath;
}
