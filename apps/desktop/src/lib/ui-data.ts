import fs from "node:fs";
import path from "node:path";
import { resolveCatalogManifestPaths } from "@/lib/session-catalog";
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

function buildLibraryFromManifests(manifests: SessionManifest[]): LibraryIndex {
  const sessions = manifests
    .map((manifest) => ({
      ...manifest.session,
      manifest_path: path.join(manifest.session.output_dir, "ui_session_manifest.json"),
    }))
    .sort((a, b) => fileMtimeMs(b.manifest_path) - fileMtimeMs(a.manifest_path));

  return {
    schema_version: "1.0.0",
    kind: "divesensei.ui-library",
    generated_at: new Date().toISOString(),
    session_count: sessions.length,
    sessions,
  };
}

const fallbackManifest: SessionManifest = {
  schema_version: "1.0.0",
  kind: "divesensei.ui-session",
  generated_at: "2026-03-11T11:31:00Z",
  session: {
    id: "demo_session",
    title: "Benchmark workspace preview",
    session_name: "Benchmark workspace preview · 2026-03-11 11:31",
    profile: "long-session",
    source_video_path: "/srv/nas/videos/Eindhoven 2026/IMG_8281.MOV",
    output_dir: path.join(repoRoot, "outputs", "demo_session"),
    status: "ready_proxy_pending",
    created_at: "2026-03-11T11:31:00Z",
    updated_at: "2026-03-11T11:31:00Z",
    candidate_count: 3,
    extracted_count: 3,
    manifest_path: path.join(repoRoot, "outputs", "demo_session", "ui_session_manifest.json"),
    timestamp_range: { first: 4.192, last: 88.604 },
    telemetry: {
      detector_seconds: 5.8,
      extract_seconds: 4.2,
      total_runtime_seconds: 10.4,
      peak_rss_kb: 1011192,
    },
  },
  artifacts: {
    session_pipeline_report: path.join(repoRoot, "outputs", "demo_session", "session_pipeline_report.json"),
    session_debug_summary: path.join(repoRoot, "outputs", "demo_session", "session_debug_summary.json"),
    session_pipeline_log: path.join(repoRoot, "outputs", "demo_session", "session_pipeline.log.jsonl"),
    detections_csv: path.join(repoRoot, "outputs", "demo_session", "detections.csv"),
  },
  detections: [
    {
      id: "det-0001",
      index: 1,
      timestamp_seconds: 4.192,
      start_time_seconds: 0.0,
      end_time_seconds: 7.192,
      duration_seconds: 7.192,
      confidence: "high",
      scores: { audio: 9.91, video: 0, combined: 9.91, audio_model_probability: 0, audio_clip_probability: 0 },
      features: { spectral_flux: 309.7, rms: 0.169, local_prominence: 10.29, nearby_peaks_8s: 2 },
      clip: { path: null, filename: "dive_splash_1_t4.2s.mp4" },
    },
    {
      id: "det-0002",
      index: 2,
      timestamp_seconds: 42.624,
      start_time_seconds: 36.624,
      end_time_seconds: 45.624,
      duration_seconds: 9,
      confidence: "medium",
      scores: { audio: 7.2, video: 0, combined: 7.2, audio_model_probability: 0, audio_clip_probability: 0 },
      features: { spectral_flux: 201.5, rms: 0.102, local_prominence: 7.1, nearby_peaks_8s: 1 },
      clip: { path: null, filename: "dive_splash_2_t42.6s_medium.mp4" },
    },
    {
      id: "det-0003",
      index: 3,
      timestamp_seconds: 88.604,
      start_time_seconds: 82.604,
      end_time_seconds: 91.604,
      duration_seconds: 9,
      confidence: "low",
      scores: { audio: 5.8, video: 0, combined: 5.8, audio_model_probability: 0, audio_clip_probability: 0 },
      features: { spectral_flux: 121.4, rms: 0.073, local_prominence: 4.2, nearby_peaks_8s: 3 },
      clip: { path: null, filename: "dive_splash_3_t88.6s_low.mp4" },
    },
  ],
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
  const discoveredManifestPaths = discoverManifestPaths(repoRoot);
  const catalogManifestPaths = resolveCatalogManifestPaths(discoveredManifestPaths);
  const catalogPathSet = new Set(catalogManifestPaths);
  const requestedFallbackPath = selectedSessionId
    ? discoveredManifestPaths.find((manifestPath) => {
        const manifest = readJsonFile<SessionManifest>(manifestPath);
        if (!manifest) return false;
        return (
          manifest.session.id === selectedSessionId ||
          path.basename(manifest.session.output_dir) === selectedSessionId ||
          manifest.session.title === selectedSessionId
        );
      })
    : undefined;
  const manifestPaths = requestedFallbackPath && !catalogPathSet.has(requestedFallbackPath)
    ? [...catalogManifestPaths, requestedFallbackPath]
    : catalogManifestPaths;
  const discoveredManifests = manifestPaths.map((manifestPath) => readJsonFile<SessionManifest>(manifestPath)).filter((manifest): manifest is SessionManifest => manifest !== null);
  const manifest = discoveredManifests.length > 0 ? pickPrimaryManifest(discoveredManifests, selectedSessionId) : fallbackManifest;
  const library = discoveredManifests.length > 0 ? buildLibraryFromManifests(discoveredManifests) : buildLibraryFromManifests([fallbackManifest]);
  return {
    library,
    manifest,
    selectedSessionId: manifest.session.id,
    logs: readLogs(manifest.artifacts.session_pipeline_log),
    artifactsPreview: {
      session_pipeline_report: safePreview(manifest.artifacts.session_pipeline_report),
      session_debug_summary: safePreview(manifest.artifacts.session_debug_summary),
    },
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
  if (status === "ready_proxy_pending") return "Ready for review";
  if (status === "complete_proxy_error") return "Needs attention";
  if (status === "complete_with_errors") return "Needs attention";
  if (status === "complete") return "Ready";
  return status.replaceAll("_", " ");
}

export function sessionIsReviewReady(status: string): boolean {
  return ["ready_proxy_pending", "complete", "complete_with_errors", "complete_proxy_error"].includes(status);
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
