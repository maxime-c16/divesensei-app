import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import type { SessionManifest } from "@/types/ui";
import { ensureRuntimeDirs, exportJobStatePath, exportJobsRoot } from "@/lib/runtime-config";

export type ExportJobStatus = "queued" | "running" | "completed" | "failed";
export type ExportJobPhase = "queued" | "preparing" | "exporting" | "completed" | "failed";

export interface ExportJobRecord {
  id: string;
  analysisRunId: string;
  manifestPath: string;
  outputDir: string;
  detectionIds: string[];
  status: ExportJobStatus;
  phase: ExportJobPhase;
  phaseLabel: string;
  progressDetail?: string;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  durationSeconds?: number;
  exitCode?: number;
  error?: string;
  exportedPaths: string[];
  logTail: string[];
}

const jobs = new Map<string, ExportJobRecord>();

function persistJob(record: ExportJobRecord): void {
  fs.writeFileSync(exportJobStatePath(record.id), JSON.stringify(record, null, 2));
}

function loadManifest(manifestPath: string): SessionManifest {
  return JSON.parse(fs.readFileSync(manifestPath, "utf-8")) as SessionManifest;
}

function sanitizeStem(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 48) || "session";
}

export function getExportJob(jobId: string): ExportJobRecord | null {
  const existing = jobs.get(jobId);
  if (existing) return existing;
  const persistedPath = exportJobStatePath(jobId);
  if (!fs.existsSync(persistedPath)) return null;
  const record = JSON.parse(fs.readFileSync(persistedPath, "utf-8")) as ExportJobRecord;
  jobs.set(jobId, record);
  return record;
}

export function listExportedPathsForAnalysisRun(analysisRunId: string): string[] {
  ensureRuntimeDirs();
  const records = fs.readdirSync(exportJobsRoot)
    .filter((entry) => entry.endsWith(".json"))
    .map((entry) => {
      const filePath = path.join(exportJobsRoot, entry);
      try {
        return JSON.parse(fs.readFileSync(filePath, "utf-8")) as ExportJobRecord;
      } catch {
        return null;
      }
    })
    .filter((record): record is ExportJobRecord =>
      Boolean(record) && record.analysisRunId === analysisRunId && record.status === "completed" && record.exportedPaths.length > 0
    )
    .sort((left, right) => {
      const leftTime = Date.parse(left.finishedAt ?? left.createdAt ?? "") || 0;
      const rightTime = Date.parse(right.finishedAt ?? right.createdAt ?? "") || 0;
      return rightTime - leftTime;
    });

  const dedupedByName = new Map<string, string>();
  for (const record of records) {
    for (const exportedPath of record.exportedPaths) {
      const basename = path.basename(exportedPath);
      if (!dedupedByName.has(basename)) {
        dedupedByName.set(basename, exportedPath);
      }
    }
  }

  return Array.from(dedupedByName.values());
}

export function startExportJob(analysisRunId: string, manifestPath: string, detectionIds: string[]): ExportJobRecord {
  const manifest = loadManifest(manifestPath);
  const selectedDetections = manifest.detections.filter((item) => detectionIds.includes(item.id));
  if (selectedDetections.length === 0) {
    throw new Error("No selected detections were found in the session manifest.");
  }

  const timestamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
  const outputDir = path.join(manifest.session.output_dir, "exports", `kept_${timestamp}`);
  fs.mkdirSync(outputDir, { recursive: true });

  const record: ExportJobRecord = {
    id: `export_${timestamp}_${sanitizeStem(analysisRunId)}`,
    analysisRunId,
    manifestPath,
    outputDir,
    detectionIds,
    status: "queued",
    phase: "queued",
    phaseLabel: "Queued",
    progressDetail: "Waiting to export kept clips.",
    createdAt: new Date().toISOString(),
    exportedPaths: [],
    logTail: [],
  };

  ensureRuntimeDirs();
  jobs.set(record.id, record);
  persistJob(record);

  const startMs = Date.now();
  const child = spawn("ffmpeg", ["-version"], { stdio: ["ignore", "pipe", "pipe"] });
  child.on("error", (error) => {
    record.status = "failed";
    record.phase = "failed";
    record.phaseLabel = "Failed";
    record.progressDetail = error.message;
    record.finishedAt = new Date().toISOString();
    record.durationSeconds = (Date.now() - startMs) / 1000;
    record.error = error.message;
    persistJob(record);
  });
  child.on("close", (code) => {
    if (code !== 0) {
      record.status = "failed";
      record.phase = "failed";
      record.phaseLabel = "Failed";
      record.progressDetail = "ffmpeg is not available for export.";
      record.finishedAt = new Date().toISOString();
      record.durationSeconds = (Date.now() - startMs) / 1000;
      record.error = "ffmpeg missing";
      persistJob(record);
      return;
    }

    void runExportSequence(record, manifest, selectedDetections, startMs).catch((error: unknown) => {
      record.status = "failed";
      record.phase = "failed";
      record.phaseLabel = "Failed";
      record.progressDetail = error instanceof Error ? error.message : "Export failed.";
      record.error = record.progressDetail;
      record.finishedAt = new Date().toISOString();
      record.durationSeconds = (Date.now() - startMs) / 1000;
      persistJob(record);
    });
  });

  return record;
}

async function runExportSequence(
  record: ExportJobRecord,
  manifest: SessionManifest,
  detections: SessionManifest["detections"],
  startMs: number,
): Promise<void> {
  const proxySourcePath = path.join(manifest.session.output_dir, "web", "session_source_review.mp4");
  const exportSourcePath = fs.existsSync(proxySourcePath) ? proxySourcePath : manifest.session.source_video_path;
  record.status = "running";
  record.phase = "preparing";
  record.phaseLabel = "Preparing exports";
  record.progressDetail = `${detections.length} kept clip${detections.length === 1 ? "" : "s"} queued for export from ${fs.existsSync(proxySourcePath) ? "the review proxy" : "the source video"}.`;
  record.startedAt = new Date().toISOString();
  persistJob(record);

  for (const [index, detection] of detections.entries()) {
    const outputPath = path.join(record.outputDir, detection.clip.filename ?? `${detection.id}.mp4`);
    record.phase = "exporting";
    record.phaseLabel = "Exporting clips";
    record.progressDetail = `Rendering ${index + 1}/${detections.length}: ${detection.id}`;
    record.logTail.push(JSON.stringify({ event: "export_start", detectionId: detection.id, outputPath }));
    record.logTail = record.logTail.slice(-40);
    persistJob(record);

    await runFfmpegExport(exportSourcePath, outputPath, detection.start_time_seconds, detection.end_time_seconds);

    record.exportedPaths.push(outputPath);
    record.logTail.push(JSON.stringify({ event: "export_complete", detectionId: detection.id, outputPath }));
    record.logTail = record.logTail.slice(-40);
    persistJob(record);
  }

  record.status = "completed";
  record.phase = "completed";
  record.phaseLabel = "Completed";
  record.progressDetail = `${record.exportedPaths.length} clip${record.exportedPaths.length === 1 ? "" : "s"} exported.`;
  record.finishedAt = new Date().toISOString();
  record.durationSeconds = (Date.now() - startMs) / 1000;
  persistJob(record);
}

function runFfmpegExport(videoPath: string, outputPath: string, startTime: number, endTime: number): Promise<void> {
  const duration = Math.max(0.25, Number(endTime) - Number(startTime));
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", [
      "-v",
      "error",
      "-nostdin",
      "-y",
      "-ss",
      `${Number(startTime).toFixed(3)}`,
      "-i",
      videoPath,
      "-t",
      `${duration.toFixed(3)}`,
      "-c:v",
      "libx264",
      "-profile:v",
      "high",
      "-level:v",
      "4.1",
      "-pix_fmt",
      "yuv420p",
      "-preset",
      "veryfast",
      "-crf",
      "22",
      "-c:a",
      "aac",
      "-b:a",
      "160k",
      "-movflags",
      "+faststart",
      outputPath,
    ]);

    let stderr = "";
    ffmpeg.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf-8");
    });
    ffmpeg.on("error", reject);
    ffmpeg.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(stderr.trim() || `ffmpeg exited with code ${code ?? "unknown"}`));
    });
  });
}
