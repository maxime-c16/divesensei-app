import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { allowedMediaRoots, ensureRuntimeDirs, isAllowedLocalPath, jobStatePath, outputsRoot, preferredPython, repoRoot } from "@/lib/runtime-config";
import { registerSessionManifest } from "@/lib/session-catalog";

export type AnalysisJobStatus = "queued" | "running" | "completed" | "failed";
export type AnalysisJobPhase =
  | "queued"
  | "scanning-audio"
  | "review-ready"
  | "building-review-proxy"
  | "finalizing"
  | "completed"
  | "failed";

export interface AnalysisJobRecord {
  id: string;
  videoPath: string;
  sessionName?: string;
  profile: string;
  detectorId: string;
  status: AnalysisJobStatus;
  phase?: AnalysisJobPhase;
  phaseLabel?: string;
  progressDetail?: string;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  outputDir: string;
  sessionId: string;
  manifestPath: string;
  durationSeconds?: number;
  exitCode?: number;
  error?: string;
  logTail: string[];
  logMtimeMs?: number;
}

const jobs = new Map<string, AnalysisJobRecord>();

interface ProgressSnapshot {
  phase: AnalysisJobPhase;
  phaseLabel: string;
  progressDetail?: string;
}

function formatSeconds(value: unknown): string | null {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue) || numberValue < 0) return null;
  if (numberValue >= 60) {
    const minutes = Math.floor(numberValue / 60);
    const seconds = Math.round(numberValue % 60);
    return `${minutes}m ${seconds}s`;
  }
  return `${Math.round(numberValue)}s`;
}

function persistJob(record: AnalysisJobRecord): void {
  fs.writeFileSync(jobStatePath(record.id), JSON.stringify(record, null, 2));
}

function deriveFailureMessage(record: AnalysisJobRecord): string | null {
  const lines = [...(record.logTail ?? [])].reverse();
  for (const line of lines) {
    const trimmed = String(line ?? "").trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("{")) continue;
    if (/^traceback/i.test(trimmed)) continue;
    if (trimmed.includes("RuntimeError:")) return trimmed;
    if (trimmed.includes("Error:")) return trimmed;
  }
  return null;
}

function sanitizeStem(videoPath: string): string {
  return path.basename(videoPath, path.extname(videoPath)).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 48) || "session";
}

export function listAllowedRoots(): string[] {
  return allowedMediaRoots.slice();
}

function applySnapshot(record: AnalysisJobRecord, snapshot: ProgressSnapshot): void {
  record.phase = snapshot.phase;
  record.phaseLabel = snapshot.phaseLabel;
  record.progressDetail = snapshot.progressDetail;
}

function parseProgressLog(record: AnalysisJobRecord): void {
  const logPath = path.join(record.outputDir, "session_pipeline.log.jsonl");
  if (!fs.existsSync(logPath)) return;
  const logStat = fs.statSync(logPath);
  if (record.logMtimeMs === logStat.mtimeMs && record.logTail.length > 0) return;

  const lines = fs.readFileSync(logPath, "utf-8").split("\n").filter(Boolean);
  record.logTail = lines.slice(-40);
  record.logMtimeMs = logStat.mtimeMs;

  let lastSnapshot: ProgressSnapshot | null = null;
  for (const line of lines) {
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }

    const event = String(parsed.event ?? "");
    if (event === "session_start" || event === "detection_start") {
      lastSnapshot = {
        phase: "scanning-audio",
        phaseLabel: "Analyzing video",
        progressDetail: "Scanning the video for dive attempts.",
      };
    } else if (event === "audio_decode_started") {
      const timeout = formatSeconds(parsed.timeout_seconds);
      lastSnapshot = {
        phase: "scanning-audio",
        phaseLabel: "Analyzing video",
        progressDetail: timeout
          ? `Decoding the audio track. Timeout window: ${timeout}.`
          : "Decoding the audio track.",
      };
    } else if (event === "audio_decode_progress") {
      const elapsed = formatSeconds(parsed.elapsed_seconds);
      const timeout = formatSeconds(parsed.timeout_seconds);
      lastSnapshot = {
        phase: "scanning-audio",
        phaseLabel: "Analyzing video",
        progressDetail: elapsed && timeout
          ? `Decoding the audio track. ${elapsed} elapsed of ${timeout} timeout window.`
          : elapsed
            ? `Decoding the audio track. ${elapsed} elapsed.`
            : "Decoding the audio track.",
      };
    } else if (event === "audio_decode_complete") {
      const elapsed = formatSeconds(parsed.elapsed_seconds);
      lastSnapshot = {
        phase: "scanning-audio",
        phaseLabel: "Analyzing video",
        progressDetail: elapsed
          ? `Audio decoded in ${elapsed}. Detecting dive attempts now.`
          : "Audio decoded. Detecting dive attempts now.",
      };
    } else if (event === "detection_complete") {
      const candidateCount = Number(parsed.candidate_count ?? 0);
      lastSnapshot = {
        phase: "scanning-audio",
        phaseLabel: "Preparing review",
        progressDetail: candidateCount > 0
          ? `${candidateCount} attempts found. Preparing the review view.`
          : "No attempts found. Preparing the review view.",
      };
    } else if (event === "clip_extraction_start") {
      const candidateCount = Number(parsed.candidate_count ?? 0);
      lastSnapshot = {
        phase: "finalizing",
        phaseLabel: "Finishing",
        progressDetail: candidateCount > 0 ? `Creating ${candidateCount} clip${candidateCount === 1 ? "" : "s"}.` : "Creating clips.",
      };
    } else if (event === "clip_extracted") {
      const completedClips = Number(parsed.completed_clips ?? 0);
      const totalClips = Number(parsed.total_clips ?? completedClips);
      lastSnapshot = {
        phase: "finalizing",
        phaseLabel: "Finishing",
        progressDetail: totalClips > 0 ? `${completedClips}/${totalClips} clips ready.` : `Clip ${completedClips} ready.`,
      };
    } else if (event === "review_ready") {
      lastSnapshot = {
        phase: "review-ready",
        phaseLabel: "Ready for review",
        progressDetail: "Attempts are ready. The review video is still being prepared.",
      };
    } else if (event === "review_proxy_start") {
      lastSnapshot = {
        phase: "building-review-proxy",
        phaseLabel: "Preparing review video",
        progressDetail: "Creating the browser-ready review video.",
      };
    } else if (event === "session_complete") {
      lastSnapshot = {
        phase: "finalizing",
        phaseLabel: "Finishing",
        progressDetail: "Saving the final session data.",
      };
    }
  }

  if (lastSnapshot) {
    applySnapshot(record, lastSnapshot);
  }
}

export function getAnalysisJob(jobId: string): AnalysisJobRecord | null {
  const inMemory = jobs.get(jobId);
  if (inMemory) {
    parseProgressLog(inMemory);
    persistJob(inMemory);
    return inMemory;
  }
  const persistedPath = jobStatePath(jobId);
  if (!fs.existsSync(persistedPath)) return null;
  const record = JSON.parse(fs.readFileSync(persistedPath, "utf-8")) as AnalysisJobRecord;
  parseProgressLog(record);
  jobs.set(jobId, record);
  return record;
}

function preferredAudioClipModelPath(): string | null {
  const candidate = path.join(repoRoot, ".divesensei-runtime", "models", "audio_clip_model.json");
  return fs.existsSync(candidate) ? candidate : null;
}

export function startAnalysisJob(videoPath: string, profile: string, detectorId: string, sessionName = ""): AnalysisJobRecord {
  const resolvedVideoPath = path.resolve(videoPath);
  if (!isAllowedLocalPath(resolvedVideoPath)) {
    throw new Error("Video location is outside the allowed folders.");
  }
  if (!fs.existsSync(resolvedVideoPath)) {
    throw new Error("Video file was not found.");
  }

  const timestamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
  const stem = sanitizeStem(resolvedVideoPath);
  const jobId = `job_${timestamp}_${stem}`;
  const sessionId = `.tmp_ui_run_${timestamp}_${stem}`;
  const outputDir = path.join(outputsRoot, sessionId);
  const manifestPath = path.join(outputDir, "ui_session_manifest.json");
  fs.mkdirSync(outputDir, { recursive: true });

  const record: AnalysisJobRecord = {
    id: jobId,
    videoPath: resolvedVideoPath,
    sessionName: sessionName.trim() || undefined,
    profile,
    detectorId,
    status: "queued",
    phase: "queued",
    phaseLabel: "Queued",
    progressDetail: "Waiting to start.",
    createdAt: new Date().toISOString(),
    outputDir,
    sessionId,
    manifestPath,
    logTail: [],
  };
  ensureRuntimeDirs();
  jobs.set(jobId, record);
  persistJob(record);

  const startMs = Date.now();
  const detectArgs = [
    "-m",
    "divesensei.cli",
    "detect",
    resolvedVideoPath,
    "--profile",
    profile,
    "--detector-id",
    detectorId,
    "--output-dir",
    outputDir,
    "--review-only",
  ];
  if (sessionName.trim()) {
    detectArgs.push("--session-name", sessionName.trim());
  }
  const clipModelPath = preferredAudioClipModelPath();
  if (clipModelPath && detectorId !== "audio_v1_heuristic") {
    detectArgs.push("--audio-clip-model-path", clipModelPath);
  }
  const child = spawn(preferredPython, detectArgs, {
    cwd: repoRoot,
    env: {
      ...process.env,
      DIVESENSEI_EMIT_PROGRESS: "1",
      PYTHONPATH: path.join(repoRoot, "src"),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  record.status = "running";
  record.phase = "scanning-audio";
  record.phaseLabel = "Analyzing video";
  record.progressDetail = "Scanning the video for dive attempts.";
  record.startedAt = new Date().toISOString();
  persistJob(record);

  const pushLog = (chunk: Buffer) => {
    const text = chunk.toString("utf-8").trim();
    if (!text) return;
    for (const line of text.split("\n")) {
      record.logTail.push(line);
      let parsed: Record<string, unknown> | null = null;
      try {
        parsed = JSON.parse(line) as Record<string, unknown>;
      } catch {
        parsed = null;
      }
      if (line.includes("\"event\": \"audio_decode_started\"")) {
        const timeout = formatSeconds(parsed?.timeout_seconds);
        applySnapshot(record, {
          phase: "scanning-audio",
          phaseLabel: "Analyzing video",
          progressDetail: timeout
            ? `Decoding the audio track. Timeout window: ${timeout}.`
            : "Decoding the audio track.",
        });
      } else if (line.includes("\"event\": \"audio_decode_progress\"")) {
        const elapsed = formatSeconds(parsed?.elapsed_seconds);
        const timeout = formatSeconds(parsed?.timeout_seconds);
        applySnapshot(record, {
          phase: "scanning-audio",
          phaseLabel: "Analyzing video",
          progressDetail: elapsed && timeout
            ? `Decoding the audio track. ${elapsed} elapsed of ${timeout} timeout window.`
            : elapsed
              ? `Decoding the audio track. ${elapsed} elapsed.`
              : "Decoding the audio track.",
        });
      } else if (line.includes("\"event\": \"audio_decode_complete\"")) {
        const elapsed = formatSeconds(parsed?.elapsed_seconds);
        applySnapshot(record, {
          phase: "scanning-audio",
          phaseLabel: "Analyzing video",
          progressDetail: elapsed
            ? `Audio decoded in ${elapsed}. Detecting dive attempts now.`
            : "Audio decoded. Detecting dive attempts now.",
        });
      } else if (line.includes("\"event\": \"detection_complete\"")) {
        const candidateCount = Number(parsed?.candidate_count ?? 0);
        applySnapshot(record, {
          phase: "scanning-audio",
          phaseLabel: "Preparing review",
          progressDetail: candidateCount > 0
            ? `Detection finished. Preparing ${candidateCount} review attempts.`
            : "Detection finished. Preparing the review view.",
        });
      } else if (line.includes("\"event\": \"clip_extracted\"")) {
        parseProgressLog(record);
      } else if (line.includes("\"event\": \"review_ready\"")) {
        applySnapshot(record, {
          phase: "review-ready",
          phaseLabel: "Ready for review",
          progressDetail: "Attempts are ready. The review video is still being prepared.",
        });
        registerSessionManifest(record.manifestPath);
      } else if (line.includes("\"event\": \"review_proxy_start\"")) {
        applySnapshot(record, {
          phase: "building-review-proxy",
          phaseLabel: "Preparing review video",
          progressDetail: "Creating the browser-ready review video.",
        });
      } else if (line.includes("\"event\": \"session_complete\"")) {
        applySnapshot(record, {
          phase: "finalizing",
          phaseLabel: "Finishing",
          progressDetail: "Saving the final session data.",
        });
        registerSessionManifest(record.manifestPath);
      }
    }
    record.logTail = record.logTail.slice(-40);
    persistJob(record);
  };

  child.stdout.on("data", pushLog);
  child.stderr.on("data", pushLog);
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
    record.exitCode = code ?? undefined;
    record.finishedAt = new Date().toISOString();
    record.durationSeconds = (Date.now() - startMs) / 1000;
    if (code === 0 && fs.existsSync(manifestPath)) {
      record.status = "completed";
      record.phase = "completed";
      record.phaseLabel = "Ready";
      record.progressDetail = "Review video is ready. You can export kept attempts.";
      registerSessionManifest(manifestPath);
      persistJob(record);
      return;
    }
    record.status = "failed";
    record.phase = "failed";
    record.phaseLabel = "Failed";
    const derived = deriveFailureMessage(record);
    if (!record.error) {
      record.error = derived ?? `Pipeline exited with code ${code ?? "unknown"}.`;
    }
    record.progressDetail = record.error;
    persistJob(record);
  });

  return record;
}
