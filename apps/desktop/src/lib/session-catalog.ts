import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ReviewDecision, SessionManifest } from "@/types/ui";
import { ensureRuntimeDirs, outputsRoot, repoRoot, sessionCatalogDbPath, sessionCatalogPath } from "@/lib/runtime-config";

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

function openCatalog(): DatabaseSync {
  ensureRuntimeDirs();
  const db = new DatabaseSync(sessionCatalogDbPath);
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS media_sources (
      id TEXT PRIMARY KEY,
      source_path TEXT NOT NULL UNIQUE,
      source_name TEXT NOT NULL,
      file_size_bytes INTEGER,
      modified_time_ms INTEGER,
      profile_hint TEXT,
      availability_status TEXT NOT NULL DEFAULT 'available',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_seen_at TEXT
    );

    CREATE TABLE IF NOT EXISTS analysis_runs (
      id TEXT PRIMARY KEY,
      media_source_id TEXT NOT NULL,
      title TEXT NOT NULL,
      profile TEXT NOT NULL,
      manifest_path TEXT NOT NULL UNIQUE,
      output_dir TEXT NOT NULL UNIQUE,
      status TEXT NOT NULL,
      candidate_count INTEGER NOT NULL DEFAULT 0,
      extracted_count INTEGER NOT NULL DEFAULT 0,
      started_at TEXT,
      finished_at TEXT,
      last_opened_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY (media_source_id) REFERENCES media_sources(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS review_decisions (
      id TEXT PRIMARY KEY,
      analysis_run_id TEXT NOT NULL,
      detection_id TEXT NOT NULL,
      label TEXT NOT NULL,
      notes TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(analysis_run_id, detection_id),
      FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
    );
  `);
  return db;
}

function readManifest(manifestPath: string): SessionManifest | null {
  try {
    return JSON.parse(fs.readFileSync(manifestPath, "utf-8")) as SessionManifest;
  } catch {
    return null;
  }
}

function statFile(filePath: string): { size: number | null; mtimeMs: number | null } {
  try {
    const stat = fs.statSync(filePath);
    return { size: stat.size, mtimeMs: stat.mtimeMs };
  } catch {
    return { size: null, mtimeMs: null };
  }
}

function includeInBootstrap(manifestPath: string): boolean {
  const dirName = path.basename(path.dirname(manifestPath));
  return dirName.startsWith(".tmp_ui_run_") || manifestPath.startsWith(outputsRoot) || manifestPath.startsWith(repoRoot);
}

function migrateLegacyJsonIfPresent(db: DatabaseSync): void {
  if (!fs.existsSync(sessionCatalogPath)) return;
  try {
    const raw = JSON.parse(fs.readFileSync(sessionCatalogPath, "utf-8")) as {
      sessions?: Array<{ manifestPath?: string }>;
    };
    for (const entry of raw.sessions ?? []) {
      if (entry.manifestPath && fs.existsSync(entry.manifestPath)) {
        registerSessionManifest(entry.manifestPath, db);
      }
    }
    fs.renameSync(sessionCatalogPath, `${sessionCatalogPath}.migrated`);
  } catch {
    // Ignore invalid transitional files.
  }
}

function refreshAvailabilityForPath(db: DatabaseSync, sourcePath: string): void {
  const now = new Date().toISOString();
  const stat = statFile(sourcePath);
  db.prepare(`
    UPDATE media_sources
    SET
      file_size_bytes = ?,
      modified_time_ms = ?,
      availability_status = ?,
      updated_at = ?,
      last_seen_at = ?
    WHERE source_path = ?
  `).run(
    stat.size,
    stat.mtimeMs,
    fs.existsSync(sourcePath) ? "available" : "missing",
    now,
    fs.existsSync(sourcePath) ? now : null,
    sourcePath,
  );
}

export function registerSessionManifest(manifestPath: string, existingDb?: DatabaseSync): void {
  const manifest = readManifest(manifestPath);
  if (!manifest) return;

  const db = existingDb ?? openCatalog();
  migrateLegacyJsonIfPresent(db);
  const now = new Date().toISOString();
  const sourcePath = manifest.session.source_video_path;
  const sourceStat = statFile(sourcePath);
  const mediaSourceId = sourcePath;

  db.prepare(`
    INSERT INTO media_sources (
      id, source_path, source_name, file_size_bytes, modified_time_ms, profile_hint,
      availability_status, created_at, updated_at, last_seen_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(source_path) DO UPDATE SET
      source_name = excluded.source_name,
      file_size_bytes = excluded.file_size_bytes,
      modified_time_ms = excluded.modified_time_ms,
      profile_hint = excluded.profile_hint,
      availability_status = excluded.availability_status,
      updated_at = excluded.updated_at,
      last_seen_at = excluded.last_seen_at
  `).run(
    mediaSourceId,
    sourcePath,
    path.basename(sourcePath),
    sourceStat.size,
    sourceStat.mtimeMs,
    manifest.session.profile,
    fs.existsSync(sourcePath) ? "available" : "missing",
    now,
    now,
    now,
  );

  db.prepare(`
    INSERT INTO analysis_runs (
      id, media_source_id, title, profile, manifest_path, output_dir, status,
      candidate_count, extracted_count, started_at, finished_at, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      media_source_id = excluded.media_source_id,
      title = excluded.title,
      profile = excluded.profile,
      manifest_path = excluded.manifest_path,
      output_dir = excluded.output_dir,
      status = excluded.status,
      candidate_count = excluded.candidate_count,
      extracted_count = excluded.extracted_count,
      started_at = excluded.started_at,
      finished_at = excluded.finished_at,
      updated_at = excluded.updated_at
  `).run(
    manifest.session.id,
    mediaSourceId,
    manifest.session.session_name ?? manifest.session.title,
    manifest.session.profile,
    manifestPath,
    manifest.session.output_dir,
    manifest.session.status,
    manifest.session.candidate_count,
    manifest.session.extracted_count,
    manifest.session.created_at ?? manifest.generated_at,
    manifest.session.updated_at ?? manifest.generated_at,
    manifest.session.created_at ?? now,
    manifest.session.updated_at ?? now,
  );

  if (!existingDb) db.close();
}

export function listCatalogSessions(): CatalogSessionRecord[] {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const rows = db.prepare(`
    SELECT
      ar.id AS sessionId,
      ar.media_source_id AS mediaSourceId,
      ar.title,
      ar.title AS sessionName,
      ar.profile,
      ar.manifest_path AS manifestPath,
      ar.output_dir AS outputDir,
      ar.status,
      ms.source_path AS sourceVideoPath,
      ms.availability_status AS sourceAvailability,
      ms.source_name AS sourceName,
      ms.file_size_bytes AS fileSizeBytes,
      ms.modified_time_ms AS modifiedTimeMs,
      ar.candidate_count AS candidateCount,
      ar.extracted_count AS extractedCount,
      ar.created_at AS createdAt,
      ar.updated_at AS updatedAt,
      ar.last_opened_at AS lastOpenedAt
    FROM analysis_runs ar
    JOIN media_sources ms ON ms.id = ar.media_source_id
    ORDER BY COALESCE(ar.last_opened_at, ar.updated_at) DESC, ar.updated_at DESC
  `).all() as unknown as CatalogSessionRecord[];
  db.close();
  return rows;
}

export function saveReviewDecision(
  analysisRunId: string,
  detectionId: string,
  label: ReviewDecision["label"],
  notes = "",
): ReviewDecision {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const run = db.prepare("SELECT id FROM analysis_runs WHERE id = ?").get(analysisRunId) as { id: string } | undefined;
  if (!run) {
    db.close();
    throw new Error("Analysis run not found.");
  }

  const existing = db.prepare(`
    SELECT created_at AS createdAt
    FROM review_decisions
    WHERE analysis_run_id = ? AND detection_id = ?
  `).get(analysisRunId, detectionId) as { createdAt: string } | undefined;

  const now = new Date().toISOString();
  const id = `${analysisRunId}:${detectionId}`;
  db.prepare(`
    INSERT INTO review_decisions (
      id, analysis_run_id, detection_id, label, notes, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(analysis_run_id, detection_id) DO UPDATE SET
      label = excluded.label,
      notes = excluded.notes,
      updated_at = excluded.updated_at
  `).run(id, analysisRunId, detectionId, label, notes, existing?.createdAt ?? now, now);
  db.close();

  return {
    id,
    analysisRunId,
    detectionId,
    label,
    notes,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  };
}

export function listReviewDecisions(analysisRunId: string): ReviewDecision[] {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const rows = db.prepare(`
    SELECT
      id,
      analysis_run_id AS analysisRunId,
      detection_id AS detectionId,
      label,
      notes,
      created_at AS createdAt,
      updated_at AS updatedAt
    FROM review_decisions
    WHERE analysis_run_id = ?
    ORDER BY updated_at DESC, detection_id ASC
  `).all(analysisRunId) as unknown as ReviewDecision[];
  db.close();
  return rows;
}

export function markSessionOpened(analysisRunId: string): void {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const now = new Date().toISOString();
  db.prepare(`
    UPDATE analysis_runs
    SET last_opened_at = ?, updated_at = ?
    WHERE id = ?
  `).run(now, now, analysisRunId);
  db.close();
}

export function renameSessionRun(analysisRunId: string, nextName: string): void {
  const trimmed = nextName.trim();
  if (!trimmed) {
    throw new Error("Session name cannot be empty.");
  }

  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const run = db.prepare(`
    SELECT manifest_path AS manifestPath
    FROM analysis_runs
    WHERE id = ?
  `).get(analysisRunId) as { manifestPath: string } | undefined;

  if (!run) {
    db.close();
    throw new Error("Analysis run not found.");
  }

  const now = new Date().toISOString();
  db.prepare(`
    UPDATE analysis_runs
    SET title = ?, updated_at = ?
    WHERE id = ?
  `).run(trimmed, now, analysisRunId);
  db.close();

  const manifest = readManifest(run.manifestPath);
  if (!manifest) return;
  manifest.session.session_name = trimmed;
  manifest.session.updated_at = now;
  fs.writeFileSync(run.manifestPath, JSON.stringify(manifest, null, 2));
}

export function getManifestPathForAnalysisRun(analysisRunId: string): string | null {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const row = db.prepare(`
    SELECT manifest_path AS manifestPath
    FROM analysis_runs
    WHERE id = ?
  `).get(analysisRunId) as { manifestPath: string } | undefined;
  db.close();
  return row?.manifestPath ?? null;
}

export function deleteSessionRun(analysisRunId: string): boolean {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const run = db.prepare(`
    SELECT
      id,
      output_dir AS outputDir,
      manifest_path AS manifestPath,
      media_source_id AS mediaSourceId
    FROM analysis_runs
    WHERE id = ?
  `).get(analysisRunId) as { id: string; outputDir: string; manifestPath: string; mediaSourceId: string } | undefined;

  if (!run) {
    db.close();
    return false;
  }

  const cleanupTargets = [run.outputDir];
  if (run.manifestPath && !run.manifestPath.startsWith(`${run.outputDir}${path.sep}`)) {
    cleanupTargets.push(run.manifestPath);
  }

  for (const target of cleanupTargets) {
    if (!target || !fs.existsSync(target)) continue;
    try {
      fs.rmSync(target, { recursive: true, force: true, maxRetries: 2 });
    } catch (error) {
      db.close();
      throw new Error(`Could not remove generated files for ${analysisRunId}: ${error instanceof Error ? error.message : "unknown cleanup error"}`);
    }
  }

  db.prepare("DELETE FROM analysis_runs WHERE id = ?").run(analysisRunId);
  const remaining = db.prepare(`
    SELECT COUNT(*) AS count
    FROM analysis_runs
    WHERE media_source_id = ?
  `).get(run.mediaSourceId) as { count: number };

  if ((remaining?.count ?? 0) === 0) {
    db.prepare("DELETE FROM media_sources WHERE id = ?").run(run.mediaSourceId);
  }
  db.close();
  return true;
}

export function refreshCatalogAvailability(): CatalogSessionRecord[] {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const rows = db.prepare("SELECT source_path AS sourcePath FROM media_sources").all() as Array<{ sourcePath: string }>;
  for (const row of rows) {
    refreshAvailabilityForPath(db, row.sourcePath);
  }
  db.close();
  return listCatalogSessions();
}

export function relinkCatalogSource(mediaSourceId: string, nextPath: string): void {
  const resolvedPath = path.resolve(nextPath);
  if (!fs.existsSync(resolvedPath)) {
    throw new Error("Target path does not exist.");
  }

  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);
  const row = db.prepare(`
    SELECT id
    FROM media_sources
    WHERE id = ?
  `).get(mediaSourceId) as { id: string } | undefined;

  if (!row) {
    db.close();
    throw new Error("Media source not found.");
  }

  const stat = statFile(resolvedPath);
  const now = new Date().toISOString();
  db.prepare(`
    UPDATE media_sources
    SET
      source_path = ?,
      source_name = ?,
      file_size_bytes = ?,
      modified_time_ms = ?,
      availability_status = 'available',
      updated_at = ?,
      last_seen_at = ?
    WHERE id = ?
  `).run(
    resolvedPath,
    path.basename(resolvedPath),
    stat.size,
    stat.mtimeMs,
    now,
    now,
    mediaSourceId,
  );

  const manifestRows = db.prepare(`
    SELECT manifest_path AS manifestPath
    FROM analysis_runs
    WHERE media_source_id = ?
  `).all(mediaSourceId) as Array<{ manifestPath: string }>;
  db.close();

  for (const manifestRow of manifestRows) {
    const manifest = readManifest(manifestRow.manifestPath);
    if (!manifest) continue;
    manifest.session.source_video_path = resolvedPath;
    fs.writeFileSync(manifestRow.manifestPath, JSON.stringify(manifest, null, 2));
  }
}

export function resolveCatalogManifestPaths(discoveredManifestPaths: string[]): string[] {
  const db = openCatalog();
  migrateLegacyJsonIfPresent(db);

  const catalogPaths = db
    .prepare(`
      SELECT manifest_path
      FROM analysis_runs
      ORDER BY COALESCE(last_opened_at, updated_at) DESC, updated_at DESC
    `)
    .all() as Array<{ manifest_path: string }>;

  const existingCatalogPaths = catalogPaths
    .map((entry) => entry.manifest_path)
    .filter((manifestPath) => fs.existsSync(manifestPath));

  if (existingCatalogPaths.length > 0) {
    db.close();
    return existingCatalogPaths;
  }

  const bootstrapCandidates = discoveredManifestPaths.filter(includeInBootstrap);
  for (const manifestPath of bootstrapCandidates) {
    registerSessionManifest(manifestPath, db);
  }

  const bootstrappedPaths = (db
    .prepare(`
      SELECT manifest_path
      FROM analysis_runs
      ORDER BY updated_at DESC
    `)
    .all() as Array<{ manifest_path: string }>)
    .map((entry) => entry.manifest_path)
    .filter((manifestPath) => fs.existsSync(manifestPath));

  db.close();
  return bootstrappedPaths;
}
