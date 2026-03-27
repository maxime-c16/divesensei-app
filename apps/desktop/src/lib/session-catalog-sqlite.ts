import fs from "node:fs";
import path from "node:path";
import BetterSqlite3 from "better-sqlite3";
import type { ReviewDecision } from "@/types/ui";
import { ensureRuntimeDirs, sessionCatalogDbPath } from "@/lib/runtime-config";
import {
  type CatalogBackend,
  type CatalogSessionRecord,
  type CatalogState,
  includeInBootstrap,
  readManifest,
  removeGeneratedTargets,
  sortRuns,
  statFile,
  updateManifestSourceVideoPath,
} from "@/lib/session-catalog-core";
import { loadLegacyCatalogState } from "@/lib/session-catalog-json";

type Database = BetterSqlite3.Database;

const CATALOG_SCHEMA_VERSION = "1";

function openCatalog(): Database {
  ensureRuntimeDirs();
  const db = new BetterSqlite3(sessionCatalogDbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.pragma("busy_timeout = 5000");
  db.exec(`
    CREATE TABLE IF NOT EXISTS catalog_meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS media_sources (
      id TEXT PRIMARY KEY,
      source_path TEXT NOT NULL,
      source_name TEXT NOT NULL,
      file_size_bytes INTEGER,
      modified_time_ms INTEGER,
      profile_hint TEXT,
      availability_status TEXT NOT NULL DEFAULT 'available',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_seen_at TEXT
    );

    CREATE UNIQUE INDEX IF NOT EXISTS media_sources_source_path_idx
      ON media_sources(source_path);

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
  initializeCatalogMeta(db);
  migrateLegacyJsonIfNeeded(db);
  reconcileCatalog(db);
  return db;
}

function initializeCatalogMeta(db: Database): void {
  const now = new Date().toISOString();
  db.prepare(`
    INSERT INTO catalog_meta (key, value)
    VALUES ('schema_version', ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value
  `).run(CATALOG_SCHEMA_VERSION);
  db.prepare(`
    INSERT INTO catalog_meta (key, value)
    VALUES ('last_opened_at', ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value
  `).run(now);
}

function catalogIsEmpty(db: Database): boolean {
  const row = db.prepare("SELECT COUNT(*) AS count FROM analysis_runs").get() as { count: number };
  return (row?.count ?? 0) === 0;
}

function migrateLegacyJsonIfNeeded(db: Database): void {
  if (!catalogIsEmpty(db)) return;
  const legacy = loadLegacyCatalogState();
  if (legacy.analysisRuns.length === 0 && legacy.mediaSources.length === 0 && legacy.reviewDecisions.length === 0) {
    return;
  }
  const tx = db.transaction((state: CatalogState) => {
    for (const source of state.mediaSources) {
      db.prepare(`
        INSERT OR REPLACE INTO media_sources (
          id, source_path, source_name, file_size_bytes, modified_time_ms, profile_hint,
          availability_status, created_at, updated_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        source.id,
        source.sourcePath,
        source.sourceName,
        source.fileSizeBytes,
        source.modifiedTimeMs,
        source.profileHint ?? null,
        source.availabilityStatus,
        source.createdAt,
        source.updatedAt,
        source.lastSeenAt,
      );
    }

    for (const run of state.analysisRuns) {
      db.prepare(`
        INSERT OR REPLACE INTO analysis_runs (
          id, media_source_id, title, profile, manifest_path, output_dir, status,
          candidate_count, extracted_count, started_at, finished_at, last_opened_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        run.id,
        run.mediaSourceId,
        run.title,
        run.profile,
        run.manifestPath,
        run.outputDir,
        run.status,
        run.candidateCount,
        run.extractedCount,
        run.startedAt,
        run.finishedAt,
        run.lastOpenedAt,
        run.createdAt,
        run.updatedAt,
      );
    }

    for (const decision of state.reviewDecisions) {
      db.prepare(`
        INSERT OR REPLACE INTO review_decisions (
          id, analysis_run_id, detection_id, label, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
      `).run(
        decision.id,
        decision.analysisRunId,
        decision.detectionId,
        decision.label,
        decision.notes,
        decision.createdAt,
        decision.updatedAt,
      );
    }
  });
  tx(legacy);
}

function registerSessionManifestWithDb(db: Database, manifestPath: string): void {
  const manifest = readManifest(manifestPath);
  if (!manifest) return;

  const now = new Date().toISOString();
  const sourcePath = manifest.session.source_video_path;
  const sourceStat = statFile(sourcePath);
  const mediaSourceId =
    (db.prepare(`
      SELECT media_source_id AS mediaSourceId
      FROM analysis_runs
      WHERE manifest_path = ? OR output_dir = ? OR id = ?
      LIMIT 1
    `).get(manifestPath, manifest.session.output_dir, manifest.session.id) as { mediaSourceId?: string } | undefined)?.mediaSourceId
    ?? (db.prepare(`
      SELECT id
      FROM media_sources
      WHERE source_path = ?
      LIMIT 1
    `).get(sourcePath) as { id?: string } | undefined)?.id
    ?? sourcePath;

  db.prepare(`
    INSERT INTO media_sources (
      id, source_path, source_name, file_size_bytes, modified_time_ms, profile_hint,
      availability_status, created_at, updated_at, last_seen_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      source_path = excluded.source_path,
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
    fs.existsSync(sourcePath) ? now : null,
  );

  db.prepare(`
    INSERT INTO analysis_runs (
      id, media_source_id, title, profile, manifest_path, output_dir, status,
      candidate_count, extracted_count, started_at, finished_at, last_opened_at, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    (db.prepare("SELECT last_opened_at AS lastOpenedAt FROM analysis_runs WHERE id = ?").get(manifest.session.id) as { lastOpenedAt?: string } | undefined)?.lastOpenedAt ?? null,
    (db.prepare("SELECT created_at AS createdAt FROM analysis_runs WHERE id = ?").get(manifest.session.id) as { createdAt?: string } | undefined)?.createdAt ?? (manifest.session.created_at ?? now),
    manifest.session.updated_at ?? now,
  );
}

function reconcileCatalog(db: Database): void {
  const runs = db.prepare(`
    SELECT id, manifest_path AS manifestPath, output_dir AS outputDir, media_source_id AS mediaSourceId
    FROM analysis_runs
  `).all() as Array<{ id: string; manifestPath: string; outputDir: string; mediaSourceId: string }>;

  const tx = db.transaction(() => {
    for (const run of runs) {
      if (!fs.existsSync(run.manifestPath) || !readManifest(run.manifestPath)) {
        db.prepare("DELETE FROM analysis_runs WHERE id = ?").run(run.id);
        continue;
      }
      registerSessionManifestWithDb(db, run.manifestPath);
    }

    db.prepare(`
      DELETE FROM review_decisions
      WHERE analysis_run_id NOT IN (SELECT id FROM analysis_runs)
    `).run();

    db.prepare(`
      DELETE FROM media_sources
      WHERE id NOT IN (SELECT DISTINCT media_source_id FROM analysis_runs)
    `).run();
  });
  tx();
}

function registerDiscoveredManifests(db: Database, discoveredManifestPaths: string[]): void {
  const tx = db.transaction(() => {
    for (const manifestPath of discoveredManifestPaths.filter(includeInBootstrap)) {
      registerSessionManifestWithDb(db, manifestPath);
    }
  });
  tx();
}

function toCatalogSessionRows(db: Database): CatalogSessionRecord[] {
  return db.prepare(`
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
  `).all() as CatalogSessionRecord[];
}

export const sqliteCatalogBackend: CatalogBackend = {
  kind: "sqlite",

  registerSessionManifest(manifestPath) {
    const db = openCatalog();
    registerSessionManifestWithDb(db, manifestPath);
    db.close();
  },

  listCatalogSessions() {
    const db = openCatalog();
    const rows = toCatalogSessionRows(db);
    db.close();
    return rows;
  },

  saveReviewDecision(analysisRunId, detectionId, label, notes = "") {
    const db = openCatalog();
    const run = db.prepare("SELECT id FROM analysis_runs WHERE id = ?").get(analysisRunId) as { id?: string } | undefined;
    if (!run) {
      db.close();
      throw new Error("Analysis run not found.");
    }
    const existing = db.prepare(`
      SELECT created_at AS createdAt
      FROM review_decisions
      WHERE analysis_run_id = ? AND detection_id = ?
    `).get(analysisRunId, detectionId) as { createdAt?: string } | undefined;
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
  },

  listReviewDecisions(analysisRunId) {
    const db = openCatalog();
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
    `).all(analysisRunId) as ReviewDecision[];
    db.close();
    return rows;
  },

  markSessionOpened(analysisRunId) {
    const db = openCatalog();
    const now = new Date().toISOString();
    db.prepare(`
      UPDATE analysis_runs
      SET last_opened_at = ?, updated_at = ?
      WHERE id = ?
    `).run(now, now, analysisRunId);
    db.close();
  },

  renameSessionRun(analysisRunId, nextName) {
    const trimmed = nextName.trim();
    if (!trimmed) {
      throw new Error("Session name cannot be empty.");
    }
    const db = openCatalog();
    const run = db.prepare(`
      SELECT manifest_path AS manifestPath
      FROM analysis_runs
      WHERE id = ?
    `).get(analysisRunId) as { manifestPath?: string } | undefined;
    if (!run?.manifestPath) {
      db.close();
      throw new Error("Analysis run not found.");
    }
    const now = new Date().toISOString();
    db.prepare(`
      UPDATE analysis_runs
      SET title = ?, updated_at = ?
      WHERE id = ?
    `).run(trimmed, now, analysisRunId);
    const manifest = readManifest(run.manifestPath);
    if (manifest) {
      manifest.session.title = trimmed;
      manifest.session.session_name = trimmed;
      manifest.session.updated_at = now;
      fs.writeFileSync(run.manifestPath, JSON.stringify(manifest, null, 2));
      registerSessionManifestWithDb(db, run.manifestPath);
    }
    db.close();
  },

  getManifestPathForAnalysisRun(analysisRunId) {
    const db = openCatalog();
    const row = db.prepare(`
      SELECT manifest_path AS manifestPath
      FROM analysis_runs
      WHERE id = ?
    `).get(analysisRunId) as { manifestPath?: string } | undefined;
    db.close();
    return row?.manifestPath ?? null;
  },

  deleteSessionRun(analysisRunId) {
    const db = openCatalog();
    const run = db.prepare(`
      SELECT
        id,
        output_dir AS outputDir,
        manifest_path AS manifestPath,
        media_source_id AS mediaSourceId
      FROM analysis_runs
      WHERE id = ?
    `).get(analysisRunId) as { id?: string; outputDir: string; manifestPath: string; mediaSourceId: string } | undefined;
    if (!run?.id) {
      db.close();
      return false;
    }

    removeGeneratedTargets(analysisRunId, run);
    const tx = db.transaction(() => {
      db.prepare("DELETE FROM analysis_runs WHERE id = ?").run(analysisRunId);
      const remaining = db.prepare(`
        SELECT COUNT(*) AS count
        FROM analysis_runs
        WHERE media_source_id = ?
      `).get(run.mediaSourceId) as { count?: number };
      if ((remaining?.count ?? 0) === 0) {
        db.prepare("DELETE FROM media_sources WHERE id = ?").run(run.mediaSourceId);
      }
    });
    tx();
    db.close();
    return true;
  },

  refreshCatalogAvailability() {
    const db = openCatalog();
    const now = new Date().toISOString();
    const rows = db.prepare(`
      SELECT id, source_path AS sourcePath
      FROM media_sources
    `).all() as Array<{ id: string; sourcePath: string }>;
    const tx = db.transaction(() => {
      for (const row of rows) {
        const stat = statFile(row.sourcePath);
        const exists = fs.existsSync(row.sourcePath);
        db.prepare(`
          UPDATE media_sources
          SET
            file_size_bytes = ?,
            modified_time_ms = ?,
            availability_status = ?,
            updated_at = ?,
            last_seen_at = ?
          WHERE id = ?
        `).run(stat.size, stat.mtimeMs, exists ? "available" : "missing", now, exists ? now : null, row.id);
      }
    });
    tx();
    const sessions = toCatalogSessionRows(db);
    db.close();
    return sessions;
  },

  relinkCatalogSource(mediaSourceId, nextPath) {
    const resolvedPath = path.resolve(nextPath);
    if (!fs.existsSync(resolvedPath)) {
      throw new Error("Target path does not exist.");
    }

    const db = openCatalog();
    const source = db.prepare(`
      SELECT id
      FROM media_sources
      WHERE id = ?
    `).get(mediaSourceId) as { id?: string } | undefined;
    if (!source?.id) {
      db.close();
      throw new Error("Media source not found.");
    }

    const now = new Date().toISOString();
    const stat = statFile(resolvedPath);
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

    const manifests = db.prepare(`
      SELECT manifest_path AS manifestPath
      FROM analysis_runs
      WHERE media_source_id = ?
    `).all(mediaSourceId) as Array<{ manifestPath: string }>;
    db.close();

    for (const item of manifests) {
      updateManifestSourceVideoPath(item.manifestPath, resolvedPath);
    }
  },

  resolveCatalogManifestPaths(discoveredManifestPaths) {
    const db = openCatalog();
    registerDiscoveredManifests(db, discoveredManifestPaths);
    const manifestPaths = sortRuns(
      (db.prepare(`
        SELECT
          manifest_path AS manifestPath,
          updated_at AS updatedAt,
          last_opened_at AS lastOpenedAt
        FROM analysis_runs
      `).all() as Array<{ manifestPath: string; updatedAt: string; lastOpenedAt: string | null }>)
    )
      .map((entry) => entry.manifestPath)
      .filter((manifestPath) => fs.existsSync(manifestPath));
    db.close();
    return manifestPaths;
  },
};
