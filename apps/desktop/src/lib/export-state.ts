import fs from "node:fs";
import path from "node:path";

export interface ExportedClipRecord {
  detectionId: string;
  path: string;
  fileName: string;
  batchName: string;
  mtimeMs: number;
}

export interface SessionExportState {
  exportedClips: ExportedClipRecord[];
  exportedPaths: string[];
  exportedDetectionIds: string[];
  exportedCount: number;
}

function readExportBatchPaths(exportsRoot: string): string[] {
  if (!exportsRoot || !fs.existsSync(exportsRoot)) return [];
  const batchDirs = fs.readdirSync(exportsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(exportsRoot, entry.name));
  const clipPaths: string[] = [];
  for (const batchDir of batchDirs) {
    for (const entry of fs.readdirSync(batchDir, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const filePath = path.join(batchDir, entry.name);
      if (!/\.(mp4|m4v|mov|webm)$/i.test(filePath)) continue;
      clipPaths.push(filePath);
    }
  }
  return clipPaths;
}

function toExportedClipRecord(filePath: string): ExportedClipRecord | null {
  try {
    const stat = fs.statSync(filePath);
    const fileName = path.basename(filePath);
    const detectionId = fileName.replace(/\.[^.]+$/, "");
    return {
      detectionId,
      path: filePath,
      fileName,
      batchName: path.basename(path.dirname(filePath)),
      mtimeMs: stat.mtimeMs,
    };
  } catch {
    return null;
  }
}

export function getSessionExportState(outputDir: string): SessionExportState {
  const emptyState: SessionExportState = {
    exportedClips: [],
    exportedPaths: [],
    exportedDetectionIds: [],
    exportedCount: 0,
  };
  if (!outputDir) return emptyState;

  const exportsRoot = path.join(outputDir, "exports");
  const latestByDetectionId = new Map<string, ExportedClipRecord>();

  for (const clipPath of readExportBatchPaths(exportsRoot)) {
    const record = toExportedClipRecord(clipPath);
    if (!record) continue;
    const existing = latestByDetectionId.get(record.detectionId);
    if (!existing || record.mtimeMs > existing.mtimeMs || (record.mtimeMs === existing.mtimeMs && record.path > existing.path)) {
      latestByDetectionId.set(record.detectionId, record);
    }
  }

  const exportedClips = Array.from(latestByDetectionId.values()).sort((left, right) => {
    if (left.mtimeMs === right.mtimeMs) return left.fileName.localeCompare(right.fileName);
    return right.mtimeMs - left.mtimeMs;
  });

  return {
    exportedClips,
    exportedPaths: exportedClips.map((item) => item.path),
    exportedDetectionIds: exportedClips.map((item) => item.detectionId),
    exportedCount: exportedClips.length,
  };
}
