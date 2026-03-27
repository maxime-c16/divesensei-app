import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const currentDir = path.dirname(fileURLToPath(import.meta.url));

function resolveRepoRoot(startDir: string): string {
  let current = startDir;
  for (let index = 0; index < 8; index += 1) {
    const hasPythonProject = fs.existsSync(path.join(current, "pyproject.toml"));
    const hasSourceTree = fs.existsSync(path.join(current, "src", "divesensei"));
    if (hasPythonProject && hasSourceTree) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return path.resolve(startDir, "../../../..");
}

export const repoRoot = resolveRepoRoot(currentDir);
export const runtimeRoot = path.join(repoRoot, ".divesensei-runtime");
export const outputsRoot = path.join(repoRoot, "outputs");
export const analysisJobsRoot = path.join(runtimeRoot, "analysis-jobs");
export const exportJobsRoot = path.join(runtimeRoot, "export-jobs");
export const importsRoot = path.join(runtimeRoot, "imports");
export const sessionCatalogPath = path.join(runtimeRoot, "session-catalog.json");
export const sessionCatalogDbPath = path.join(runtimeRoot, "session-catalog.sqlite");
export const allowedMediaRoots = [
  repoRoot,
  path.join(os.homedir(), "Movies"),
  path.join(os.homedir(), "Videos"),
  "/srv/nas/videos",
];

export const preferredPython = [
  path.join(repoRoot, ".venv", "bin", "python"),
  path.join(repoRoot, ".venv-detect", "bin", "python"),
  "python3",
].find((candidate) => candidate === "python3" || fs.existsSync(candidate)) ?? "python3";

export function ensureRuntimeDirs(): void {
  fs.mkdirSync(runtimeRoot, { recursive: true });
  fs.mkdirSync(outputsRoot, { recursive: true });
  fs.mkdirSync(analysisJobsRoot, { recursive: true });
  fs.mkdirSync(exportJobsRoot, { recursive: true });
  fs.mkdirSync(importsRoot, { recursive: true });
}

export function isAllowedLocalPath(targetPath: string): boolean {
  const resolved = path.resolve(targetPath);
  return allowedMediaRoots.some((root) => resolved === root || resolved.startsWith(`${root}${path.sep}`));
}

export function jobStatePath(jobId: string): string {
  ensureRuntimeDirs();
  return path.join(analysisJobsRoot, `${jobId}.json`);
}

export function exportJobStatePath(jobId: string): string {
  ensureRuntimeDirs();
  return path.join(exportJobsRoot, `${jobId}.json`);
}
