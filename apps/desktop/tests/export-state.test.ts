import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "bun:test";
import { getSessionExportState } from "../src/lib/export-state";

const tempRoots: string[] = [];

function createTempSessionDir(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "divesensei-export-state-"));
  tempRoots.push(root);
  return root;
}

function writeClip(filePath: string, mtimeMs: number): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, "clip");
  const stamp = new Date(mtimeMs);
  fs.utimesSync(filePath, stamp, stamp);
}

afterEach(() => {
  while (tempRoots.length > 0) {
    fs.rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
});

describe("getSessionExportState", () => {
  it("returns the latest exported clip per detection id", () => {
    const sessionDir = createTempSessionDir();
    const exportsDir = path.join(sessionDir, "exports");

    writeClip(path.join(exportsDir, "kept_20260331121947", "det-0004.mp4"), 1000);
    writeClip(path.join(exportsDir, "kept_20260331121947", "det-0005.mp4"), 1000);
    writeClip(path.join(exportsDir, "kept_20260331125445", "det-0004.mp4"), 2000);
    writeClip(path.join(exportsDir, "kept_20260331125445", "det-0007.mp4"), 2000);

    const state = getSessionExportState(sessionDir);

    expect(state.exportedCount).toBe(3);
    expect(state.exportedDetectionIds).toEqual(["det-0004", "det-0007", "det-0005"]);
    expect(state.exportedPaths).toEqual([
      path.join(exportsDir, "kept_20260331125445", "det-0004.mp4"),
      path.join(exportsDir, "kept_20260331125445", "det-0007.mp4"),
      path.join(exportsDir, "kept_20260331121947", "det-0005.mp4"),
    ]);
  });

  it("ignores missing export folders and non-video files", () => {
    const sessionDir = createTempSessionDir();
    const exportsDir = path.join(sessionDir, "exports", "kept_20260331130000");
    fs.mkdirSync(exportsDir, { recursive: true });
    fs.writeFileSync(path.join(exportsDir, "notes.txt"), "ignore");

    const state = getSessionExportState(sessionDir);

    expect(state.exportedCount).toBe(0);
    expect(state.exportedPaths).toEqual([]);
  });
});
