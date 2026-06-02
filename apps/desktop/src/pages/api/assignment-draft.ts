import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import type { APIRoute } from "astro";
import { outputsRoot, preferredPython, repoRoot } from "@/lib/runtime-config";

const NO_STORE_HEADERS = { "Cache-Control": "no-store, max-age=0" };
const ASSIGNMENT_DRAFT_ROOT = path.join(outputsRoot, "assignment_drafts");
const ASSIGNMENT_VALIDATION_ROOT = path.join(outputsRoot, "assignment_validation", "snmt");
const MOBILE_SHARE_ROOT = path.join(outputsRoot, "mobile_share");

const DRAFT_CSV = "clip_assignments_draft.csv";
const DRAFT_METADATA = "draft_metadata.json";
const DRAFT_WARNINGS = "draft_warnings.json";
const DRAFT_ERRORS = "draft_errors.json";

const ALLOWED_UPDATE_FIELDS = new Set([
  "diver_query",
  "training_group",
  "apparatus_type",
  "apparatus_height_m",
  "apparatus_id",
  "dive_code",
  "assignment_status",
  "share_status",
  "notes",
  "manual_series_override",
  "manual_attempt_override",
  "autofill_confirmed",
  "bulk_operation_id",
  "bulk_operation_label",
]);

const APPLY_PREVIOUS_FIELDS = [
  "diver_query",
  "training_group",
  "apparatus_type",
  "apparatus_height_m",
  "apparatus_id",
  "dive_code",
  "assignment_status",
  "share_status",
];

type CsvRow = Record<string, string>;

function jsonResponse(data: unknown, status = 200): Response {
  return Response.json(data, { status, headers: NO_STORE_HEADERS });
}

function resolveUnder(rawPath: string, root: string, label: string): string {
  const resolved = path.resolve(rawPath);
  const allowed = path.resolve(root);
  if (resolved !== allowed && !resolved.startsWith(`${allowed}${path.sep}`)) {
    throw new Error(`${label}_outside_allowed_root`);
  }
  return resolved;
}

function draftDirFromRequest(rawPath: unknown): string {
  if (typeof rawPath !== "string" || !rawPath.trim()) {
    throw new Error("draftDir_required");
  }
  return resolveUnder(rawPath, ASSIGNMENT_DRAFT_ROOT, "draftDir");
}

function mobilePackageFromRequest(rawPath: unknown): string {
  if (typeof rawPath !== "string" || !rawPath.trim()) {
    throw new Error("mobilePackage_required");
  }
  return resolveUnder(rawPath, MOBILE_SHARE_ROOT, "mobilePackage");
}

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (quoted && char === "\"" && next === "\"") {
      current += "\"";
      index += 1;
    } else if (char === "\"") {
      quoted = !quoted;
    } else if (!quoted && char === ",") {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function readCsv(filePath: string): { fields: string[]; rows: CsvRow[] } {
  const text = fs.readFileSync(filePath, "utf-8");
  const lines = text.split(/\r?\n/).filter((line) => line.length > 0);
  if (lines.length === 0) return { fields: [], rows: [] };
  const fields = parseCsvLine(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row: CsvRow = {};
    fields.forEach((field, index) => {
      row[field] = values[index] ?? "";
    });
    return row;
  });
  return { fields, rows };
}

function csvEscape(value: unknown): string {
  const raw = value == null ? "" : String(value);
  if (/[",\n\r]/.test(raw)) {
    return `"${raw.replaceAll("\"", "\"\"")}"`;
  }
  return raw;
}

function writeCsv(filePath: string, fields: string[], rows: CsvRow[]): void {
  const lines = [
    fields.map(csvEscape).join(","),
    ...rows.map((row) => fields.map((field) => csvEscape(row[field] ?? "")).join(",")),
  ];
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf-8");
}

function readJsonIfExists(filePath: string): Record<string, unknown> {
  if (!fs.existsSync(filePath)) return {};
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as Record<string, unknown>;
}

function draftSummary(draftDir: string): Record<string, unknown> {
  const csvPath = path.join(draftDir, DRAFT_CSV);
  const initialized = fs.existsSync(csvPath);
  const rows = initialized ? readCsv(csvPath).rows : [];
  const stat = initialized ? fs.statSync(csvPath) : null;
  const warnings = readJsonIfExists(path.join(draftDir, DRAFT_WARNINGS));
  const errors = readJsonIfExists(path.join(draftDir, DRAFT_ERRORS));
  const metadata = readJsonIfExists(path.join(draftDir, DRAFT_METADATA));
  return {
    session_slug: path.basename(path.dirname(draftDir)),
    run_id: path.basename(draftDir),
    draft_path: draftDir,
    row_count: rows.length,
    draft_initialized: initialized,
    last_modified: stat ? stat.mtime.toISOString() : null,
    warning_count: Array.isArray(warnings.warnings) ? warnings.warnings.length : 0,
    error_count: Array.isArray(errors.errors) ? errors.errors.length : 0,
    validation_status: rows.length > 0 ? "draft_loaded" : "not_initialized",
    metadata,
  };
}

function rowKeyFor(row: CsvRow): string {
  return row.source_mobile_filename || row.mobile_filename || "";
}

function hasPreviousAssignmentContext(row: CsvRow): boolean {
  if (!row) return false;
  const source = row.assignment_source ?? "";
  if (source && source !== "draft_init") return true;
  return APPLY_PREVIOUS_FIELDS.some((field) => {
    const value = row[field] ?? "";
    if (!value) return false;
    if (field === "assignment_status") return value !== "a_identifier";
    if (field === "share_status") return value !== "a_verifier";
    return true;
  });
}

function loadDraftRows(draftDir: string): { csvPath: string; fields: string[]; rows: CsvRow[] } {
  const csvPath = path.join(draftDir, DRAFT_CSV);
  if (!fs.existsSync(csvPath)) {
    throw new Error("draft_not_initialized");
  }
  return { csvPath, ...readCsv(csvPath) };
}

function validateUpdates(updates: unknown): Record<string, string> {
  if (!updates || typeof updates !== "object" || Array.isArray(updates)) {
    throw new Error("updates_required");
  }
  const out: Record<string, string> = {};
  for (const [field, value] of Object.entries(updates as Record<string, unknown>)) {
    if (!ALLOWED_UPDATE_FIELDS.has(field)) {
      throw new Error(`invalid_update_field:${field}`);
    }
    out[field] = value == null ? "" : String(value);
  }
  return out;
}

function updateDraftRow(body: Record<string, unknown>): Record<string, unknown> {
  const draftDir = draftDirFromRequest(body.draftDir);
  const key = typeof body.rowKey === "string" ? body.rowKey : "";
  if (!key) throw new Error("rowKey_required");
  const updates = validateUpdates(body.updates);
  const explicit = new Set(Array.isArray(body.explicitOverwriteFields) ? body.explicitOverwriteFields.map(String) : []);
  const { csvPath, fields, rows } = loadDraftRows(draftDir);
  const index = rows.findIndex((row) => rowKeyFor(row) === key);
  if (index < 0) throw new Error("row_not_found");

  const row = { ...rows[index] };
  for (const [field, value] of Object.entries(updates)) {
    const existing = row[field] ?? "";
    if (existing && existing !== value && !explicit.has(field)) {
      throw new Error(`confirmed_field_requires_explicit_overwrite:${field}`);
    }
    row[field] = value;
  }
  row.last_modified_at = new Date().toISOString();
  row.last_modified_by = typeof body.lastModifiedBy === "string" && body.lastModifiedBy.trim() ? body.lastModifiedBy : "desktop_assignment_draft_api";
  row.assignment_source = typeof body.assignmentSource === "string" && body.assignmentSource.trim() ? body.assignmentSource : "api_update";
  if (typeof body.autofillSource === "string") row.autofill_source = body.autofillSource;
  rows[index] = row;
  writeCsv(csvPath, fields, rows);
  return { row, summary: draftSummary(draftDir) };
}

function applyPreviousAssignment(body: Record<string, unknown>): Record<string, unknown> {
  const draftDir = draftDirFromRequest(body.draftDir);
  const key = typeof body.rowKey === "string" ? body.rowKey : "";
  if (!key) throw new Error("rowKey_required");
  const confirmOverwrite = body.confirmOverwrite === true;
  const { csvPath, fields, rows } = loadDraftRows(draftDir);
  const index = rows.findIndex((row) => rowKeyFor(row) === key);
  if (index < 0) throw new Error("row_not_found");
  const previousIndex = rows.slice(0, index).findLastIndex(hasPreviousAssignmentContext);
  if (previousIndex < 0) throw new Error("previous_assignment_context_not_found");
  const previous = rows[previousIndex];
  const before = { ...rows[index] };
  const proposed: Record<string, string> = {};
  const conflicts: Array<Record<string, string>> = [];

  for (const field of APPLY_PREVIOUS_FIELDS) {
    const value = previous[field] ?? "";
    if (!value) continue;
    proposed[field] = value;
    const existing = before[field] ?? "";
    if (existing && existing !== value) {
      conflicts.push({ field, before: existing, after: value });
    }
  }

  if (Object.keys(proposed).length === 0) {
    throw new Error("previous_assignment_context_empty");
  }

  if (conflicts.length > 0 && !confirmOverwrite) {
    return {
      requires_confirmation: true,
      conflicts,
      previous_row_key: rowKeyFor(previous),
      proposed_updates: proposed,
      row: before,
      summary: draftSummary(draftDir),
    };
  }

  const row = { ...before, ...proposed };
  row.autofill_source = `previous_clip:${rowKeyFor(previous)}`;
  row.autofill_confirmed = "true";
  row.assignment_source = "apply_previous_confirmed";
  row.last_modified_at = new Date().toISOString();
  row.last_modified_by = typeof body.lastModifiedBy === "string" && body.lastModifiedBy.trim()
    ? body.lastModifiedBy
    : "desktop_assignment_panel_apply_previous";
  rows[index] = row;
  writeCsv(csvPath, fields, rows);
  return {
    row,
    before,
    previous_row_key: rowKeyFor(previous),
    applied_updates: proposed,
    undo_snapshot: before,
    summary: draftSummary(draftDir),
  };
}

function validationPreview(body: Record<string, unknown>): Record<string, unknown> {
  const draftDir = draftDirFromRequest(body.draftDir);
  const mobilePackage = mobilePackageFromRequest(body.mobilePackage);
  const runId = typeof body.runId === "string" && body.runId.trim()
    ? body.runId.trim()
    : `assignment-draft-api-${Date.now()}`;
  const draftCsv = path.join(draftDir, DRAFT_CSV);
  if (!fs.existsSync(draftCsv)) throw new Error("draft_not_initialized");
  const args = [
    "-m",
    "divesensei.cli",
    "validate-clip-assignments",
    "--mobile-package",
    mobilePackage,
    "--assignments",
    draftCsv,
    "--output-root",
    ASSIGNMENT_VALIDATION_ROOT,
    "--run-id",
    runId,
    "--force",
    "--json",
  ];
  const env = { ...process.env, PYTHONPATH: path.join(repoRoot, "src") };
  const completed = spawnSync(preferredPython, args, { cwd: repoRoot, env, encoding: "utf-8" });
  if (completed.status !== 0) {
    return {
      export_ready: false,
      blocking_errors: [{ type: "validator_failed", message: completed.stderr || completed.stdout }],
      warnings: [],
      row_count: 0,
      packageable_count: 0,
      a_identifier_count: 0,
      a_verifier_count: 0,
      near_duplicate_diver_warnings: 0,
      missing_dive_code_warnings: 0,
    };
  }
  let summary: Record<string, unknown> = {};
  try {
    summary = JSON.parse(completed.stdout) as Record<string, unknown>;
  } catch {
    summary = {};
  }
  const outputDir = String(summary.output_dir ?? "");
  const errors = readJsonIfExists(path.join(outputDir, "assignment_errors.json"));
  const warnings = readJsonIfExists(path.join(outputDir, "assignment_warnings.json"));
  const assignments = readJsonIfExists(path.join(outputDir, "validated_assignments.json"));
  const assignmentRows = Array.isArray(assignments.assignments) ? assignments.assignments as Array<Record<string, unknown>> : [];
  const warningRows = Array.isArray(warnings.warnings) ? warnings.warnings as Array<Record<string, unknown>> : [];
  const errorRows = Array.isArray(errors.errors) ? errors.errors as Array<Record<string, unknown>> : [];
  const result = {
    validation_output_dir: outputDir,
    blocking_errors: errorRows,
    warnings: warningRows,
    row_count: assignmentRows.length,
    packageable_count: assignmentRows.filter((row) => row.packageable === true).length,
    a_identifier_count: assignmentRows.filter((row) => row.assignment_status === "a_identifier").length,
    a_verifier_count: assignmentRows.filter((row) => row.share_status === "a_verifier").length,
    near_duplicate_diver_warnings: warningRows.filter((row) => row.type === "diver_resolution").length,
    missing_dive_code_warnings: warningRows.filter((row) => row.type === "missing_dive_code").length,
    export_ready: errorRows.length === 0,
  };
  return result;
}

function listRecentValues(body: Record<string, unknown>): Record<string, unknown> {
  const draftDir = draftDirFromRequest(body.draftDir);
  const { rows } = loadDraftRows(draftDir);
  const fields = ["diver_query", "training_group", "apparatus_height_m", "dive_code"];
  const values: Record<string, string[]> = {};
  for (const field of fields) {
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const row of [...rows].reverse()) {
      const value = (row[field] ?? "").trim();
      if (!value || seen.has(value)) continue;
      seen.add(value);
      ordered.push(value);
      if (ordered.length >= 12) break;
    }
    values[field] = ordered;
  }
  values.apparatus_type = ["springboard", "platform", "unknown"];
  return {
    values,
    source: "assignment_draft_recent_values",
    inference_performed: false,
  };
}

export const GET: APIRoute = async ({ url }) => {
  try {
    const draftDir = draftDirFromRequest(url.searchParams.get("draftDir"));
    const rowKey = url.searchParams.get("rowKey");
    const summary = draftSummary(draftDir);
    if (!rowKey) return jsonResponse({ summary });
    const { rows } = loadDraftRows(draftDir);
    const row = rows.find((item) => rowKeyFor(item) === rowKey);
    if (!row) return jsonResponse({ error: "row_not_found" }, 404);
    return jsonResponse({ summary, row });
  } catch (error) {
    return jsonResponse({ error: error instanceof Error ? error.message : "assignment_draft_get_failed" }, 400);
  }
};

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  if (!body || typeof body !== "object") {
    return jsonResponse({ error: "json_body_required" }, 400);
  }
  try {
    if (body.action === "update-row") {
      return jsonResponse(updateDraftRow(body));
    }
    if (body.action === "apply-previous") {
      return jsonResponse(applyPreviousAssignment(body));
    }
    if (body.action === "validate-preview") {
      return jsonResponse(validationPreview(body));
    }
    if (body.action === "list-recent-values") {
      return jsonResponse(listRecentValues(body));
    }
    return jsonResponse({ error: "unsupported_action" }, 400);
  } catch (error) {
    return jsonResponse({ error: error instanceof Error ? error.message : "assignment_draft_post_failed" }, 400);
  }
};
