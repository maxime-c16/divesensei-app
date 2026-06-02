from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from divesensei.workflows.export_rough_clips import sanitize_slug
from divesensei.workflows.validate_clip_assignments import validate_clip_assignments


DEFAULT_OUTPUT_ROOT = Path("outputs/assignment_drafts/snmt")
ALLOWED_OUTPUT_ROOT = Path("outputs/assignment_drafts")

CORE_FIELDS = [
    "clip_number",
    "mobile_filename",
    "timestamp_seconds",
    "timestamp_friendly",
    "clip_window_friendly",
    "diver_query",
    "diver_label",
    "training_group",
    "apparatus_type",
    "apparatus_height_m",
    "apparatus_id",
    "dive_code",
    "assignment_status",
    "share_status",
    "notes",
]

AUDIT_FIELDS = [
    "assignment_source",
    "typed_diver_query",
    "resolved_diver_id",
    "resolved_diver_label",
    "diver_match_type",
    "diver_match_score",
    "manual_alias_created",
    "autofill_source",
    "autofill_confirmed",
    "bulk_operation_id",
    "bulk_operation_label",
    "manual_series_override",
    "manual_attempt_override",
    "computed_attempt_version",
    "last_modified_at",
    "last_modified_by",
]

TRACE_FIELDS = [
    "source_mobile_package",
    "source_mobile_clip_path",
    "source_mobile_filename",
    "candidate_id",
    "source_manifest_path",
]

DRAFT_FIELDS = CORE_FIELDS + AUDIT_FIELDS + TRACE_FIELDS
PRESERVE_FIELDS = [
    "diver_query",
    "diver_label",
    "training_group",
    "apparatus_type",
    "apparatus_height_m",
    "apparatus_id",
    "dive_code",
    "assignment_status",
    "share_status",
    "notes",
    *AUDIT_FIELDS,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei manage-assignment-draft",
        description="Initialize or preview safe assignment draft files from a mobile share package.",
    )
    subparsers = parser.add_subparsers(dest="action")

    init = subparsers.add_parser("init", help="Create a validator-compatible draft assignment CSV/JSON")
    init.add_argument("--mobile-package", required=True, type=Path)
    init.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=Path)
    init.add_argument("--run-id", required=True)
    init.add_argument("--existing-draft", type=Path)
    init.add_argument("--force", action="store_true")
    init.add_argument("--last-modified-by", default="local_cli_agent")
    init.add_argument("--json", action="store_true")

    preview = subparsers.add_parser("validate-preview", help="Run validator against an assignment draft")
    preview.add_argument("--draft", required=True, type=Path)
    preview.add_argument("--mobile-package", required=True, type=Path)
    preview.add_argument("--divers", type=Path)
    preview.add_argument("--output-root", default="outputs/assignment_validation/snmt", type=Path)
    preview.add_argument("--run-id", required=True)
    preview.add_argument("--force", action="store_true")
    preview.add_argument("--json", action="store_true")
    return parser


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_output_root(output_root: Path) -> Path:
    root = resolve(output_root)
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Output root must stay under {allowed}") from exc
    return root


def ensure_output_dir(path: Path, force: bool) -> None:
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        path.resolve().relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Draft directory must stay under {allowed}") from exc
    if path.exists():
        if not force:
            raise FileExistsError(f"Draft directory exists; pass --force or use --existing-draft: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def mobile_clip_list_path(mobile_package: Path) -> Path:
    return mobile_package / "_technique" / "LISTE_DES_CLIPS.csv"


def source_manifest_path(mobile_package: Path) -> str:
    provenance = mobile_package / "_technique" / "manifest_source.json"
    if not provenance.exists():
        return ""
    try:
        data = read_json(provenance)
    except json.JSONDecodeError:
        return ""
    return str(data.get("source_manifest_path") or "")


def session_slug_for_mobile_package(mobile_package: Path) -> str:
    return sanitize_slug(mobile_package.parent.name)


def load_mobile_rows(mobile_package: Path) -> list[dict[str, Any]]:
    mobile_package = resolve(mobile_package)
    path = mobile_clip_list_path(mobile_package)
    if not path.exists():
        raise FileNotFoundError(f"Missing mobile clip list: {path}")
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Mobile clip list is empty: {path}")
    return rows


def load_existing_draft(path: Path | None) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    if not path:
        return {}, []
    path = resolve(path)
    rows = read_csv(path)
    by_key: dict[str, dict[str, str]] = {}
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        key = str(row.get("source_mobile_filename") or row.get("mobile_filename") or "").strip()
        if not key:
            issues.append({"type": "missing_key", "row": index})
            continue
        if key in by_key:
            issues.append({"type": "duplicate_existing_draft_key", "key": key, "row": index})
            continue
        by_key[key] = row
    return by_key, issues


def build_draft_rows(
    mobile_package: Path,
    mobile_rows: list[dict[str, Any]],
    existing_by_key: dict[str, dict[str, str]],
    *,
    modified_by: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = source_manifest_path(mobile_package)
    draft_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    used_existing: set[str] = set()
    duplicate_source_keys: set[str] = set()
    seen_source_keys: set[str] = set()

    for index, source in enumerate(mobile_rows, start=1):
        source_name = str(source.get("mobile_filename") or "").strip()
        if not source_name:
            warnings.append({"type": "missing_mobile_filename", "row": index})
            continue
        if source_name in seen_source_keys:
            duplicate_source_keys.add(source_name)
            warnings.append({"type": "duplicate_source_row", "mobile_filename": source_name})
        seen_source_keys.add(source_name)
        existing = existing_by_key.get(source_name)
        if existing:
            used_existing.add(source_name)
        row = {field: "" for field in DRAFT_FIELDS}
        row.update(
            {
                "clip_number": source.get("clip_number") or index,
                "mobile_filename": source_name,
                "timestamp_seconds": source.get("timestamp_seconds") or "",
                "timestamp_friendly": source.get("timestamp_friendly") or "",
                "clip_window_friendly": source.get("clip_window_friendly") or "",
                "assignment_status": "a_identifier",
                "share_status": "a_verifier",
                "assignment_source": "draft_init",
                "autofill_confirmed": "false",
                "computed_attempt_version": "s015_v1",
                "last_modified_at": now,
                "last_modified_by": modified_by,
                "source_mobile_package": str(mobile_package),
                "source_mobile_clip_path": source.get("mobile_clip_path") or str((mobile_package / "clips" / source_name).resolve()),
                "source_mobile_filename": source_name,
                "candidate_id": source.get("candidate_id") or "",
                "source_manifest_path": manifest_path,
            }
        )
        if existing:
            for field in PRESERVE_FIELDS:
                value = existing.get(field)
                if value not in (None, ""):
                    row[field] = value
            row["assignment_source"] = existing.get("assignment_source") or "draft_merge_preserved"
            if existing.get("last_modified_at"):
                row["last_modified_at"] = existing["last_modified_at"]
            if existing.get("last_modified_by"):
                row["last_modified_by"] = existing["last_modified_by"]
        draft_rows.append(row)

    missing_existing = sorted(set(existing_by_key) - used_existing)
    for key in missing_existing:
        warnings.append({"type": "missing_source_clip_for_existing_draft", "mobile_filename": key})

    diff = {
        "new_clip_count": len([row for row in draft_rows if row.get("source_mobile_filename") not in existing_by_key]),
        "preserved_clip_count": len(used_existing),
        "missing_clip_count": len(missing_existing),
        "missing_clips": missing_existing,
        "duplicate_source_rows": sorted(duplicate_source_keys),
        "duplicate_source_row_count": len(duplicate_source_keys),
    }
    return draft_rows, warnings, diff


def future_ui_contract(out_dir: Path, mobile_package: Path) -> dict[str, Any]:
    draft_csv = out_dir / "clip_assignments_draft.csv"
    return {
        "autosave_target": str(draft_csv),
        "manual_save_target": str(draft_csv),
        "row_key": "source_mobile_filename",
        "validate_command": (
            "PYTHONPATH=src python3 -m divesensei.cli validate-clip-assignments "
            f"--mobile-package {mobile_package} --assignments {draft_csv} "
            "--output-root outputs/assignment_validation/snmt --run-id <run-id> --force"
        ),
        "export_command": "Run export-recipient-packages only after validator output has zero hard errors.",
        "conflict_behavior": "Preserve existing coach-entered fields; report new/missing/duplicate clip keys.",
        "warning_display": "Show unknown diver, blank code, near duplicate, and unconfirmed autofill warnings inline.",
        "error_display": "Block package export for validator hard errors.",
        "undo_metadata": ["bulk_operation_id", "bulk_operation_label", "last_modified_at", "last_modified_by"],
        "bulk_operation_metadata": ["bulk_operation_id", "bulk_operation_label", "autofill_source", "autofill_confirmed"],
        "safety": "Assignment draft is separate from review labels and approval/FN semantics.",
    }


def write_draft_outputs(out_dir: Path, mobile_package: Path, rows: list[dict[str, Any]], warnings: list[dict[str, Any]], errors: list[dict[str, Any]], diff: dict[str, Any]) -> dict[str, Any]:
    draft_csv = out_dir / "clip_assignments_draft.csv"
    draft_json = out_dir / "clip_assignments_draft.json"
    metadata_path = out_dir / "draft_metadata.json"
    write_csv(draft_csv, rows, DRAFT_FIELDS)
    write_json(draft_json, {"schema_version": "s022_assignment_draft_v1", "rows": rows})
    metadata = {
        "schema_version": "s022_assignment_draft_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mobile_package": str(mobile_package),
        "row_count": len(rows),
        "default_assignment_status": "a_identifier",
        "default_share_status": "a_verifier",
        "validator_compatible": True,
        "future_ui_contract": future_ui_contract(out_dir, mobile_package),
    }
    write_json(metadata_path, metadata)
    write_json(out_dir / "draft_warnings.json", {"warnings": warnings})
    write_json(out_dir / "draft_errors.json", {"errors": errors})
    write_json(out_dir / "draft_diff_preview.json", diff)
    (out_dir / "validator_command.txt").write_text(metadata["future_ui_contract"]["validate_command"] + "\n", encoding="utf-8")
    (out_dir / "recipient_export_command.txt").write_text(metadata["future_ui_contract"]["export_command"] + "\n", encoding="utf-8")
    (out_dir / "DRAFT_SUMMARY.md").write_text(
        "\n".join(
            [
                "# Assignment Draft Summary",
                "",
                f"Rows: {len(rows)}",
                "Default status: a_identifier / a_verifier",
                "Identity and dive code are not inferred.",
                "This draft is separate from review labels and approval/FN state.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "draft_dir": str(out_dir),
        "draft_csv": str(draft_csv),
        "draft_json": str(draft_json),
        "draft_metadata": str(metadata_path),
        "row_count": len(rows),
        "warning_count": len(warnings),
        "error_count": len(errors),
        "diff": diff,
    }


def init_draft(args: argparse.Namespace) -> dict[str, Any]:
    mobile_package = resolve(args.mobile_package)
    rows = load_mobile_rows(mobile_package)
    existing_by_key, existing_issues = load_existing_draft(args.existing_draft)
    session_slug = session_slug_for_mobile_package(mobile_package)
    output_root = safe_output_root(args.output_root)
    out_dir = output_root / session_slug / args.run_id
    ensure_output_dir(out_dir, bool(args.force))
    draft_rows, warnings, diff = build_draft_rows(mobile_package, rows, existing_by_key, modified_by=args.last_modified_by)
    warnings = [*existing_issues, *warnings]
    return write_draft_outputs(out_dir, mobile_package, draft_rows, warnings, [], diff)


def validate_preview(args: argparse.Namespace) -> dict[str, Any]:
    summary = validate_clip_assignments(
        mobile_package=args.mobile_package,
        assignments=args.draft,
        divers=args.divers,
        output_root=args.output_root,
        run_id=args.run_id,
        force=args.force,
    )
    return {
        "draft": str(resolve(args.draft)),
        "mobile_package": str(resolve(args.mobile_package)),
        "validation_output_dir": summary["output_dir"],
        "assignment_count": summary["assignment_count"],
        "warning_count": summary["warning_count"],
        "error_count": summary["error_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "init":
        summary = init_draft(args)
    elif args.action == "validate-preview":
        summary = validate_preview(args)
    else:
        parser.print_help()
        return 0
    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Draft action: {args.action}")
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
