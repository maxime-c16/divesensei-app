from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from divesensei.io.media_io import extract_clip_ffmpeg, probe_media_duration_seconds


DEFAULT_OUTPUT_ROOT = Path("outputs/clip_exports/snmt_rough")
ALLOWED_OUTPUT_ROOT = Path("outputs/clip_exports")
DEFAULT_PRE_ROLL = 2.5
DEFAULT_POST_ROLL = 4.0
DEFAULT_MIN_DURATION = 1.0
DEFAULT_MAX_DURATION = 12.0

MEDIA_MODES = {"auto", "source", "proxy"}
CANDIDATE_SOURCES = {
    "auto",
    "ui-manifest",
    "reviewed-candidates",
    "event-review-support",
    "event-reviewed-manifest",
    "session-report",
}


@dataclass(frozen=True)
class CandidateSource:
    source_id: str
    artifact_path: Path
    rows: list[dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-rough-clips",
        description="Create a dry-run-first rough, unreviewed clip export plan for an evaluation session.",
    )
    parser.add_argument("evaluation_root", help="Evaluation session output directory")
    parser.add_argument("--media", choices=sorted(MEDIA_MODES), default="auto")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pre-roll", type=float, default=DEFAULT_PRE_ROLL)
    parser.add_argument("--post-roll", type=float, default=DEFAULT_POST_ROLL)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--candidate-source", choices=sorted(CANDIDATE_SOURCES), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; this is the default unless --execute is passed")
    parser.add_argument("--execute", action="store_true", help="Render clips. Required for any media output.")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output run directory")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_existing_path(value: Any, base: Path) -> Path | None:
    if not value:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = base / p
    return p


def resolve_evaluation_root(raw: str) -> Path:
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Evaluation root not found: {root}")
    return root


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._").lower()
    return cleaned[:80] or "session"


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def stable_candidate_id(row: dict[str, Any], index: int) -> str:
    for key in (
        "candidate_id",
        "source_candidate_id",
        "legacy_candidate_id",
        "id",
        "detectionId",
        "proposal_id",
        "legacy_candidate_label",
        "review_annotation_id",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"candidate-{index:04d}"


def optional_context(row: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in (
        "anchor_quality",
        "anchorQuality",
        "event_label",
        "suggested_event_label",
        "candidate_usefulness",
        "usefulness",
        "clip_quality",
        "timing_semantics_shadow",
        "candidate_quality_shadow",
        "candidate_quality_shadow_score",
        "review_relevance_score",
    ):
        if key in row:
            context[key] = row.get(key)
    for key, value in row.items():
        if key.startswith("candidate_quality_shadow"):
            context[key] = value
    return context


def source_video_from_manifest(evaluation_root: Path) -> Path | None:
    manifest_path = evaluation_root / "ui_session_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        return resolve_existing_path(manifest.get("session", {}).get("source_video_path"), evaluation_root)
    report_path = evaluation_root / "session_pipeline_report.json"
    if report_path.exists():
        report = read_json(report_path)
        return resolve_existing_path(report.get("video_path"), evaluation_root)
    return None


def proxy_video_for_session(evaluation_root: Path) -> Path | None:
    report_path = evaluation_root / "session_pipeline_report.json"
    if report_path.exists():
        report = read_json(report_path)
        candidate = resolve_existing_path(report.get("review_proxy_path"), evaluation_root)
        if candidate and candidate.exists():
            return candidate
    for rel in ("web/session_source_review.mp4", "web/session_source_review.part.mp4"):
        candidate = evaluation_root / rel
        if candidate.exists():
            return candidate
    return None


def resolve_media(evaluation_root: Path, media_mode: str) -> tuple[Path, str]:
    source = source_video_from_manifest(evaluation_root)
    proxy = proxy_video_for_session(evaluation_root)
    source_exists = bool(source and source.exists())
    proxy_exists = bool(proxy and proxy.exists())
    if media_mode == "source":
        if not source_exists:
            raise ValueError("Source media requested but no existing source video path was found.")
        return source, "source"
    if media_mode == "proxy":
        if not proxy_exists:
            raise ValueError("Proxy media requested but no existing proxy/review video path was found.")
        return proxy, "proxy"
    if source_exists:
        return source, "source"
    if proxy_exists:
        return proxy, "proxy"
    raise ValueError("No existing source or proxy media path could be resolved.")


def load_candidate_source(evaluation_root: Path, requested: str) -> CandidateSource:
    candidates: list[tuple[str, Path, str]] = [
        ("ui-manifest", evaluation_root / "ui_session_manifest.json", "detections"),
        ("reviewed-candidates", evaluation_root / "exports/evaluation-review/reviewed_candidates.jsonl", "jsonl"),
        ("event-review-support", evaluation_root / "exports/event-review-support/event_review_support.jsonl", "jsonl"),
        ("event-reviewed-manifest", evaluation_root / "exports/event-reviewed-manifest/event_reviewed_manifest.jsonl", "jsonl"),
        ("session-report", evaluation_root / "session_pipeline_report.json", "candidates"),
    ]
    for source_id, path, kind in candidates:
        if requested != "auto" and requested != source_id:
            continue
        if not path.exists():
            continue
        if kind == "jsonl":
            rows = read_jsonl(path)
        else:
            data = read_json(path)
            rows = data.get(kind) or []
        if isinstance(rows, list) and rows:
            return CandidateSource(source_id=source_id, artifact_path=path, rows=[row for row in rows if isinstance(row, dict)])
    raise ValueError(f"No candidate rows found for candidate source: {requested}")


def timestamp_for_row(row: dict[str, Any]) -> float | None:
    for key in ("timestamp_seconds", "timestamp", "event_anchor_timestamp_seconds"):
        value = finite_float(row.get(key))
        if value is not None:
            return value
    return None


def boundary_pair(row: dict[str, Any], keys: tuple[str, str]) -> tuple[float, float] | None:
    start = finite_float(row.get(keys[0]))
    end = finite_float(row.get(keys[1]))
    if start is None or end is None:
        return None
    start = max(0.0, start)
    if end > start:
        return start, end
    return None


def resolve_window(
    row: dict[str, Any],
    *,
    pre_roll: float,
    post_roll: float,
    min_duration: float,
    max_duration: float,
    media_duration: float | None,
) -> tuple[float | None, float | None, list[str], str | None]:
    warnings: list[str] = []
    source = None
    window = (
        boundary_pair(row, ("start_time_seconds", "end_time_seconds"))
        or boundary_pair(row, ("start_time", "end_time"))
        or boundary_pair(row, ("event_window_start_seconds", "event_window_end_seconds"))
    )
    if window is not None:
        source = "explicit_boundary"
    else:
        timestamp = timestamp_for_row(row)
        if timestamp is None:
            return None, None, warnings, "missing_timestamp_or_sane_boundary"
        source = "timestamp_fallback"
        window = (max(0.0, timestamp - pre_roll), timestamp + post_roll)
    start, end = window
    if end - start > max_duration:
        if source == "explicit_boundary":
            timestamp = timestamp_for_row(row)
            if timestamp is not None:
                start, end = max(0.0, timestamp - pre_roll), timestamp + post_roll
                source = "timestamp_fallback_after_oversize_boundary"
                warnings.append("explicit_boundary_exceeded_max_duration")
            else:
                return None, None, warnings, "clip_window_exceeds_max_duration"
        if end - start > max_duration:
            end = start + max_duration
            warnings.append("clip_window_clamped_to_max_duration")
    if media_duration is not None and end > media_duration:
        end = media_duration
        warnings.append("clip_end_clamped_to_media_duration")
    start = max(0.0, start)
    if end - start < min_duration:
        return None, None, warnings, "clip_window_shorter_than_min_duration"
    return round(start, 3), round(end, 3), warnings, None


def ensure_safe_output_root(output_root: Path) -> Path:
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root = output_root.resolve()
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Output root must be under {allowed}") from exc
    return output_root


def ensure_run_dir(run_dir: Path, *, force: bool) -> None:
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        run_dir.resolve().relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Run directory must be under {allowed}") from exc
    if run_dir.exists():
        if not force:
            raise ValueError(f"Output run directory already exists; use --force to replace: {run_dir}")
        shutil.rmtree(run_dir)
    (run_dir / "clips").mkdir(parents=True, exist_ok=False)


def equivalent_ffmpeg_command(media_path: Path, output_path: Path, start: float, end: float) -> list[str]:
    duration = max(0.25, end - start)
    return [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(media_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale/format via divesensei.io.media_io.extract_clip_ffmpeg",
        str(output_path),
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_root = resolve_evaluation_root(args.evaluation_root)
    media_path, selected_media_mode = resolve_media(evaluation_root, args.media)
    media_duration = probe_media_duration_seconds(media_path)
    candidate_source = load_candidate_source(evaluation_root, args.candidate_source)
    session_slug = sanitize_slug(evaluation_root.name)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = ensure_safe_output_root(Path(args.output_root))
    run_dir = output_root / session_slug / run_id
    rows = candidate_source.rows[: args.limit] if args.limit is not None else candidate_source.rows

    id_counts: dict[str, int] = {}
    planned: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    used_output_paths: set[str] = set()
    previous_end: float | None = None
    for zero_index, row in enumerate(rows):
        index = zero_index + 1
        raw_candidate_id = stable_candidate_id(row, index)
        safe_candidate_id = sanitize_slug(raw_candidate_id)
        id_counts[safe_candidate_id] = id_counts.get(safe_candidate_id, 0) + 1
        file_candidate_id = safe_candidate_id
        if id_counts[safe_candidate_id] > 1:
            file_candidate_id = f"{safe_candidate_id}__dup{id_counts[safe_candidate_id]:02d}"
        start, end, warnings, error = resolve_window(
            row,
            pre_roll=float(args.pre_roll),
            post_roll=float(args.post_roll),
            min_duration=float(args.min_duration),
            max_duration=float(args.max_duration),
            media_duration=media_duration,
        )
        timestamp = timestamp_for_row(row)
        base: dict[str, Any] = {
            "run_id": run_id,
            "session_slug": session_slug,
            "evaluation_root": str(evaluation_root),
            "media_path": str(media_path),
            "media_mode": selected_media_mode,
            "candidate_source": candidate_source.source_id,
            "candidate_source_artifact": str(candidate_source.artifact_path),
            "index": index,
            "candidate_id": raw_candidate_id,
            "source_candidate_id": row.get("source_candidate_id") or row.get("id") or row.get("detectionId"),
            "timestamp_seconds": timestamp,
            "rough_status": "rough_unreviewed",
            "rough_unreviewed": True,
            "optional_context": optional_context(row),
            "warnings": list(warnings),
        }
        if error or start is None or end is None:
            failed.append({**base, "status": "skipped", "error": error or "invalid_window"})
            continue
        if previous_end is not None and start < previous_end:
            base["warnings"].append("overlaps_previous_clip")
        previous_end = max(previous_end or 0.0, end)
        start_ms = int(round(start * 1000))
        end_ms = int(round(end * 1000))
        filename = f"{session_slug}__{index:04d}__{file_candidate_id}__{start_ms}-{end_ms}__rough.mp4"
        output_path = (run_dir / "clips" / filename).resolve()
        try:
            output_path.relative_to(run_dir.resolve())
        except ValueError:
            failed.append({**base, "status": "skipped", "error": "unsafe_output_path"})
            continue
        if str(output_path) in used_output_paths:
            failed.append({**base, "status": "skipped", "error": "duplicate_output_path"})
            continue
        used_output_paths.add(str(output_path))
        planned.append(
            {
                **base,
                "clip_start_seconds": start,
                "clip_end_seconds": end,
                "duration_seconds": round(end - start, 3),
                "output_path": str(output_path),
                "status": "planned",
                "ffmpeg_command": equivalent_ffmpeg_command(media_path, output_path, start, end),
            }
        )

    existing_collisions = [row["output_path"] for row in planned if Path(row["output_path"]).exists()]
    if existing_collisions:
        for row in planned:
            if row["output_path"] in existing_collisions:
                row["warnings"].append("output_file_already_exists")
    return {
        "schema_version": "s003_rough_clip_export_v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not bool(args.execute),
        "execute": bool(args.execute),
        "evaluation_root": str(evaluation_root),
        "session_slug": session_slug,
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "clips_dir": str(run_dir / "clips"),
        "media_path": str(media_path),
        "media_mode_requested": args.media,
        "media_mode_selected": selected_media_mode,
        "media_duration_seconds": media_duration,
        "candidate_source": candidate_source.source_id,
        "candidate_source_artifact": str(candidate_source.artifact_path),
        "candidate_rows_loaded": len(candidate_source.rows),
        "candidate_rows_considered": len(rows),
        "planned_clips": planned,
        "failed_clips": failed,
        "collisions": existing_collisions,
        "summary": {
            "planned_clip_count": len(planned),
            "failed_clip_count": len(failed),
            "collision_count": len(existing_collisions),
            "estimated_total_clip_duration_seconds": round(sum(float(row["duration_seconds"]) for row in planned), 3),
            "rough_status": "rough_unreviewed",
        },
    }


CSV_FIELDS = [
    "run_id",
    "session_slug",
    "evaluation_root",
    "media_path",
    "media_mode",
    "candidate_source",
    "index",
    "candidate_id",
    "source_candidate_id",
    "timestamp_seconds",
    "clip_start_seconds",
    "clip_end_seconds",
    "duration_seconds",
    "output_path",
    "status",
    "rough_status",
    "rough_unreviewed",
    "warnings",
    "ffmpeg_command",
]


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (list, dict)) else row.get(key)
                    for key in CSV_FIELDS
                }
            )


def write_summary_md(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# Rough Clip Export Summary",
        "",
        f"Run id: `{plan['run_id']}`",
        f"Evaluation root: `{plan['evaluation_root']}`",
        f"Media: `{plan['media_path']}` ({plan['media_mode_selected']})",
        f"Candidate source: `{plan['candidate_source']}`",
        f"Mode: `{'execute' if plan['execute'] else 'dry-run'}`",
        f"Planned clips: {plan['summary']['planned_clip_count']}",
        f"Failed/skipped rows: {plan['summary']['failed_clip_count']}",
        f"Estimated total duration: {plan['summary']['estimated_total_clip_duration_seconds']}s",
        "",
        "Status: rough_unreviewed",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(plan: dict[str, Any]) -> None:
    run_dir = Path(plan["run_dir"])
    all_rows = plan["planned_clips"] + plan["failed_clips"]
    write_json(run_dir / "dry_run_plan.json", plan)
    write_json(run_dir / "manifest.json", {**plan, "manifest_rows": all_rows})
    write_manifest_csv(run_dir / "manifest.csv", all_rows)
    write_summary_md(run_dir / "EXPORT_SUMMARY.md", plan)
    if plan["failed_clips"]:
        write_json(run_dir / "failed_clips.json", plan["failed_clips"])


def execute_plan(plan: dict[str, Any], *, fail_fast: bool) -> None:
    for row in plan["planned_clips"]:
        output_path = Path(row["output_path"])
        try:
            extract_clip_ffmpeg(
                plan["media_path"],
                output_path,
                float(row["clip_start_seconds"]),
                float(row["clip_end_seconds"]),
            )
            row["status"] = "exported"
            row["output_exists"] = output_path.exists()
            row["output_size_bytes"] = output_path.stat().st_size if output_path.exists() else None
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            plan["failed_clips"].append(
                {
                    "index": row["index"],
                    "candidate_id": row["candidate_id"],
                    "output_path": row["output_path"],
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if fail_fast:
                break
    plan["summary"]["exported_clip_count"] = sum(1 for row in plan["planned_clips"] if row.get("status") == "exported")
    plan["summary"]["failed_clip_count"] = len(plan["failed_clips"])


def export_rough_clips(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.dry_run and args.execute:
        raise ValueError("Pass either --dry-run or --execute, not both.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer when provided.")
    if args.min_duration <= 0 or args.max_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError("--min-duration and --max-duration must be positive, with max >= min.")
    if args.pre_roll < 0 or args.post_roll <= 0:
        raise ValueError("--pre-roll must be >= 0 and --post-roll must be > 0.")

    plan = build_plan(args)
    run_dir = Path(plan["run_dir"])
    ensure_run_dir(run_dir, force=bool(args.force))
    if plan["collisions"] and not args.force:
        raise ValueError("Output file collisions detected; use --force with a fresh/intentional run id.")
    if args.execute:
        execute_plan(plan, fail_fast=bool(args.fail_fast))
    write_outputs(plan)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    try:
        plan = export_rough_clips(argv)
    except Exception as exc:
        print(f"export-rough-clips failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "run_dir": plan["run_dir"],
        "mode": "execute" if plan["execute"] else "dry-run",
        **plan["summary"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
