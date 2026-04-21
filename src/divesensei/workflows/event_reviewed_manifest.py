from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from divesensei.workflows.evaluation_session_support import load_jsonl, read_json, resolve_evaluation_session_paths, write_json, write_jsonl

DEFAULT_OUTPUT_ROOT = "exports/event-reviewed-manifest"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-event-reviewed-manifest",
        description="Export human-reviewed event labels for dataset construction.",
    )
    parser.add_argument("session_path", help="Evaluation session output dir, ui_session_manifest.json, or session_pipeline_report.json")
    parser.add_argument("--output-dir", default="", help="Optional export directory. Defaults to <session>/exports/event-reviewed-manifest")
    parser.add_argument("--preview-json", default="outputs/event_reviewed_manifest_preview.json")
    parser.add_argument("--preview-md", default="outputs/event_reviewed_manifest_preview.md")
    return parser


def _load_event_support(export_dir: Path) -> list[dict[str, Any]]:
    support_path = export_dir / "exports" / "event-review-support" / "event_review_support.jsonl"
    return load_jsonl(support_path)


def _load_decisions(review_path: Path) -> dict[str, dict[str, Any]]:
    store = read_json(review_path) or {"decisions": [], "falseNegatives": []}
    decisions = {}
    for row in store.get("decisions", []):
        detection_id = str(row.get("detectionId") or "")
        if detection_id:
            decisions[detection_id] = row
    for row in store.get("falseNegatives", []):
        annotation_id = str(row.get("id") or "")
        if annotation_id:
            decisions[annotation_id] = {
                "id": annotation_id,
                "detectionId": annotation_id,
                "label": row.get("label") or "false_negative",
                "eventLabel": row.get("eventLabel"),
                "subtype": row.get("subtype"),
                "notes": row.get("notes"),
                "createdAt": row.get("createdAt"),
                "updatedAt": row.get("updatedAt"),
            }
    return decisions


def _final_event_label(decision: dict[str, Any] | None) -> tuple[str | None, str | None, bool]:
    if not decision:
        return None, None, False
    event_label = decision.get("eventLabel")
    if not event_label:
        return None, None, False
    return str(event_label), "human_reviewed", True


def build_reviewed_manifest(session_path: str, output_dir: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = resolve_evaluation_session_paths(session_path)
    manifest = read_json(paths["manifest_path"])
    if not manifest:
        raise FileNotFoundError(f"Could not read session manifest: {paths['manifest_path']}")
    export_dir = Path(str(paths["output_dir"]))
    event_support_rows = _load_event_support(export_dir)
    decisions = _load_decisions(Path(str(paths["output_dir"])) / "evaluation_review.json")
    source_root = str(Path(session_path).resolve())
    rows: list[dict[str, Any]] = []
    for row in event_support_rows:
        legacy_candidate_id = str(row.get("legacy_candidate_id") or "")
        decision = decisions.get(legacy_candidate_id) if legacy_candidate_id else None
        final_event_label, final_provenance, human_reviewed = _final_event_label(decision)
        rows.append(
            {
                "row_key": f"{manifest['session']['id']}::{legacy_candidate_id}" if legacy_candidate_id else None,
                "source_session_root": source_root,
                "source_session_id": manifest["session"]["id"],
                "source_video_path": manifest["session"]["source_video_path"],
                "session_type": row.get("suggested_session_type_context"),
                "session_type_provenance": row.get("session_type_provenance"),
                "legacy_candidate_id": row.get("legacy_candidate_id"),
                "legacy_decision_label": row.get("legacy_candidate_label"),
                "legacy_subtype": row.get("legacy_non_dive_subtype"),
                "suggested_event_label": row.get("suggested_event_label"),
                "suggested_event_label_confidence": row.get("suggested_event_label_confidence"),
                "suggested_event_label_reason": row.get("suggested_event_label_reason"),
                "has_preceding_rebound_context": row.get("has_preceding_rebound_context"),
                "has_delayed_entry_candidate": row.get("has_delayed_entry_candidate"),
                "no_rebound_context_detected": row.get("no_rebound_context_detected"),
                "event_anchor_timestamp_seconds": row.get("event_anchor_timestamp_seconds"),
                "event_anchor_strategy": row.get("event_anchor_strategy"),
                "event_anchor_strategy_rationale": row.get("event_anchor_strategy_rationale"),
                "event_window_start_seconds": row.get("event_window_start_seconds"),
                "event_window_end_seconds": row.get("event_window_end_seconds"),
                "proposal_timestamp_seconds": row.get("proposal_timestamp_seconds"),
                "proposal_frontend": row.get("proposal_frontend"),
                "clip_probability": row.get("clip_probability"),
                "detector_scores": row.get("detector_scores"),
                "manual_correction_type": row.get("manual_correction_type"),
                "manual_correction_rationale": row.get("manual_correction_rationale"),
                "final_human_event_label": final_event_label,
                "final_human_event_label_provenance": final_provenance,
                "human_reviewed_at_event_level": human_reviewed,
                "review_decision_label": decision.get("label") if decision else None,
                "review_subtype": decision.get("subtype") if decision else None,
                "review_notes": decision.get("notes") if decision else None,
                "legacy_decision_human_reviewed": decision is not None,
            }
        )

    summary = {
        "session_id": manifest["session"]["id"],
        "row_count": len(rows),
        "reviewed_count": sum(1 for row in rows if row.get("human_reviewed_at_event_level")),
        "missing_event_label_count": sum(1 for row in rows if not row.get("final_human_event_label")),
        "final_label_counts": dict(Counter(str(row.get("final_human_event_label")) for row in rows if row.get("final_human_event_label"))),
        "legacy_label_counts": dict(Counter(str(row.get("legacy_decision_label")) for row in rows)),
        "agreement_counts": {
            "agree": sum(
                1
                for row in rows
                if row.get("final_human_event_label")
                and str(row.get("final_human_event_label")) == str(row.get("suggested_event_label"))
            ),
            "disagree": sum(
                1
                for row in rows
                if row.get("final_human_event_label")
                and str(row.get("final_human_event_label")) != str(row.get("suggested_event_label"))
            ),
        },
        "source_root": source_root,
    }
    return rows, summary


def write_preview_md(path: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> Path:
    lines = [
        "# Event Reviewed Manifest Preview",
        "",
        f"- session: `{summary['session_id']}`",
        f"- rows: `{summary['row_count']}`",
        f"- reviewed rows: `{summary['reviewed_count']}`",
        f"- missing final event labels: `{summary['missing_event_label_count']}`",
        "",
        "## Counts",
        "",
        f"- final labels: `{json.dumps(summary['final_label_counts'], sort_keys=True)}`",
        f"- legacy labels: `{json.dumps(summary['legacy_label_counts'], sort_keys=True)}`",
        f"- suggestion agreement: `{json.dumps(summary['agreement_counts'], sort_keys=True)}`",
        "",
        "## Example Reviewed Rows",
        "",
    ]
    for row in rows[:5]:
        lines.append(
            f"- `{row.get('legacy_candidate_id')}` -> final `{row.get('final_human_event_label')}` / suggestion `{row.get('suggested_event_label')}`"
        )
    lines.append("")
    lines.extend([
        "## Notes",
        "",
        "- final event labels are human-reviewed values from the desktop review store",
        "- legacy detector decisions and suggestion fields are preserved",
        "- missing rows remain visible for dataset completeness tracking",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv or []))
    rows, summary = build_reviewed_manifest(args.session_path, args.output_dir or None)
    paths = resolve_evaluation_session_paths(args.session_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(str(paths["output_dir"])) / DEFAULT_OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "event_reviewed_manifest.jsonl"
    md_path = output_dir / "event_reviewed_manifest_summary.md"
    write_jsonl(json_path, rows)
    write_preview_md(md_path, summary, rows)
    write_json(Path(args.preview_json), summary)
    write_preview_md(Path(args.preview_md), summary, rows)
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "summary": summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
