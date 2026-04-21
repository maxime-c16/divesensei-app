from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from divesensei.workflows.evaluation_session_support import load_jsonl, read_json, resolve_evaluation_session_paths, write_jsonl


PRIMARY_ANCHOR_STRATEGY = "proposal_centered"
BACKUP_ANCHOR_STRATEGY = "earliest_strong_peak_in_local_cluster"
SPRINGBOARD_STRONG_PEAK_OFFSET_SECONDS = 0.35

SPRINGBOARD_SESSION_KEYS = ("insep_15min", "Champigny")
PLATFORM_SESSION_KEYS = ("IMG_9015",)


@dataclass(frozen=True)
class SessionTypeInfo:
    session_type: str
    provenance: str
    confidence: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-event-window-manifest",
        description="Export a preview event-window manifest from reviewed evaluation sessions.",
    )
    parser.add_argument(
        "session_roots",
        nargs="*",
        help="Evaluation session roots or manifest/report paths. If omitted, a default preview set is used.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/event_window_manifest_preview.jsonl",
        help="Path to write the preview manifest JSONL.",
    )
    parser.add_argument(
        "--summary-md",
        default="outputs/event_window_manifest_preview_summary.md",
        help="Path to write the preview summary markdown.",
    )
    return parser


def _default_preview_roots() -> list[str]:
    return [
        "outputs/evaluation_insep_15min_validated",
        "outputs/evaluation_champigny_20260406-labelling",
        "outputs/evaluation_insep_quick_9015_20260409_ui",
    ]


def _session_type_for_source_video(source_video_path: str) -> SessionTypeInfo:
    lowered = source_video_path.lower()
    if any(key.lower() in lowered for key in SPRINGBOARD_SESSION_KEYS):
        if "insep_15min" in lowered:
            return SessionTypeInfo("springboard", "direct_review", "high")
        return SessionTypeInfo("springboard", "session_type_inferred", "medium")
    if any(key.lower() in lowered for key in PLATFORM_SESSION_KEYS):
        return SessionTypeInfo("platform", "session_type_inferred", "medium")
    if "insep quick" in lowered or "img_9015" in lowered:
        return SessionTypeInfo("platform", "session_type_inferred", "medium")
    return SessionTypeInfo("unknown", "uncertain", "low")


def _event_label_for_row(session_type: str, row: dict[str, Any]) -> tuple[str, str, bool]:
    review_label = str(row.get("review_label") or "")
    subtype = str(row.get("subtype") or "")
    human_label = str(row.get("human_label") or "")
    if review_label == "dive" or human_label == "dive":
        if session_type == "springboard":
            return "springboard_dive", "session_type_inferred", False
        if session_type == "platform":
            return "platform_dive", "session_type_inferred", False
        return "platform_dive", "uncertain", True
    if review_label == "non_dive" and subtype == "board_rebound":
        return "springboard_rebound_only", "subtype_mapped", False
    if review_label == "non_dive" and subtype in {"voice_whistle", "handling_noise", "non_dive_splash"}:
        return "noise_or_other", "subtype_mapped", False
    if review_label == "non_dive" and not subtype:
        return "noise_or_other", "uncertain", True
    if review_label == "false_negative" or human_label == "dive":
        if session_type == "springboard":
            return "springboard_dive", "session_type_inferred", False
        if session_type == "platform":
            return "platform_dive", "session_type_inferred", False
        return "platform_dive", "uncertain", True
    return "noise_or_other", "uncertain", True


def _anchor_timestamp(row: dict[str, Any]) -> float:
    for key in ("proposal_timestamp_seconds", "timestamp_seconds", "timestamp", "matched_proposal_delta_seconds"):
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    return 0.0


def _legacy_scores(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_timestamp_seconds": row.get("proposal_timestamp_seconds", row.get("timestamp_seconds", row.get("timestamp"))),
        "proposal_frontend": row.get("proposal_frontend"),
        "clip_probability": row.get("audio_clip_probability"),
        "audio_score": row.get("audio_score", row.get("raw_proposal_score")),
        "combined_score": row.get("combined_score", row.get("final_combined_score")),
        "governed_r9_score": row.get("governed_r9_score"),
        "audio_model_probability": row.get("audio_model_probability"),
        "audio_clip_probability": row.get("audio_clip_probability"),
        "visual_late_fusion_logreg_c0.5": row.get("visual_late_fusion_logreg_c0.5"),
        "raw_proposal_score": row.get("raw_proposal_score"),
        "threshold_passed": row.get("threshold_passed"),
    }


def _manifest_row(
    *,
    source_root: str,
    source_session_id: str,
    source_video_path: str,
    session_type_info: SessionTypeInfo,
    row: dict[str, Any],
    is_false_negative: bool,
    anchor_strategy: str,
) -> dict[str, Any]:
    event_label, event_label_provenance, uncertainty = _event_label_for_row(session_type_info.session_type, row)
    proposal_timestamp = float(row.get("proposal_timestamp_seconds") or row.get("timestamp_seconds") or row.get("timestamp") or _anchor_timestamp(row))
    anchor_timestamp = proposal_timestamp
    window_pre = 0.75 if anchor_strategy == PRIMARY_ANCHOR_STRATEGY else 1.0
    window_post = 2.25 if anchor_strategy == PRIMARY_ANCHOR_STRATEGY else 3.0
    if is_false_negative:
        anchor_timestamp = float(row.get("timestamp_seconds") or row.get("timestamp") or anchor_timestamp)
    elif session_type_info.session_type == "springboard":
        anchor_timestamp = max(0.0, proposal_timestamp - SPRINGBOARD_STRONG_PEAK_OFFSET_SECONDS)
    else:
        anchor_timestamp = proposal_timestamp
    return {
        "source_session_root": source_root,
        "source_session_id": source_session_id,
        "source_video_path": source_video_path,
        "session_type": session_type_info.session_type,
        "session_type_provenance": session_type_info.provenance,
        "session_type_confidence": session_type_info.confidence,
        "legacy_candidate_id": row.get("source_candidate_id") or row.get("detectionId") or row.get("proposal_id"),
        "legacy_candidate_label": row.get("review_label") or row.get("label") or row.get("human_label"),
        "legacy_non_dive_subtype": row.get("subtype"),
        "is_false_negative_window": bool(is_false_negative),
        "event_anchor_timestamp_seconds": anchor_timestamp,
        "anchor_strategy": "earliest_strong_peak_in_local_cluster" if session_type_info.session_type == "springboard" else PRIMARY_ANCHOR_STRATEGY,
        "event_window_start_seconds": max(0.0, anchor_timestamp - window_pre),
        "event_window_end_seconds": anchor_timestamp + window_post,
        "event_label": event_label,
        "event_label_provenance": event_label_provenance,
        "uncertainty_flag": bool(uncertainty),
        **_legacy_scores(row),
    }


def _load_rows_for_session(root: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = resolve_evaluation_session_paths(root)
    manifest = read_json(paths["manifest_path"])
    export_dir = Path(str(paths["output_dir"])) / "exports" / "evaluation-review"
    reviewed_candidates = load_jsonl(export_dir / "reviewed_candidates.jsonl")
    false_negatives = load_jsonl(export_dir / "false_negatives.jsonl")
    return manifest, reviewed_candidates, false_negatives


def build_preview_rows(session_roots: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    for root in session_roots:
        manifest, reviewed_candidates, false_negatives = _load_rows_for_session(root)
        session = manifest["session"]
        session_id = str(session.get("id"))
        source_video_path = str(session.get("source_video_path"))
        session_type_info = _session_type_for_source_video(source_video_path)

        source_root = str(Path(root).resolve())
        for candidate in reviewed_candidates:
            rows.append(
                _manifest_row(
                    source_root=source_root,
                    source_session_id=session_id,
                    source_video_path=source_video_path,
                    session_type_info=session_type_info,
                    row=candidate,
                    is_false_negative=False,
                    anchor_strategy=PRIMARY_ANCHOR_STRATEGY,
                )
            )
        for false_negative in false_negatives:
            rows.append(
                _manifest_row(
                    source_root=source_root,
                    source_session_id=session_id,
                    source_video_path=source_video_path,
                    session_type_info=session_type_info,
                    row=false_negative,
                    is_false_negative=True,
                    anchor_strategy=PRIMARY_ANCHOR_STRATEGY,
                )
            )

        session_summaries.append(
            {
                "source_root": source_root,
                "source_session_id": session_id,
                "source_video_path": source_video_path,
                "session_type": session_type_info.session_type,
                "session_type_provenance": session_type_info.provenance,
                "reviewed_candidates": len(reviewed_candidates),
                "false_negatives": len(false_negatives),
            }
        )

    summary = {
        "preview_row_count": len(rows),
        "session_summaries": session_summaries,
    }
    return rows, summary


def _counts(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key)) for row in rows)
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def write_summary_md(path: Path, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> Path:
    session_summaries = summary["session_summaries"]
    lines = [
        "# Event Window Manifest Preview",
        "",
        f"- preview rows: `{len(rows)}`",
        f"- sessions used: `{len(session_summaries)}`",
        f"- primary anchor: `{PRIMARY_ANCHOR_STRATEGY}`",
        f"- backup anchor: `{BACKUP_ANCHOR_STRATEGY}`",
        "",
        "## Sessions",
        "",
        "| source root | source session | session type | provenance | reviewed candidates | false negatives |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in session_summaries:
        lines.append(
            f"| `{item['source_root']}` | `{item['source_session_id']}` | `{item['session_type']}` | `{item['session_type_provenance']}` | {item['reviewed_candidates']} | {item['false_negatives']} |"
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- event labels: `{json.dumps(_counts(rows, 'event_label'), sort_keys=True)}`",
            f"- label provenance: `{json.dumps(_counts(rows, 'event_label_provenance'), sort_keys=True)}`",
            f"- session type: `{json.dumps(_counts(rows, 'session_type'), sort_keys=True)}`",
            f"- session type provenance: `{json.dumps(_counts(rows, 'session_type_provenance'), sort_keys=True)}`",
            "",
            "## Notes",
            "",
            "- legacy candidate labels and detector scores are preserved as metadata",
            "- no legacy label semantics were rewritten in place",
            "- all event labels in this preview are provenance-tagged",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv or []))
    session_roots = list(args.session_roots) if args.session_roots else _default_preview_roots()
    rows, summary = build_preview_rows(session_roots)
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    summary_md = Path(args.summary_md).expanduser().resolve()
    write_jsonl(output_jsonl, rows)
    write_summary_md(summary_md, rows, summary)
    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "summary_md": str(summary_md),
                "preview_row_count": len(rows),
                "event_label_counts": _counts(rows, "event_label"),
                "label_provenance_counts": _counts(rows, "event_label_provenance"),
                "session_type_counts": _counts(rows, "session_type"),
                "session_type_provenance_counts": _counts(rows, "session_type_provenance"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
