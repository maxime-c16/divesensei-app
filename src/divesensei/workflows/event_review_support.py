from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from divesensei.workflows.evaluation_session_support import load_jsonl, read_json, resolve_evaluation_session_paths, update_session_artifacts, write_json, write_jsonl


DEFAULT_OUTPUT_ROOT = "exports/event-review-support"
WINDOW_PRE_SECONDS = 0.75
WINDOW_POST_SECONDS = 2.25
REBOUND_CONTEXT_SECONDS = 1.5
SPRINGBOARD_STRONG_PEAK_OFFSET_SECONDS = 0.35

SESSION_EVENT_TYPE_OVERRIDES: dict[str, dict[str, str]] = {
    "evaluation_champigny_20260406-labelling": {
        "det-0002": "platform",
        "det-0003": "platform",
        "det-0004": "platform",
        "det-0005": "platform",
        "det-0006": "platform",
        "det-0109": "platform",
        "det-0110": "platform",
        "det-0111": "platform",
    }
}


@dataclass(frozen=True)
class ReviewSupportRow:
    source_session_root: str
    source_session_id: str
    source_video_path: str
    legacy_candidate_id: str | None
    legacy_candidate_label: str | None
    legacy_non_dive_subtype: str | None
    is_false_negative_window: bool
    event_anchor_timestamp_seconds: float
    event_window_start_seconds: float
    event_window_end_seconds: float
    suggested_event_label: str | None
    suggested_event_label_confidence: str
    suggested_event_label_reason: str
    suggested_session_type_context: str
    has_preceding_rebound_context: bool
    has_delayed_entry_candidate: bool
    no_rebound_context_detected: bool
    event_label_provenance_suggestion: str
    event_anchor_strategy: str
    event_anchor_strategy_rationale: str
    uncertainty_flag: bool
    proposal_timestamp_seconds: float | None
    proposal_frontend: str | None
    clip_probability: float | None
    detector_scores: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-event-review-support",
        description="Precompute event-review hints from reviewed evaluation sessions.",
    )
    parser.add_argument("session_path", help="Evaluation session output dir, ui_session_manifest.json, or session_pipeline_report.json")
    parser.add_argument("--output-dir", default="", help="Optional export directory. Defaults to <session>/exports/event-review-support")
    parser.add_argument("--window-pre-seconds", type=float, default=WINDOW_PRE_SECONDS)
    parser.add_argument("--window-post-seconds", type=float, default=WINDOW_POST_SECONDS)
    parser.add_argument("--rebound-context-seconds", type=float, default=REBOUND_CONTEXT_SECONDS)
    return parser


def _infer_session_type(source_video_path: str) -> tuple[str, str]:
    lowered = source_video_path.lower()
    if "insep_15min" in lowered:
        return "springboard", "direct_review"
    if "champigny" in lowered:
        return "springboard", "direct_review"
    if "img_9015" in lowered or "insep quick" in lowered:
        return "platform", "session_type_inferred"
    return "unknown", "uncertain"


def _override_event_type(session_id: str, candidate_id: str | None, fallback: str, fallback_provenance: str) -> tuple[str, str]:
    if not candidate_id:
        return fallback, fallback_provenance
    override = SESSION_EVENT_TYPE_OVERRIDES.get(session_id, {}).get(candidate_id)
    if override:
        return override, "manual_session_override"
    return fallback, fallback_provenance


def _suggested_label(
    row: dict[str, Any],
    session_type: str,
    session_type_provenance: str,
    preceding_rebound: bool,
    delayed_entry: bool,
) -> tuple[str, str, str, bool, str]:
    review_label = str(row.get("review_label") or "")
    subtype = str(row.get("subtype") or "")
    human_label = str(row.get("human_label") or "")
    trustworthy_session_type = session_type in {"springboard", "platform"} and session_type_provenance in {"direct_review", "manual_session_override"}
    session_type_is_conservative = session_type in {"springboard", "platform"} and session_type_provenance not in {"direct_review", "manual_session_override"}
    no_rebound_context = not preceding_rebound
    platform_session_trustworthy = session_type == "platform" and session_type_provenance in {"session_type_inferred", "manual_session_override"}
    if review_label == "dive" or human_label == "dive":
        if session_type == "springboard" and trustworthy_session_type:
            if preceding_rebound:
                return "springboard_dive", "rebound_context_plus_delayed_entry", "high", False, "session_type_inferred"
            return "springboard_dive", "springboard_dive_without_rebound_context", "low", False, "session_type_inferred"
        if session_type == "platform" and (trustworthy_session_type or platform_session_trustworthy):
            if no_rebound_context:
                return "platform_dive", "platform_session_dive_without_rebound_context", "high", False, "session_type_inferred"
            return "platform_dive", "platform_session_dive_with_rebound_context", "high", False, "session_type_inferred"
        if session_type_is_conservative:
            return "uncertain", "insufficient_context_uncertain", "low", True, "uncertain"
        return "uncertain", "insufficient_context_uncertain", "low", True, "uncertain"
    if review_label == "non_dive" and subtype == "board_rebound":
        if session_type == "platform" and (trustworthy_session_type or platform_session_trustworthy):
            return "springboard_rebound_only", "board_rebound_platform_session", "high", False, "subtype_mapped"
        if delayed_entry:
            return "springboard_rebound_only", "board_rebound_with_delayed_entry_context", "medium", False, "subtype_mapped"
        if session_type == "springboard" and trustworthy_session_type:
            return "springboard_rebound_only", "board_rebound_without_delayed_entry", "high", False, "subtype_mapped"
        return "uncertain", "insufficient_context_uncertain", "low", True, "uncertain"
    if review_label == "non_dive" and subtype in {"voice_whistle", "handling_noise", "non_dive_splash"}:
        return "noise_or_other", f"negative_subtype_{subtype}", "high", False, "subtype_mapped"
    if review_label == "non_dive" and not subtype:
        if session_type == "platform" and (trustworthy_session_type or platform_session_trustworthy):
            return "noise_or_other", "platform_session_generic_non_dive", "medium", False, "subtype_mapped"
        return "uncertain", "insufficient_context_uncertain", "low", True, "uncertain"
    if review_label == "false_negative":
        if session_type == "springboard" and trustworthy_session_type:
            return "springboard_dive", "false_negative_dive_springboard_session", "medium", False, "session_type_inferred"
        if session_type == "platform" and (trustworthy_session_type or platform_session_trustworthy):
            return "platform_dive", "false_negative_dive_platform_session", "high", False, "session_type_inferred"
        return "uncertain", "insufficient_context_uncertain", "low", True, "uncertain"
    return "uncertain", "insufficient_context_uncertain", "low", True, "uncertain"


def _event_window(anchor: float, pre_seconds: float, post_seconds: float) -> tuple[float, float]:
    return max(0.0, anchor - pre_seconds), anchor + post_seconds


def _candidate_rows(export_dir: Path) -> list[dict[str, Any]]:
    return load_jsonl(export_dir / "reviewed_candidates.jsonl")


def _false_negative_rows(export_dir: Path) -> list[dict[str, Any]]:
    return load_jsonl(export_dir / "false_negatives.jsonl")


def _apply_latest_review_store(output_dir: Path, candidate_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    review_store_path = output_dir / "evaluation_review.json"
    if not review_store_path.exists():
        return list(candidate_rows)
    review_store = read_json(review_store_path)
    decisions = {
        str(item.get("detectionId")): item
        for item in (review_store.get("decisions") or [])
        if item.get("detectionId")
    }
    merged_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        merged = dict(row)
        candidate_id = str(row.get("source_candidate_id") or row.get("detectionId") or row.get("proposal_id") or "")
        decision = decisions.get(candidate_id)
        if decision:
            merged["review_label"] = decision.get("label", merged.get("review_label"))
            merged["subtype"] = decision.get("subtype", merged.get("subtype"))
            merged["manual_anchor_timestamp_seconds"] = decision.get("manualAnchorTimestampSeconds")
            merged["manual_window_start_seconds"] = decision.get("manualWindowStartSeconds")
            merged["manual_window_end_seconds"] = decision.get("manualWindowEndSeconds")
            merged["manual_correction_type"] = decision.get("manualCorrectionType")
            merged["manual_correction_rationale"] = decision.get("manualCorrectionRationale")
        merged_rows.append(merged)
    return merged_rows


def _support_row(
    *,
    source_root: str,
    source_session_id: str,
    source_video_path: str,
    row: dict[str, Any],
    is_false_negative_window: bool,
    session_type: str,
    session_type_provenance: str,
    pre_seconds: float,
    post_seconds: float,
    rebound_context_seconds: float,
    candidate_rows: Sequence[dict[str, Any]],
    false_negative_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    proposal_timestamp = float(row.get("proposal_timestamp_seconds") or row.get("timestamp_seconds") or row.get("timestamp") or 0.0)
    anchor = proposal_timestamp
    if is_false_negative_window:
        anchor = float(row.get("timestamp_seconds") or anchor)
    window_start, window_end = _event_window(anchor, pre_seconds, post_seconds)
    if is_false_negative_window:
        candidate_id = row.get("review_annotation_id") or row.get("source_candidate_id") or row.get("proposal_id")
    else:
        candidate_id = row.get("source_candidate_id") or row.get("detectionId") or row.get("proposal_id")
    event_type_context, event_type_provenance = _override_event_type(
        source_session_id,
        str(candidate_id) if candidate_id is not None else None,
        session_type,
        session_type_provenance,
    )
    preceding_rebound = any(
        cand.get("review_label") == "non_dive"
        and str(cand.get("subtype") or "") == "board_rebound"
        and 0.0 <= anchor - float(cand.get("timestamp_seconds") or cand.get("timestamp") or 0.0) <= rebound_context_seconds
        for cand in candidate_rows
    )
    delayed_entry = any(
        cand.get("review_label") == "dive"
        and 0.0 <= float(cand.get("timestamp_seconds") or cand.get("timestamp") or 0.0) - anchor <= post_seconds
        for cand in candidate_rows
    ) or any(
        fn.get("human_label") == "dive"
        and 0.0 <= float(fn.get("timestamp_seconds") or 0.0) - anchor <= post_seconds
        for fn in false_negative_rows
    )
    suggestion, reason, confidence, uncertainty, label_provenance_suggestion = _suggested_label(
        row, event_type_context, event_type_provenance, preceding_rebound, delayed_entry
    )
    if event_type_context == "springboard":
        anchor_strategy = "earliest_strong_peak_in_local_cluster"
        anchor = max(0.0, proposal_timestamp - SPRINGBOARD_STRONG_PEAK_OFFSET_SECONDS)
        window_start, window_end = _event_window(anchor, pre_seconds, post_seconds)
        anchor_rationale = "springboard_rows_shift_to_earliest_strong_peak_proxy"
    elif event_type_context == "platform":
        anchor_strategy = "proposal_centered"
        anchor_rationale = "platform_rows_remain_proposal_centered"
    else:
        anchor_strategy = "proposal_centered"
        anchor_rationale = "unknown_rows_use_proposal_centered_fallback"
    manual_anchor = row.get("manual_anchor_timestamp_seconds")
    manual_window_start = row.get("manual_window_start_seconds")
    manual_window_end = row.get("manual_window_end_seconds")
    manual_correction_type = row.get("manual_correction_type")
    manual_correction_rationale = row.get("manual_correction_rationale")
    if manual_anchor is not None or manual_window_start is not None or manual_window_end is not None:
        anchor = float(manual_anchor if manual_anchor is not None else anchor)
        window_start = float(manual_window_start if manual_window_start is not None else window_start)
        window_end = float(manual_window_end if manual_window_end is not None else window_end)
        anchor_strategy = "manual_review_override"
        anchor_rationale = str(manual_correction_rationale or manual_correction_type or "manual_review_override")
    return {
        "source_session_root": source_root,
        "source_session_id": source_session_id,
        "source_video_path": source_video_path,
        "legacy_candidate_id": candidate_id,
        "legacy_candidate_label": row.get("review_label") or row.get("label") or row.get("human_label"),
        "legacy_non_dive_subtype": row.get("subtype"),
        "is_false_negative_window": bool(is_false_negative_window),
        "event_anchor_timestamp_seconds": anchor,
        "event_anchor_strategy": anchor_strategy,
        "event_anchor_strategy_rationale": anchor_rationale,
        "manual_correction_type": manual_correction_type,
        "manual_correction_rationale": manual_correction_rationale,
        "event_window_start_seconds": window_start,
        "event_window_end_seconds": window_end,
        "suggested_event_label": suggestion,
        "suggested_event_label_confidence": confidence,
        "suggested_event_label_reason": reason,
        "suggested_session_type_context": event_type_context,
        "has_preceding_rebound_context": preceding_rebound,
        "has_delayed_entry_candidate": delayed_entry,
        "no_rebound_context_detected": not preceding_rebound,
        "event_label_provenance_suggestion": label_provenance_suggestion,
        "uncertainty_flag": uncertainty,
        "proposal_timestamp_seconds": proposal_timestamp,
        "proposal_frontend": row.get("proposal_frontend"),
        "clip_probability": row.get("audio_clip_probability"),
        "detector_scores": {
            "audio_score": row.get("audio_score"),
            "combined_score": row.get("combined_score"),
            "governed_r9_score": row.get("governed_r9_score"),
            "audio_model_probability": row.get("audio_model_probability"),
            "audio_clip_probability": row.get("audio_clip_probability"),
            "visual_late_fusion_logreg_c0.5": row.get("visual_late_fusion_logreg_c0.5"),
            "raw_proposal_score": row.get("raw_proposal_score"),
            "threshold_passed": row.get("threshold_passed"),
        },
        "session_type_provenance": event_type_provenance,
    }


def build_review_support(session_path: str, output_dir: str | None, pre_seconds: float, post_seconds: float, rebound_context_seconds: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = resolve_evaluation_session_paths(session_path)
    manifest = read_json(paths["manifest_path"])
    output_root = Path(str(paths["output_dir"]))
    export_dir = output_root / "exports" / "evaluation-review"
    candidate_rows = _apply_latest_review_store(output_root, _candidate_rows(export_dir))
    false_negative_rows = _false_negative_rows(export_dir)
    source_video_path = str(manifest["session"]["source_video_path"])
    session_type, session_type_provenance = _infer_session_type(source_video_path)
    source_root = str(Path(session_path).resolve())
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        rows.append(
            _support_row(
                source_root=source_root,
                source_session_id=str(manifest["session"]["id"]),
                source_video_path=source_video_path,
                row=row,
                is_false_negative_window=False,
                session_type=session_type,
                session_type_provenance=session_type_provenance,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                rebound_context_seconds=rebound_context_seconds,
                candidate_rows=candidate_rows,
                false_negative_rows=false_negative_rows,
            )
        )
    for row in false_negative_rows:
        rows.append(
            _support_row(
                source_root=source_root,
                source_session_id=str(manifest["session"]["id"]),
                source_video_path=source_video_path,
                row=row,
                is_false_negative_window=True,
                session_type=session_type,
                session_type_provenance=session_type_provenance,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                rebound_context_seconds=rebound_context_seconds,
                candidate_rows=candidate_rows,
                false_negative_rows=false_negative_rows,
            )
        )

    summary = {
        "session_id": manifest["session"]["id"],
        "session_type": session_type,
        "session_type_provenance": session_type_provenance,
        "row_count": len(rows),
        "suggested_event_label_counts": dict(Counter(str(row.get("suggested_event_label")) for row in rows)),
        "suggested_event_label_confidence_counts": dict(Counter(str(row.get("suggested_event_label_confidence")) for row in rows)),
        "suggested_event_label_reason_counts": dict(Counter(str(row.get("suggested_event_label_reason")) for row in rows)),
        "has_preceding_rebound_context_count": sum(1 for row in rows if row.get("has_preceding_rebound_context")),
        "has_delayed_entry_candidate_count": sum(1 for row in rows if row.get("has_delayed_entry_candidate")),
        "no_rebound_context_detected_count": sum(1 for row in rows if row.get("no_rebound_context_detected")),
        "uncertainty_count": sum(1 for row in rows if row.get("uncertainty_flag")),
        "session_aware_downgrade_count": sum(1 for row in rows if row.get("uncertainty_flag") and row.get("suggested_event_label") is None),
        "source_root": source_root,
    }
    return rows, summary


def write_summary_md(path: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> Path:
    examples = {
        "springboard_dive": [row for row in rows if row.get("suggested_event_label") == "springboard_dive"][:3],
        "springboard_rebound_only": [row for row in rows if row.get("suggested_event_label") == "springboard_rebound_only"][:3],
        "platform_dive": [row for row in rows if row.get("suggested_event_label") == "platform_dive"][:3],
        "noise_or_other": [row for row in rows if row.get("suggested_event_label") == "noise_or_other"][:3],
        "uncertain": [row for row in rows if row.get("uncertainty_flag")][:3],
    }
    lines = [
        "# Event Review Support Preview",
        "",
        f"- session: `{summary['session_id']}`",
        f"- session type: `{summary['session_type']}`",
        f"- session type provenance: `{summary['session_type_provenance']}`",
        f"- rows: `{summary['row_count']}`",
        "",
        "## Counts",
        "",
        f"- suggestion types: `{json.dumps(summary['suggested_event_label_counts'], sort_keys=True)}`",
        f"- suggestion confidence: `{json.dumps(summary['suggested_event_label_confidence_counts'], sort_keys=True)}`",
        f"- suggestion reasons: `{json.dumps(summary['suggested_event_label_reason_counts'], sort_keys=True)}`",
        f"- rebound-context hints: `{summary['has_preceding_rebound_context_count']}`",
        f"- delayed-entry hints: `{summary['has_delayed_entry_candidate_count']}`",
        f"- no-rebound-context hints: `{summary['no_rebound_context_detected_count']}`",
        f"- uncertain rows: `{summary['uncertainty_count']}`",
        f"- session-aware downgrades to uncertain: `{summary['session_aware_downgrade_count']}`",
        "",
        "## Examples",
        "",
    ]
    for label, items in examples.items():
        lines.append(f"### {label}")
        if not items:
            lines.append("- none")
        for item in items:
            lines.append(
                f"- `{item.get('source_session_id')}` @ `{item.get('event_anchor_timestamp_seconds'):.3f}` -> `{item.get('suggested_event_label')}` | {item.get('suggested_event_label_reason')}"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- suggestions are machine-generated hints only",
            "- legacy candidate labels and detector scores are preserved",
            "- human review remains the final source of truth",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv or []))
    session_path = args.session_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    rows, summary = build_review_support(session_path, str(output_dir) if output_dir else None, args.window_pre_seconds, args.window_post_seconds, args.rebound_context_seconds)
    if output_dir is None:
        paths = resolve_evaluation_session_paths(session_path)
        output_dir = Path(str(paths["output_dir"])) / DEFAULT_OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "event_review_support.jsonl"
    md_path = output_dir / "event_review_support_summary.md"
    write_jsonl(json_path, rows)
    write_summary_md(md_path, summary, rows)
    update_session_artifacts(
        resolve_evaluation_session_paths(session_path),
        {
            "event_review_support": str(json_path),
            "event_review_support_summary": str(md_path),
        },
    )
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "row_count": len(rows),
                "summary": summary,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
