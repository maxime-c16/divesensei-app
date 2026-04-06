#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from divesensei.detection.audio_clip_model import AudioClipModel
from divesensei.detection.audio_features import AUDIO_CLIP_FEATURES
from divesensei.workflows.evaluation_session_support import (
    infer_domain_metadata,
    load_evaluation_review_data,
    load_jsonl,
    normalize_non_dive_subtype,
    resolve_evaluation_session_paths,
    update_session_artifacts,
    write_json,
    write_jsonl,
    read_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-evaluation-review",
        description="Export reviewed evaluation sessions into hard-negative, diagnostics, and summary artifacts.",
    )
    parser.add_argument("session_path", help="Evaluation session output dir, ui_session_manifest.json, or session_pipeline_report.json")
    parser.add_argument("--output-dir", default="", help="Optional export directory. Defaults to <session>/exports/evaluation-review")
    parser.add_argument("--hard-negative-pre-seconds", type=float, default=2.0)
    parser.add_argument("--hard-negative-post-seconds", type=float, default=2.0)
    parser.add_argument("--false-negative-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--recall-floor", type=float, default=0.9)
    parser.add_argument("--hist-bins", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _probability_histogram(values: Sequence[float], bins: int) -> dict[str, list[float] | list[int]]:
    if not values:
        return {"bins": [], "counts": []}
    hist, edges = np.histogram(np.array(values, dtype=np.float32), bins=max(2, bins), range=(0.0, 1.0))
    return {
        "bins": [float(edge) for edge in edges.tolist()],
        "counts": [int(count) for count in hist.tolist()],
    }


def _score_thresholds(rows: Sequence[dict[str, Any]], recall_floor: float) -> dict[str, Any]:
    labeled = [
        row
        for row in rows
        if row.get("human_label") in {"dive", "non-dive"} and row.get("audio_clip_probability") is not None
    ]
    if len({row["human_label"] for row in labeled}) < 2:
        return {"status": "insufficient_reviewed_scores"}
    thresholds = sorted({float(row["audio_clip_probability"]) for row in labeled} | {0.0, 0.5, 1.0})
    best_precision: dict[str, Any] | None = None
    best_f1: dict[str, Any] | None = None
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for row in labeled:
            predicted_positive = float(row["audio_clip_probability"]) >= threshold
            truth_positive = row["human_label"] == "dive"
            if predicted_positive and truth_positive:
                tp += 1
            elif predicted_positive and not truth_positive:
                fp += 1
            elif truth_positive:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        entry = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if recall >= recall_floor and (
            best_precision is None
            or precision > best_precision["precision"]
            or (math.isclose(precision, best_precision["precision"]) and f1 > best_precision["f1"])
        ):
            best_precision = entry
        if best_f1 is None or f1 > best_f1["f1"] or (math.isclose(f1, best_f1["f1"]) and precision > best_f1["precision"]):
            best_f1 = entry
    return {
        "status": "ok",
        "rows": len(labeled),
        "best_precision_under_recall_floor": best_precision,
        "best_f1": best_f1,
        "recall_floor": recall_floor,
    }


def _feature_contribution_summary(model: AudioClipModel | None, rows: Sequence[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if model is None:
        return {"status": "model_unavailable", "top_features": []}
    reviewed_non_dives = [row for row in rows if row.get("review_label") == "non_dive"]
    if not reviewed_non_dives:
        return {"status": "no_reviewed_non_dives", "top_features": []}
    totals = {name: 0.0 for name in model.feature_names}
    counts = {name: 0 for name in model.feature_names}
    sample_rows: list[dict[str, Any]] = []
    for row in reviewed_non_dives:
        feature_map = {name: float(row.get(name, 0.0) or 0.0) for name in model.feature_names}
        contributions = model.explain_feature_map(feature_map)
        for item in contributions:
            feature = str(item["feature"])
            totals[feature] += float(item["contribution"])
            counts[feature] += 1
        sample_rows.append(
            {
                "detection_id": row.get("source_candidate_id"),
                "timestamp": row.get("timestamp_seconds"),
                "subtype": row.get("subtype"),
                "top_features": contributions[: min(5, len(contributions))],
            }
        )
    ranked = sorted(
        (
            {
                "feature": feature,
                "mean_contribution": totals[feature] / max(1, counts[feature]),
                "observations": counts[feature],
            }
            for feature in model.feature_names
        ),
        key=lambda item: abs(float(item["mean_contribution"])),
        reverse=True,
    )
    return {"status": "ok", "top_features": ranked[:top_k], "example_rows": sample_rows[:top_k]}


def _top_rows(rows: Sequence[dict[str, Any]], limit: int, key_name: str, reverse: bool = True) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key_name, 0.0) or 0.0), reverse=reverse)[:limit]


def _label_audio_command(row: dict[str, Any], *, pre_seconds: float, post_seconds: float) -> str:
    note_bits = ["hard negative mined from evaluation"]
    subtype = row.get("subtype")
    if subtype:
        note_bits.append(f"subtype={subtype}")
    note_bits.append(f"session={row.get('session_id')}")
    note_bits.append(f"candidate={row.get('source_candidate_id')}")
    note = " | ".join(note_bits).replace('"', "'")
    subtype_flag = f" --subtype {subtype}" if subtype else ""
    return (
        "divesensei label-audio "
        f"\"{row.get('source_video_path')}\" "
        f"{float(row.get('timestamp_seconds', 0.0)):.3f} "
        f"--label non-dive{subtype_flag} --pre-seconds {float(pre_seconds):.3f} --post-seconds {float(post_seconds):.3f} "
        f"--notes \"{note}\""
    )


def _load_model(report: dict[str, Any]) -> AudioClipModel | None:
    config = dict(report.get("config", {}) or {})
    model_path = str(config.get("audio_clip_model_path", "") or "").strip()
    if not model_path:
        return None
    candidate = Path(model_path).expanduser().resolve()
    if not candidate.exists():
        return None
    try:
        return AudioClipModel.load(candidate)
    except Exception:
        return None


def _merge_reviewed_candidates(
    manifest: dict[str, Any],
    report: dict[str, Any],
    decisions: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions_by_id = {str(item.get("detectionId")): item for item in decisions}
    proposal_by_detection = {
        str(row.get("final_detection_id")): row
        for row in proposal_rows
        if row.get("final_detection_id")
    }
    session = manifest["session"]
    session_id = session["id"]
    domain = infer_domain_metadata(str(session["source_video_path"]))
    rows: list[dict[str, Any]] = []
    for detection in manifest.get("detections", []):
        detection_id = str(detection["id"])
        decision = decisions_by_id.get(detection_id)
        proposal = proposal_by_detection.get(detection_id, {})
        review_label = decision.get("label") if decision else None
        subtype = normalize_non_dive_subtype(decision.get("subtype") if decision else None)
        human_label = "dive" if review_label == "dive" else "non-dive" if review_label == "non_dive" else None
        payload = {
            "entry_type": "reviewed_candidate",
            "session_id": session_id,
            "session_name": session.get("session_name"),
            "source_video_path": session["source_video_path"],
            "source_audio_path": manifest.get("artifacts", {}).get("source_audio"),
            "source_file": Path(str(session["source_video_path"])).name,
            "source_candidate_id": detection_id,
            "timestamp_seconds": float(detection["timestamp_seconds"]),
            "start_time_seconds": float(detection["start_time_seconds"]),
            "end_time_seconds": float(detection["end_time_seconds"]),
            "review_window_start_seconds": float(detection.get("review_start_seconds") or detection["start_time_seconds"]),
            "review_window_end_seconds": float(detection.get("review_end_seconds") or detection["end_time_seconds"]),
            "confidence": detection.get("confidence"),
            "audio_score": float(detection.get("scores", {}).get("audio", 0.0) or 0.0),
            "video_score": float(detection.get("scores", {}).get("video", 0.0) or 0.0),
            "combined_score": float(detection.get("scores", {}).get("combined", 0.0) or 0.0),
            "audio_model_probability": proposal.get("audio_model_probability", detection.get("scores", {}).get("audio_model_probability")),
            "audio_clip_probability": proposal.get("audio_clip_probability", detection.get("scores", {}).get("audio_clip_probability")),
            "proposal_id": proposal.get("proposal_id"),
            "proposal_frontend": proposal.get("proposal_frontend"),
            "proposal_stage": proposal.get("pipeline_stage", "final_selected"),
            "review_state": "reviewed" if decision else "pending",
            "review_label": review_label,
            "review_notes": decision.get("notes") if decision else "",
            "review_created_at": decision.get("createdAt") if decision else None,
            "review_updated_at": decision.get("updatedAt") if decision else None,
            "subtype": subtype,
            "human_label": human_label,
            **domain,
        }
        features = dict(detection.get("features", {}) or {})
        for feature_name in AUDIO_CLIP_FEATURES:
            payload[feature_name] = proposal.get(feature_name, features.get(feature_name))
        rows.append(payload)
    return rows


def _attribute_false_negative(
    annotation: dict[str, Any],
    proposal_rows: Sequence[dict[str, Any]],
    *,
    session_id: str,
    source_video_path: str,
    tolerance_seconds: float,
) -> dict[str, Any]:
    timestamp = float(annotation.get("timestampSeconds", 0.0) or 0.0)
    nearby = [
        row
        for row in proposal_rows
        if abs(float(row.get("timestamp", 0.0) or 0.0) - timestamp) <= tolerance_seconds
    ]
    nearby.sort(key=lambda row: (abs(float(row.get("timestamp", 0.0) or 0.0) - timestamp), -float(row.get("raw_proposal_score", 0.0) or 0.0)))
    matched = nearby[0] if nearby else None
    if matched is None:
        failure_type = "no_proposal_generated"
    else:
        stage = str(matched.get("pipeline_stage", "classifier_rejected"))
        if stage == "ambiguous_case":
            failure_type = "ambiguous_case"
        elif stage == "threshold_rejected":
            failure_type = "threshold_rejected"
        elif stage == "final_selected":
            failure_type = "reviewed_false_negative"
        else:
            failure_type = "classifier_rejected"
    payload = {
        "entry_type": "false_negative",
        "session_id": session_id,
        "source_video_path": source_video_path,
        "source_file": Path(source_video_path).name,
        "source_candidate_id": matched.get("final_detection_id") if matched else None,
        "proposal_id": matched.get("proposal_id") if matched else None,
        "timestamp_seconds": timestamp,
        "review_window_start_seconds": float(annotation.get("reviewStartSeconds", max(0.0, timestamp - 2.0))),
        "review_window_end_seconds": float(annotation.get("reviewEndSeconds", timestamp + 2.0)),
        "review_label": "false_negative",
        "review_state": "reviewed_false_negative",
        "review_notes": annotation.get("notes", ""),
        "review_created_at": annotation.get("createdAt"),
        "review_updated_at": annotation.get("updatedAt"),
        "subtype": normalize_non_dive_subtype(annotation.get("subtype")),
        "human_label": "dive",
        "failure_type": failure_type,
        "matched_proposal_delta_seconds": abs(float(matched.get("timestamp", 0.0)) - timestamp) if matched else None,
        "matched_proposal_probability": matched.get("audio_clip_probability") if matched else None,
        "matched_proposal_stage": matched.get("pipeline_stage") if matched else None,
        "matched_proposal_frontend": matched.get("proposal_frontend") if matched else None,
    }
    if matched is not None:
        for feature_name in AUDIO_CLIP_FEATURES:
            payload[feature_name] = matched.get(feature_name)
    return payload


def _build_candidate_diagnostics(
    proposal_rows: Sequence[dict[str, Any]],
    reviewed_candidates: Sequence[dict[str, Any]],
    false_negatives: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    reviewed_by_candidate = {str(row.get("source_candidate_id")): row for row in reviewed_candidates}
    false_negative_by_proposal = {str(row.get("proposal_id")): row for row in false_negatives if row.get("proposal_id")}
    diagnostics: list[dict[str, Any]] = []
    for row in proposal_rows:
        review_row = reviewed_by_candidate.get(str(row.get("final_detection_id")))
        fn_row = false_negative_by_proposal.get(str(row.get("proposal_id")))
        payload = {
            **row,
            "review_state": review_row.get("review_state") if review_row else fn_row.get("review_state") if fn_row else "unreviewed_proposal",
            "review_label": review_row.get("review_label") if review_row else fn_row.get("review_label") if fn_row else None,
            "human_label": review_row.get("human_label") if review_row else fn_row.get("human_label") if fn_row else None,
            "subtype": review_row.get("subtype") if review_row else fn_row.get("subtype") if fn_row else None,
            "failure_type": fn_row.get("failure_type") if fn_row else None,
        }
        diagnostics.append(payload)
    return diagnostics


def _per_session_metrics(
    manifest: dict[str, Any],
    reviewed_candidates: Sequence[dict[str, Any]],
    false_negatives: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    duration_seconds = float(manifest.get("session", {}).get("session_duration_seconds") or 0.0)
    duration_minutes = duration_seconds / 60.0 if duration_seconds > 0 else None
    reviewed_non_dives = sum(1 for row in reviewed_candidates if row.get("review_label") == "non_dive")
    reviewed_dives = sum(1 for row in reviewed_candidates if row.get("review_label") == "dive")
    unsure = sum(1 for row in reviewed_candidates if row.get("review_label") == "unsure")
    pending = sum(1 for row in reviewed_candidates if row.get("review_state") == "pending")
    return {
        "session_id": manifest.get("session", {}).get("id"),
        "session_name": manifest.get("session", {}).get("session_name"),
        "duration_seconds": duration_seconds,
        "candidate_count": len(reviewed_candidates),
        "reviewed_dive_count": reviewed_dives,
        "reviewed_non_dive_count": reviewed_non_dives,
        "reviewed_unsure_count": unsure,
        "pending_candidate_count": pending,
        "false_negative_count": len(false_negatives),
        "reviewed_false_positives_per_minute": (reviewed_non_dives / duration_minutes) if duration_minutes else None,
        "reviewed_false_negatives_per_minute": (len(false_negatives) / duration_minutes) if duration_minutes else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    session_paths = resolve_evaluation_session_paths(args.session_path)
    manifest = read_json(session_paths["manifest_path"])
    if str(manifest.get("session", {}).get("mode", "standard")) != "evaluation":
        print("This command only supports evaluation sessions.")
        return 1
    report = read_json(session_paths["report_path"])
    review_payload = load_evaluation_review_data(session_paths["review_path"])
    proposal_rows = load_jsonl(session_paths["proposal_path"])
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session_paths["output_dir"] / "exports" / "evaluation-review"
    output_dir.mkdir(parents=True, exist_ok=True)

    reviewed_candidates = _merge_reviewed_candidates(
        manifest,
        report,
        review_payload.get("decisions", []),
        proposal_rows,
    )
    false_negatives = [
        _attribute_false_negative(
            annotation,
            proposal_rows,
            session_id=str(manifest["session"]["id"]),
            source_video_path=str(manifest["session"]["source_video_path"]),
            tolerance_seconds=float(args.false_negative_tolerance_seconds),
        )
        for annotation in review_payload.get("falseNegatives", [])
    ]
    candidate_diagnostics = _build_candidate_diagnostics(proposal_rows, reviewed_candidates, false_negatives)
    hard_negative_rows = [row for row in reviewed_candidates if row.get("review_label") == "non_dive"]
    reviewed_rows = [row for row in reviewed_candidates if row.get("review_state") == "reviewed"]
    threshold_summary = _score_thresholds(reviewed_candidates, float(args.recall_floor))
    model = _load_model(report)
    feature_summary = _feature_contribution_summary(model, reviewed_candidates, int(args.top_k))
    failure_counts: dict[str, int] = {}
    for row in false_negatives:
        failure_counts[str(row.get("failure_type"))] = failure_counts.get(str(row.get("failure_type")), 0) + 1

    ranking_reports = {
        "top_reviewed_false_positives": _top_rows(hard_negative_rows, int(args.top_k), "audio_clip_probability"),
        "top_false_negatives": _top_rows(false_negatives, int(args.top_k), "matched_proposal_probability"),
        "top_ambiguous_cases": _top_rows(
            [row for row in reviewed_candidates if row.get("review_label") == "unsure"] + [row for row in false_negatives if row.get("failure_type") == "ambiguous_case"],
            int(args.top_k),
            "audio_clip_probability",
        ),
    }

    summary = {
        "session_id": manifest["session"]["id"],
        "session_name": manifest["session"].get("session_name"),
        "source_video_path": manifest["session"]["source_video_path"],
        "source_audio_path": manifest.get("artifacts", {}).get("source_audio"),
        "reviewed_candidate_count": len(reviewed_rows),
        "hard_negative_count": len(hard_negative_rows),
        "false_negative_count": len(false_negatives),
        "reviewed_label_counts": {
            "dive": sum(1 for row in reviewed_candidates if row.get("review_label") == "dive"),
            "non_dive": len(hard_negative_rows),
            "unsure": sum(1 for row in reviewed_candidates if row.get("review_label") == "unsure"),
            "pending": sum(1 for row in reviewed_candidates if row.get("review_state") == "pending"),
        },
        "hard_negative_subtype_counts": {
            subtype: sum(1 for row in hard_negative_rows if row.get("subtype") == subtype)
            for subtype in ["board_rebound", "board_slap", "non_dive_splash", "voice_whistle", "handling_noise", "unknown_transient"]
        },
        "failure_attribution": failure_counts,
        "per_session_metrics": [_per_session_metrics(manifest, reviewed_candidates, false_negatives)],
        "score_distributions": {
            "dive": _probability_histogram([float(row["audio_clip_probability"]) for row in reviewed_candidates if row.get("review_label") == "dive" and row.get("audio_clip_probability") is not None], int(args.hist_bins)),
            "non_dive": _probability_histogram([float(row["audio_clip_probability"]) for row in hard_negative_rows if row.get("audio_clip_probability") is not None], int(args.hist_bins)),
            "unsure": _probability_histogram([float(row["audio_clip_probability"]) for row in reviewed_candidates if row.get("review_label") == "unsure" and row.get("audio_clip_probability") is not None], int(args.hist_bins)),
        },
        "threshold_recommendation": threshold_summary,
        "feature_diagnostics": feature_summary,
    }

    reviewed_candidates_path = write_jsonl(output_dir / "reviewed_candidates.jsonl", reviewed_candidates)
    false_negatives_path = write_jsonl(output_dir / "false_negatives.jsonl", false_negatives)
    hard_negative_manifest_path = write_jsonl(output_dir / "mined_hard_negatives.jsonl", hard_negative_rows)
    commands_path = output_dir / "label_audio_hard_negatives.sh"
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(
            _label_audio_command(
                row,
                pre_seconds=float(args.hard_negative_pre_seconds),
                post_seconds=float(args.hard_negative_post_seconds),
            )
            for row in hard_negative_rows
        )
        + ("\n" if hard_negative_rows else ""),
        encoding="utf-8",
    )
    candidate_diagnostics_path = write_jsonl(output_dir / "candidate_diagnostics.jsonl", candidate_diagnostics)
    ranking_reports_path = write_json(output_dir / "ranking_reports.json", ranking_reports)
    summary_path = write_json(output_dir / "evaluation_export_summary.json", summary)

    update_session_artifacts(
        session_paths,
        {
            "evaluation_export_summary": str(summary_path),
            "reviewed_candidates": str(reviewed_candidates_path),
            "hard_negative_manifest": str(hard_negative_manifest_path),
            "hard_negative_commands": str(commands_path),
            "candidate_diagnostics": str(candidate_diagnostics_path),
            "ranking_reports": str(ranking_reports_path),
        },
    )

    print(
        json.dumps(
            {
                "session_id": manifest["session"]["id"],
                "reviewed_candidate_count": len(reviewed_rows),
                "hard_negative_count": len(hard_negative_rows),
                "false_negative_count": len(false_negatives),
                "failure_attribution": failure_counts,
                "threshold_recommendation": threshold_summary,
                "summary_path": str(summary_path),
                "hard_negative_manifest_path": str(hard_negative_manifest_path),
                "hard_negative_commands_path": str(commands_path),
            },
            indent=2,
            default=_json_safe,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
