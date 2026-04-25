from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(".")
TOLERANCE_SECONDS = 2.0
POSITIVE_SCORE_THRESHOLD = 0.845
EVALUATED_SESSION_IDS = [
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_Compete-16-11-2025-first-10min_20260422-154957",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
    "evaluation_insep_plateform_mixed_sound",
]
VISUAL_ARTIFACT_ROOTS = {
    "evaluation_CAO-1st-15min_20260421-072906": Path(
        "/Users/mcauchy/Downloads/r52_visual_recovery_scoring/r52_visual_recovery_scoring/evaluation_CAO-1st-15min_20260421-072906/audio_gated_full_frame_1p0fps"
    ),
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927": Path(
        "/Users/mcauchy/Downloads/r52_visual_recovery_scoring/r52_visual_recovery_scoring/evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927/audio_gated_full_frame_1p0fps"
    ),
    "evaluation_SNMT-WED-8:4:26_20260419-142758": Path(
        "/Users/mcauchy/Downloads/r52_visual_recovery_scoring/r52_visual_recovery_scoring/evaluation_SNMT-WED-8:4:26_20260419-142758/audio_gated_full_frame_1p0fps"
    ),
    "evaluation_Compete-16-11-2025-first-10min_20260422-154957": Path(
        "/Users/mcauchy/Downloads/r41_remote_gpu_results/audio_gated_full_frame_1fps"
    ),
    "evaluation_insep_plateform_mixed_sound": Path(
        "/Users/mcauchy/Downloads/r42_visual_full_frame_control/audio_gated_full_frame_1p0fps"
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def safe_median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def pct(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def anchors_for_session(session: Path) -> list[dict[str, Any]]:
    reviewed = read_jsonl(session / "exports/evaluation-review/reviewed_candidates.jsonl")
    fns = read_jsonl(session / "exports/evaluation-review/false_negatives.jsonl")
    anchors = []
    for row in reviewed:
        if row.get("review_label") == "dive":
            anchors.append(
                {
                    "anchor_id": str(row.get("proposal_id") or row.get("source_candidate_id")),
                    "timestamp_seconds": float(row["timestamp_seconds"]),
                    "source": "reviewed_audio_dive_candidate",
                    "audio_matched": True,
                    "raw": row,
                }
            )
    for idx, row in enumerate(fns, start=1):
        nearest = row.get("nearest_proposal_offset_seconds")
        audio_matched = nearest is not None and abs(float(nearest)) <= TOLERANCE_SECONDS
        anchors.append(
            {
                "anchor_id": str(row.get("proposal_id") or f"fn-{idx:04d}"),
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "source": "reviewed_false_negative",
                "audio_matched": audio_matched,
                "nearest_proposal_offset_seconds": nearest,
                "raw": row,
            }
        )
    anchors.sort(key=lambda row: row["timestamp_seconds"])
    return anchors


def load_predictions(artifact_root: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(artifact_root / "visual_frame_predictions.jsonl")
        if row.get("prompt_id") == "diving_attempt"
        and row.get("decision_rule") == "yes_no_first_token_margin"
    ]
    rows.sort(key=lambda row: float(row["timestamp_seconds"]))
    for row in rows:
        row["_score"] = float(row.get("score") or 0.0)
        row["_margin"] = float(row.get("yes_no_first_token_margin") or 0.0)
        row["_timestamp"] = float(row["timestamp_seconds"])
        row["_positive_for_recipe"] = bool(row.get("is_positive")) and row["_score"] >= POSITIVE_SCORE_THRESHOLD
    return rows


def load_visual_intervals(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in predictions if row["_positive_for_recipe"]]
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        ts = row["_timestamp"]
        if not groups or ts - groups[-1][-1]["_timestamp"] > 3.0:
            groups.append([row])
        else:
            groups[-1].append(row)
    intervals = []
    for idx, group in enumerate(groups, start=1):
        start = max(0.0, group[0]["_timestamp"] - 1.5)
        end = group[-1]["_timestamp"] + 3.0
        intervals.append(
            {
                "visual_interval_id": f"r53-vis-{idx:04d}",
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "anchor_timestamp_seconds": round(group[len(group) // 2]["_timestamp"], 3),
                "positive_frame_count": len(group),
                "max_score": max(row["_score"] for row in group),
            }
        )
    return intervals


def match_visual(anchor_ts: float, intervals: list[dict[str, Any]]) -> dict[str, Any] | None:
    containing = [row for row in intervals if float(row["start_seconds"]) <= anchor_ts <= float(row["end_seconds"])]
    if containing:
        return min(containing, key=lambda row: abs(float(row["anchor_timestamp_seconds"]) - anchor_ts))
    nearby = [row for row in intervals if abs(float(row["anchor_timestamp_seconds"]) - anchor_ts) <= TOLERANCE_SECONDS]
    if nearby:
        return min(nearby, key=lambda row: abs(float(row["anchor_timestamp_seconds"]) - anchor_ts))
    return None


def nearest_prediction(anchor_ts: float, predictions: list[dict[str, Any]], positive_only: bool = False) -> dict[str, Any] | None:
    rows = [row for row in predictions if row["_positive_for_recipe"]] if positive_only else predictions
    if not rows:
        return None
    return min(rows, key=lambda row: abs(row["_timestamp"] - anchor_ts))


def frames_within(anchor_ts: float, predictions: list[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    return [row for row in predictions if abs(row["_timestamp"] - anchor_ts) <= radius]


def category_name(audio_matched: bool, visual_matched: bool) -> str:
    if audio_matched and visual_matched:
        return "audio_matched_visual_matched"
    if audio_matched and not visual_matched:
        return "audio_matched_visual_missed"
    if not audio_matched and visual_matched:
        return "audio_missed_visual_recovered"
    return "audio_missed_visual_also_missed"


def likely_failure_reason(
    session_id: str,
    category: str,
    anchor_ts: float,
    predictions: list[dict[str, Any]],
    nearest_positive: dict[str, Any] | None,
) -> str:
    if category == "audio_missed_visual_recovered":
        return "recovered_by_visual_branch"
    if session_id == "evaluation_Compete-16-11-2025-first-10min_20260422-154957":
        return "audio_already_solved_session"
    has_frame_2 = bool(frames_within(anchor_ts, predictions, 2.0))
    has_frame_6 = bool(frames_within(anchor_ts, predictions, 6.0))
    if not has_frame_6:
        return "sampling_missed_visual_evidence"
    if nearest_positive and abs(nearest_positive["_timestamp"] - anchor_ts) <= 6.0:
        return "interval_geometry_missed_positive"
    if has_frame_2:
        return "vlm_prompt_false_negative"
    return "sampling_missed_visual_evidence"


def audit_session(session_id: str) -> dict[str, Any]:
    session = ROOT / "outputs" / session_id
    artifact_root = VISUAL_ARTIFACT_ROOTS[session_id]
    anchors = anchors_for_session(session)
    predictions = load_predictions(artifact_root)
    intervals = load_visual_intervals(predictions)
    positive_rows = [row for row in predictions if row["_positive_for_recipe"]]
    interval_ids_matched: set[str] = set()
    anchor_rows = []
    categories: dict[str, list[dict[str, Any]]] = {
        "audio_matched_visual_matched": [],
        "audio_matched_visual_missed": [],
        "audio_missed_visual_recovered": [],
        "audio_missed_visual_also_missed": [],
    }
    for anchor in anchors:
        ts = float(anchor["timestamp_seconds"])
        visual_match = match_visual(ts, intervals)
        visual_matched = visual_match is not None
        if visual_match:
            interval_ids_matched.add(str(visual_match["visual_interval_id"]))
        nearest_any = nearest_prediction(ts, predictions)
        nearest_pos = nearest_prediction(ts, predictions, positive_only=True)
        nearby_2 = frames_within(ts, predictions, 2.0)
        nearby_4 = frames_within(ts, predictions, 4.0)
        nearby_6 = frames_within(ts, predictions, 6.0)
        nearby_neg = [row for row in nearby_6 if not row["_positive_for_recipe"]]
        cat = category_name(bool(anchor["audio_matched"]), visual_matched)
        audit_row = {
            "session_id": session_id,
            "anchor_id": anchor["anchor_id"],
            "timestamp_seconds": round(ts, 3),
            "anchor_source": anchor["source"],
            "audio_matched": bool(anchor["audio_matched"]),
            "visual_matched": visual_matched,
            "category": cat,
            "visual_interval_id": visual_match["visual_interval_id"] if visual_match else None,
            "nearest_sampled_frame_timestamp": round(nearest_any["_timestamp"], 3) if nearest_any else None,
            "nearest_sampled_frame_delta_seconds": round(nearest_any["_timestamp"] - ts, 3) if nearest_any else None,
            "nearest_sampled_frame_score": round(nearest_any["_score"], 4) if nearest_any else None,
            "nearest_sampled_frame_margin": round(nearest_any["_margin"], 4) if nearest_any else None,
            "nearest_sampled_frame_positive": bool(nearest_any["_positive_for_recipe"]) if nearest_any else None,
            "nearest_positive_frame_timestamp": round(nearest_pos["_timestamp"], 3) if nearest_pos else None,
            "nearest_positive_frame_abs_delta_seconds": round(abs(nearest_pos["_timestamp"] - ts), 3) if nearest_pos else None,
            "nearest_positive_score": round(nearest_pos["_score"], 4) if nearest_pos else None,
            "nearest_positive_margin": round(nearest_pos["_margin"], 4) if nearest_pos else None,
            "sampled_frame_within_2s": bool(nearby_2),
            "sampled_frame_within_4s": bool(nearby_4),
            "sampled_frame_within_6s": bool(nearby_6),
            "positive_frame_within_2s": any(row["_positive_for_recipe"] for row in nearby_2),
            "positive_frame_within_4s": any(row["_positive_for_recipe"] for row in nearby_4),
            "positive_frame_within_6s": any(row["_positive_for_recipe"] for row in nearby_6),
            "nearby_negative_prediction_within_6s": bool(nearby_neg),
            "nearby_negative_best_score_within_6s": round(max((row["_score"] for row in nearby_neg), default=0.0), 4)
            if nearby_neg
            else None,
            "nearest_raw_response": nearest_any.get("raw_response") if nearest_any else None,
            "likely_failure_reason": likely_failure_reason(session_id, cat, ts, predictions, nearest_pos),
        }
        anchor_rows.append(audit_row)
        categories[cat].append(audit_row)
    unmatched_intervals = [row for row in intervals if str(row["visual_interval_id"]) not in interval_ids_matched]
    prediction_duration = (
        max(row["_timestamp"] for row in predictions) - min(row["_timestamp"] for row in predictions)
        if len(predictions) > 1
        else 0.0
    )
    positive_scores = [row["_score"] for row in positive_rows]
    margins = [row["_margin"] for row in predictions]
    audio_misses = [row for row in anchor_rows if not row["audio_matched"]]
    audio_misses_with_pos6 = [row for row in audio_misses if row["positive_frame_within_6s"]]
    audio_misses_with_frames_no_pos6 = [
        row for row in audio_misses if row["sampled_frame_within_6s"] and not row["positive_frame_within_6s"]
    ]
    category_summary = {}
    for name, rows in categories.items():
        deltas = [row["nearest_positive_frame_abs_delta_seconds"] for row in rows if row["nearest_positive_frame_abs_delta_seconds"] is not None]
        scores = [row["nearest_positive_score"] for row in rows if row["nearest_positive_score"] is not None]
        margins_for_rows = [row["nearest_positive_margin"] for row in rows if row["nearest_positive_margin"] is not None]
        category_summary[name] = {
            "count": len(rows),
            "percentage_of_anchors": pct(len(rows), len(anchor_rows)),
            "nearest_visual_positive_abs_delta_median_seconds": safe_median(deltas),
            "nearest_visual_positive_score_median": safe_median(scores),
            "nearest_visual_positive_margin_median": safe_median(margins_for_rows),
            "sampled_frame_within_2s_count": sum(1 for row in rows if row["sampled_frame_within_2s"]),
            "sampled_frame_within_4s_count": sum(1 for row in rows if row["sampled_frame_within_4s"]),
            "sampled_frame_within_6s_count": sum(1 for row in rows if row["sampled_frame_within_6s"]),
            "nearby_negative_prediction_within_6s_count": sum(1 for row in rows if row["nearby_negative_prediction_within_6s"]),
        }
    examples = select_examples(session_id, anchor_rows, unmatched_intervals, predictions)
    return {
        "session_id": session_id,
        "session_root": str(session),
        "visual_artifact_root": str(artifact_root),
        "anchors": anchor_rows,
        "category_summary": category_summary,
        "session_diagnostics": {
            "reviewed_dive_anchors": len(anchor_rows),
            "audio_miss_count": len(audio_misses),
            "prediction_rows": len(predictions),
            "visual_positive_frame_count": len(positive_rows),
            "visual_positive_frame_density_per_minute": round(len(positive_rows) / (prediction_duration / 60.0), 4)
            if prediction_duration > 0
            else None,
            "visual_interval_count": len(intervals),
            "positive_score_mean": safe_mean(positive_scores),
            "positive_score_median": safe_median(positive_scores),
            "yes_no_margin_mean": safe_mean(margins),
            "yes_no_margin_median": safe_median(margins),
            "audio_misses_with_positive_within_6s_count": len(audio_misses_with_pos6),
            "audio_misses_with_positive_within_6s_proportion": pct(len(audio_misses_with_pos6), len(audio_misses)),
            "audio_misses_with_sampled_frames_but_no_positive_within_6s_count": len(audio_misses_with_frames_no_pos6),
            "audio_misses_with_sampled_frames_but_no_positive_within_6s_proportion": pct(
                len(audio_misses_with_frames_no_pos6), len(audio_misses)
            ),
            "unmatched_visual_interval_count": len(unmatched_intervals),
            "unmatched_visual_intervals": unmatched_intervals,
            "prediction_time_span_seconds": round(prediction_duration, 3),
        },
        "representative_examples": examples,
    }


def nearest_prediction_summary(anchor_ts: float, predictions: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    rows = sorted(predictions, key=lambda row: abs(row["_timestamp"] - anchor_ts))[:limit]
    return [
        {
            "timestamp_seconds": round(row["_timestamp"], 3),
            "delta_seconds": round(row["_timestamp"] - anchor_ts, 3),
            "score": round(row["_score"], 4),
            "margin": round(row["_margin"], 4),
            "positive_for_recipe": bool(row["_positive_for_recipe"]),
            "raw_response": row.get("raw_response"),
        }
        for row in rows
    ]


def select_examples(
    session_id: str,
    anchor_rows: list[dict[str, Any]],
    unmatched_intervals: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    categories = [
        "audio_missed_visual_recovered",
        "audio_missed_visual_also_missed",
        "audio_matched_visual_missed",
    ]
    for cat in categories:
        candidates = [row for row in anchor_rows if row["category"] == cat]
        if not candidates:
            continue
        if cat == "audio_missed_visual_recovered":
            chosen = min(candidates, key=lambda row: row["nearest_positive_frame_abs_delta_seconds"] or math.inf)
        elif cat == "audio_missed_visual_also_missed":
            chosen = max(candidates, key=lambda row: abs(row["nearest_sampled_frame_score"] or 0.0))
        else:
            chosen = min(candidates, key=lambda row: abs(row["nearest_sampled_frame_delta_seconds"] or math.inf))
        ts = float(chosen["timestamp_seconds"])
        examples.append(
            {
                "example_type": cat,
                "session_id": session_id,
                "anchor_id": chosen["anchor_id"],
                "anchor_timestamp_seconds": ts,
                "nearest_sampled_frames": nearest_prediction_summary(ts, predictions),
                "visual_interval_overlap_status": chosen["visual_interval_id"] or "none",
                "likely_failure_reason": chosen["likely_failure_reason"],
            }
        )
    for interval in unmatched_intervals[:1]:
        ts = float(interval["anchor_timestamp_seconds"])
        examples.append(
            {
                "example_type": "unmatched_visual_proposal",
                "session_id": session_id,
                "anchor_id": None,
                "anchor_timestamp_seconds": ts,
                "visual_interval": interval,
                "nearest_sampled_frames": nearest_prediction_summary(ts, predictions),
                "visual_interval_overlap_status": "unmatched_visual_interval",
                "likely_failure_reason": "visual_false_positive_or_unreviewed_event",
            }
        )
    return examples


def extract_example_frames(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_root = Path("outputs/r53_visual_failure_examples")
    output_root.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    for idx, example in enumerate(examples, start=1):
        session_id = str(example["session_id"])
        video = ROOT / "outputs" / session_id / "web/session_source_review.mp4"
        if not video.exists():
            extracted.append({**example, "still_frame_path": None, "still_frame_status": "missing_review_proxy"})
            continue
        timestamp = float(example["anchor_timestamp_seconds"])
        out = output_root / f"{idx:02d}_{session_id}_{example['example_type']}_{timestamp:.3f}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-y",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=20)
            status = "extracted" if out.exists() else "not_written"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            status = f"extract_failed:{type(exc).__name__}"
        extracted.append({**example, "still_frame_path": str(out) if out.exists() else None, "still_frame_status": status})
    return extracted


def render_anchor_audit_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R53 Visual Recovery Anchor Audit",
        "",
        "| Session | A+V+ | A+V- | A-V+ | A-V- |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["sessions"]:
        summary = row["category_summary"]
        lines.append(
            f"| `{row['session_id']}` | {summary['audio_matched_visual_matched']['count']} | {summary['audio_matched_visual_missed']['count']} | {summary['audio_missed_visual_recovered']['count']} | {summary['audio_missed_visual_also_missed']['count']} |"
        )
    lines.extend(
        [
            "",
            "Legend: `A+V+` audio matched + visual matched, `A+V-` audio matched + visual missed, `A-V+` audio missed + visual recovered, `A-V-` audio missed + visual also missed.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_diagnosis_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R53 Visual Recovery Failure Mode Diagnosis",
        "",
        "| Session | Positive frames | Pos/min | Intervals | Audio misses with pos <=6s | Sampled no-pos misses <=6s | Unmatched intervals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["sessions"]:
        diag = row["session_diagnostics"]
        lines.append(
            f"| `{row['session_id']}` | {diag['visual_positive_frame_count']} | {diag['visual_positive_frame_density_per_minute']} | {diag['visual_interval_count']} | {diag['audio_misses_with_positive_within_6s_count']} / {diag['audio_miss_count']} | {diag['audio_misses_with_sampled_frames_but_no_positive_within_6s_count']} / {diag['audio_miss_count']} | {diag['unmatched_visual_interval_count']} |"
        )
    lines.extend(
        [
            "",
            f"- main cause: `{payload['main_cause_of_low_visual_recall']}`",
            f"- recommended next lever: `{payload['recommended_next_lever']}`",
            f"- visual retained: `{payload['visual_recovery_retained']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_next_lever_md(payload: dict[str, Any]) -> str:
    return (
        "# R53 Visual Recovery Next Lever\n\n"
        f"- decision: `{payload['decision']}`\n"
        f"- primary next lever: `{payload['recommended_next_lever']}`\n"
        f"- rationale: {payload['rationale']}\n"
        f"- product default changes: `{payload['product_default_changes']}`\n"
    )


def main() -> int:
    sessions = [audit_session(session_id) for session_id in EVALUATED_SESSION_IDS]
    all_examples = []
    for session in sessions:
        all_examples.extend(session["representative_examples"])
    extracted_examples = extract_example_frames(all_examples[:18])
    totals = {
        "audio_matched_visual_matched": 0,
        "audio_matched_visual_missed": 0,
        "audio_missed_visual_recovered": 0,
        "audio_missed_visual_also_missed": 0,
    }
    for session in sessions:
        for key in totals:
            totals[key] += session["category_summary"][key]["count"]
    total_anchors = sum(totals.values())
    aggregate_category_summary = {
        key: {"count": count, "percentage_of_anchors": pct(count, total_anchors)}
        for key, count in totals.items()
    }
    total_audio_misses = totals["audio_missed_visual_recovered"] + totals["audio_missed_visual_also_missed"]
    recovered_audio_misses = totals["audio_missed_visual_recovered"]
    sampled_no_positive_misses = sum(
        session["session_diagnostics"]["audio_misses_with_sampled_frames_but_no_positive_within_6s_count"]
        for session in sessions
    )
    diagnosis = {
        "benchmark_id": "r53_visual_recovery_failure_mode_diagnosis",
        "recipe": {
            "roi": "full_frame",
            "fps": 1.0,
            "prompt": "diving_attempt",
            "decision_rule": "yes_no_first_token_margin",
            "interval_geometry": "split_internal_gap_3s",
            "trust_level": "research_only_visual_recovery",
        },
        "sessions": [
            {
                "session_id": session["session_id"],
                "category_summary": session["category_summary"],
                "session_diagnostics": session["session_diagnostics"],
                "representative_examples": session["representative_examples"],
            }
            for session in sessions
        ],
        "aggregate_category_summary": aggregate_category_summary,
        "main_cause_of_low_visual_recall": "low_temporal_sampling_density_and_prompt_false_negatives_near_audio_misses",
        "evidence": {
            "total_audio_misses": total_audio_misses,
            "audio_misses_recovered_by_visual": recovered_audio_misses,
            "sampled_audio_misses_without_positive_within_6s": sampled_no_positive_misses,
            "interpretation": "The fixed 1 FPS audio-gated stream often samples near missed anchors but does not produce recipe-positive frames; this points to temporal sparsity plus VLM/prompt false negatives, not interval geometry as the next primary lever.",
        },
        "recommended_next_lever": "NEXT_LEVER_AUDIO_LOCAL_VISUAL_BURST",
        "visual_recovery_retained": True,
        "product_default_changes": "none",
    }
    anchor_audit = {
        "benchmark_id": "r53_visual_recovery_anchor_audit",
        "sessions": [
            {
                "session_id": session["session_id"],
                "anchors": session["anchors"],
                "category_summary": session["category_summary"],
            }
            for session in sessions
        ],
        "aggregate_category_summary": aggregate_category_summary,
        "still_frame_examples": extracted_examples,
    }
    next_lever = {
        "benchmark_id": "r53_visual_recovery_next_lever",
        "decision": "R53_VISUAL_RECOVERY_FAILURE_MODE_IDENTIFIED",
        "recommended_next_lever": "NEXT_LEVER_AUDIO_LOCAL_VISUAL_BURST",
        "rationale": (
            "The weak sessions do not mainly fail because of ROI or interval shaping. They either have no need for recovery "
            "(Compete), or have audio-missed anchors with sampled nearby frames that remain VLM-negative. The next bounded "
            "test should increase temporal evidence locally around audio-miss neighborhoods, e.g. 4 FPS or 8 FPS in +/-3s "
            "diagnostic windows, before changing prompts or model family."
        ),
        "visual_recovery_retained": True,
        "product_default_changes": "none",
    }
    outputs = Path("outputs")
    write_json(outputs / "r53_visual_recovery_failure_mode_diagnosis.json", diagnosis)
    write_json(outputs / "r53_visual_recovery_anchor_audit.json", anchor_audit)
    write_json(outputs / "r53_visual_recovery_next_lever.json", next_lever)
    write_md(outputs / "r53_visual_recovery_failure_mode_diagnosis.md", render_diagnosis_md(diagnosis))
    write_md(outputs / "r53_visual_recovery_anchor_audit.md", render_anchor_audit_md(anchor_audit))
    write_md(outputs / "r53_visual_recovery_next_lever.md", render_next_lever_md(next_lever))
    write_md(
        Path("docs/research/R53_VISUAL_RECOVERY_FAILURE_MODE_DIAGNOSIS.md"),
        "\n".join(
            [
                "# R53 Visual Recovery Failure Mode Diagnosis",
                "",
                "This pass diagnoses the fixed r52 visual recovery recipe without changing prompt, ROI, FPS, geometry, product defaults, or approve policy.",
                "",
                f"- aggregate category summary: `{aggregate_category_summary}`",
                f"- main cause: `{diagnosis['main_cause_of_low_visual_recall']}`",
                f"- recommended next lever: `{next_lever['recommended_next_lever']}`",
                "- product defaults: unchanged",
                "- visual recovery remains research-only and retained for detection.",
            ]
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "aggregate_category_summary": aggregate_category_summary,
                "main_cause": diagnosis["main_cause_of_low_visual_recall"],
                "next_lever": next_lever["recommended_next_lever"],
                "still_frames": len([row for row in extracted_examples if row.get("still_frame_path")]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
