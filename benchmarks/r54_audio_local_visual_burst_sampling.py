from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(".")
BURST_ROOT = Path("/Users/mcauchy/Downloads/r54_audio_local_visual_burst/r54_audio_local_visual_burst")
FPS_VALUES = [4.0, 8.0]
SESSION_IDS = [
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
    "evaluation_insep_plateform_mixed_sound",
]
POSITIVE_SCORE_THRESHOLD = 0.845
R52_AGGREGATE = {
    "reviewed_anchors": 288,
    "audio_matched": 230,
    "union_matched": 240,
    "audio_recall": 0.7986,
    "union_recall": 0.8333,
    "recovered_anchors": 10,
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


def safe_rate(numer: int, denom: int) -> float:
    return round(numer / denom, 4) if denom else 0.0


def target_root_for_session(session_id: str) -> Path:
    return Path("kaggle/r54_visual_burst_sessions/r54_visual_burst_sessions/outputs") / session_id


def load_targets(session_id: str) -> list[dict[str, Any]]:
    return read_json(target_root_for_session(session_id) / "r54_burst_targets.json").get("targets", [])


def prediction_positive(row: dict[str, Any]) -> bool:
    return (
        row.get("prompt_id") == "diving_attempt"
        and row.get("decision_rule") == "yes_no_first_token_margin"
        and bool(row.get("is_positive"))
        and float(row.get("score") or 0.0) >= POSITIVE_SCORE_THRESHOLD
    )


def load_predictions(session_id: str, fps: float) -> list[dict[str, Any]]:
    fps_label = str(fps).replace(".", "p")
    path = BURST_ROOT / session_id / f"burst_full_frame_{fps_label}fps" / "visual_frame_predictions.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row["_timestamp"] = float(row["timestamp_seconds"])
        row["_score"] = float(row.get("score") or 0.0)
        row["_margin"] = float(row.get("yes_no_first_token_margin") or 0.0)
        row["_positive_for_recipe"] = prediction_positive(row)
    return rows


def load_summary(session_id: str, fps: float) -> dict[str, Any]:
    fps_label = str(fps).replace(".", "p")
    path = BURST_ROOT / session_id / f"burst_full_frame_{fps_label}fps" / "visual_vlm_proposal_summary.json"
    return read_json(path) if path.exists() else {}


def predictions_near_target(predictions: list[dict[str, Any]], target: dict[str, Any], radius: float = 3.0) -> list[dict[str, Any]]:
    ts = float(target["timestamp_seconds"])
    return [row for row in predictions if abs(row["_timestamp"] - ts) <= radius]


def intervals_from_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [row for row in predictions if row["_positive_for_recipe"]]
    positives.sort(key=lambda row: row["_timestamp"])
    groups: list[list[dict[str, Any]]] = []
    for row in positives:
        if not groups or row["_timestamp"] - groups[-1][-1]["_timestamp"] > 3.0:
            groups.append([row])
        else:
            groups[-1].append(row)
    intervals = []
    for idx, group in enumerate(groups, start=1):
        intervals.append(
            {
                "visual_interval_id": f"r54-burst-{idx:04d}",
                "start_seconds": round(max(0.0, group[0]["_timestamp"] - 1.5), 3),
                "end_seconds": round(group[-1]["_timestamp"] + 3.0, 3),
                "anchor_timestamp_seconds": round(group[len(group) // 2]["_timestamp"], 3),
                "duration_seconds": round(group[-1]["_timestamp"] + 3.0 - max(0.0, group[0]["_timestamp"] - 1.5), 3),
                "positive_frame_count": len(group),
                "max_score": max(row["_score"] for row in group),
            }
        )
    return intervals


def interval_matches_target(interval: dict[str, Any], target: dict[str, Any], radius: float = 3.0) -> bool:
    ts = float(target["timestamp_seconds"])
    return (
        float(interval["start_seconds"]) <= ts <= float(interval["end_seconds"])
        or abs(float(interval["anchor_timestamp_seconds"]) - ts) <= radius
    )


def evaluate_session_fps(session_id: str, fps: float) -> dict[str, Any]:
    targets = load_targets(session_id)
    predictions = load_predictions(session_id, fps)
    intervals = intervals_from_predictions(predictions)
    summary = load_summary(session_id, fps)
    rows: list[dict[str, Any]] = []
    converted_primary = 0
    preserved_controls = 0
    false_control_positives = 0
    primary_targets = [row for row in targets if row["target_type"] == "audio_missed_visual_also_missed"]
    positive_controls = [row for row in targets if row["target_type"] == "audio_missed_visual_recovered_control"]
    unmatched_controls = [row for row in targets if row["target_type"] == "unmatched_visual_proposal_control"]
    for target in targets:
        near = predictions_near_target(predictions, target)
        positives = [row for row in near if row["_positive_for_recipe"]]
        converted = bool(positives)
        best = max(near, key=lambda row: row["_score"], default=None)
        if target["target_type"] == "audio_missed_visual_also_missed" and converted:
            converted_primary += 1
        if target["target_type"] == "audio_missed_visual_recovered_control" and converted:
            preserved_controls += 1
        if target["target_type"] == "unmatched_visual_proposal_control" and converted:
            false_control_positives += 1
        rows.append(
            {
                **target,
                "fps": fps,
                "sampled_frame_count": len(near),
                "positive_frame_count": len(positives),
                "converted_or_positive": converted,
                "best_score": round(best["_score"], 4) if best else None,
                "best_margin": round(best["_margin"], 4) if best else None,
                "best_timestamp_seconds": round(best["_timestamp"], 3) if best else None,
                "positive_timestamps": [round(row["_timestamp"], 3) for row in positives[:10]],
            }
        )
    matched_intervals: set[str] = set()
    for interval in intervals:
        for target in targets:
            if target.get("expected_positive") and interval_matches_target(interval, target):
                matched_intervals.add(str(interval["visual_interval_id"]))
                break
    false_intervals = [row for row in intervals if str(row["visual_interval_id"]) not in matched_intervals]
    total_window_seconds = len(targets) * 6.0
    return {
        "session_id": session_id,
        "fps": fps,
        "target_count": len(targets),
        "primary_av_minus_v_minus_targets": len(primary_targets),
        "primary_converted": converted_primary,
        "primary_conversion_rate": safe_rate(converted_primary, len(primary_targets)),
        "positive_control_count": len(positive_controls),
        "positive_controls_preserved": preserved_controls,
        "positive_control_preservation_rate": safe_rate(preserved_controls, len(positive_controls)),
        "unmatched_visual_control_count": len(unmatched_controls),
        "unmatched_visual_controls_positive": false_control_positives,
        "prediction_rows": len(predictions),
        "positive_prediction_rows": sum(1 for row in predictions if row["_positive_for_recipe"]),
        "interval_count": len(intervals),
        "new_false_visual_intervals": len(false_intervals),
        "false_visual_per_burst_minute": round(len(false_intervals) / (total_window_seconds / 60.0), 4) if total_window_seconds else 0.0,
        "total_burst_window_seconds": total_window_seconds,
        "runtime_seconds": summary.get("elapsed_seconds"),
        "sampled_frame_count_remote": summary.get("sampled_frame_count"),
        "target_rows": rows,
        "false_intervals": false_intervals,
    }


def aggregate(rows: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    fps_rows = [row for row in rows if row["fps"] == fps]
    primary_targets = sum(row["primary_av_minus_v_minus_targets"] for row in fps_rows)
    primary_converted = sum(row["primary_converted"] for row in fps_rows)
    controls = sum(row["positive_control_count"] for row in fps_rows)
    controls_preserved = sum(row["positive_controls_preserved"] for row in fps_rows)
    false_intervals = sum(row["new_false_visual_intervals"] for row in fps_rows)
    window_seconds = sum(row["total_burst_window_seconds"] for row in fps_rows)
    added_union = R52_AGGREGATE["union_matched"] + primary_converted
    return {
        "fps": fps,
        "sessions": len(fps_rows),
        "primary_av_minus_v_minus_targets": primary_targets,
        "primary_converted": primary_converted,
        "primary_conversion_rate": safe_rate(primary_converted, primary_targets),
        "positive_controls": controls,
        "positive_controls_preserved": controls_preserved,
        "positive_control_preservation_rate": safe_rate(controls_preserved, controls),
        "new_false_visual_intervals": false_intervals,
        "false_visual_per_burst_minute": round(false_intervals / (window_seconds / 60.0), 4) if window_seconds else 0.0,
        "prediction_rows": sum(row["prediction_rows"] for row in fps_rows),
        "runtime_seconds": round(sum(float(row["runtime_seconds"] or 0.0) for row in fps_rows), 3),
        "recovered_anchors_added_over_r52": primary_converted,
        "r52_union_recall": R52_AGGREGATE["union_recall"],
        "new_union_matched_anchors_estimate": added_union,
        "new_union_recall_estimate": round(added_union / R52_AGGREGATE["reviewed_anchors"], 4),
        "review_burden_delta_false_intervals": false_intervals,
    }


def failure_update(rows: list[dict[str, Any]], best_fps: float) -> list[dict[str, Any]]:
    updates = []
    for session in [row for row in rows if row["fps"] == best_fps]:
        remaining = [
            target
            for target in session["target_rows"]
            if target["target_type"] == "audio_missed_visual_also_missed" and not target["converted_or_positive"]
        ]
        for target in remaining:
            if target["sampled_frame_count"] == 0:
                reason = "unknown"
            elif target["best_score"] is not None and target["best_score"] >= 0.7:
                reason = "prompt_or_model_false_negative"
            else:
                reason = "visual_scene_not_informative"
            updates.append(
                {
                    "session_id": session["session_id"],
                    "fps": best_fps,
                    "anchor_id": target["anchor_id"],
                    "timestamp_seconds": target["timestamp_seconds"],
                    "best_score": target["best_score"],
                    "best_margin": target["best_margin"],
                    "sampled_frame_count": target["sampled_frame_count"],
                    "updated_failure_reason": reason,
                }
            )
    return updates


def render_results_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R54 Audio-Local Visual Burst Results",
        "",
        "| Session | FPS | A-V- targets | Converted | Rate | Controls preserved | False intervals | Frames | Runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["session_results"]:
        lines.append(
            f"| `{row['session_id']}` | {row['fps']} | {row['primary_av_minus_v_minus_targets']} | {row['primary_converted']} | {row['primary_conversion_rate']} | {row['positive_controls_preserved']}/{row['positive_control_count']} | {row['new_false_visual_intervals']} | {row['prediction_rows']} | {row['runtime_seconds']} |"
        )
    lines.extend(["", "## Aggregate"])
    for row in payload["aggregate_results"]:
        lines.append(
            f"- `{row['fps']} FPS`: converted `{row['primary_converted']}/{row['primary_av_minus_v_minus_targets']}`, new union recall estimate `{row['new_union_recall_estimate']}`, false intervals `{row['new_false_visual_intervals']}`, frames `{row['prediction_rows']}`"
        )
    lines.append("")
    lines.append(f"- decision: `{payload['decision']}`")
    lines.append(f"- next lever: `{payload['next_lever']}`")
    return "\n".join(lines) + "\n"


def render_failure_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R54 Visual Burst Failure Update",
        "",
        "| Session | Anchor | Timestamp | Best score | Samples | Updated reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["remaining_failures"][:80]:
        lines.append(
            f"| `{row['session_id']}` | `{row['anchor_id']}` | {row['timestamp_seconds']} | {row['best_score']} | {row['sampled_frame_count']} | `{row['updated_failure_reason']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    health = read_json(BURST_ROOT / "r54_remote_burst_health.json")
    remote_summary = read_json(BURST_ROOT / "r54_remote_burst_summary.json")
    session_results = []
    for fps in FPS_VALUES:
        for session_id in SESSION_IDS:
            session_results.append(evaluate_session_fps(session_id, fps))
    aggregate_results = [aggregate(session_results, fps) for fps in FPS_VALUES]
    best = max(
        aggregate_results,
        key=lambda row: (row["primary_converted"], -row["new_false_visual_intervals"], -row["prediction_rows"]),
    )
    gain = best["primary_converted"] > 0
    next_lever = "VISUAL_RECOVERY_NEXT_RECIPE_AUDIO_LOCAL_BURST" if gain else "NEXT_LEVER_PROMPT_OR_MODEL_REVISION"
    results = {
        "benchmark_id": "r54_audio_local_visual_burst_results",
        "remote_scoring_health": health,
        "remote_scoring_summary": remote_summary,
        "r52_reference": R52_AGGREGATE,
        "session_results": session_results,
        "aggregate_results": aggregate_results,
        "best_fps": best["fps"],
        "best_result": best,
        "decision": "R54_AUDIO_LOCAL_VISUAL_BURST_GAIN" if gain else "R54_AUDIO_LOCAL_VISUAL_BURST_NO_CLEAR_GAIN",
        "next_lever": next_lever,
        "visual_recovery_decision": "VISUAL_RECOVERY_RETAINED_FOR_DETECTION",
        "product_default_changes": "none",
    }
    failure = {
        "benchmark_id": "r54_visual_burst_failure_update",
        "best_fps": best["fps"],
        "remaining_failures": failure_update(session_results, best["fps"]),
        "failure_mode_interpretation": (
            "Remaining A-V- failures after burst scoring are no longer primarily a 1 FPS sampling issue if sampled frames exist; "
            "they point to prompt/model false negatives or visually uninformative/camera-framing-limited windows."
        ),
    }
    write_json(Path("outputs/r54_audio_local_visual_burst_results.json"), results)
    write_md(Path("outputs/r54_audio_local_visual_burst_results.md"), render_results_md(results))
    write_json(Path("outputs/r54_visual_burst_failure_update.json"), failure)
    write_md(Path("outputs/r54_visual_burst_failure_update.md"), render_failure_md(failure))
    write_md(
        Path("docs/research/R54_AUDIO_LOCAL_VISUAL_BURST_SAMPLING.md"),
        "\n".join(
            [
                "# R54 Audio-Local Visual Burst Sampling",
                "",
                "This pass tests 4 FPS and 8 FPS PaliGemma burst scoring in +/-3s windows around r53 hard anchors and controls. It does not change product defaults or approve policy.",
                "",
                f"- remote GPU devices: `{health.get('cuda_devices')}`",
                f"- best FPS: `{best['fps']}`",
                f"- converted A-V- hard anchors: `{best['primary_converted']}/{best['primary_av_minus_v_minus_targets']}`",
                f"- new union recall estimate: `{best['new_union_recall_estimate']}`",
                f"- decision: `{results['decision']}`",
                f"- next lever: `{next_lever}`",
            ]
        )
        + "\n",
    )
    print(json.dumps({"best": best, "decision": results["decision"], "next_lever": next_lever}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
