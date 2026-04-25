from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable


BURST_ROOT = Path("/Users/mcauchy/Downloads/r54_audio_local_visual_burst/r54_audio_local_visual_burst")
TARGET_ROOT = Path("kaggle/r54_visual_burst_sessions/r54_visual_burst_sessions/outputs")
SESSIONS = [
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
    "evaluation_insep_plateform_mixed_sound",
]
FPS_VALUES = [4.0, 8.0]
R52 = {"anchors": 288, "union_matched": 240, "union_recall": 0.8333}
R54_BEST = {"fps": 8.0, "converted": 13, "union_matched": 253, "union_recall": 0.8785, "false_intervals": 3}


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


def load_targets(session_id: str) -> list[dict[str, Any]]:
    return read_json(TARGET_ROOT / session_id / "r54_burst_targets.json")["targets"]


def load_predictions(session_id: str, fps: float) -> dict[str, list[dict[str, Any]]]:
    fps_label = str(fps).replace(".", "p")
    rows = read_jsonl(BURST_ROOT / session_id / f"burst_full_frame_{fps_label}fps" / "visual_frame_predictions.jsonl")
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row["_score"] = float(row.get("score") or 0.0)
        row["_margin"] = float(row.get("yes_no_first_token_margin") or 0.0)
        row["_timestamp"] = float(row.get("timestamp_seconds") or 0.0)
        by_reason.setdefault(str(row.get("window_reason")), []).append(row)
    for group in by_reason.values():
        group.sort(key=lambda row: row["_timestamp"])
    return by_reason


def top_mean(values: list[float], n: int) -> float:
    if not values:
        return 0.0
    return sum(sorted(values, reverse=True)[:n]) / min(n, len(values))


def session_stats(predictions: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    scores = [row["_score"] for rows in predictions.values() for row in rows]
    if not scores:
        return {"mean": 0.0, "std": 0.0, "median": 0.0}
    mean = sum(scores) / len(scores)
    var = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {"mean": mean, "std": var**0.5, "median": statistics.median(scores)}


def candidate_defs(stats: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"name": "current_recipe_any_score_ge_0.845", "kind": "any_score", "score_threshold": 0.845, "min_count": 1},
        {"name": "lower_score_any_ge_0.80", "kind": "any_score", "score_threshold": 0.80, "min_count": 1},
        {"name": "lower_score_any_ge_0.75", "kind": "any_score", "score_threshold": 0.75, "min_count": 1},
        {"name": "lower_score_any_ge_0.70", "kind": "any_score", "score_threshold": 0.70, "min_count": 1},
        {"name": "lower_score_any_ge_0.65", "kind": "any_score", "score_threshold": 0.65, "min_count": 1},
        {"name": "top2_mean_score_ge_0.70", "kind": "top_mean", "score_threshold": 0.70, "top_n": 2},
        {"name": "top2_mean_score_ge_0.65", "kind": "top_mean", "score_threshold": 0.65, "top_n": 2},
        {"name": "top3_mean_score_ge_0.65", "kind": "top_mean", "score_threshold": 0.65, "top_n": 3},
        {"name": "two_frames_score_ge_0.70", "kind": "any_score", "score_threshold": 0.70, "min_count": 2},
        {"name": "two_frames_score_ge_0.65", "kind": "any_score", "score_threshold": 0.65, "min_count": 2},
        {
            "name": "session_normalized_score_ge_mean_plus_0p5std",
            "kind": "any_score",
            "score_threshold": stats["mean"] + 0.5 * stats["std"],
            "min_count": 1,
            "session_normalized": True,
        },
    ]


def fires(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    scores = [row["_score"] for row in rows if row["_margin"] > 0.0]
    margins = [row["_margin"] for row in rows]
    threshold = float(candidate["score_threshold"])
    if candidate["kind"] == "any_score":
        count = sum(1 for score in scores if score >= threshold)
        decision = count >= int(candidate.get("min_count", 1))
    elif candidate["kind"] == "top_mean":
        value = top_mean(scores, int(candidate.get("top_n", 2)))
        count = sum(1 for score in scores if score >= threshold)
        decision = value >= threshold
    else:
        raise ValueError(candidate["kind"])
    return decision, {
        "max_score": max(scores) if scores else 0.0,
        "top2_mean": top_mean(scores, 2),
        "top3_mean": top_mean(scores, 3),
        "max_margin": max(margins) if margins else 0.0,
        "frame_count": len(rows),
        "near_positive_count": sum(1 for score in scores if score >= threshold),
    }


def evaluate_candidate(candidate: dict[str, Any], fps: float) -> dict[str, Any]:
    session_rows = []
    target_rows = []
    for session_id in SESSIONS:
        targets = load_targets(session_id)
        predictions = load_predictions(session_id, fps)
        primary = [row for row in targets if row["target_type"] == "audio_missed_visual_also_missed"]
        controls = [row for row in targets if row["target_type"] == "audio_missed_visual_recovered_control"]
        false_controls = [row for row in targets if row["target_type"] == "unmatched_visual_proposal_control"]
        converted = preserved = false_hits = 0
        for target in targets:
            decision, metrics = fires(candidate, predictions.get(str(target["target_id"]), []))
            if target["target_type"] == "audio_missed_visual_also_missed" and decision:
                converted += 1
            if target["target_type"] == "audio_missed_visual_recovered_control" and decision:
                preserved += 1
            if target["target_type"] == "unmatched_visual_proposal_control" and decision:
                false_hits += 1
            target_rows.append(
                {
                    "session_id": session_id,
                    "fps": fps,
                    "candidate": candidate["name"],
                    **target,
                    "decision": decision,
                    **{key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()},
                }
            )
        session_rows.append(
            {
                "session_id": session_id,
                "fps": fps,
                "candidate": candidate["name"],
                "primary_targets": len(primary),
                "primary_converted": converted,
                "primary_conversion_rate": round(converted / len(primary), 4) if primary else 0.0,
                "positive_controls": len(controls),
                "positive_controls_preserved": preserved,
                "positive_control_preservation_rate": round(preserved / len(controls), 4) if controls else 0.0,
                "unmatched_controls": len(false_controls),
                "unmatched_controls_positive": false_hits,
            }
        )
    primary_targets = sum(row["primary_targets"] for row in session_rows)
    converted = sum(row["primary_converted"] for row in session_rows)
    controls = sum(row["positive_controls"] for row in session_rows)
    preserved = sum(row["positive_controls_preserved"] for row in session_rows)
    false_hits = sum(row["unmatched_controls_positive"] for row in session_rows)
    union_matched = R52["union_matched"] + converted
    return {
        "candidate": candidate["name"],
        "fps": fps,
        "score_threshold": round(float(candidate["score_threshold"]), 4),
        "primary_targets": primary_targets,
        "primary_converted": converted,
        "primary_conversion_rate": round(converted / primary_targets, 4) if primary_targets else 0.0,
        "positive_controls": controls,
        "positive_controls_preserved": preserved,
        "positive_control_preservation_rate": round(preserved / controls, 4) if controls else 0.0,
        "unmatched_controls_positive": false_hits,
        "estimated_union_matched": union_matched,
        "estimated_union_recall": round(union_matched / R52["anchors"], 4),
        "added_recovered_over_r54": converted - R54_BEST["converted"] if fps == 8.0 else None,
        "session_rows": session_rows,
        "target_rows": target_rows,
    }


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Avoid choosing the most permissive threshold as the recipe recommendation.
    # The r55 objective is a usable score rule, not a pure recall maximizer.
    candidates = [
        row
        for row in rows
        if row["positive_control_preservation_rate"] >= 1.0
        and row["fps"] == 8.0
        and float(row.get("score_threshold") or 0.0) >= 0.70
    ]
    return max(
        candidates or rows,
        key=lambda row: (
            row["primary_converted"],
            -row["unmatched_controls_positive"],
            row["estimated_union_recall"],
            1 if row["fps"] == 8.0 else 0,
        ),
    )


def remaining_failures(best: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for row in best["target_rows"]:
        if row["target_type"] != "audio_missed_visual_also_missed" or row["decision"]:
            continue
        if row["max_score"] >= 0.7:
            reason = "recoverable_by_threshold"
        elif row["max_score"] >= 0.55:
            reason = "likely_prompt_false_negative"
        elif row["frame_count"] > 0:
            reason = "visually_uninformative"
        else:
            reason = "uncertain"
        failures.append(
            {
                "session_id": row["session_id"],
                "anchor_id": row["anchor_id"],
                "timestamp_seconds": row["timestamp_seconds"],
                "best_score": row["max_score"],
                "top2_mean": row["top2_mean"],
                "max_margin": row["max_margin"],
                "diagnosis": reason,
            }
        )
    return failures


def render_comparison_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# R55 Burst Threshold Candidate Comparison",
        "",
        "| Candidate | FPS | Converted | Rate | Controls | False controls | Est. union recall | Added over r54 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["fps"], -r["primary_converted"], r["unmatched_controls_positive"], r["candidate"])):
        lines.append(
            f"| `{row['candidate']}` | {row['fps']} | {row['primary_converted']}/{row['primary_targets']} | {row['primary_conversion_rate']} | {row['positive_controls_preserved']}/{row['positive_controls']} | {row['unmatched_controls_positive']} | {row['estimated_union_recall']} | {row['added_recovered_over_r54']} |"
        )
    return "\n".join(lines) + "\n"


def render_diagnosis_md(payload: dict[str, Any]) -> str:
    best = payload["best_candidate"]
    lines = [
        "# R55 Audio-Local Burst Prompt/Score Diagnosis",
        "",
        f"- remote VLM rerun: `{payload['remote_vlm_rerun']}`",
        f"- best candidate: `{best['candidate']}` at `{best['fps']}` FPS",
        f"- converted: `{best['primary_converted']}/{best['primary_targets']}`",
        f"- estimated union recall: `{best['estimated_union_recall']}`",
        f"- added recovered over r54: `{best['added_recovered_over_r54']}`",
        f"- false unmatched controls: `{best['unmatched_controls_positive']}`",
        "",
        "## CAO Finding",
        payload["cao_finding"],
        "",
        "## SNMT / INSEP Finding",
        payload["snmt_insep_finding"],
        "",
        f"- decision: `{payload['decision']}`",
        f"- next recipe: `{payload['next_recipe']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    all_results = []
    for fps in FPS_VALUES:
        # Session-normalized candidates need per-session stats; evaluate with a representative aggregate threshold
        base_stats = {"mean": 0.0, "std": 0.0, "median": 0.0}
        candidates = candidate_defs(base_stats)
        for candidate in candidates:
            if candidate.get("session_normalized"):
                # Use all-frame global stats for this FPS; keep it explicit as a weak diagnostic.
                all_preds = []
                for session_id in SESSIONS:
                    for group in load_predictions(session_id, fps).values():
                        all_preds.extend(group)
                stats = session_stats({"all": all_preds})
                candidate = dict(candidate)
                candidate["score_threshold"] = stats["mean"] + 0.5 * stats["std"]
            all_results.append(evaluate_candidate(candidate, fps))
    best = choose_best(all_results)
    current_8 = next(row for row in all_results if row["fps"] == 8.0 and row["candidate"] == "current_recipe_any_score_ge_0.845")
    failures = remaining_failures(best)
    from collections import Counter

    failure_counts = dict(Counter(row["diagnosis"] for row in failures))
    cao_sessions = {"evaluation_CAO-1st-15min_20260421-072906", "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927"}
    cao_rows = [row for row in best["session_rows"] if row["session_id"] in cao_sessions]
    snmt_insep_rows = [row for row in best["session_rows"] if row["session_id"] not in cao_sessions]
    cao_converted = sum(row["primary_converted"] for row in cao_rows)
    cao_targets = sum(row["primary_targets"] for row in cao_rows)
    si_converted = sum(row["primary_converted"] for row in snmt_insep_rows)
    si_targets = sum(row["primary_targets"] for row in snmt_insep_rows)
    gain = (
        best["fps"] == 8.0
        and best["primary_converted"] > R54_BEST["converted"]
        and best["unmatched_controls_positive"] <= current_8["unmatched_controls_positive"]
    )
    next_recipe = (
        "VISUAL_RECOVERY_RECIPE_AUDIO_LOCAL_BURST_WITH_SCORE_AGGREGATION"
        if gain
        else "NEXT_LEVER_PROMPT_OR_MODEL_REVISION"
    )
    diagnosis = {
        "benchmark_id": "r55_audio_local_burst_prompt_score_diagnosis",
        "remote_vlm_rerun": False,
        "source_artifacts": str(BURST_ROOT),
        "r54_reference": R54_BEST,
        "best_candidate": {k: v for k, v in best.items() if k not in {"session_rows", "target_rows"}},
        "raw_max_recall_candidate": {
            k: v
            for k, v in max(all_results, key=lambda row: (row["primary_converted"], -row["unmatched_controls_positive"])).items()
            if k not in {"session_rows", "target_rows"}
        },
        "current_8fps_reference_candidate": {k: v for k, v in current_8.items() if k not in {"session_rows", "target_rows"}},
        "cao_finding": (
            f"Best score-only rule converts {cao_converted}/{cao_targets} CAO A-V- targets. "
            "CAO improves only when thresholds are substantially relaxed, indicating prompt/model weakness remains."
        ),
        "snmt_insep_finding": (
            f"Best score-only rule converts {si_converted}/{si_targets} SNMT/INSEP A-V- targets and preserves positive controls. "
            "These sessions are more score-threshold recoverable than CAO."
        ),
        "remaining_failure_counts": failure_counts,
        "decision": "R55_BURST_SCORE_AGGREGATION_GAIN" if gain else "R55_BURST_SCORE_AGGREGATION_NO_CLEAR_GAIN",
        "next_recipe": next_recipe,
        "visual_recovery_decision": "VISUAL_RECOVERY_RETAINED_FOR_DETECTION",
        "product_default_changes": "none",
    }
    comparison = {
        "benchmark_id": "r55_burst_threshold_candidate_comparison",
        "candidates": [{k: v for k, v in row.items() if k not in {"session_rows", "target_rows"}} for row in all_results],
        "best_candidate": diagnosis["best_candidate"],
    }
    failure_payload = {
        "benchmark_id": "r55_remaining_visual_failure_diagnosis",
        "best_candidate": diagnosis["best_candidate"],
        "remaining_failures": failures,
        "failure_counts": failure_counts,
    }
    write_json(Path("outputs/r55_audio_local_burst_prompt_score_diagnosis.json"), diagnosis)
    write_json(Path("outputs/r55_burst_threshold_candidate_comparison.json"), comparison)
    write_json(Path("outputs/r55_remaining_visual_failure_diagnosis.json"), failure_payload)
    write_md(Path("outputs/r55_audio_local_burst_prompt_score_diagnosis.md"), render_diagnosis_md(diagnosis))
    write_md(Path("outputs/r55_burst_threshold_candidate_comparison.md"), render_comparison_md(comparison["candidates"]))
    lines = [
        "# R55 Remaining Visual Failure Diagnosis",
        "",
        f"- failure counts: `{failure_counts}`",
        "",
        "| Session | Anchor | Timestamp | Best score | Top2 | Diagnosis |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in failures:
        lines.append(
            f"| `{row['session_id']}` | `{row['anchor_id']}` | {row['timestamp_seconds']} | {row['best_score']} | {row['top2_mean']} | `{row['diagnosis']}` |"
        )
    write_md(Path("outputs/r55_remaining_visual_failure_diagnosis.md"), "\n".join(lines) + "\n")
    write_md(
        Path("docs/research/R55_AUDIO_LOCAL_BURST_PROMPT_SCORE_DIAGNOSIS.md"),
        render_diagnosis_md(diagnosis),
    )
    print(json.dumps({"best": diagnosis["best_candidate"], "decision": diagnosis["decision"], "failure_counts": failure_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
