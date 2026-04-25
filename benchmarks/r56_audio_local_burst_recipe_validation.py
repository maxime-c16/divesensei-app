from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


R54_ROOT = Path("/Users/mcauchy/Downloads/r54_audio_local_visual_burst/r54_audio_local_visual_burst")
R56_ROOT = Path("/Users/mcauchy/Downloads/r56_burst_validation/r56_burst_validation")
R54_TARGET_ROOT = Path("kaggle/r54_visual_burst_sessions/r54_visual_burst_sessions/outputs")
R56_TARGET_ROOT = Path("kaggle/r56_burst_validation_sessions/r56_burst_validation_sessions/outputs")
SESSIONS = [
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
    "evaluation_insep_plateform_mixed_sound",
]
R52 = {"anchors": 288, "union_matched": 240, "union_recall": 0.8333}


RECIPES = [
    {"name": "r55_burst_8fps_any_score_ge_0.70", "fps": 8.0, "threshold": 0.70},
    {"name": "r55_burst_4fps_any_score_ge_0.70", "fps": 4.0, "threshold": 0.70},
    {"name": "aggressive_8fps_any_score_ge_0.65", "fps": 8.0, "threshold": 0.65},
    {"name": "aggressive_4fps_any_score_ge_0.65", "fps": 4.0, "threshold": 0.65},
]


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


def load_targets(root: Path, session_id: str, filename: str) -> list[dict[str, Any]]:
    return read_json(root / session_id / filename)["targets"]


def load_predictions(root: Path, session_id: str, fps: float, prefix: str) -> dict[str, list[dict[str, Any]]]:
    fps_label = str(fps).replace(".", "p")
    path = root / session_id / f"{prefix}_{fps_label}fps" / "visual_frame_predictions.jsonl"
    rows = read_jsonl(path)
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["_score"] = float(row.get("score") or 0.0)
        row["_margin"] = float(row.get("yes_no_first_token_margin") or 0.0)
        row["_timestamp"] = float(row.get("timestamp_seconds") or 0.0)
        by_reason[str(row.get("window_reason"))].append(row)
    return by_reason


def fires(rows: list[dict[str, Any]], threshold: float) -> tuple[bool, dict[str, Any]]:
    eligible = [row for row in rows if row["_margin"] > 0.0]
    max_score = max((row["_score"] for row in eligible), default=0.0)
    count = sum(1 for row in eligible if row["_score"] >= threshold)
    return count > 0, {"max_score": round(max_score, 4), "positive_frame_count": count, "sampled_frame_count": len(rows)}


def evaluate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    fps = float(recipe["fps"])
    threshold = float(recipe["threshold"])
    session_rows = []
    false_rows = []
    subtype_counter: Counter[str] = Counter()
    hard_total = hard_converted = controls_total = controls_preserved = 0
    false_total = false_positive = 0
    random_total = random_positive = 0
    reviewed_nuis_total = reviewed_nuis_positive = 0
    frame_count = 0
    for session_id in SESSIONS:
        hard_targets = load_targets(R54_TARGET_ROOT, session_id, "r54_burst_targets.json")
        hard_preds = load_predictions(R54_ROOT, session_id, fps, "burst_full_frame")
        false_targets = load_targets(R56_TARGET_ROOT, session_id, "r56_validation_targets.json")
        false_preds = load_predictions(R56_ROOT, session_id, fps, "false_controls_full_frame")
        s_hard_total = s_hard_converted = s_control_total = s_controls_preserved = 0
        s_false_total = s_false_positive = 0
        s_frames = 0
        for target in hard_targets:
            decision, metrics = fires(hard_preds.get(str(target["target_id"]), []), threshold)
            s_frames += metrics["sampled_frame_count"]
            if target["target_type"] == "audio_missed_visual_also_missed":
                s_hard_total += 1
                if decision:
                    s_hard_converted += 1
            elif target["target_type"] == "audio_missed_visual_recovered_control":
                s_control_total += 1
                if decision:
                    s_controls_preserved += 1
            elif target["target_type"] == "unmatched_visual_proposal_control":
                s_false_total += 1
                if decision:
                    s_false_positive += 1
                    false_rows.append({"session_id": session_id, "source": "r54_unmatched_visual_control", **target, **metrics})
        for target in false_targets:
            decision, metrics = fires(false_preds.get(str(target["target_id"]), []), threshold)
            s_frames += metrics["sampled_frame_count"]
            s_false_total += 1
            if target["target_type"] == "random_negative_away_from_reviewed_dives":
                random_total += 1
            else:
                reviewed_nuis_total += 1
            if decision:
                s_false_positive += 1
                false_positive += 1
                subtype_counter[str(target.get("subtype") or "unknown")] += 1
                false_rows.append({"session_id": session_id, "source": "r56_false_control", **target, **metrics})
                if target["target_type"] == "random_negative_away_from_reviewed_dives":
                    random_positive += 1
                else:
                    reviewed_nuis_positive += 1
        hard_total += s_hard_total
        hard_converted += s_hard_converted
        controls_total += s_control_total
        controls_preserved += s_controls_preserved
        false_total += s_false_total
        frame_count += s_frames
        session_rows.append(
            {
                "session_id": session_id,
                "hard_targets": s_hard_total,
                "hard_converted": s_hard_converted,
                "conversion_rate": round(s_hard_converted / s_hard_total, 4) if s_hard_total else 0.0,
                "positive_controls": s_control_total,
                "positive_controls_preserved": s_controls_preserved,
                "false_control_windows": s_false_total,
                "false_positive_windows": s_false_positive,
                "false_positive_rate": round(s_false_positive / s_false_total, 4) if s_false_total else 0.0,
                "frame_count": s_frames,
            }
        )
    union_matched = R52["union_matched"] + hard_converted
    total_false_windows = false_total
    total_window_minutes = total_false_windows * 6.0 / 60.0
    return {
        "recipe": recipe["name"],
        "fps": fps,
        "threshold": threshold,
        "hard_targets": hard_total,
        "hard_converted": hard_converted,
        "conversion_rate": round(hard_converted / hard_total, 4) if hard_total else 0.0,
        "positive_controls": controls_total,
        "positive_controls_preserved": controls_preserved,
        "positive_control_preservation_rate": round(controls_preserved / controls_total, 4) if controls_total else 0.0,
        "false_control_windows": total_false_windows,
        "false_positive_windows": false_positive + sum(1 for row in false_rows if row.get("source") == "r54_unmatched_visual_control"),
        "r56_false_positive_windows": false_positive,
        "false_positive_rate": round(false_positive / total_false_windows, 4) if total_false_windows else 0.0,
        "false_positive_rate_per_minute": round(false_positive / total_window_minutes, 4) if total_window_minutes else 0.0,
        "reviewed_nuisance_false_positive": reviewed_nuis_positive,
        "reviewed_nuisance_total": reviewed_nuis_total,
        "random_negative_false_positive": random_positive,
        "random_negative_total": random_total,
        "nuisance_false_positives_by_subtype": dict(sorted(subtype_counter.items())),
        "estimated_union_matched": union_matched,
        "estimated_union_recall": round(union_matched / R52["anchors"], 4),
        "review_burden_delta_windows": false_positive,
        "frame_count": frame_count,
        "session_rows": session_rows,
        "false_positive_rows": false_rows,
    }


def render_comparison_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R56 Burst Recipe Candidate Comparison",
        "",
        "| Recipe | Hard converted | Controls | R56 FP | FP rate | FP/min | Est union recall | Frames |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"| `{row['recipe']}` | {row['hard_converted']}/{row['hard_targets']} | {row['positive_controls_preserved']}/{row['positive_controls']} | {row['r56_false_positive_windows']}/{row['false_control_windows']} | {row['false_positive_rate']} | {row['false_positive_rate_per_minute']} | {row['estimated_union_recall']} | {row['frame_count']} |"
        )
    return "\n".join(lines) + "\n"


def render_validation_md(payload: dict[str, Any]) -> str:
    best = payload["selected_recipe"]
    return (
        "# R56 Audio-Local Burst Recipe Validation\n\n"
        f"- selected recipe: `{best['recipe']}`\n"
        f"- hard recovery: `{best['hard_converted']}/{best['hard_targets']}`\n"
        f"- false controls: `{best['r56_false_positive_windows']}/{best['false_control_windows']}`\n"
        f"- estimated union recall: `{best['estimated_union_recall']}`\n"
        f"- decision: `{payload['decision']}`\n"
        f"- next recipe decision: `{payload['next_recipe_decision']}`\n"
        f"- main remaining risk: `{payload['main_remaining_risk']}`\n"
    )


def render_false_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R56 Burst False-Control Audit",
        "",
        "| Recipe | Reviewed nuisance FP | Random FP | Subtypes |",
        "|---|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"| `{row['recipe']}` | {row['reviewed_nuisance_false_positive']}/{row['reviewed_nuisance_total']} | {row['random_negative_false_positive']}/{row['random_negative_total']} | `{row['nuisance_false_positives_by_subtype']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    health = read_json(R56_ROOT / "r56_remote_validation_health.json")
    remote_summary = read_json(R56_ROOT / "r56_remote_validation_summary.json")
    candidates = [evaluate_recipe(recipe) for recipe in RECIPES]
    primary = next(row for row in candidates if row["recipe"] == "r55_burst_8fps_any_score_ge_0.70")
    efficiency = next(row for row in candidates if row["recipe"] == "r55_burst_4fps_any_score_ge_0.70")
    # Validation is intentionally conservative: require high recovery, preserved controls, and bounded new false controls.
    if primary["conversion_rate"] >= 0.75 and primary["positive_control_preservation_rate"] >= 1.0 and primary["false_positive_rate"] <= 0.20:
        decision = "R56_BURST_RECIPE_VALIDATED"
        next_recipe = "VISUAL_RECOVERY_RECIPE_AUDIO_LOCAL_BURST_8FPS_SCORE_070"
        selected = primary
    elif efficiency["conversion_rate"] >= 0.75 and efficiency["false_positive_rate"] <= 0.20:
        decision = "R56_BURST_RECIPE_VALIDATED"
        next_recipe = "VISUAL_RECOVERY_RECIPE_AUDIO_LOCAL_BURST_4FPS_SCORE_070"
        selected = efficiency
    else:
        decision = "R56_BURST_RECIPE_NEEDS_HARDENING"
        next_recipe = "VISUAL_RECOVERY_RECIPE_NEEDS_FALSE_POSITIVE_HARDENING"
        selected = primary
    validation = {
        "benchmark_id": "r56_audio_local_burst_recipe_validation",
        "remote_vlm_rerun": True,
        "remote_scoring_health": health,
        "remote_scoring_summary": remote_summary,
        "validation_set_composition": read_json(Path("outputs/r56_burst_false_control_audit.json")),
        "r52_reference": R52,
        "candidates": [{k: v for k, v in row.items() if k not in {"session_rows", "false_positive_rows"}} for row in candidates],
        "selected_recipe": {k: v for k, v in selected.items() if k not in {"session_rows", "false_positive_rows"}},
        "decision": decision,
        "next_recipe_decision": next_recipe,
        "main_remaining_risk": "false_positive_rate_on_broader_nuisance_controls",
        "recommended_next_detection_step": (
            "Harden the burst recipe with a second-stage false-positive guard before adopting it as the next research recipe."
            if decision == "R56_BURST_RECIPE_NEEDS_HARDENING"
            else "Use the selected burst recipe as the next research-only visual recovery recipe and validate on one more fresh reviewed session."
        ),
        "product_default_changes": "none",
        "visual_recovery_decision": "VISUAL_RECOVERY_RETAINED_FOR_DETECTION",
    }
    comparison = {
        "benchmark_id": "r56_burst_recipe_candidate_comparison",
        "candidates": validation["candidates"],
        "selected_recipe": validation["selected_recipe"],
    }
    false_audit = {
        "benchmark_id": "r56_burst_false_control_audit",
        "remote_vlm_rerun": True,
        "validation_set_composition": validation["validation_set_composition"],
        "candidates": validation["candidates"],
        "false_positive_rows_by_recipe": {
            row["recipe"]: row["false_positive_rows"]
            for row in candidates
        },
    }
    write_json(Path("outputs/r56_audio_local_burst_recipe_validation.json"), validation)
    write_json(Path("outputs/r56_burst_recipe_candidate_comparison.json"), comparison)
    write_json(Path("outputs/r56_burst_false_control_audit.json"), false_audit)
    write_md(Path("outputs/r56_audio_local_burst_recipe_validation.md"), render_validation_md(validation))
    write_md(Path("outputs/r56_burst_recipe_candidate_comparison.md"), render_comparison_md(comparison))
    write_md(Path("outputs/r56_burst_false_control_audit.md"), render_false_md(false_audit))
    write_md(
        Path("docs/research/R56_AUDIO_LOCAL_BURST_RECIPE_VALIDATION.md"),
        render_validation_md(validation),
    )
    print(json.dumps({"decision": decision, "next_recipe": next_recipe, "selected": validation["selected_recipe"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
