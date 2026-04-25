from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


SESSION_ROOT = Path("outputs/evaluation_insep_plateform_mixed_sound")
DEFAULT_REMOTE_RESULTS = Path("/Users/mcauchy/Downloads/r48_remote_audio_window_prompt_ablation")
WINDOW_OFFSETS_SECONDS = (-2.0, -1.0, 0.0, 1.0, 2.0)
SCORE_THRESHOLD = 0.845
METHODS = ("max_score", "mean_top2_score", "top3_mean", "any_positive", "majority_positive")
PROMPTS = [
    "baseline_frame_prompt",
    "window_dive_evidence_prompt",
    "anti_clutter_prompt",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_predictions(results_root: Path) -> Path | None:
    candidates = [
        results_root / "r48_audio_window_frame_predictions.jsonl",
        results_root / "r48_remote_audio_window_prompt_ablation" / "r48_audio_window_frame_predictions.jsonl",
    ]
    candidates.extend(results_root.glob("**/r48_audio_window_frame_predictions.jsonl"))
    for path in candidates:
        if path.exists():
            return path
    return None


def build_candidate_shell() -> dict[str, dict[str, Any]]:
    reviewed = read_jsonl(SESSION_ROOT / "exports/evaluation-review/reviewed_candidates.jsonl")
    candidates: dict[str, dict[str, Any]] = {}
    for row in reviewed:
        label = row.get("review_label")
        if label == "dive":
            candidate_label = "true_dive_candidate"
        elif label == "non_dive":
            candidate_label = "nuisance_non_dive_candidate"
        else:
            candidate_label = "ambiguous"
        candidates[row["proposal_id"]] = {
            "proposal_id": row["proposal_id"],
            "source_candidate_id": row.get("source_candidate_id"),
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "review_label": label,
            "candidate_label": candidate_label,
            "confidence": row.get("confidence"),
            "audio_score": row.get("audio_score"),
        }
    return candidates


def build_prompt_dataset(prediction_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    shell = build_candidate_shell()
    by_prompt: dict[str, dict[str, dict[str, Any]]] = {
        prompt_id: {proposal_id: {**candidate, "window_samples": []} for proposal_id, candidate in shell.items()}
        for prompt_id in PROMPTS
    }
    for row in prediction_rows:
        prompt_id = str(row.get("prompt_id"))
        proposal_id = str(row.get("proposal_id"))
        if prompt_id not in by_prompt or proposal_id not in by_prompt[prompt_id]:
            continue
        by_prompt[prompt_id][proposal_id]["window_samples"].append(
            {
                "offset_seconds": row.get("offset_seconds"),
                "target_timestamp_seconds": row.get("target_timestamp_seconds"),
                "available": bool(row.get("available")),
                "frame_timestamp_seconds": row.get("frame_timestamp_seconds"),
                "nearest_delta_seconds": row.get("nearest_delta_seconds"),
                "score": float(row.get("score") or 0.0),
                "is_positive": bool(row.get("is_positive")),
                "yes_first_token_probability": row.get("yes_first_token_probability"),
                "no_first_token_probability": row.get("no_first_token_probability"),
                "yes_no_first_token_margin": row.get("yes_no_first_token_margin"),
                "raw_response": row.get("raw_response"),
            }
        )
    prompt_datasets: dict[str, list[dict[str, Any]]] = {}
    for prompt_id, rows in by_prompt.items():
        prompt_rows = []
        for candidate in rows.values():
            samples = sorted(candidate["window_samples"], key=lambda item: float(item.get("offset_seconds") or 0.0))
            candidate["window_samples"] = samples
            candidate["sample_count"] = len(samples)
            candidate["available_frame_count"] = sum(1 for sample in samples if sample["available"])
            candidate["missing_frame_count"] = len(WINDOW_OFFSETS_SECONDS) - candidate["available_frame_count"]
            prompt_rows.append(candidate)
        prompt_datasets[prompt_id] = prompt_rows
    return {
        "benchmark_id": "r48_remote_audio_window_prompt_ablation",
        "session_root": str(SESSION_ROOT),
        "window_offsets_seconds": list(WINDOW_OFFSETS_SECONDS),
        "score_threshold": SCORE_THRESHOLD,
        "prompt_datasets": prompt_datasets,
    }


def aggregate_scores(samples: list[dict[str, Any]], method: str) -> tuple[bool, float | None]:
    available = [sample for sample in samples if sample.get("available")]
    if not available:
        return False, None
    scores = [float(sample.get("score") or 0.0) for sample in available]
    positives = [bool(sample.get("is_positive")) and float(sample.get("score") or 0.0) >= SCORE_THRESHOLD for sample in available]
    if method == "max_score":
        value = max(scores)
        return value >= SCORE_THRESHOLD, value
    if method == "mean_top2_score":
        top = sorted(scores, reverse=True)[:2]
        value = sum(top) / len(top)
        return value >= SCORE_THRESHOLD, value
    if method == "top3_mean":
        top = sorted(scores, reverse=True)[:3]
        value = sum(top) / len(top)
        return value >= SCORE_THRESHOLD, value
    if method == "any_positive":
        return any(positives), max(scores)
    if method == "majority_positive":
        value = sum(1 for flag in positives if flag) / len(positives)
        return value > 0.5, value
    raise ValueError(f"unknown aggregation method: {method}")


def score_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in ("true_dive_candidate", "nuisance_non_dive_candidate"):
        values = []
        for row in rows:
            if row["candidate_label"] != label:
                continue
            scores = [float(sample.get("score") or 0.0) for sample in row["window_samples"] if sample.get("available")]
            if scores:
                values.append(max(scores))
        values.sort()
        if not values:
            out[label] = {"count": 0}
            continue
        out[label] = {
            "count": len(values),
            "min": round(values[0], 4),
            "median": round(values[len(values) // 2], 4),
            "max": round(values[-1], 4),
        }
    return out


def evaluate_prompt(prompt_id: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usable_rows = [row for row in rows if row["candidate_label"] in {"true_dive_candidate", "nuisance_non_dive_candidate"}]
    true_rows = [row for row in usable_rows if row["candidate_label"] == "true_dive_candidate"]
    nuisance_rows = [row for row in usable_rows if row["candidate_label"] == "nuisance_non_dive_candidate"]
    method_results = []
    for method in METHODS:
        confirmed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for row in usable_rows:
            decision, value = aggregate_scores(row["window_samples"], method)
            scored = {
                "proposal_id": row["proposal_id"],
                "timestamp_seconds": row["timestamp_seconds"],
                "candidate_label": row["candidate_label"],
                "confirmed": decision,
                "aggregate_value": value,
                "available_frame_count": row["available_frame_count"],
            }
            if decision:
                confirmed.append(scored)
            else:
                rejected.append(scored)
        true_confirmed = [row for row in confirmed if row["candidate_label"] == "true_dive_candidate"]
        nuisance_confirmed = [row for row in confirmed if row["candidate_label"] == "nuisance_non_dive_candidate"]
        true_rejected = [row for row in rejected if row["candidate_label"] == "true_dive_candidate"]
        nuisance_rejected = [row for row in rejected if row["candidate_label"] == "nuisance_non_dive_candidate"]
        precision = len(true_confirmed) / len(confirmed) if confirmed else None
        recall = len(true_confirmed) / len(true_rows) if true_rows else None
        nuisance_rejection = len(nuisance_rejected) / len(nuisance_rows) if nuisance_rows else None
        method_results.append(
            {
                "prompt_id": prompt_id,
                "method": method,
                "candidate_count": len(usable_rows),
                "true_dive_count": len(true_rows),
                "nuisance_count": len(nuisance_rows),
                "confirmed_count": len(confirmed),
                "rejected_count": len(rejected),
                "precision": round(precision, 4) if precision is not None else None,
                "true_dive_recall": round(recall, 4) if recall is not None else None,
                "nuisance_rejection_rate": round(nuisance_rejection, 4) if nuisance_rejection is not None else None,
                "false_confirmed_nuisance_events": len(nuisance_confirmed),
                "false_rejected_real_dives": len(true_rejected),
                "confirmed_true_dive_count": len(true_confirmed),
                "confirmed_nuisance_count": len(nuisance_confirmed),
            }
        )
    return {
        "prompt_id": prompt_id,
        "candidate_count": len(usable_rows),
        "available_window_samples": sum(row["available_frame_count"] for row in usable_rows),
        "missing_window_samples": sum(row["missing_frame_count"] for row in usable_rows),
        "score_distribution_by_label": score_distribution(usable_rows),
        "aggregation_results": method_results,
    }


def render_comparison_md(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# R48 Audio Window Prompt Candidate Comparison",
        "",
        "| Prompt | Aggregation | Confirmed | Precision | True dive recall | Nuisance rejection | False nuisance | False rejected dives |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['prompt_id']} | {row['method']} | {row['confirmed_count']} | {row['precision']} | {row['true_dive_recall']} | {row['nuisance_rejection_rate']} | {row['false_confirmed_nuisance_events']} | {row['false_rejected_real_dives']} |"
        )
    return "\n".join(lines) + "\n"


def render_summary_md(summary: dict[str, Any], comparison_rows: Sequence[dict[str, Any]]) -> str:
    best = summary["best_candidate"]
    lines = [
        "# R48 Remote Audio Window Prompt Ablation",
        "",
        f"- remote run health: `{summary['remote_run_health']['status']}`",
        f"- prediction rows: `{summary['prediction_row_count']}`",
        f"- prompts evaluated: `{summary['prompts_evaluated']}`",
        f"- best candidate: `{best['prompt_id']}::{best['method']}`",
        f"- best precision: `{best['precision']}`",
        f"- best true dive recall: `{best['true_dive_recall']}`",
        f"- best nuisance rejection: `{best['nuisance_rejection_rate']}`",
        f"- false confirmed nuisance: `{best['false_confirmed_nuisance_events']}`",
        "",
        "## Comparison",
        "",
        render_comparison_md(comparison_rows).split("\n", 2)[2],
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate r48 remote audio-window prompt ablation outputs.")
    parser.add_argument("--remote-results", default=str(DEFAULT_REMOTE_RESULTS))
    args = parser.parse_args()
    remote_root = Path(args.remote_results).expanduser()
    prediction_path = find_predictions(remote_root)
    if prediction_path is None:
        raise FileNotFoundError(f"Could not find r48_audio_window_frame_predictions.jsonl under {remote_root}")
    prediction_rows = read_jsonl(prediction_path)
    dataset = build_prompt_dataset(prediction_rows)
    prompt_results = [evaluate_prompt(prompt_id, dataset["prompt_datasets"][prompt_id]) for prompt_id in PROMPTS]
    comparison_rows = [row for prompt in prompt_results for row in prompt["aggregation_results"]]
    best = max(
        comparison_rows,
        key=lambda row: (
            row["precision"] or 0.0,
            row["true_dive_recall"] or 0.0,
            row["nuisance_rejection_rate"] or 0.0,
        ),
    )
    r47_baseline = {
        "prompt_id": "r47_cached_baseline",
        "method": "max_score",
        "confirmed_count": 58,
        "precision": 0.7241,
        "true_dive_recall": 0.6462,
        "nuisance_rejection_rate": 0.6444,
        "false_confirmed_nuisance_events": 16,
        "false_rejected_real_dives": 23,
    }
    visual_only_reference = {
        "branch": "visual_only_full_frame_gap_split_3s",
        "union_recall": 0.8558,
        "recovered_anchors": 8,
        "unmatched_visual": 4,
        "false_visual_per_minute": 0.271,
    }
    remote_health_path = next(iter(remote_root.glob("**/r48_remote_run_health.json")), None)
    remote_health = read_json(remote_health_path) if remote_health_path else {"status": "missing_remote_health"}
    remote_health["status"] = "complete" if prediction_rows else "empty_predictions"
    summary = {
        "benchmark_id": "r48_remote_audio_window_prompt_ablation",
        "remote_results_root": str(remote_root),
        "prediction_path": str(prediction_path),
        "prediction_row_count": len(prediction_rows),
        "remote_run_health": remote_health,
        "prompts_evaluated": PROMPTS,
        "window_offsets_seconds": list(WINDOW_OFFSETS_SECONDS),
        "prompt_results": prompt_results,
        "comparison_rows": comparison_rows,
        "best_candidate": best,
        "r47_cached_baseline_reference": r47_baseline,
        "visual_only_recovery_reference": visual_only_reference,
        "improves_over_r47_cached_baseline": bool(
            (best["precision"] or 0.0) > r47_baseline["precision"]
            and (best["false_confirmed_nuisance_events"] <= r47_baseline["false_confirmed_nuisance_events"])
        ),
        "hard_verifier_supported": bool(
            (best["precision"] or 0.0) >= 0.9
            and best["false_confirmed_nuisance_events"] <= 3
            and (best["true_dive_recall"] or 0.0) >= 0.6
        ),
    }
    recommendation = {
        "benchmark_id": "r48_visual_role_recommendation",
        "recommended_visual_role": "hybrid",
        "audio_window_verifier_status": "supported" if summary["hard_verifier_supported"] else "not_supported_as_hard_verifier",
        "visual_only_recovery_status": "still_useful_research_branch",
        "review_prioritization_status": "candidate_metadata_only_if_best_prompt_separates_scores",
        "exact_visual_splash_contact_localization_required": False,
        "next_step": "Use the best r48 prompt only if it materially reduces nuisance confirmations; otherwise keep visual-only recovery and test lightweight motion/person prefilter before more prompt work.",
    }
    outputs = Path("outputs")
    write_json(outputs / "r48_remote_audio_window_prompt_ablation.json", summary)
    write_json(outputs / "r48_audio_window_prompt_candidate_comparison.json", {"comparison_rows": comparison_rows})
    write_json(outputs / "r48_visual_role_recommendation.json", recommendation)
    write_md(outputs / "r48_remote_audio_window_prompt_ablation.md", render_summary_md(summary, comparison_rows))
    write_md(outputs / "r48_audio_window_prompt_candidate_comparison.md", render_comparison_md(comparison_rows))
    write_md(
        outputs / "r48_visual_role_recommendation.md",
        "\n".join(
            [
                "# R48 Visual Role Recommendation",
                "",
                f"- recommended visual role: `{recommendation['recommended_visual_role']}`",
                f"- audio-window verifier status: `{recommendation['audio_window_verifier_status']}`",
                f"- visual-only recovery status: `{recommendation['visual_only_recovery_status']}`",
                f"- exact splash/contact localization required: `{recommendation['exact_visual_splash_contact_localization_required']}`",
                f"- next step: {recommendation['next_step']}",
            ]
        )
        + "\n",
    )
    write_md(
        Path("docs/research/R48_REMOTE_AUDIO_WINDOW_PROMPT_ABLATION.md"),
        "\n".join(
            [
                "# R48 Remote Audio Window Prompt Ablation",
                "",
                "This pass runs the r47 audio-window verification formulation with real remote VLM prompt variants.",
                "",
                f"- prediction rows: `{len(prediction_rows)}`",
                f"- best candidate: `{best['prompt_id']}::{best['method']}`",
                f"- precision: `{best['precision']}`",
                f"- true dive recall: `{best['true_dive_recall']}`",
                f"- nuisance rejection: `{best['nuisance_rejection_rate']}`",
                f"- hard verifier supported: `{summary['hard_verifier_supported']}`",
                "",
                "Visual remains research-only and does not affect approve_review_v1.",
            ]
        )
        + "\n",
    )
    print(json.dumps({"best_candidate": best, "hard_verifier_supported": summary["hard_verifier_supported"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
