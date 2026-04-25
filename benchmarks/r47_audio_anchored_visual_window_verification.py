from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


SESSION_ROOT = Path("outputs/evaluation_insep_plateform_mixed_sound")
FULL_FRAME_ROOT = Path("/Users/mcauchy/Downloads/r42_visual_full_frame_control/audio_gated_full_frame_1p0fps")
WINDOW_OFFSETS_SECONDS = (-2.0, -1.0, 0.0, 1.0, 2.0)
FRAME_MATCH_TOLERANCE_SECONDS = 0.55
SCORE_THRESHOLD = 0.845


PROMPT_VARIANTS = [
    {
        "prompt_id": "baseline_frame_prompt",
        "prompt": "answer en is this frame part of a diving attempt into water?\n",
        "source": "cached_r42_full_frame_predictions",
        "status": "executed",
    },
    {
        "prompt_id": "window_dive_evidence_prompt",
        "prompt": "answer en does this frame show visual evidence of a real dive attempt into the water?\n",
        "source": "requires_new_vlm_inference",
        "status": "not_executed_in_r47_cached_pass",
    },
    {
        "prompt_id": "anti_clutter_prompt",
        "prompt": "answer en does this frame show a real dive attempt, not poolside activity, standing, walking, talking, or unrelated splash?\n",
        "source": "requires_new_vlm_inference",
        "status": "not_executed_in_r47_cached_pass",
    },
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


def nearest_frame(frame_rows: list[dict[str, Any]], timestamp: float) -> tuple[dict[str, Any] | None, float | None]:
    if not frame_rows:
        return None, None
    row = min(frame_rows, key=lambda item: abs(float(item["timestamp_seconds"]) - timestamp))
    delta = float(row["timestamp_seconds"]) - timestamp
    if abs(delta) <= FRAME_MATCH_TOLERANCE_SECONDS:
        return row, delta
    return None, delta


def build_dataset() -> dict[str, Any]:
    reviewed = read_jsonl(SESSION_ROOT / "exports/evaluation-review/reviewed_candidates.jsonl")
    frame_rows = [
        row
        for row in read_jsonl(FULL_FRAME_ROOT / "visual_frame_predictions.jsonl")
        if row.get("prompt_id") == "diving_attempt" and row.get("decision_rule") == "yes_no_first_token_margin"
    ]
    frame_rows.sort(key=lambda row: float(row["timestamp_seconds"]))

    candidates: list[dict[str, Any]] = []
    for row in reviewed:
        label = row.get("review_label")
        if label == "dive":
            candidate_label = "true_dive_candidate"
        elif label == "non_dive":
            candidate_label = "nuisance_non_dive_candidate"
        else:
            candidate_label = "ambiguous"
        timestamp = float(row["timestamp_seconds"])
        samples: list[dict[str, Any]] = []
        for offset in WINDOW_OFFSETS_SECONDS:
            target = timestamp + offset
            frame, delta = nearest_frame(frame_rows, target)
            sample: dict[str, Any] = {
                "offset_seconds": offset,
                "target_timestamp_seconds": round(target, 3),
                "available": frame is not None,
                "nearest_delta_seconds": round(delta, 3) if delta is not None else None,
            }
            if frame is not None:
                sample.update(
                    {
                        "frame_timestamp_seconds": float(frame["timestamp_seconds"]),
                        "score": float(frame.get("score") or 0.0),
                        "is_positive": bool(frame.get("is_positive")),
                        "yes_first_token_probability": frame.get("yes_first_token_probability"),
                        "no_first_token_probability": frame.get("no_first_token_probability"),
                        "yes_no_first_token_margin": frame.get("yes_no_first_token_margin"),
                        "raw_response": frame.get("raw_response"),
                    }
                )
            samples.append(sample)
        candidates.append(
            {
                "proposal_id": row["proposal_id"],
                "source_candidate_id": row.get("source_candidate_id"),
                "timestamp_seconds": timestamp,
                "review_label": label,
                "candidate_label": candidate_label,
                "confidence": row.get("confidence"),
                "audio_score": row.get("audio_score"),
                "sample_count": len(samples),
                "available_frame_count": sum(1 for sample in samples if sample["available"]),
                "missing_frame_count": sum(1 for sample in samples if not sample["available"]),
                "window_samples": samples,
            }
        )

    counts: dict[str, int] = {}
    for row in candidates:
        counts[row["candidate_label"]] = counts.get(row["candidate_label"], 0) + 1
    return {
        "benchmark_id": "r47_audio_anchored_visual_window_verification",
        "session_root": str(SESSION_ROOT),
        "visual_frame_source": str(FULL_FRAME_ROOT / "visual_frame_predictions.jsonl"),
        "window_offsets_seconds": list(WINDOW_OFFSETS_SECONDS),
        "frame_match_tolerance_seconds": FRAME_MATCH_TOLERANCE_SECONDS,
        "prompt_variants": PROMPT_VARIANTS,
        "candidate_count": len(candidates),
        "label_counts": counts,
        "total_window_samples": sum(row["sample_count"] for row in candidates),
        "available_window_samples": sum(row["available_frame_count"] for row in candidates),
        "missing_window_samples": sum(row["missing_frame_count"] for row in candidates),
        "candidates": candidates,
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
    if method == "any_positive":
        return any(positives), max(scores)
    if method == "majority_positive":
        value = sum(1 for flag in positives if flag) / len(positives)
        return value > 0.5, value
    raise ValueError(f"unknown aggregation method: {method}")


def evaluate_method(dataset: dict[str, Any], method: str) -> dict[str, Any]:
    rows = [row for row in dataset["candidates"] if row["candidate_label"] in {"true_dive_candidate", "nuisance_non_dive_candidate"}]
    true_rows = [row for row in rows if row["candidate_label"] == "true_dive_candidate"]
    nuisance_rows = [row for row in rows if row["candidate_label"] == "nuisance_non_dive_candidate"]
    confirmed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        decision, value = aggregate_scores(row["window_samples"], method)
        scored = {
            "proposal_id": row["proposal_id"],
            "timestamp_seconds": row["timestamp_seconds"],
            "candidate_label": row["candidate_label"],
            "confirmed": decision,
            "aggregate_value": value,
            "available_frame_count": row["available_frame_count"],
        }
        scored_rows.append(scored)
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
    return {
        "method": method,
        "candidate_count": len(rows),
        "true_dive_count": len(true_rows),
        "nuisance_count": len(nuisance_rows),
        "confirmed_count": len(confirmed),
        "rejected_count": len(rejected),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "true_dive_confirm_rate": round(recall, 4) if recall is not None else None,
        "nuisance_rejection_rate": round(nuisance_rejection, 4) if nuisance_rejection is not None else None,
        "false_rejection_real_audio_dives": len(true_rejected),
        "false_confirmation_nuisance_audio_events": len(nuisance_confirmed),
        "confirmed_true_dive_count": len(true_confirmed),
        "confirmed_nuisance_count": len(nuisance_confirmed),
        "scored_rows": scored_rows,
    }


def render_dataset_md(dataset: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R47 Audio-Anchored Visual Window Dataset",
            "",
            f"- session: `{dataset['session_root']}`",
            f"- candidates: `{dataset['candidate_count']}`",
            f"- label counts: `{dataset['label_counts']}`",
            f"- window offsets seconds: `{dataset['window_offsets_seconds']}`",
            f"- available window samples: `{dataset['available_window_samples']}`",
            f"- missing window samples: `{dataset['missing_window_samples']}`",
            "- prompt status: baseline cached prompt executed; window evidence and anti-clutter prompts require a follow-up remote VLM run.",
        ]
    ) + "\n"


def render_verification_md(results: dict[str, Any]) -> str:
    lines = [
        "# R47 Audio-Anchored Visual Window Verification",
        "",
        "| Aggregation | Confirmed | Precision | Recall | Nuisance rejection | False reject dives | False confirm nuisance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["aggregation_results"]:
        lines.append(
            f"| {row['method']} | {row['confirmed_count']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['nuisance_rejection_rate']:.4f} | {row['false_rejection_real_audio_dives']} | {row['false_confirmation_nuisance_audio_events']} |"
        )
    lines.extend(
        [
            "",
            f"- best aggregation: `{results['best_aggregation']['method']}`",
            f"- supported: `{results['audio_anchored_visual_verification_supported']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    dataset = build_dataset()
    methods = ["max_score", "mean_top2_score", "any_positive", "majority_positive"]
    aggregation_results = [evaluate_method(dataset, method) for method in methods]
    best = max(
        aggregation_results,
        key=lambda row: (
            row["precision"] or 0.0,
            row["recall"] or 0.0,
            row["nuisance_rejection_rate"] or 0.0,
        ),
    )
    visual_only_reference = {
        "branch": "visual_only_full_frame_gap_split_3s",
        "union_recall": 0.8558,
        "recovered_anchors": 8,
        "unmatched_visual": 4,
        "false_visual_per_minute": 0.271,
        "role": "recovery_proposal_branch",
    }
    verification = {
        "benchmark_id": "r47_audio_anchored_visual_window_verification",
        "prompt_results": [
            {
                "prompt_id": "baseline_frame_prompt",
                "status": "executed_from_cached_r42_predictions",
                "aggregation_results": aggregation_results,
            },
            {
                "prompt_id": "window_dive_evidence_prompt",
                "status": "not_executed_requires_remote_vlm_prompt_run",
                "aggregation_results": [],
            },
            {
                "prompt_id": "anti_clutter_prompt",
                "status": "not_executed_requires_remote_vlm_prompt_run",
                "aggregation_results": [],
            },
        ],
        "aggregation_results": aggregation_results,
        "best_aggregation": best,
        "visual_only_reference": visual_only_reference,
        "audio_anchored_visual_verification_supported": False,
        "support_reason": "Baseline cached frame prompt confirms almost every audio candidate window, including nuisance candidates, so it is not yet a useful verifier.",
    }
    recommendation = {
        "benchmark_id": "r47_visual_product_architecture_recommendation",
        "recommended_architecture": "hybrid",
        "primary_visual_role": "recovery_branch_pending_window_prompt_validation",
        "audio_role": "primary_temporal_anchor_generator",
        "visual_only_branch": "keep_for_audio_gap_recovery_research",
        "audio_window_verification": "not_supported_by_cached_baseline_prompt_yet",
        "exact_visual_splash_contact_localization_required": False,
        "clip_extraction_recommendation": "derive clip start/end from audio anchor plus buffers; use visual evidence to confirm or recover candidate windows, not to locate exact splash contact.",
        "next_required_evidence": "Run the two window-compatible prompts remotely on the same audio-window frame set before promoting visual verification as the main visual path.",
    }

    outputs = Path("outputs")
    write_json(outputs / "r47_audio_anchored_visual_window_dataset.json", dataset)
    write_json(outputs / "r47_audio_anchored_visual_window_verification.json", verification)
    write_json(outputs / "r47_visual_product_architecture_recommendation.json", recommendation)
    write_md(outputs / "r47_audio_anchored_visual_window_dataset.md", render_dataset_md(dataset))
    write_md(outputs / "r47_audio_anchored_visual_window_verification.md", render_verification_md(verification))
    write_md(
        outputs / "r47_visual_product_architecture_recommendation.md",
        "\n".join(
            [
                "# R47 Visual Product Architecture Recommendation",
                "",
                "- recommended architecture: `hybrid`",
                "- audio remains the primary temporal anchor generator.",
                "- visual-only remains a research recovery branch for audio gaps.",
                "- audio-window verification is product-aligned, but not supported by the cached baseline prompt yet.",
                "- exact visual splash/contact localization is not necessary for useful dive review clips.",
                "- next evidence: run window-compatible and anti-clutter prompts remotely on the audio-window frame set.",
            ]
        )
        + "\n",
    )
    write_md(
        Path("docs/research/R47_AUDIO_ANCHORED_VISUAL_WINDOW_VERIFICATION.md"),
        "\n".join(
            [
                "# R47 Audio-Anchored Visual Window Verification",
                "",
                "This pass reframes the visual task around candidate-window verification rather than exact splash/contact localization.",
                "",
                f"- audio candidates: `{dataset['candidate_count']}`",
                f"- best cached-prompt aggregation: `{best['method']}`",
                f"- best cached-prompt precision: `{best['precision']}`",
                f"- best cached-prompt recall: `{best['recall']}`",
                f"- nuisance rejection: `{best['nuisance_rejection_rate']}`",
                "",
                "The cached baseline frame prompt is too permissive for verification. The product framing is correct, but the available cached prompt evidence does not yet support switching visual's primary role to verifier.",
            ]
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
