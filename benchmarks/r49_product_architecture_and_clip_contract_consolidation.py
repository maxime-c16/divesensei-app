from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SESSION_ROOT = Path("outputs/evaluation_insep_plateform_mixed_sound")
REVIEWED_PATH = SESSION_ROOT / "exports/evaluation-review/reviewed_candidates.jsonl"
FN_PATH = SESSION_ROOT / "exports/evaluation-review/false_negatives.jsonl"
R48_PATH = Path("outputs/r48_remote_audio_window_prompt_ablation.json")


CLIP_PRESETS = {
    "short": {"pre_seconds": 4.0, "post_seconds": 3.0},
    "medium": {"pre_seconds": 6.0, "post_seconds": 4.0},
    "long": {"pre_seconds": 8.0, "post_seconds": 5.0},
}


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


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def evaluate_clip_presets(reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dives = [row for row in reviewed if row.get("review_label") == "dive"]
    nuisance = [row for row in reviewed if row.get("review_label") == "non_dive"]
    out: list[dict[str, Any]] = []
    for name, config in CLIP_PRESETS.items():
        pre = float(config["pre_seconds"])
        post = float(config["post_seconds"])
        fully_covered = 0
        anchor_covered = 0
        overlap_ratios = []
        clip_lengths = []
        for row in dives:
            anchor = float(row["timestamp_seconds"])
            clip_start = max(0.0, anchor - pre)
            clip_end = anchor + post
            clip_lengths.append(clip_end - clip_start)
            review_start = float(row.get("review_window_start_seconds") or anchor)
            review_end = float(row.get("review_window_end_seconds") or anchor)
            if clip_start <= anchor <= clip_end:
                anchor_covered += 1
            if clip_start <= review_start and review_end <= clip_end:
                fully_covered += 1
            denom = max(0.001, review_end - review_start)
            overlap_ratios.append(overlap_seconds(clip_start, clip_end, review_start, review_end) / denom)
        total_clip_minutes = sum(clip_lengths) / 60.0
        out.append(
            {
                "preset": name,
                "pre_seconds": pre,
                "post_seconds": post,
                "nominal_clip_length_seconds": pre + post,
                "reviewed_audio_dive_candidates": len(dives),
                "reviewed_nuisance_candidates": len(nuisance),
                "anchor_coverage_count": anchor_covered,
                "anchor_coverage_rate": round(anchor_covered / len(dives), 4) if dives else None,
                "review_window_full_coverage_count": fully_covered,
                "review_window_full_coverage_rate": round(fully_covered / len(dives), 4) if dives else None,
                "mean_review_window_overlap_rate": round(sum(overlap_ratios) / len(overlap_ratios), 4) if overlap_ratios else None,
                "total_review_clip_minutes_for_audio_dive_candidates": round(total_clip_minutes, 3),
                "review_ui_false_candidate_count_if_applied_to_all_audio_candidates": len(nuisance),
                "review_ui_false_candidate_minutes_if_applied_to_all_audio_candidates": round(len(nuisance) * (pre + post) / 60.0, 3),
                "manual_signal_gap": "Review windows are 4s bounded review contexts, not exact body-action start/end labels.",
            }
        )
    return out


def product_task_taxonomy(r48: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task": "audio_candidate_detection",
            "input": "session video/audio",
            "output": "timestamped audio candidate anchors",
            "success_metric": "reviewed anchor recall, timing accuracy, manageable candidate count",
            "current_best_evidence": "audio remains the primary temporal anchor generator; hard echo/rebound sessions still motivate recovery support",
            "status": "product_ready_core_with_monitoring",
        },
        {
            "task": "safe_auto_approve_lane",
            "input": "governed r9 score on reviewed candidate rows",
            "output": "narrow auto-approved lane plus metadata",
            "success_metric": "zero dangerous approvals, high precision, monitored coverage",
            "current_best_evidence": "approve_review_v1 remains the only valid live/default policy",
            "status": "product_ready_narrow_lane",
        },
        {
            "task": "human_review_queue",
            "input": "all non-auto-approved candidates plus visual recovery proposals",
            "output": "coach/diver review decisions and corrected labels",
            "success_metric": "low missed-event burden, clear provenance, review throughput",
            "current_best_evidence": "app is a review workflow with a safe trusted fast lane, not full autonomous triage",
            "status": "product_ready_primary_workflow",
        },
        {
            "task": "visual_only_recovery_branch",
            "input": "session review proxy video, sampled full-frame VLM predictions",
            "output": "visual recovery proposals for audio gaps",
            "success_metric": "incremental anchor recovery vs false visual proposals/min",
            "current_best_evidence": "full-frame 1 FPS gap-split 3s recovered 8 anchors with 4 unmatched visual proposals and 0.271 false visual/min",
            "status": "research_only_useful",
        },
        {
            "task": "clip_extraction",
            "input": "audio anchor timestamp plus preset pre/post buffers",
            "output": "short/medium/long review clips",
            "success_metric": "review-window coverage, clip length, review burden, user-perceived clip quality",
            "current_best_evidence": "current reviewed windows can be covered from audio anchors without exact visual splash localization",
            "status": "ready_for_product_aligned_engineering",
        },
        {
            "task": "optional_visual_metadata_review_prioritization",
            "input": "visual-window prompt scores around audio anchors",
            "output": "weak evidence metadata, not hard gating",
            "success_metric": "separation useful for sorting without false rejection of real dives",
            "current_best_evidence": f"r48 best hard-verifier-like candidate precision={r48['best_candidate']['precision']} recall={r48['best_candidate']['true_dive_recall']} false rejected dives={r48['best_candidate']['false_rejected_real_dives']}",
            "status": "research_only_not_hard_gate",
        },
    ]


def render_architecture_md(contract: dict[str, Any]) -> str:
    lines = [
        "# R49 Product Architecture Contract",
        "",
        f"- product definition: {contract['product_definition']}",
        f"- primary anchor: `{contract['primary_temporal_anchor']}`",
        f"- exact visual splash/contact localization required: `{contract['exact_visual_splash_contact_localization_required']}`",
        "",
        "## Product Tasks",
        "",
        "| Task | Output | Success metric | Status |",
        "|---|---|---|---|",
    ]
    for task in contract["product_task_taxonomy"]:
        lines.append(f"| {task['task']} | {task['output']} | {task['success_metric']} | {task['status']} |")
    return "\n".join(lines) + "\n"


def render_clip_md(contract: dict[str, Any]) -> str:
    lines = [
        "# R49 Clip Extraction Contract",
        "",
        "| Preset | Window | Anchor coverage | Review-window full coverage | Mean review-window overlap | Dive clip minutes | Nuisance minutes if all reviewed audio candidates clipped |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in contract["preset_evaluation"]:
        lines.append(
            f"| {row['preset']} | -{row['pre_seconds']}s/+{row['post_seconds']}s | {row['anchor_coverage_rate']} | {row['review_window_full_coverage_rate']} | {row['mean_review_window_overlap_rate']} | {row['total_review_clip_minutes_for_audio_dive_candidates']} | {row['review_ui_false_candidate_minutes_if_applied_to_all_audio_candidates']} |"
        )
    lines.extend(
        [
            "",
            f"- recommended default preset: `{contract['recommended_default_preset']}`",
            f"- recommended alternate presets: `{contract['recommended_alternate_presets']}`",
            f"- manual signal gap: {contract['manual_signal_gap']}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_visual_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R49 Visual Branch Reclassification",
        "",
        "| Branch | Classification | Evidence | Product action |",
        "|---|---|---|---|",
    ]
    for row in payload["branches"]:
        lines.append(f"| {row['branch']} | {row['classification']} | {row['evidence']} | {row['product_action']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    reviewed = read_jsonl(REVIEWED_PATH)
    false_negatives = read_jsonl(FN_PATH)
    r48 = read_json(R48_PATH)
    clip_eval = evaluate_clip_presets(reviewed)
    architecture = {
        "benchmark_id": "r49_product_architecture_contract",
        "product_definition": "DiveSensei is an audio-anchored dive clip extraction and review system, with visual recovery support.",
        "primary_temporal_anchor": "audio_candidate_anchor",
        "active_default_policy": "approve_review_v1",
        "approve_policy_change": "none",
        "exact_visual_splash_contact_localization_required": False,
        "product_task_taxonomy": product_task_taxonomy(r48),
        "recommended_product_workflow": {
            "app_shows": ["Auto-approved narrow lane", "Needs review primary queue", "Visual recovery suggestions as separate review-support lane"],
            "auto_approved": "Only approve_review_v1 rows; no visual branch changes approval.",
            "sent_to_review": "All non-v1-approved audio candidates plus visual-only recovery proposals.",
            "visual_recovery_presentation": "Show as visually suggested clips with explicit visual_vlm provenance and no auto-approval implication.",
            "coach_diver_experience": "Review audio-anchored clips first; inspect visual recovery suggestions for possible missed dives; confirm labels manually.",
        },
    }
    clip_contract = {
        "benchmark_id": "r49_clip_extraction_contract",
        "session_root": str(SESSION_ROOT),
        "reviewed_audio_candidates": len(reviewed),
        "reviewed_audio_dive_candidates": sum(1 for row in reviewed if row.get("review_label") == "dive"),
        "reviewed_nuisance_candidates": sum(1 for row in reviewed if row.get("review_label") == "non_dive"),
        "known_false_negative_rows": len(false_negatives),
        "preset_evaluation": clip_eval,
        "recommended_default_preset": "medium",
        "recommended_alternate_presets": {
            "short": "fast review / compact clip list",
            "long": "coach export or cases where takeoff/setup context matters",
        },
        "false_candidates_in_review_ui": "acceptable if provenance stays explicit and rows are not silently auto-approved or auto-excluded",
        "exact_visual_splash_contact_localization_needed": False,
        "manual_signal_gap": "Current labels provide anchors and 4s review windows, not exact takeoff/entry start/end. A clip-quality review UI should capture whether the preset contains enough setup, takeoff, entry, and aftermath.",
    }
    visual = {
        "benchmark_id": "r49_visual_branch_reclassification",
        "branches": [
            {
                "branch": "visual_only_recovery_branch",
                "classification": "research_only_useful_recovery",
                "evidence": "full-frame 1 FPS gap split 3s: union recall 0.8558, recovered anchors 8, unmatched visual 4, false visual/min 0.271",
                "product_action": "retain as separate review-support recovery lane; do not auto-approve",
            },
            {
                "branch": "audio_window_visual_metadata",
                "classification": "metadata_only_research",
                "evidence": "r48 baseline sees many dives but confirms nuisance; window prompt rejects clutter but recall collapses",
                "product_action": "may be stored/displayed as weak evidence or sorting metadata only",
            },
            {
                "branch": "audio_window_hard_verifier",
                "classification": "not_supported",
                "evidence": "best r48 candidate precision 0.8000, recall 0.1846, false rejected real dives 53",
                "product_action": "must not reject or confirm audio candidates",
            },
        ],
        "no_visual_branch_should_reject_audio_candidates": True,
        "recommended_visual_role": "visual recovery support plus optional weak metadata",
    }
    doc = "\n".join(
        [
            "# R49 Product Architecture And Clip Contract",
            "",
            "DiveSensei should be evaluated as an audio-anchored dive clip extraction and review system, with visual recovery support.",
            "",
            "## Decisions",
            "",
            "- Audio anchors are primary.",
            "- Clip extraction should use audio anchor plus configurable buffers.",
            "- Exact visual splash/contact localization is not required.",
            "- Visual-only recovery remains useful as a research/support branch.",
            "- Audio-window hard verification is rejected after r48.",
            "",
            "## Best Next Engineering Step",
            "",
            "Implement clip extraction presets around audio anchors and add a clip-quality review signal. This is more product-aligned than more prompt hunting.",
        ]
    ) + "\n"

    outputs = Path("outputs")
    write_json(outputs / "r49_product_architecture_contract.json", architecture)
    write_json(outputs / "r49_clip_extraction_contract.json", clip_contract)
    write_json(outputs / "r49_visual_branch_reclassification.json", visual)
    write_md(outputs / "r49_product_architecture_contract.md", render_architecture_md(architecture))
    write_md(outputs / "r49_clip_extraction_contract.md", render_clip_md(clip_contract))
    write_md(outputs / "r49_visual_branch_reclassification.md", render_visual_md(visual))
    write_md(Path("docs/research/R49_PRODUCT_ARCHITECTURE_AND_CLIP_CONTRACT.md"), doc)
    print(json.dumps({"clip_presets": clip_eval, "visual_role": visual["recommended_visual_role"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
