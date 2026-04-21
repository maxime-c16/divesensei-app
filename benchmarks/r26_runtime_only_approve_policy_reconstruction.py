from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]

OUT_AUDIT_JSON = ROOT / "outputs/r26_runtime_score_plumbing_audit.json"
OUT_AUDIT_MD = ROOT / "outputs/r26_runtime_score_plumbing_audit.md"
OUT_CANDIDATES_JSON = ROOT / "outputs/r26_runtime_only_approve_candidates.json"
OUT_CANDIDATES_MD = ROOT / "outputs/r26_runtime_only_approve_candidates.md"
OUT_BENCHMARK_JSON = ROOT / "outputs/r26_runtime_only_policy_benchmark.json"
OUT_BENCHMARK_MD = ROOT / "outputs/r26_runtime_only_policy_benchmark.md"
OUT_DOC = ROOT / "docs/research/R26_RUNTIME_ONLY_APPROVE_POLICY_RECONSTRUCTION.md"

V1_R9_MIN = 0.92158
R24_V2_R9_FLOOR = 0.84
R24_V2_VISUAL_MIN = 0.55

# Bounded runtime-only nuisance-risk proxy thresholds.
FLAT_GATE_LO = 0.4819070100784302
FLAT_GATE_HI = 0.5605325698852539
POST_FLUX_GATE = 2.988285779953003
HF_VOICE_PROXY_MAX = 0.49950724840164185


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def score_value(row: dict[str, Any]) -> float | None:
    governed = row.get("governed_r9_score")
    if governed is not None:
        return float(governed)
    fallback = row.get("audio_model_probability")
    if fallback is None:
        return None
    return float(fallback)


def collect_runtime_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_dir in sorted((ROOT / "outputs").glob("evaluation_*")):
        manifest_path = session_dir / "ui_session_manifest.json"
        review_path = session_dir / "evaluation_review.json"
        if not manifest_path.exists() or not review_path.exists():
            continue
        manifest = load_json(manifest_path)
        review = load_json(review_path)
        decisions = {
            str(item.get("detectionId")): item
            for item in review.get("decisions", [])
            if item.get("label") in {"dive", "non_dive"}
        }
        for detection in manifest.get("detections", []):
            detection_id = str(detection.get("id"))
            decision = decisions.get(detection_id)
            if decision is None:
                continue
            scores = dict(detection.get("scores", {}) or {})
            features = dict(detection.get("features", {}) or {})
            rows.append(
                {
                    "source_session_id": session_dir.name,
                    "detection_id": detection_id,
                    "label": "platform_dive" if decision.get("label") == "dive" else "noise_or_other",
                    "reviewed_subtype": decision.get("subtype"),
                    "audio_score": float(scores.get("audio", 0.0) or 0.0),
                    "combined_score": float(scores.get("combined", 0.0) or 0.0),
                    "governed_r9_score": scores.get("governed_r9_score"),
                    "audio_model_probability": scores.get("audio_model_probability"),
                    "audio_clip_probability": scores.get("audio_clip_probability"),
                    "visual_late_fusion_logreg_c0.5": features.get("visual_late_fusion_logreg_c0.5"),
                    "spectral_flatness": features.get("spectral_flatness"),
                    "post_flux_ratio": features.get("post_flux_ratio"),
                    "hf_ratio": features.get("hf_ratio"),
                }
            )
    return rows


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_session_id": row["source_session_id"],
        "detection_id": row["detection_id"],
        "label": row["label"],
        "reviewed_subtype": row.get("reviewed_subtype"),
        "governed_r9_score": row.get("governed_r9_score"),
        "audio_model_probability": row.get("audio_model_probability"),
        "visual_late_fusion_logreg_c0.5": row.get("visual_late_fusion_logreg_c0.5"),
        "combined_score": row.get("combined_score"),
        "spectral_flatness": row.get("spectral_flatness"),
        "post_flux_ratio": row.get("post_flux_ratio"),
    }


def summarize(rows: list[dict[str, Any]], approve: Callable[[dict[str, Any]], bool], description: str) -> dict[str, Any]:
    approved = [row for row in rows if approve(row)]
    dangerous = [row for row in approved if row.get("label") != "platform_dive"]
    precision = safe_div(sum(1 for row in approved if row.get("label") == "platform_dive"), len(approved))
    return {
        "description": description,
        "row_count": len(rows),
        "approve_count": len(approved),
        "approve_coverage": safe_div(len(approved), len(rows)) or 0.0,
        "approve_precision": precision,
        "dangerous_auto_approves": len(dangerous),
        "approved_label_counts": dict(sorted(Counter(str(row.get("label")) for row in approved).items())),
        "source_approve_counts": dict(sorted(Counter(str(row.get("source_session_id")) for row in approved).items())),
        "dangerous_rows": [compact(row) for row in dangerous[:25]],
    }


def main() -> None:
    rows = collect_runtime_rows()
    if not rows:
        raise RuntimeError("No reviewed runtime rows found in outputs/evaluation_*/.")

    sampled_manifests = sorted(
        path for path in (ROOT / "outputs").glob("evaluation_*/ui_session_manifest.json") if path.exists()
    )
    manifest_availability = []
    for path in sampled_manifests:
        data = load_json(path)
        detections = data.get("detections", [])
        manifest_availability.append(
            {
                "session_id": path.parent.name,
                "row_count": len(detections),
                "governed_r9_score_present_count": sum(
                    1 for row in detections if (row.get("scores") or {}).get("governed_r9_score") not in (None, "")
                ),
                "audio_model_probability_nonzero_count": sum(
                    1 for row in detections if float((row.get("scores") or {}).get("audio_model_probability") or 0.0) > 0.0
                ),
                "visual_late_fusion_present_count": sum(
                    1 for row in detections if (row.get("features") or {}).get("visual_late_fusion_logreg_c0.5") not in (None, "")
                ),
            }
        )

    availability_rollup = {
        "reviewed_runtime_row_count": len(rows),
        "reviewed_label_counts": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "governed_r9_score_present_count": sum(1 for row in rows if row.get("governed_r9_score") not in (None, "")),
        "audio_model_probability_nonzero_count": sum(
            1 for row in rows if float(row.get("audio_model_probability") or 0.0) > 0.0
        ),
        "visual_late_fusion_present_count": sum(
            1 for row in rows if row.get("visual_late_fusion_logreg_c0.5") not in (None, "")
        ),
    }

    plumbing_audit = {
        "experiment_name": "r26_runtime_score_plumbing_audit_and_runtime_only_policy_reconstruction",
        "accepted_truth": {
            "active_default_policy": "approve_review_v1",
            "r24_rollout_candidate": False,
            "r25_classification": "mixed_runtime_and_reviewed_leakage",
        },
        "score_audit": {
            "governed_r9_score": {
                "currently_produced_where": [
                    "offline benchmark/scoring path in r20-r24 row banks (`r9_score` in benchmark artifacts)",
                    "detector proposal diagnostics details when available (`build_proposal_diagnostics`)",
                ],
                "why_missing_from_live_pre_review_manifests": (
                    "normal evaluate-session detections do not currently populate `details.governed_r9_score`; "
                    "manifest writer had no explicit governed score field until this pass."
                ),
                "code_path_to_change": [
                    "src/divesensei/metadata/ui_contract.py (scores field pass-through)",
                    "src/divesensei/workflows/evaluation_session_support.py (proposal diagnostics pass-through)",
                    "src/divesensei/workflows/export_evaluation_review.py (review export pass-through)",
                    "apps/desktop/src/lib/approve-review-policy.ts (prefer governed score with fallback)",
                ],
                "change_size": "low-risk plumbing for pass-through; larger pipeline work still required to compute/populate the score in detector runtime details.",
                "bounded_implementation_applied_now": True,
                "residual_blocker": (
                    "runtime detector path still does not emit governed r9 score values for candidates; "
                    "plumbing now forwards values if/when detector provides them."
                ),
                "can_emit_pre_review": True,
            },
            "visual_late_fusion_logreg_c0.5": {
                "currently_produced_where": [
                    "offline r23/r24 shadow backfill jobs (CLIP/morphology extraction + late fusion)",
                    "manifest/export paths only as pass-through when value exists",
                ],
                "why_missing_from_live_pre_review_manifests": (
                    "normal evaluate-session flow does not run the visual backfill computation before review, so this field is usually null."
                ),
                "code_path_to_change": [
                    "pre-review evaluate-session pipeline to compute visual verifier features for candidate windows",
                    "existing pass-through field kept in ui_contract/exports and preserved in this pass",
                ],
                "change_size": "larger pipeline change (video decode + feature extraction + late-fusion scoring).",
                "bounded_implementation_applied_now": True,
                "residual_blocker": "runtime visual feature/scoring generation is not wired into normal pre-review evaluate-session execution.",
                "can_emit_pre_review": True,
            },
        },
        "runtime_manifest_sampling": manifest_availability,
        "availability_rollup": availability_rollup,
    }

    def approve_review_v1(row: dict[str, Any]) -> bool:
        score = score_value(row)
        return score is not None and score >= V1_R9_MIN

    def runtime_audio_visual_strict_gate(row: dict[str, Any]) -> bool:
        score = score_value(row)
        visual = row.get("visual_late_fusion_logreg_c0.5")
        if score is None or visual is None:
            return False
        return float(score) >= R24_V2_R9_FLOOR and float(visual) >= R24_V2_VISUAL_MIN

    def runtime_nuisance_risk_flat_postflux_gate(row: dict[str, Any]) -> bool:
        flat = row.get("spectral_flatness")
        post_flux = row.get("post_flux_ratio")
        if flat is None or post_flux is None:
            return False
        return float(flat) >= FLAT_GATE_LO and float(post_flux) >= POST_FLUX_GATE

    def runtime_nuisance_risk_high_confidence_gate(row: dict[str, Any]) -> bool:
        flat = row.get("spectral_flatness")
        post_flux = row.get("post_flux_ratio")
        if flat is None or post_flux is None:
            return False
        return float(flat) >= FLAT_GATE_HI and float(post_flux) >= POST_FLUX_GATE

    def runtime_voice_whistle_proxy_suppressor(row: dict[str, Any]) -> bool:
        flat = row.get("spectral_flatness")
        post_flux = row.get("post_flux_ratio")
        hf_ratio = row.get("hf_ratio")
        if flat is None or post_flux is None or hf_ratio is None:
            return False
        return float(flat) >= FLAT_GATE_LO and float(post_flux) >= POST_FLUX_GATE and float(hf_ratio) <= HF_VOICE_PROXY_MAX

    families: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("approve_review_v1_runtime", "Current default score gate on governed_r9_score (fallback audio_model_probability).", approve_review_v1),
        (
            "runtime_audio_visual_strict_gate",
            "Strict runtime-only audio+visual gate; requires pre-review governed score and visual late-fusion score.",
            runtime_audio_visual_strict_gate,
        ),
        (
            "runtime_nuisance_risk_flat_postflux_gate",
            "Runtime nuisance-risk proxy: approve only high-flatness + strong post-flux rows (subtype-free).",
            runtime_nuisance_risk_flat_postflux_gate,
        ),
        (
            "runtime_nuisance_risk_high_confidence_gate",
            "Stricter nuisance-risk proxy using a higher flatness floor.",
            runtime_nuisance_risk_high_confidence_gate,
        ),
        (
            "runtime_voice_whistle_proxy_suppressor",
            "Runtime-only voice/whistle suppressor proxy layered onto nuisance-risk gate.",
            runtime_voice_whistle_proxy_suppressor,
        ),
    ]

    candidate_results = []
    for candidate_id, description, fn in families:
        summary = summarize(rows, fn, description)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "runtime_only": candidate_id != "approve_review_v1_runtime",
                "summary": summary,
                "safe_vs_v1": False,
                "recovered_r24_coverage_ratio": None,
            }
        )

    by_id = {row["candidate_id"]: row for row in candidate_results}
    v1_summary = by_id["approve_review_v1_runtime"]["summary"]

    r24 = load_json(ROOT / "outputs/r24_voice_whistle_hardened_approve_policy.json")
    r24_upper = {
        "reference_name": "r24_shadow_policy_upper_reference_only",
        "candidate": r24["best"]["candidate"],
        "approve_count": r24["best"]["overall"]["approve_count"],
        "approve_precision": r24["best"]["overall"]["approve_precision"],
        "approve_coverage": r24["best"]["overall"]["approve_coverage"],
        "dangerous_auto_approves": r24["best"]["overall"]["dangerous_approves"],
        "suspicious_added_approval_count": r24["best"]["overall"]["suspicious_added_approval_count"],
        "note": "Reference only; uses reviewed-data-only shadow path and is not runtime-valid as live policy input.",
    }

    for result in candidate_results:
        summary = result["summary"]
        safe = (
            result["candidate_id"] != "approve_review_v1_runtime"
            and summary["dangerous_auto_approves"] == 0
            and (summary["approve_precision"] or 0.0) >= 0.90
            and summary["approve_count"] > v1_summary["approve_count"]
        )
        result["safe_vs_v1"] = safe
        result["recovered_r24_coverage_ratio"] = safe_div(summary["approve_coverage"], r24_upper["approve_coverage"] or 0.0)

    runtime_candidates = [row for row in candidate_results if row["candidate_id"] != "approve_review_v1_runtime"]
    best_runtime = max(
        runtime_candidates,
        key=lambda row: (
            row["safe_vs_v1"],
            -row["summary"]["dangerous_auto_approves"],
            row["summary"]["approve_count"],
            row["summary"]["approve_precision"] or -1.0,
        ),
    )
    improves = bool(best_runtime["safe_vs_v1"])

    if improves:
        classification = "runtime_only_candidate_promising"
    elif availability_rollup["governed_r9_score_present_count"] == 0 and availability_rollup["visual_late_fusion_present_count"] == 0:
        classification = "runtime_only_candidate_blocked_by_missing_score_path"
    else:
        classification = "runtime_only_candidate_no_clear_gain"

    candidates_payload = {
        "experiment_name": "r26_runtime_only_approve_candidates",
        "candidate_family_focus": "runtime-safe subtype-veto replacements using only pre-review fields",
        "runtime_only_requirements": [
            "no reviewed subtype dependence",
            "no post-review metadata join",
            "all inputs must be available in ui_session_manifest rows before review",
        ],
        "candidates": candidate_results,
        "best_runtime_candidate": best_runtime,
        "safe_improvement_over_approve_review_v1": improves,
    }

    benchmark_payload = {
        "experiment_name": "r26_runtime_only_policy_benchmark",
        "question": "Can we recover some r24 gain without reviewed metadata?",
        "baseline": {
            "policy_id": "approve_review_v1",
            "summary": v1_summary,
        },
        "upper_reference_only": r24_upper,
        "runtime_candidate_comparison": runtime_candidates,
        "best_runtime_candidate": best_runtime,
        "safe_runtime_improvement_over_v1": improves,
        "classification": classification,
        "final_decisions": [
            "R26_RUNTIME_ONLY_RECONSTRUCTION_PROGRESS" if classification != "runtime_only_candidate_blocked_by_missing_score_path" else "R26_RUNTIME_ONLY_RECONSTRUCTION_BLOCKED",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }

    OUT_AUDIT_JSON.write_text(json.dumps(plumbing_audit, indent=2), encoding="utf-8")
    OUT_CANDIDATES_JSON.write_text(json.dumps(candidates_payload, indent=2), encoding="utf-8")
    OUT_BENCHMARK_JSON.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")

    sample_table = [
        "| session | rows | governed_r9 present | audio_model_probability>0 | visual present |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in manifest_availability:
        sample_table.append(
            f"| `{sample['session_id']}` | {sample['row_count']} | {sample['governed_r9_score_present_count']} | "
            f"{sample['audio_model_probability_nonzero_count']} | {sample['visual_late_fusion_present_count']} |"
        )
    candidate_table = [
        "| candidate | approve_count | precision | coverage | dangerous_auto_approves | safe_vs_v1 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in candidate_results:
        summary = row["summary"]
        candidate_table.append(
            f"| `{row['candidate_id']}` | {summary['approve_count']} | "
            f"{summary['approve_precision'] if summary['approve_precision'] is not None else 'n/a'} | "
            f"{summary['approve_coverage']:.4f} | {summary['dangerous_auto_approves']} | `{row['safe_vs_v1']}` |"
        )

    audit_md = [
        "# r26 Runtime Score Plumbing Audit",
        "",
        "- accepted default policy: `approve_review_v1`",
        "- r25 classification retained: `mixed_runtime_and_reviewed_leakage`",
        "",
        "## Runtime Availability Sampling",
        "",
        *sample_table,
        "",
        "## Result",
        "",
        "- `governed_r9_score` and `visual_late_fusion_logreg_c0.5` are now plumbed through manifest/export paths when present.",
        "- normal evaluate-session runtime still does not compute/populate these values, so they remain absent in current live pre-review rows.",
    ]
    candidates_md = [
        "# r26 Runtime-Only Approve Candidates",
        "",
        "Candidate family is bounded to runtime-only subtype-veto replacements.",
        "",
        *candidate_table,
        "",
        f"- best runtime candidate: `{best_runtime['candidate_id']}`",
        f"- safe improvement over v1: `{improves}`",
    ]
    benchmark_md = [
        "# r26 Runtime-Only Policy Benchmark",
        "",
        f"- baseline (`approve_review_v1`) coverage: `{v1_summary['approve_coverage']:.4f}`",
        f"- baseline dangerous auto-approves: `{v1_summary['dangerous_auto_approves']}`",
        f"- leaked r24 upper reference coverage: `{r24_upper['approve_coverage']}`",
        f"- leaked r24 upper reference dangerous: `{r24_upper['dangerous_auto_approves']}`",
        f"- best runtime-only candidate: `{best_runtime['candidate_id']}`",
        f"- best runtime-only coverage: `{best_runtime['summary']['approve_coverage']:.4f}`",
        f"- best runtime-only precision: `{best_runtime['summary']['approve_precision']}`",
        f"- best runtime-only dangerous auto-approves: `{best_runtime['summary']['dangerous_auto_approves']}`",
        f"- classification: `{classification}`",
        "",
        "No reviewed subtype metadata was used by runtime-only candidates.",
    ]
    doc_md = [
        "# R26 Runtime-Only Approve Policy Reconstruction",
        "",
        "This pass reconstructs a bounded approve-candidate family using only pre-review/runtime fields.",
        "",
        "## Accepted Current State",
        "",
        "- `approve_review_v1` remains the only valid live default.",
        "- `r24` remains a reviewed-data shadow upper reference, not a live rollout candidate.",
        "- r25 classification remains `mixed_runtime_and_reviewed_leakage`.",
        "",
        "## r26 Findings",
        "",
        f"- `governed_r9_score` pre-review emit path: `plumbed`, but runtime generation still missing in normal evaluate-session flow.",
        f"- `visual_late_fusion_logreg_c0.5` pre-review emit path: `plumbed`, but runtime generation still missing in normal evaluate-session flow.",
        f"- best runtime-only subtype-veto replacement: `{best_runtime['candidate_id']}`.",
        f"- safe runtime-only improvement over v1: `{improves}`.",
        f"- final classification: `{classification}`.",
        "",
        "## Final Decisions",
        "",
        f"- `{'R26_RUNTIME_ONLY_RECONSTRUCTION_PROGRESS' if classification != 'runtime_only_candidate_blocked_by_missing_score_path' else 'R26_RUNTIME_ONLY_RECONSTRUCTION_BLOCKED'}`",
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`",
    ]

    OUT_AUDIT_MD.write_text("\n".join(audit_md) + "\n", encoding="utf-8")
    OUT_CANDIDATES_MD.write_text("\n".join(candidates_md) + "\n", encoding="utf-8")
    OUT_BENCHMARK_MD.write_text("\n".join(benchmark_md) + "\n", encoding="utf-8")
    OUT_DOC.write_text("\n".join(doc_md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "wrote": [
                    str(OUT_AUDIT_JSON),
                    str(OUT_AUDIT_MD),
                    str(OUT_CANDIDATES_JSON),
                    str(OUT_CANDIDATES_MD),
                    str(OUT_BENCHMARK_JSON),
                    str(OUT_BENCHMARK_MD),
                    str(OUT_DOC),
                ],
                "classification": classification,
                "best_runtime_candidate": best_runtime["candidate_id"],
                "safe_runtime_improvement_over_v1": improves,
                "final_decisions": benchmark_payload["final_decisions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
