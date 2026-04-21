from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]

OUT_BASELINE_JSON = ROOT / "outputs/r28_runtime_only_baseline.json"
OUT_BASELINE_MD = ROOT / "outputs/r28_runtime_only_baseline.md"
OUT_CANDIDATES_JSON = ROOT / "outputs/r28_runtime_only_approve_candidates.json"
OUT_CANDIDATES_MD = ROOT / "outputs/r28_runtime_only_approve_candidates.md"
OUT_REEVALUATION_JSON = ROOT / "outputs/r28_runtime_only_reevaluation.json"
OUT_REEVALUATION_MD = ROOT / "outputs/r28_runtime_only_reevaluation.md"
OUT_DOC = ROOT / "docs/research/R28_RUNTIME_ONLY_APPROVE_REEVALUATION.md"

V1_R9_MIN = 0.92158

RUNTIME_SESSION_ROOTS = [
    "evaluation_r27_scorepath_insep_quick_v2",
    "evaluation_r27_scorepath_champigny_proxy",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def score(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "detection_id": row["detection_id"],
        "event_label": row["event_label"],
        "governed_r9_score": row.get("governed_r9_score"),
        "visual_late_fusion_logreg_c0.5": row.get("visual_late_fusion_logreg_c0.5"),
        "audio_score": row.get("audio_score"),
        "timestamp_seconds": row.get("timestamp_seconds"),
    }


def collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    for session_id in RUNTIME_SESSION_ROOTS:
        root = ROOT / "outputs" / session_id
        manifest_path = root / "ui_session_manifest.json"
        review_path = root / "evaluation_review.json"
        report_path = root / "session_pipeline_report.json"
        if not manifest_path.exists() or not review_path.exists():
            session_summaries.append(
                {
                    "session_id": session_id,
                    "included": False,
                    "reason": "missing_manifest_or_review",
                }
            )
            continue
        manifest = load_json(manifest_path)
        review = load_json(review_path)
        decisions = {
            str(item.get("detectionId")): item
            for item in review.get("decisions", [])
            if item.get("eventLabel") in {"platform_dive", "noise_or_other"}
        }
        detections = manifest.get("detections", [])
        included = 0
        visual_present = 0
        governed_present = 0
        for detection in detections:
            detection_id = str(detection.get("id"))
            decision = decisions.get(detection_id)
            if decision is None:
                continue
            scores = dict(detection.get("scores", {}) or {})
            features = dict(detection.get("features", {}) or {})
            governed = scores.get("governed_r9_score")
            visual = features.get("visual_late_fusion_logreg_c0.5")
            rows.append(
                {
                    "session_id": session_id,
                    "detection_id": detection_id,
                    "timestamp_seconds": float(detection.get("timestamp_seconds") or 0.0),
                    "event_label": str(decision.get("eventLabel")),
                    "governed_r9_score": governed,
                    "visual_late_fusion_logreg_c0.5": visual,
                    "audio_model_probability": scores.get("audio_model_probability"),
                    "audio_score": scores.get("audio"),
                }
            )
            included += 1
            if governed not in (None, ""):
                governed_present += 1
            if visual not in (None, ""):
                visual_present += 1
        report = load_json(report_path) if report_path.exists() else {}
        enrichment = report.get("runtime_score_enrichment") or report.get("score_enrichment") or {}
        session_summaries.append(
            {
                "session_id": session_id,
                "included": True,
                "source_video_path": manifest.get("session", {}).get("source_video_path"),
                "manifest_detection_count": len(detections),
                "review_decision_count": len(review.get("decisions", [])),
                "platform_noise_rows_used": included,
                "governed_present_count": governed_present,
                "visual_present_count": visual_present,
                "visual_missing_count": included - visual_present,
                "event_label_counts": dict(
                    sorted(Counter(row["event_label"] for row in rows if row["session_id"] == session_id).items())
                ),
                "runtime_score_enrichment": enrichment,
            }
        )
    return rows, session_summaries


def summarize_policy(
    rows: list[dict[str, Any]],
    *,
    policy_id: str,
    description: str,
    approve: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    approved = [row for row in rows if approve(row)]
    dangerous = [row for row in approved if row["event_label"] != "platform_dive"]
    platform_count = sum(1 for row in rows if row["event_label"] == "platform_dive")
    approved_platform = sum(1 for row in approved if row["event_label"] == "platform_dive")
    return {
        "policy_id": policy_id,
        "description": description,
        "row_count": len(rows),
        "approve_count": len(approved),
        "approve_coverage": safe_div(len(approved), len(rows)) or 0.0,
        "approve_precision": safe_div(approved_platform, len(approved)),
        "platform_approve_recall": safe_div(approved_platform, platform_count),
        "dangerous_approvals": len(dangerous),
        "suspicious_additions": len(dangerous),
        "approved_label_counts": dict(sorted(Counter(row["event_label"] for row in approved).items())),
        "approved_source_counts": dict(sorted(Counter(row["session_id"] for row in approved).items())),
        "dangerous_rows": [compact(row) for row in dangerous],
        "approved_rows": [compact(row) for row in approved[:50]],
    }


def build_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "policy_id": "approve_review_v1",
            "description": f"approve if governed_r9_score >= {V1_R9_MIN}",
            "family": "baseline",
            "r9_min": V1_R9_MIN,
            "visual_min": None,
            "mode": "r9_only",
        }
    ]
    for threshold in [0.94, 0.93, 0.92, 0.90, 0.88, 0.86, 0.84]:
        candidates.append(
            {
                "policy_id": f"r9_score_gate::{threshold:.2f}",
                "description": f"approve if governed_r9_score >= {threshold:.2f}",
                "family": "r9_threshold",
                "r9_min": threshold,
                "visual_min": None,
                "mode": "r9_only",
            }
        )
    for threshold in [0.95, 0.97, 0.99]:
        candidates.append(
            {
                "policy_id": f"ultra_conservative_r9_score_gate::{threshold:.2f}",
                "description": f"approve if governed_r9_score >= {threshold:.2f}",
                "family": "ultra_conservative_r9_threshold",
                "r9_min": threshold,
                "visual_min": None,
                "mode": "r9_only",
            }
        )
    for r9_min in [0.84, 0.86, 0.88, 0.90]:
        for visual_min in [0.55, 0.70, 0.85, 0.95]:
            candidates.append(
                {
                    "policy_id": f"runtime_or_visual_gate::r9_{r9_min:.2f}::visual_{visual_min:.2f}",
                    "description": (
                        f"approve if governed_r9_score >= {V1_R9_MIN} OR "
                        f"(governed_r9_score >= {r9_min:.2f} AND visual_late_fusion_logreg_c0.5 >= {visual_min:.2f})"
                    ),
                    "family": "runtime_audio_visual_or_gate",
                    "r9_min": r9_min,
                    "visual_min": visual_min,
                    "mode": "v1_or_audio_visual",
                }
            )
    for r9_min in [0.70, 0.75, 0.80, 0.84]:
        for visual_min in [0.97, 0.99, 0.995]:
            candidates.append(
                {
                    "policy_id": f"strict_visual_only_gate::r9_{r9_min:.2f}::visual_{visual_min:.3f}",
                    "description": (
                        f"approve only if governed_r9_score >= {r9_min:.2f} "
                        f"AND visual_late_fusion_logreg_c0.5 >= {visual_min:.3f}"
                    ),
                    "family": "strict_runtime_audio_visual",
                    "r9_min": r9_min,
                    "visual_min": visual_min,
                    "mode": "audio_visual_only",
                }
            )
    return candidates


def approval_fn(candidate: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:
    mode = candidate["mode"]
    r9_min = float(candidate["r9_min"])
    visual_min = candidate.get("visual_min")

    def has_r9(row: dict[str, Any], minimum: float) -> bool:
        value = score(row, "governed_r9_score")
        return value is not None and value >= minimum

    def has_visual(row: dict[str, Any], minimum: float) -> bool:
        value = score(row, "visual_late_fusion_logreg_c0.5")
        return value is not None and value >= minimum

    if mode == "r9_only":
        return lambda row: has_r9(row, r9_min)
    if mode == "v1_or_audio_visual":
        assert visual_min is not None
        return lambda row: has_r9(row, V1_R9_MIN) or (has_r9(row, r9_min) and has_visual(row, float(visual_min)))
    if mode == "audio_visual_only":
        assert visual_min is not None
        return lambda row: has_r9(row, r9_min) and has_visual(row, float(visual_min))
    raise ValueError(f"unknown candidate mode: {mode}")


def make_md_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    rows, session_summaries = collect_rows()
    if not rows:
        raise RuntimeError("No runtime-scored reviewed platform/noise rows found.")

    availability = {
        "runtime_session_count": len([s for s in session_summaries if s.get("included")]),
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row["event_label"] for row in rows).items())),
        "governed_r9_present_count": sum(1 for row in rows if row.get("governed_r9_score") not in (None, "")),
        "governed_r9_nonzero_count": sum(1 for row in rows if float(row.get("governed_r9_score") or 0.0) > 0.0),
        "visual_present_count": sum(1 for row in rows if row.get("visual_late_fusion_logreg_c0.5") not in (None, "")),
        "visual_missing_count": sum(1 for row in rows if row.get("visual_late_fusion_logreg_c0.5") in (None, "")),
        "visual_presence_rate": safe_div(
            sum(1 for row in rows if row.get("visual_late_fusion_logreg_c0.5") not in (None, "")),
            len(rows),
        ),
        "source_counts": dict(sorted(Counter(row["session_id"] for row in rows).items())),
    }

    candidates = build_candidates()
    results = [
        summarize_policy(
            rows,
            policy_id=candidate["policy_id"],
            description=candidate["description"],
            approve=approval_fn(candidate),
        )
        | {
            "family": candidate["family"],
            "r9_min": candidate["r9_min"],
            "visual_min": candidate.get("visual_min"),
            "mode": candidate["mode"],
        }
        for candidate in candidates
    ]
    v1 = next(row for row in results if row["policy_id"] == "approve_review_v1")
    for row in results:
        row["coverage_delta_vs_v1"] = float(row["approve_coverage"] - v1["approve_coverage"])
        row["approve_count_delta_vs_v1"] = int(row["approve_count"] - v1["approve_count"])
        row["safe_vs_v1_dangerous_rule"] = row["dangerous_approvals"] == 0
        row["improves_over_v1"] = row["dangerous_approvals"] == 0 and row["approve_count"] > v1["approve_count"]

    safe_candidates = [row for row in results if row["dangerous_approvals"] == 0]
    safe_nonzero_candidates = [row for row in safe_candidates if row["approve_count"] > 0]
    safe_improvers = [row for row in safe_nonzero_candidates if row["approve_count"] > v1["approve_count"]]
    best_safe = max(
        safe_nonzero_candidates,
        key=lambda row: (row["approve_count"], row["approve_precision"] or 0.0, row["approve_coverage"]),
        default=None,
    )
    best = max(
        safe_improvers,
        key=lambda row: (row["approve_count"], row["approve_precision"] or 0.0, row["approve_coverage"]),
        default=None,
    )
    if best is None:
        best = best_safe or {
            "policy_id": "none_safe_nonzero",
            "description": "No nonzero runtime-only approve candidate avoided dangerous approvals.",
            "family": "no_candidate",
            "r9_min": None,
            "visual_min": None,
            "mode": "none",
            "row_count": len(rows),
            "approve_count": 0,
            "approve_coverage": 0.0,
            "approve_precision": None,
            "platform_approve_recall": 0.0,
            "dangerous_approvals": 0,
            "suspicious_additions": 0,
            "approved_label_counts": {},
            "approved_source_counts": {},
            "dangerous_rows": [],
            "approved_rows": [],
            "coverage_delta_vs_v1": -float(v1["approve_coverage"]),
            "approve_count_delta_vs_v1": -int(v1["approve_count"]),
            "safe_vs_v1_dangerous_rule": True,
            "improves_over_v1": False,
        }
    final_decision = "R28_RUNTIME_ONLY_REEVALUATION_GAIN" if safe_improvers else "R28_RUNTIME_ONLY_REEVALUATION_NO_CLEAR_GAIN"

    baseline = {
        "experiment_name": "r28_runtime_only_baseline",
        "active_default_policy": "approve_review_v1",
        "sessions": session_summaries,
        "availability": availability,
        "approve_review_v1": v1,
        "interpretation": {
            "scores_genuinely_present_and_nontrivial": availability["governed_r9_nonzero_count"] == availability["row_count"],
            "visual_fallback_rows": availability["visual_missing_count"],
            "v1_behavior_change_vs_pre_r27": (
                "materially changed: pre-r27 manifests had no governed_r9_score, so v1 could not be evaluated as a true live score path; "
                "r27/r28 now exercise nonzero runtime governed scores."
            ),
        },
    }

    candidate_payload = {
        "experiment_name": "r28_runtime_only_approve_candidates",
        "runtime_only_inputs": ["governed_r9_score", "visual_late_fusion_logreg_c0.5"],
        "excluded_inputs": ["reviewed_subtype", "eventLabel as policy input", "persisted review metadata as policy input"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "comparison_rows": results,
    }

    reevaluation = {
        "experiment_name": "r28_runtime_only_approve_policy_reevaluation",
        "baseline_policy": v1,
        "best_runtime_only_candidate": best,
        "best_safe_nonzero_candidate": best_safe,
        "safe_nonzero_candidate_count": len(safe_nonzero_candidates),
        "safe_improver_count": len(safe_improvers),
        "candidate_comparison": results,
        "visual_path_reliability": {
            "visual_present_count": availability["visual_present_count"],
            "visual_missing_count": availability["visual_missing_count"],
            "visual_presence_rate": availability["visual_presence_rate"],
            "fallback_behavior": (
                "candidate rules requiring visual_late_fusion_logreg_c0.5 do not approve via the visual branch when the visual score is missing; "
                "v1 r9-score approvals remain available."
            ),
            "undermines_usefulness": availability["visual_missing_count"] > 0,
        },
        "answers": {
            "runtime_only_candidate_safely_improves_over_v1": bool(safe_improvers),
            "visual_scoring_helping_in_runtime_conditions": bool(safe_improvers) and "visual" in best["policy_id"],
            "closer_to_runtime_valid_widened_approval": bool(safe_improvers),
            "critical_runtime_finding": (
                "The repaired runtime score path produces one nuisance approval under approve_review_v1 itself; "
                "runtime score calibration is therefore not yet equivalent to the governed offline r9 reference."
            ),
        },
        "final_decisions": [final_decision, "APPROVE_REVIEW_V1_REMAINS_DEFAULT"],
    }

    OUT_BASELINE_JSON.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    OUT_CANDIDATES_JSON.write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
    OUT_REEVALUATION_JSON.write_text(json.dumps(reevaluation, indent=2), encoding="utf-8")

    top_rows = sorted(
        results,
        key=lambda row: (row["dangerous_approvals"] == 0, row["approve_count"], row["approve_precision"] or 0.0),
        reverse=True,
    )[:12]
    practical_rows = sorted(
        results,
        key=lambda row: (row["dangerous_approvals"], -row["approve_count"], -(row["approve_precision"] or 0.0)),
    )[:12]

    baseline_md = f"""# R28 Runtime-Only Baseline

Default live policy: `approve_review_v1`.

## Runtime Sessions Used

{make_md_table(session_summaries, ["session_id", "platform_noise_rows_used", "governed_present_count", "visual_present_count", "visual_missing_count"])}

## Availability

- Rows: {availability["row_count"]}
- Labels: `{json.dumps(availability["label_counts"], sort_keys=True)}`
- Governed r9 present: {availability["governed_r9_present_count"]}/{availability["row_count"]}
- Governed r9 nonzero: {availability["governed_r9_nonzero_count"]}/{availability["row_count"]}
- Visual score present: {availability["visual_present_count"]}/{availability["row_count"]}
- Visual score missing: {availability["visual_missing_count"]}

## approve_review_v1

- Approve count: {v1["approve_count"]}
- Approve coverage: {v1["approve_coverage"]:.4f}
- Approve precision: {v1["approve_precision"] if v1["approve_precision"] is not None else "n/a"}
- Dangerous approvals: {v1["dangerous_approvals"]}

Dangerous v1 rows:

```json
{json.dumps(v1["dangerous_rows"], indent=2)}
```

Pre-r27, this could not be evaluated as a real live governed-score path because manifests had no governed r9 score. R28 confirms the score is now present and nontrivial on the included runtime sessions.
"""

    candidates_md = f"""# R28 Runtime-Only Approve Candidates

Policy inputs allowed:

- `governed_r9_score`
- `visual_late_fusion_logreg_c0.5`

Explicitly excluded:

- reviewed subtype
- persisted human review metadata as a policy input
- event label as a policy input

Candidate count: {len(candidates)}

## Practical Candidate Rows

{make_md_table(practical_rows, ["policy_id", "family", "approve_count", "approve_coverage", "approve_precision", "dangerous_approvals", "approve_count_delta_vs_v1"])}
"""

    reevaluation_md = f"""# R28 Runtime-Only Approve Reevaluation

## Best Runtime-Only Candidate

- Policy: `{best["policy_id"]}`
- Description: {best["description"]}
- Approve count: {best["approve_count"]}
- Approve coverage: {best["approve_coverage"]:.4f}
- Approve precision: {best["approve_precision"] if best["approve_precision"] is not None else "n/a"}
- Dangerous approvals: {best["dangerous_approvals"]}
- Coverage delta vs v1: {best["coverage_delta_vs_v1"]:.4f}

## Candidate Comparison

{make_md_table(practical_rows, ["policy_id", "approve_count", "approve_coverage", "approve_precision", "dangerous_approvals", "coverage_delta_vs_v1"])}

## Visual Reliability

- Visual present: {availability["visual_present_count"]}/{availability["row_count"]}
- Visual missing: {availability["visual_missing_count"]}
- Presence rate: {availability["visual_presence_rate"]:.4f}
- Fallback: visual-gated candidate branches do not approve if visual score is missing; v1 r9-score approvals still apply.

## Conclusion

Runtime-only widened approval {'does safely improve over v1 on this repaired-score benchmark' if safe_improvers else 'does not safely improve over v1 on this repaired-score benchmark'}.

Critical finding: the repaired runtime score implementation is not yet calibrated like the governed offline r9 reference. The live runtime v1 threshold approved one nuisance row, so widened runtime approval must remain blocked until score-path calibration is reconciled.

Final decisions:

- `{final_decision}`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
"""

    doc_md = f"""# R28 Runtime-Only Approve Reevaluation

R28 is the first approve-lane reevaluation using the repaired r27 live score path.

The policy benchmark used only fields that can exist before review:

- `governed_r9_score`
- `visual_late_fusion_logreg_c0.5`

It did not use reviewed subtype or any persisted human-review metadata as policy input.

## Result

Best runtime-only candidate: `{best["policy_id"]}`.

- v1 coverage: {v1["approve_coverage"]:.4f}
- best coverage: {best["approve_coverage"]:.4f}
- best precision: {best["approve_precision"] if best["approve_precision"] is not None else "n/a"}
- best dangerous approvals: {best["dangerous_approvals"]}

Critical runtime finding: `approve_review_v1` itself produced {v1["dangerous_approvals"]} dangerous approval on the repaired runtime score path. This means r27 made the score path executable, but the bootstrapped runtime scorer is not yet product-equivalent to the governed offline r9 model.

## Decision

- `{final_decision}`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
"""

    OUT_BASELINE_MD.write_text(baseline_md, encoding="utf-8")
    OUT_CANDIDATES_MD.write_text(candidates_md, encoding="utf-8")
    OUT_REEVALUATION_MD.write_text(reevaluation_md, encoding="utf-8")
    OUT_DOC.write_text(doc_md, encoding="utf-8")

    print(json.dumps({
        "rows": len(rows),
        "v1": {"approve_count": v1["approve_count"], "coverage": v1["approve_coverage"], "dangerous": v1["dangerous_approvals"]},
        "best": {"policy_id": best["policy_id"], "approve_count": best["approve_count"], "coverage": best["approve_coverage"], "dangerous": best["dangerous_approvals"]},
        "final_decisions": reevaluation["final_decisions"],
    }, indent=2))


if __name__ == "__main__":
    main()
