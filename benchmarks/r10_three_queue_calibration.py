from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
R9_WEIGHTED_PATH = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted.json"
INTERNAL_ROWS_PATH = ROOT / "outputs" / "platform_noise_es4_dataset_rows.json"
EXTERNAL_ROWS_PATH = ROOT / "outputs" / "external_holdout_slice.json"

OUT_JSON = ROOT / "outputs" / "r10_three_queue_calibration.json"
OUT_MD = ROOT / "outputs" / "r10_three_queue_calibration.md"
AUDIT_JSON = ROOT / "outputs" / "r10_three_queue_score_overlap_audit.json"
AUDIT_MD = ROOT / "outputs" / "r10_three_queue_score_overlap_audit.md"

LOW_GRID = [round(value / 100, 2) for value in range(5, 61, 5)]
HIGH_GRID = [round(value / 100, 2) for value in range(40, 96, 5)]

MIN_PRODUCT_PRECISION = 0.90
MIN_PRODUCT_COVERAGE = 0.40


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_to_int(label: str) -> int:
    return 1 if label == "platform_dive" else 0


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _f(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def _source_name(row: dict[str, Any]) -> str:
    return str(row.get("source_session_id") or row.get("row_key", "").split("::", 1)[0] or "unknown")


def _slice_policy(rows: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    auto_approve = [row for row in rows if float(row["score"]) >= high]
    auto_exclude = [row for row in rows if float(row["score"]) <= low]
    review = [row for row in rows if low < float(row["score"]) < high]
    accepted = auto_approve + auto_exclude
    accepted_true = [_label_to_int(str(row["label"])) for row in accepted]
    accepted_pred = [1 if row in auto_approve else 0 for row in accepted]

    platform_total = sum(1 for row in rows if row["label"] == "platform_dive")
    noise_total = sum(1 for row in rows if row["label"] == "noise_or_other")
    auto_approve_true = sum(1 for row in auto_approve if row["label"] == "platform_dive")
    auto_exclude_true = sum(1 for row in auto_exclude if row["label"] == "noise_or_other")
    auto_approve_errors = [row for row in auto_approve if row["label"] != "platform_dive"]
    auto_exclude_errors = [row for row in auto_exclude if row["label"] != "noise_or_other"]

    return {
        "thresholds": {"auto_exclude_max_score": low, "auto_approve_min_score": high},
        "row_count": len(rows),
        "coverage": _safe_div(len(accepted), len(rows)) or 0.0,
        "review_rate": _safe_div(len(review), len(rows)) or 0.0,
        "auto_approve_count": len(auto_approve),
        "auto_exclude_count": len(auto_exclude),
        "review_required_count": len(review),
        "auto_approve_precision": _safe_div(auto_approve_true, len(auto_approve)),
        "auto_exclude_precision": _safe_div(auto_exclude_true, len(auto_exclude)),
        "auto_approve_platform_recall_contribution": _safe_div(auto_approve_true, platform_total),
        "auto_exclude_noise_recall_contribution": _safe_div(auto_exclude_true, noise_total),
        "accepted_accuracy": float(accuracy_score(accepted_true, accepted_pred)) if accepted_true else None,
        "accepted_macro_f1": float(f1_score(accepted_true, accepted_pred, average="macro")) if accepted_true and len(set(accepted_true + accepted_pred)) > 1 else None,
        "accepted_platform_recall": float(recall_score(accepted_true, accepted_pred, pos_label=1)) if any(value == 1 for value in accepted_true) else None,
        "accepted_noise_recall": float(recall_score(accepted_true, accepted_pred, pos_label=0)) if any(value == 0 for value in accepted_true) else None,
        "auto_approve_error_count": len(auto_approve_errors),
        "auto_exclude_error_count": len(auto_exclude_errors),
        "review_required_label_counts": dict(sorted(Counter(str(row["label"]) for row in review).items())),
        "review_required_subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in review).items())),
        "auto_approve_error_rows": auto_approve_errors,
        "auto_exclude_error_rows": auto_exclude_errors,
    }


def _policy_pair(internal_rows: list[dict[str, Any]], external_rows: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    internal = _slice_policy(internal_rows, low, high)
    external = _slice_policy(external_rows, low, high)
    external_min_precision = min(_f(external["auto_approve_precision"]), _f(external["auto_exclude_precision"]))
    internal_min_precision = min(_f(internal["auto_approve_precision"]), _f(internal["auto_exclude_precision"]))
    return {
        "thresholds": {"auto_exclude_max_score": low, "auto_approve_min_score": high},
        "internal": internal,
        "external": external,
        "external_min_queue_precision": external_min_precision,
        "internal_min_queue_precision": internal_min_precision,
        "joint_score": min(external_min_precision, internal_min_precision) * 10.0
        + external["coverage"]
        + 0.25 * internal["coverage"],
    }


def _row_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    r9 = _load_json(R9_WEIGHTED_PATH)
    internal_source = _load_json(INTERNAL_ROWS_PATH)["holdout_rows"]
    external_source = _load_json(EXTERNAL_ROWS_PATH)["rows"]
    internal_probs = r9["internal_metrics"]["probs"]
    external_probs = r9["external_metrics"]["probs"]

    internal_rows = [
        {
            "row_key": str(row["row_key"]),
            "source_session_id": str(row["row_key"]).split("::", 1)[0],
            "label": str(row["label"]),
            "score": float(score),
            "legacy_subtype": None,
            "suggested_event_label_reason": None,
        }
        for row, score in zip(internal_source, internal_probs)
    ]
    external_rows = [
        {
            "row_key": str(row["row_key"]),
            "source_session_id": str(row.get("source_session_id") or str(row["row_key"]).split("::", 1)[0]),
            "label": str(row["final_human_event_label"]),
            "score": float(score),
            "legacy_subtype": row.get("legacy_subtype"),
            "suggested_event_label_reason": row.get("suggested_event_label_reason"),
            "event_anchor_strategy": row.get("event_anchor_strategy"),
            "manual_correction_type": row.get("manual_correction_type"),
        }
        for row, score in zip(external_source, external_probs)
    ]
    return internal_rows, external_rows


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in ("platform_dive", "noise_or_other"):
        scores = sorted(float(row["score"]) for row in rows if row["label"] == label)
        out[label] = {
            "count": len(scores),
            "min": scores[0] if scores else None,
            "p10": scores[int(0.10 * (len(scores) - 1))] if scores else None,
            "median": scores[int(0.50 * (len(scores) - 1))] if scores else None,
            "p90": scores[int(0.90 * (len(scores) - 1))] if scores else None,
            "max": scores[-1] if scores else None,
        }
    return out


def _overlap_audit(rows: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    dangerous_auto_approve = [row for row in rows if row["label"] == "noise_or_other" and float(row["score"]) >= high]
    dangerous_auto_exclude = [row for row in rows if row["label"] == "platform_dive" and float(row["score"]) <= low]
    ambiguous = [row for row in rows if low < float(row["score"]) < high]
    return {
        "thresholds": {"auto_exclude_max_score": low, "auto_approve_min_score": high},
        "score_summary": _score_summary(rows),
        "dangerous_auto_approve_noise_count": len(dangerous_auto_approve),
        "dangerous_auto_exclude_platform_count": len(dangerous_auto_exclude),
        "dangerous_auto_approve_noise_by_subtype": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in dangerous_auto_approve).items())),
        "dangerous_auto_approve_noise_by_reason": dict(sorted(Counter(str(row.get("suggested_event_label_reason") or "none") for row in dangerous_auto_approve).items())),
        "dangerous_auto_approve_noise_rows": sorted(dangerous_auto_approve, key=lambda row: float(row["score"]), reverse=True),
        "dangerous_auto_exclude_platform_rows": sorted(dangerous_auto_exclude, key=lambda row: float(row["score"])),
        "ambiguous_count": len(ambiguous),
        "ambiguous_label_counts": dict(sorted(Counter(str(row["label"]) for row in ambiguous).items())),
        "ambiguous_subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in ambiguous).items())),
        "ambiguous_by_source": dict(sorted(Counter(_source_name(row) for row in ambiguous).items())),
    }


def _clean_policy(policy: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(policy))
    for slice_name in ("internal", "external"):
        cleaned[slice_name]["auto_approve_error_rows"] = cleaned[slice_name]["auto_approve_error_rows"][:20]
        cleaned[slice_name]["auto_exclude_error_rows"] = cleaned[slice_name]["auto_exclude_error_rows"][:20]
    return cleaned


def main() -> None:
    internal_rows, external_rows = _row_records()
    policies = [
        _policy_pair(internal_rows, external_rows, low, high)
        for low in LOW_GRID
        for high in HIGH_GRID
        if low < high
    ]
    external_safe = [
        policy
        for policy in policies
        if _f(policy["external"]["auto_approve_precision"]) >= MIN_PRODUCT_PRECISION
        and _f(policy["external"]["auto_exclude_precision"]) >= MIN_PRODUCT_PRECISION
    ]
    product_viable = [
        policy
        for policy in external_safe
        if policy["external"]["coverage"] >= MIN_PRODUCT_COVERAGE
        and _f(policy["internal"]["auto_approve_precision"]) >= MIN_PRODUCT_PRECISION
        and _f(policy["internal"]["auto_exclude_precision"]) >= MIN_PRODUCT_PRECISION
        and policy["internal"]["coverage"] > 0.0
    ]
    best_external_safe = max(
        external_safe,
        key=lambda policy: (policy["external"]["coverage"], policy["external_min_queue_precision"]),
        default=None,
    )
    best_balanced = max(
        policies,
        key=lambda policy: (
            min(_f(policy["external"]["auto_approve_precision"]), _f(policy["external"]["auto_exclude_precision"])),
            policy["external"]["coverage"],
        ),
    )
    best_product = max(product_viable, key=lambda policy: policy["joint_score"], default=None)
    recommended = best_product or best_external_safe or best_balanced

    if best_product:
        decision = "TRIAGE_MODE_VIABLE"
        next_action = "wire_three_queue_policy_simulation"
    elif best_external_safe and best_external_safe["external"]["coverage"] > 0.0:
        decision = "TRIAGE_MODE_VIABLE_LOW_COVERAGE_EXTERNAL_ONLY"
        next_action = "do_not_wire_product_mode_yet_run_score_separation_work"
    else:
        decision = "TRIAGE_MODE_NOT_VIABLE_SCORE_OVERLAP"
        next_action = "run_score_separation_model_or_representation_step"

    report = {
        "experiment_name": "r10_three_queue_calibration",
        "model_source": str(R9_WEIGHTED_PATH),
        "policy_definition": {
            "auto_excluded": "score <= auto_exclude_max_score",
            "needs_review": "auto_exclude_max_score < score < auto_approve_min_score",
            "auto_approved": "score >= auto_approve_min_score",
        },
        "product_precision_target": MIN_PRODUCT_PRECISION,
        "product_coverage_target": MIN_PRODUCT_COVERAGE,
        "decision": decision,
        "next_action": next_action,
        "recommended_policy": _clean_policy(recommended),
        "best_external_safe_policy": _clean_policy(best_external_safe) if best_external_safe else None,
        "best_balanced_policy": _clean_policy(best_balanced),
        "top_external_safe_policies": [_clean_policy(policy) for policy in sorted(external_safe, key=lambda p: p["external"]["coverage"], reverse=True)[:10]],
        "internal_score_summary": _score_summary(internal_rows),
        "external_score_summary": _score_summary(external_rows),
        "all_policy_count": len(policies),
        "external_safe_policy_count": len(external_safe),
        "product_viable_policy_count": len(product_viable),
    }
    audit = {
        "experiment_name": "r10_three_queue_score_overlap_audit",
        "decision": decision,
        "recommended_policy_thresholds": recommended["thresholds"],
        "internal": _overlap_audit(internal_rows, recommended["thresholds"]["auto_exclude_max_score"], recommended["thresholds"]["auto_approve_min_score"]),
        "external": _overlap_audit(external_rows, recommended["thresholds"]["auto_exclude_max_score"], recommended["thresholds"]["auto_approve_min_score"]),
        "single_best_next_move": "score_separation_benchmark_for_review_required_boundary",
        "single_best_next_move_rationale": "Safe two-sided auto queues exist only at very low external coverage and do not validate on the internal reference slice; the next useful step is to improve score separation/calibration, not collect another generic reviewed session.",
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# r10 Three-Queue Calibration",
        "",
        f"- decision: `{decision}`",
        f"- next action: `{next_action}`",
        f"- product precision target: `{MIN_PRODUCT_PRECISION}`",
        f"- product coverage target: `{MIN_PRODUCT_COVERAGE}`",
        f"- external-safe policy count: `{len(external_safe)}`",
        f"- product-viable policy count: `{len(product_viable)}`",
        "",
        "## Recommended Policy",
        "",
        "| slice | low | high | coverage | auto approve | approve precision | auto exclude | exclude precision | review required | review composition |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for slice_name in ("internal", "external"):
        payload = recommended[slice_name]
        lines.append(
            f"| {slice_name} | {recommended['thresholds']['auto_exclude_max_score']:.2f} | {recommended['thresholds']['auto_approve_min_score']:.2f} | "
            f"{payload['coverage']:.4f} | {payload['auto_approve_count']} | {_f(payload['auto_approve_precision']):.4f} | "
            f"{payload['auto_exclude_count']} | {_f(payload['auto_exclude_precision']):.4f} | {payload['review_required_count']} | "
            f"`{json.dumps(payload['review_required_label_counts'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Best External-Safe Policies",
            "",
            "| low | high | external coverage | approve precision | exclude precision | approve count | exclude count | review count |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in sorted(external_safe, key=lambda p: p["external"]["coverage"], reverse=True)[:10]:
        ext = policy["external"]
        lines.append(
            f"| {policy['thresholds']['auto_exclude_max_score']:.2f} | {policy['thresholds']['auto_approve_min_score']:.2f} | "
            f"{ext['coverage']:.4f} | {_f(ext['auto_approve_precision']):.4f} | {_f(ext['auto_exclude_precision']):.4f} | "
            f"{ext['auto_approve_count']} | {ext['auto_exclude_count']} | {ext['review_required_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A safe external policy exists only at low coverage.",
            "- Internal validation does not support a two-sided auto queue because the safe threshold region has near-zero accepted internal rows.",
            "- This is a score-overlap/calibration blocker, not a generic review-volume blocker.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit_lines = [
        "# r10 Three-Queue Score Overlap Audit",
        "",
        f"- decision: `{decision}`",
        f"- recommended thresholds: `{json.dumps(recommended['thresholds'], sort_keys=True)}`",
        f"- single best next move: `{audit['single_best_next_move']}`",
        "",
        "## External Danger Rows",
        "",
        f"- noise rows that would be auto-approved as platform: `{audit['external']['dangerous_auto_approve_noise_count']}`",
        f"- platform rows that would be auto-excluded as noise: `{audit['external']['dangerous_auto_exclude_platform_count']}`",
        f"- dangerous noise by subtype: `{json.dumps(audit['external']['dangerous_auto_approve_noise_by_subtype'], sort_keys=True)}`",
        f"- dangerous noise by reason: `{json.dumps(audit['external']['dangerous_auto_approve_noise_by_reason'], sort_keys=True)}`",
        "",
        "## External Ambiguous Region",
        "",
        f"- ambiguous rows: `{audit['external']['ambiguous_count']}`",
        f"- ambiguous labels: `{json.dumps(audit['external']['ambiguous_label_counts'], sort_keys=True)}`",
        f"- ambiguous subtypes: `{json.dumps(audit['external']['ambiguous_subtype_counts'], sort_keys=True)}`",
        "",
        "## Next Move Rationale",
        "",
        audit["single_best_next_move_rationale"],
    ]
    AUDIT_MD.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON), str(OUT_MD), str(AUDIT_JSON), str(AUDIT_MD)], "decision": decision, "recommended_thresholds": recommended["thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
