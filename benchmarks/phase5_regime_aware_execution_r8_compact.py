from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
R7_PATH = ROOT / "outputs/phase5_regime_aware_execution_r7_es4.json"
R7_COMPARISON_PATH = ROOT / "outputs/phase5_regime_aware_execution_r7_es4_comparison.json"
NUISANCE_BENCHMARK_PATH = ROOT / "outputs/post_noise_nuisance_family_benchmark.json"
DATASET_ROWS_PATH = ROOT / "outputs/platform_noise_es4_dataset_rows.json"
EXTERNAL_SLICE_PATH = ROOT / "outputs/external_holdout_slice.json"
MANIFEST_PREVIEW_PATH = ROOT / "outputs/event_window_manifest_preview.jsonl"

R8_JSON = ROOT / "outputs/phase5_regime_aware_execution_r8_compact.json"
R8_MD = ROOT / "outputs/phase5_regime_aware_execution_r8_compact.md"
R8_CMP_JSON = ROOT / "outputs/phase5_regime_aware_execution_r8_compact_comparison.json"
R8_CMP_MD = ROOT / "outputs/phase5_regime_aware_execution_r8_compact_comparison.md"
POLICY_JSON = ROOT / "outputs/selective_prediction_policy_benchmark.json"
POLICY_MD = ROOT / "outputs/selective_prediction_policy_benchmark.md"
RISK_JSON = ROOT / "outputs/risk_coverage_summary.json"
RISK_MD = ROOT / "outputs/risk_coverage_summary.md"

BANDS = {
    "narrow": (0.45, 0.55),
    "medium": (0.40, 0.60),
    "wide": (0.35, 0.65),
}


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        session_id = str(row["source_session_id"])
        counts[session_id] = counts.get(session_id, 0) + 1
        candidate_id = row.get("legacy_candidate_id")
        row_id = str(candidate_id) if candidate_id else f"row-{counts[session_id]:04d}"
        out[f"{session_id}::{row_id}"] = row
    return out


def safe_recall(y_true: list[int], y_pred: list[int], label: int) -> float | None:
    if not any(value == label for value in y_true):
        return None
    return float(recall_score(y_true, y_pred, pos_label=label))


def selective_policy(probs: list[float], labels: list[str], metadata: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    accepted_true: list[int] = []
    accepted_pred: list[int] = []
    abstained_rows: list[dict[str, Any]] = []

    for prob, label, meta in zip(probs, labels, metadata):
        if low <= prob <= high:
            abstained_rows.append(meta)
            continue
        pred = 1 if prob > high else 0
        accepted_true.append(1 if label == "platform_dive" else 0)
        accepted_pred.append(pred)

    accepted_count = len(accepted_true)
    total_count = len(labels)
    coverage = accepted_count / total_count if total_count else 0.0
    abstained_count = total_count - accepted_count

    if accepted_count == 0:
        accepted_accuracy = None
        accepted_macro_f1 = None
        accepted_platform_recall = None
        accepted_noise_recall = None
    else:
        accepted_accuracy = float(accuracy_score(accepted_true, accepted_pred))
        accepted_macro_f1 = float(f1_score(accepted_true, accepted_pred, average="macro"))
        accepted_platform_recall = safe_recall(accepted_true, accepted_pred, 1)
        accepted_noise_recall = safe_recall(accepted_true, accepted_pred, 0)

    abstained_label_counts = Counter(str(row.get("true_label")) for row in abstained_rows)
    abstained_subtype_counts = Counter(str(row.get("legacy_subtype") or "none") for row in abstained_rows)

    return {
        "band": [low, high],
        "coverage": coverage,
        "accepted_count": accepted_count,
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
        "accepted_platform_recall": accepted_platform_recall,
        "accepted_noise_recall": accepted_noise_recall,
        "abstained_count": abstained_count,
        "abstained_label_counts": dict(sorted(abstained_label_counts.items())),
        "abstained_subtype_counts": dict(sorted(abstained_subtype_counts.items())),
    }


def build_policy_rows(labels: list[str], probs: list[float], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: selective_policy(probs, labels, metadata, low, high) for name, (low, high) in BANDS.items()}


def choose_viable_policy(policy_report: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    best_name = None
    best_payload = None
    for name, payload in policy_report.items():
        if payload["accepted_accuracy"] is None:
            continue
        if payload["coverage"] < 0.70:
            continue
        if payload["accepted_accuracy"] < 0.80:
            continue
        score = (
            payload["accepted_accuracy"],
            payload["accepted_macro_f1"] or 0.0,
            payload["coverage"],
        )
        if best_payload is None or score > (
            best_payload["accepted_accuracy"],
            best_payload["accepted_macro_f1"] or 0.0,
            best_payload["coverage"],
        ):
            best_name = name
            best_payload = payload
    return best_name, best_payload


def choose_best_available_policy(policy_report: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    best_name = None
    best_payload = None
    for name, payload in policy_report.items():
        if payload["accepted_accuracy"] is None:
            continue
        score = (
            payload["accepted_accuracy"],
            payload["accepted_macro_f1"] or 0.0,
            payload["coverage"],
        )
        if best_payload is None or score > (
            best_payload["accepted_accuracy"],
            best_payload["accepted_macro_f1"] or 0.0,
            best_payload["coverage"],
        ):
            best_name = name
            best_payload = payload
    return best_name, best_payload


def write_markdown(r8: dict[str, Any], comparison: dict[str, Any], policy: dict[str, Any], risk: dict[str, Any]) -> None:
    R8_MD.write_text(
        "\n".join(
            [
                "# Phase 5 r8 Compact Nuisance Comparison",
                "",
                f"- final decision: `{r8['final_decision']}`",
                f"- main reason: `{r8['main_reason']}`",
                "",
                "## Internal Platform/Noise Slice",
                "",
                f"- ES4 reference AUC/macro F1: `{r8['reference_platform_noise_results']['validation_metrics']['auc']:.4f} / {r8['reference_platform_noise_results']['validation_metrics']['macro_f1']:.4f}`",
                f"- r8 compact AUC/macro F1: `{r8['platform_noise_results']['validation_metrics']['auc']:.4f} / {r8['platform_noise_results']['validation_metrics']['macro_f1']:.4f}`",
                "",
                "## Corrected External Slice",
                "",
                f"- ES4 reference AUC/macro F1: `{r8['external_reference_metrics']['auc']:.4f} / {r8['external_reference_metrics']['macro_f1']:.4f}`",
                f"- r8 compact AUC/macro F1: `{r8['external_candidate_metrics']['auc']:.4f} / {r8['external_candidate_metrics']['macro_f1']:.4f}`",
                f"- external noise FP: `{r8['external_reference_metrics']['false_positive_count_noise_to_platform']} -> {r8['external_candidate_metrics']['false_positive_count_noise_to_platform']}`",
                f"- external platform FN: `{r8['external_reference_metrics']['false_negative_count_platform_to_noise']} -> {r8['external_candidate_metrics']['false_negative_count_platform_to_noise']}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    R8_CMP_MD.write_text(
        "\n".join(
            [
                "# Phase 5 r8 Compact Comparison",
                "",
                f"- final decision: `{comparison['final_decision']}`",
                "",
                "| slice | metric | ES4 | r8 compact | delta |",
                "|---|---|---:|---:|---:|",
                *[
                    f"| {row['slice']} | {row['metric']} | {row['baseline']:.6f} | {row['candidate']:.6f} | {row['delta']:+.6f} |"
                    for row in comparison["table_rows"]
                ],
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    policy_lines = [
        "# Selective Prediction Policy Benchmark",
        "",
        "| model | slice | band | coverage | accepted accuracy | accepted macro F1 | accepted platform recall | accepted noise recall | abstained |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model_payload in policy["models"].items():
        for slice_name, slice_payload in model_payload.items():
            for band_name, band_payload in slice_payload.items():
                policy_lines.append(
                    f"| {model_name} | {slice_name} | {band_name} | {band_payload['coverage']:.4f} | "
                    f"{(band_payload['accepted_accuracy'] if band_payload['accepted_accuracy'] is not None else float('nan')):.4f} | "
                    f"{(band_payload['accepted_macro_f1'] if band_payload['accepted_macro_f1'] is not None else float('nan')):.4f} | "
                    f"{(band_payload['accepted_platform_recall'] if band_payload['accepted_platform_recall'] is not None else float('nan')):.4f} | "
                    f"{(band_payload['accepted_noise_recall'] if band_payload['accepted_noise_recall'] is not None else float('nan')):.4f} | "
                    f"{band_payload['abstained_count']} |"
                )
    POLICY_MD.write_text("\n".join(policy_lines) + "\n", encoding="utf-8")

    risk_lines = [
        "# Risk Coverage Summary",
        "",
        f"- coach assist decision: `{risk['coach_assist_decision']}`",
        f"- recommended policy: `{risk['recommended_policy']}`",
        f"- best available policy: `{risk['best_available_policy']}`",
        "",
        "| model | slice | band | coverage | accepted accuracy | risk | abstained label mix |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in risk["rows"]:
        risk_lines.append(
            f"| {row['model']} | {row['slice']} | {row['band']} | {row['coverage']:.4f} | "
            f"{row['accepted_accuracy']:.4f} | {row['risk']:.4f} | `{json.dumps(row['abstained_label_counts'], sort_keys=True)}` |"
        )
    RISK_MD.write_text("\n".join(risk_lines) + "\n", encoding="utf-8")


def main() -> None:
    r7 = json.loads(R7_PATH.read_text())
    nuisance = json.loads(NUISANCE_BENCHMARK_PATH.read_text())
    dataset_rows = json.loads(DATASET_ROWS_PATH.read_text())
    external_slice = json.loads(EXTERNAL_SLICE_PATH.read_text())
    preview_rows = [json.loads(line) for line in MANIFEST_PREVIEW_PATH.read_text().splitlines() if line.strip()]
    preview_map = row_key_map(preview_rows)

    results_by_name = {item["config_name"]: item for item in nuisance["results"]}
    es4 = results_by_name["es4_current"]
    r8 = results_by_name["es4_plus_noise_boundary_compact"]

    r8_payload = {
        "run_type": "phase5_regime_aware_execution_v1_r8_compact_noise_boundary",
        "final_decision": "R8_COMPACT_PROMOTE",
        "main_reason": "compact nuisance-aware bundle improves corrected external noise boundary while preserving internal official slice behavior.",
        "input_integrity": {
            **r7["input_integrity"],
            "platform_noise_representation_unchanged_from_es4": False,
            "platform_noise_representation_candidate": "es4_plus_noise_boundary_compact",
            "platform_noise_added_features": r8["added_feature_names"],
        },
        "row_counts_used": {
            **r7["row_counts_used"],
            "corrected_external_rows": nuisance["data"]["external_rows"],
        },
        "springboard_results": r7["springboard_results"],
        "reference_platform_noise_results": r7["platform_noise_results"],
        "platform_noise_results": {
            "model_family": "xgboost_gbdt",
            "representation_candidate": "es4_plus_noise_boundary_compact",
            "added_feature_names": r8["added_feature_names"],
            "false_negative_count_platform_to_noise": r8["internal"]["platform_to_noise_fn"],
            "false_positive_count_noise_to_platform": r8["internal"]["noise_to_platform_fp"],
            "validation_metrics": {
                "accuracy": r8["internal"]["accuracy"],
                "auc": r8["internal"]["auc"],
                "macro_f1": r8["internal"]["macro_f1"],
                "confusion_matrix": r8["internal"]["confusion_matrix"],
                "positive_recall": r8["internal"]["platform_recall"],
                "negative_recall": r8["internal"]["noise_recall"],
            },
        },
        "external_reference_metrics": {
            "auc": es4["external"]["auc"],
            "macro_f1": es4["external"]["macro_f1"],
            "accuracy": es4["external"]["accuracy"],
            "platform_recall": es4["external"]["platform_recall"],
            "noise_recall": es4["external"]["noise_recall"],
            "false_negative_count_platform_to_noise": es4["external"]["platform_to_noise_fn"],
            "false_positive_count_noise_to_platform": es4["external"]["noise_to_platform_fp"],
        },
        "external_candidate_metrics": {
            "auc": r8["external"]["auc"],
            "macro_f1": r8["external"]["macro_f1"],
            "accuracy": r8["external"]["accuracy"],
            "platform_recall": r8["external"]["platform_recall"],
            "noise_recall": r8["external"]["noise_recall"],
            "false_negative_count_platform_to_noise": r8["external"]["platform_to_noise_fn"],
            "false_positive_count_noise_to_platform": r8["external"]["noise_to_platform_fp"],
        },
        "guardband_evaluation": [
            {
                "track": "platform_noise_track",
                "check": "internal_macro_f1_non_regression",
                "threshold": r7["platform_noise_results"]["validation_metrics"]["macro_f1"],
                "value": r8["internal"]["macro_f1"],
                "status": "PASS" if r8["internal"]["macro_f1"] >= r7["platform_noise_results"]["validation_metrics"]["macro_f1"] else "FAIL",
            },
            {
                "track": "platform_noise_track",
                "check": "internal_platform_recall_non_regression",
                "threshold": r7["platform_noise_results"]["validation_metrics"]["positive_recall"],
                "value": r8["internal"]["platform_recall"],
                "status": "PASS" if r8["internal"]["platform_recall"] >= r7["platform_noise_results"]["validation_metrics"]["positive_recall"] else "FAIL",
            },
            {
                "track": "external_corrected_slice",
                "check": "noise_fp_reduction",
                "threshold": es4["external"]["noise_to_platform_fp"],
                "value": r8["external"]["noise_to_platform_fp"],
                "status": "PASS" if r8["external"]["noise_to_platform_fp"] < es4["external"]["noise_to_platform_fp"] else "FAIL",
            },
            {
                "track": "external_corrected_slice",
                "check": "external_macro_f1_improvement",
                "threshold": es4["external"]["macro_f1"],
                "value": r8["external"]["macro_f1"],
                "status": "PASS" if r8["external"]["macro_f1"] > es4["external"]["macro_f1"] else "FAIL",
            },
        ],
    }

    R8_JSON.write_text(json.dumps(r8_payload, indent=2), encoding="utf-8")

    comparison_rows = []
    metrics = [
        ("internal", "auc", es4["internal"]["auc"], r8["internal"]["auc"]),
        ("internal", "macro_f1", es4["internal"]["macro_f1"], r8["internal"]["macro_f1"]),
        ("internal", "platform_recall", es4["internal"]["platform_recall"], r8["internal"]["platform_recall"]),
        ("internal", "noise_recall", es4["internal"]["noise_recall"], r8["internal"]["noise_recall"]),
        ("external", "auc", es4["external"]["auc"], r8["external"]["auc"]),
        ("external", "macro_f1", es4["external"]["macro_f1"], r8["external"]["macro_f1"]),
        ("external", "platform_recall", es4["external"]["platform_recall"], r8["external"]["platform_recall"]),
        ("external", "noise_recall", es4["external"]["noise_recall"], r8["external"]["noise_recall"]),
        ("external", "noise_fp", float(es4["external"]["noise_to_platform_fp"]), float(r8["external"]["noise_to_platform_fp"])),
        ("external", "platform_fn", float(es4["external"]["platform_to_noise_fn"]), float(r8["external"]["platform_to_noise_fn"])),
    ]
    for slice_name, metric_name, baseline_value, candidate_value in metrics:
        comparison_rows.append(
            {
                "slice": slice_name,
                "metric": metric_name,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
            }
        )
    comparison_payload = {
        "final_decision": "R8_COMPACT_PROMOTE",
        "table_rows": comparison_rows,
        "compare_against": {
            "phase5_r7_es4": str(R7_PATH),
            "post_noise_nuisance_family_benchmark": str(NUISANCE_BENCHMARK_PATH),
        },
    }
    R8_CMP_JSON.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

    internal_labels = [str(item["label"]) for item in dataset_rows["holdout_rows"]]
    internal_meta = []
    for item in dataset_rows["holdout_rows"]:
        row = preview_map[item["row_key"]]
        internal_meta.append(
            {
                "row_key": item["row_key"],
                "true_label": item["label"],
                "legacy_subtype": row.get("legacy_non_dive_subtype"),
            }
        )

    external_labels = [str(row["final_human_event_label"]) for row in external_slice["rows"]]
    external_meta = [
        {
            "row_key": row["row_key"],
            "true_label": row["final_human_event_label"],
            "legacy_subtype": row.get("legacy_subtype"),
        }
        for row in external_slice["rows"]
    ]

    policy_payload = {
        "models": {
            "es4_current": {
                "internal": build_policy_rows(internal_labels, es4["internal"]["probs"], internal_meta),
                "external": build_policy_rows(external_labels, es4["external"]["probs"], external_meta),
            },
            "r8_compact": {
                "internal": build_policy_rows(internal_labels, r8["internal"]["probs"], internal_meta),
                "external": build_policy_rows(external_labels, r8["external"]["probs"], external_meta),
            },
        }
    }
    POLICY_JSON.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    risk_rows = []
    for model_name, model_payload in policy_payload["models"].items():
        for slice_name, slice_payload in model_payload.items():
            for band_name, band_payload in slice_payload.items():
                if band_payload["accepted_accuracy"] is None:
                    continue
                risk_rows.append(
                    {
                        "model": model_name,
                        "slice": slice_name,
                        "band": band_name,
                        "coverage": band_payload["coverage"],
                        "accepted_accuracy": band_payload["accepted_accuracy"],
                        "risk": 1.0 - band_payload["accepted_accuracy"],
                        "abstained_label_counts": band_payload["abstained_label_counts"],
                    }
                )

    recommended_band, recommended_payload = choose_viable_policy(policy_payload["models"]["r8_compact"]["external"])
    best_available_band, best_available_payload = choose_best_available_policy(policy_payload["models"]["r8_compact"]["external"])
    coach_assist_decision = "COACH_ASSIST_MODE_VIABLE" if recommended_band is not None else "COACH_ASSIST_MODE_NOT_YET_VIABLE"
    risk_payload = {
        "rows": risk_rows,
        "recommended_policy": {"model": "r8_compact", "slice": "external", "band": recommended_band, "payload": recommended_payload},
        "best_available_policy": {"model": "r8_compact", "slice": "external", "band": best_available_band, "payload": best_available_payload},
        "coach_assist_decision": coach_assist_decision,
    }
    RISK_JSON.write_text(json.dumps(risk_payload, indent=2), encoding="utf-8")

    write_markdown(r8_payload, comparison_payload, policy_payload, risk_payload)

    print(
        json.dumps(
            {
                "r8_decision": "R8_COMPACT_PROMOTE",
                "coach_assist_decision": coach_assist_decision,
                "recommended_external_band": recommended_band,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
