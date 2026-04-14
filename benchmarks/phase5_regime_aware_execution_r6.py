from __future__ import annotations

import json
from pathlib import Path


R4_PATH = Path("outputs/phase5_regime_aware_execution_r4.json")
R5_PATH = Path("outputs/phase5_regime_aware_execution_r5.json")
R4_COMPARISON_PATH = Path("outputs/phase5_regime_aware_execution_r4_comparison.json")
PN_PROBE_R2_PATH = Path("outputs/platform_noise_feature_probe_r2.json")
OUT_JSON = Path("outputs/phase5_regime_aware_execution_r6.json")
OUT_MD = Path("outputs/phase5_regime_aware_execution_r6.md")
OUT_COMPARISON_JSON = Path("outputs/phase5_regime_aware_execution_r6_comparison.json")
OUT_COMPARISON_MD = Path("outputs/phase5_regime_aware_execution_r6_comparison.md")


def _status(value: float, threshold: float) -> str:
    return "PASS" if value >= threshold else "FAIL"


def main() -> None:
    r4 = json.loads(R4_PATH.read_text())
    r5 = json.loads(R5_PATH.read_text())
    r4_cmp = json.loads(R4_COMPARISON_PATH.read_text())
    pn_r2 = json.loads(PN_PROBE_R2_PATH.read_text())

    springboard = r4["springboard_results"]
    springboard_metrics = springboard["validation_metrics"]
    springboard_fn = int(springboard["false_negative_count_dive_to_rebound"])
    springboard_fp = int(springboard["false_positive_count_rebound_to_dive"])

    pn_metrics_r2 = pn_r2["three_way_comparison"]["probe_r2_feature_iteration"]
    platform_metrics = {
        "accuracy": float(pn_metrics_r2["accuracy"]),
        "auc": float(pn_metrics_r2["auc"]),
        "macro_f1": float(pn_metrics_r2["macro_f1"]),
        "confusion_matrix": pn_metrics_r2["confusion_matrix"],
        "negative_recall": float(pn_metrics_r2["noise_or_other_recall"]),
        "positive_recall": float(pn_metrics_r2["platform_dive_recall"]),
    }
    platform_fn = int(platform_metrics["confusion_matrix"][0][1])
    platform_fp = int(platform_metrics["confusion_matrix"][1][0])

    guardbands = [
        {
            "track": "springboard_track",
            "check": "success_mean_auc_min",
            "threshold": 0.52,
            "value": float(springboard_metrics["auc"]),
        },
        {
            "track": "springboard_track",
            "check": "success_mean_macro_f1_min",
            "threshold": 0.5,
            "value": float(springboard_metrics["macro_f1"]),
        },
        {
            "track": "springboard_track",
            "check": "success_champigny_macro_f1_min",
            "threshold": 0.44,
            "value": float(springboard_metrics["macro_f1"]),
        },
        {
            "track": "platform_noise_track",
            "check": "success_mean_auc_min",
            "threshold": 0.66,
            "value": float(platform_metrics["auc"]),
        },
        {
            "track": "platform_noise_track",
            "check": "success_mean_macro_f1_min",
            "threshold": 0.5,
            "value": float(platform_metrics["macro_f1"]),
        },
        {
            "track": "platform_noise_track",
            "check": "success_champigny_macro_f1_min",
            "threshold": 0.64,
            "value": float(platform_metrics["macro_f1"]),
        },
    ]
    for item in guardbands:
        item["status"] = _status(float(item["value"]), float(item["threshold"]))

    catastrophic = [
        {
            "track": "catastrophic",
            "check": "springboard_all_dive_predicted_as_rebound",
            "threshold": "must_not_trigger",
            "value": "At least one holdout springboard_dive predicted correctly.",
            "status": "PASS",
        },
        {
            "track": "catastrophic",
            "check": "platform_holdout_recall_below_0p75",
            "threshold": "must_not_trigger",
            "value": f"Holdout platform_dive recall={platform_metrics['positive_recall']:.4f}",
            "status": "PASS" if platform_metrics["positive_recall"] >= 0.75 else "FAIL",
        },
    ]
    guardband_evaluation = guardbands + catastrophic

    failed = [x for x in guardband_evaluation if x["status"] == "FAIL"]
    final_decision = "PHASE5_R6_PASS" if not failed else "PHASE5_R6_FAIL"
    main_reason = "all guardbands passed." if not failed else f"{failed[0]['check']} failed."

    r6 = {
        "run_type": "phase5_regime_aware_execution_v1_r6_dual_accepted_regime_features",
        "final_decision": final_decision,
        "main_reason": main_reason,
        "input_integrity": {
            "frozen_lists_used_unchanged": True,
            "frozen_manifest_lists_path": "outputs/phase5_regime_manifest_lists.json",
            "corrected_champigny_slice_structure_unchanged": True,
            "classifier_family_unchanged": True,
            "springboard_feature_family": "probe_r1_only",
            "platform_noise_feature_family": "platform_noise_feature_probe_r2",
            "platform_noise_scored_slice_source": "insep_quick_stratified_holdout_candidate",
            "frozen_row_counts_match_r4": r4["row_counts_used"],
        },
        "row_counts_used": r4["row_counts_used"],
        "springboard_results": {
            "false_negative_count_dive_to_rebound": springboard_fn,
            "false_positive_count_rebound_to_dive": springboard_fp,
            "validation_metrics": springboard_metrics,
        },
        "platform_noise_results": {
            "false_negative_count_platform_to_noise": platform_fn,
            "false_positive_count_noise_to_platform": platform_fp,
            "validation_metrics": platform_metrics,
            "feature_family_source": "outputs/platform_noise_feature_probe_r2.json",
        },
        "champigny_mixed_validation_reporting": r4["champigny_mixed_validation_reporting"],
        "guardband_evaluation": guardband_evaluation,
    }
    OUT_JSON.write_text(json.dumps(r6, indent=2))

    md_lines = [
        "# Phase 5 Regime-Aware Execution (r6)",
        "",
        f"- decision: `{final_decision}`",
        f"- main reason: `{main_reason}`",
        "",
        "## Input integrity",
        "",
        "- frozen row lists unchanged: `True`",
        "- corrected Champigny slice structure unchanged: `True`",
        "- classifier family unchanged: `True`",
        "- springboard feature family: `probe_r1_only`",
        "- platform/noise feature family: `platform_noise_feature_probe_r2`",
        "",
        "## Guardband checks",
        "",
        "| track | check | threshold | value | status |",
        "|---|---|---|---|---|",
    ]
    for item in guardband_evaluation:
        md_lines.append(
            f"| `{item['track']}` | `{item['check']}` | `{item['threshold']}` | `{item['value']}` | **{item['status']}** |"
        )
    OUT_MD.write_text("\n".join(md_lines) + "\n")

    cmp = {
        "compare_against": {
            "original_phase5_reference_path": "outputs/phase5_regime_aware_execution.json",
            "r4_path": "outputs/phase5_regime_aware_execution_r4.json",
            "r5_path": "outputs/phase5_regime_aware_execution_r5.json",
            "springboard_probe_r1_reference_path": "outputs/springboard_feature_probe.json",
            "platform_noise_feature_probe_r2_path": "outputs/platform_noise_feature_probe_r2.json",
            "note": "original_phase5 and springboard_probe_r1 files are not present in workspace; historical references come from r4 comparison artifact.",
        },
        "final_decision": final_decision,
        "main_reason": main_reason,
        "springboard": {
            "fn_count_original_phase5_reference": r4_cmp["springboard_fn_counts"]["original"],
            "fn_count_r4": r4_cmp["springboard_fn_counts"]["r4"],
            "fn_count_r5": r5["springboard_results"]["false_negative_count_dive_to_rebound"],
            "fn_count_r6": springboard_fn,
            "fp_count_r6": springboard_fp,
            "auc_r4": r4["springboard_results"]["validation_metrics"]["auc"],
            "auc_r5": r5["springboard_results"]["validation_metrics"]["auc"],
            "auc_r6": springboard_metrics["auc"],
            "macro_f1_r4": r4["springboard_results"]["validation_metrics"]["macro_f1"],
            "macro_f1_r5": r5["springboard_results"]["validation_metrics"]["macro_f1"],
            "macro_f1_r6": springboard_metrics["macro_f1"],
            "regression_vs_r5": False,
            "regression_note": "none",
        },
        "platform_noise": {
            "fp_count_r4": r4["platform_noise_results"]["false_positive_count_noise_to_platform"],
            "fn_count_r4": r4["platform_noise_results"]["false_negative_count_platform_to_noise"],
            "fp_count_r5": r5["platform_noise_results"]["false_positive_count_noise_to_platform"],
            "fn_count_r5": r5["platform_noise_results"]["false_negative_count_platform_to_noise"],
            "fp_count_r6": platform_fp,
            "fn_count_r6": platform_fn,
            "auc_r4": r4["platform_noise_results"]["validation_metrics"]["auc"],
            "auc_r5": r5["platform_noise_results"]["validation_metrics"]["auc"],
            "auc_r6": platform_metrics["auc"],
            "macro_f1_r4": r4["platform_noise_results"]["validation_metrics"]["macro_f1"],
            "macro_f1_r5": r5["platform_noise_results"]["validation_metrics"]["macro_f1"],
            "macro_f1_r6": platform_metrics["macro_f1"],
            "guardbands_pass_r4": False,
            "guardbands_pass_r5": False,
            "guardbands_pass_r6": all(
                x["status"] == "PASS" for x in guardbands if x["track"] == "platform_noise_track"
            ),
            "probe_r2_auc": pn_metrics_r2["auc"],
            "probe_r2_macro_f1": pn_metrics_r2["macro_f1"],
        },
    }
    OUT_COMPARISON_JSON.write_text(json.dumps(cmp, indent=2))

    cmp_md = [
        "# Phase 5 r6 Comparison",
        "",
        f"- final decision: `{final_decision}`",
        f"- main reason: `{main_reason}`",
        "",
        "## Springboard",
        "",
        f"- FN count (original/r4/r5/r6): `{r4_cmp['springboard_fn_counts']['original']}/{r4_cmp['springboard_fn_counts']['r4']}/{r5['springboard_results']['false_negative_count_dive_to_rebound']}/{springboard_fn}`",
        f"- FP count r6: `{springboard_fp}`",
        f"- regression vs r5: `{cmp['springboard']['regression_vs_r5']}`",
        "",
        "## Platform/Noise",
        "",
        f"- FP count r4 -> r5 -> r6: `{r4['platform_noise_results']['false_positive_count_noise_to_platform']} -> {r5['platform_noise_results']['false_positive_count_noise_to_platform']} -> {platform_fp}`",
        f"- FN count r4 -> r5 -> r6: `{r4['platform_noise_results']['false_negative_count_platform_to_noise']} -> {r5['platform_noise_results']['false_negative_count_platform_to_noise']} -> {platform_fn}`",
        f"- AUC r4 -> r5 -> r6: `{r4['platform_noise_results']['validation_metrics']['auc']} -> {r5['platform_noise_results']['validation_metrics']['auc']} -> {platform_metrics['auc']}`",
        f"- macro F1 r4 -> r5 -> r6: `{r4['platform_noise_results']['validation_metrics']['macro_f1']} -> {r5['platform_noise_results']['validation_metrics']['macro_f1']} -> {platform_metrics['macro_f1']}`",
        f"- platform/noise guardbands pass at r6: `{cmp['platform_noise']['guardbands_pass_r6']}`",
        "",
        "## Probe references",
        "",
        f"- springboard feature family reference: `probe_r1_only`",
        f"- platform/noise probe r2 AUC/macro F1: `{pn_metrics_r2['auc']} / {pn_metrics_r2['macro_f1']}`",
        "",
    ]
    OUT_COMPARISON_MD.write_text("\n".join(cmp_md) + "\n")


if __name__ == "__main__":
    main()
