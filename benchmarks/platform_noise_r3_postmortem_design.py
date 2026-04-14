from __future__ import annotations

import json
from pathlib import Path


R2_PATH = Path("outputs/platform_noise_feature_probe_r2.json")
R3_PATH = Path("outputs/platform_noise_feature_probe_r3.json")
R6_RESIDUAL_PATH = Path("outputs/platform_noise_r6_residual_analysis.json")

POSTMORTEM_JSON = Path("outputs/platform_noise_r3_postmortem.json")
POSTMORTEM_MD = Path("outputs/platform_noise_r3_postmortem.md")
CANDIDATES_JSON = Path("outputs/platform_noise_final_feature_candidates.json")
CANDIDATES_MD = Path("outputs/platform_noise_final_feature_candidates.md")


def write_postmortem(postmortem: dict) -> None:
    POSTMORTEM_JSON.write_text(json.dumps(postmortem, indent=2))

    lines = [
        "# Platform/Noise r3 Postmortem",
        "",
        f"- decision: `{postmortem['decision']}`",
        "",
        "## Part A — Why r3 failed",
        "",
        f"- r2 AUC/F1: `{postmortem['metrics']['r2_auc']:.4f}` / `{postmortem['metrics']['r2_macro_f1']:.4f}`",
        f"- r3 AUC/F1: `{postmortem['metrics']['r3_auc']:.4f}` / `{postmortem['metrics']['r3_macro_f1']:.4f}`",
        f"- noise->platform FP delta (r2->r3): `{postmortem['metrics']['noise_fp_delta_r2_to_r3']}`",
        "",
        "### Failure interpretation",
        "",
        f"- tonal penalty too narrow: `{postmortem['part_a_r3_failure']['tonal_penalty_too_narrow']}`",
        f"- residual rows not separated: `{postmortem['part_a_r3_failure']['rows_not_separated_by_single_penalty']}`",
        f"- redundancy / destabilization risk: `{postmortem['part_a_r3_failure']['redundancy_or_destabilization']}`",
        "",
        "## Part B — Residual cluster characterization (accepted r2 set)",
        "",
        f"- tonal-dominant FP rows: `{postmortem['part_b_residual_cluster']['tonal_dominant_fp_rows']}`",
        f"- diffuse handling-dominant FP rows: `{postmortem['part_b_residual_cluster']['diffuse_handling_fp_rows']}`",
        f"- ambiguous mixed-profile FP rows: `{postmortem['part_b_residual_cluster']['ambiguous_mixed_fp_rows']}`",
        f"- boundary platform FN rows: `{postmortem['part_b_residual_cluster']['boundary_platform_fn_rows']}`",
        "",
    ]
    POSTMORTEM_MD.write_text("\n".join(lines))


def write_candidates(candidates: dict) -> None:
    CANDIDATES_JSON.write_text(json.dumps(candidates, indent=2))

    lines = [
        "# Platform/Noise Final Feature Candidates",
        "",
        f"- decision: `{candidates['decision']}`",
        "",
        "## Part C — Online-informed final bundle",
        "",
        "### Official/primary references used",
    ]
    for ref in candidates["references"]:
        lines.append(f"- {ref['id']}: {ref['title']} ({ref['url']})")
    lines.extend(
        [
            "",
            "### Proposed small coherent bundle",
            "",
            "| feature | what it is | why it maps to residual cluster | derivable now |",
            "|---|---|---|---|",
        ]
    )
    for feat in candidates["final_candidate_bundle"]["features"]:
        lines.append(
            f"| `{feat['name']}` | {feat['what_it_is']} | {feat['why_for_residual_cluster']} | {feat['derivable_from_current_windows']} |"
        )
    lines.extend(
        [
            "",
            "## Part D — Go / no-go",
            "",
            f"- `{candidates['decision']}`",
            f"- rationale: `{candidates['decision_rationale']}`",
            "",
        ]
    )
    CANDIDATES_MD.write_text("\n".join(lines))


def main() -> None:
    r2 = json.loads(R2_PATH.read_text())
    r3 = json.loads(R3_PATH.read_text())
    residual = json.loads(R6_RESIDUAL_PATH.read_text())

    r2_metrics = r2["three_way_comparison"]["probe_r2_feature_iteration"]
    r3_metrics = r3["four_way_comparison"]["probe_r3_with_tonal_noise_penalty"]
    r2_fp = r2_metrics["false_positive_noise_to_platform_rows"]
    r2_fn = r2_metrics["false_negative_platform_to_noise_rows"]

    subtype_by_row = {
        item["row_key"]: item.get("legacy_subtype")
        for item in residual["part_a_residual_error_audit"]["false_positive_details"]
        + residual["part_a_residual_error_audit"]["false_negative_details"]
    }

    tonal_fp = [row for row in r2_fp if subtype_by_row.get(row) == "voice_whistle"]
    diffuse_fp = [row for row in r2_fp if subtype_by_row.get(row) == "handling_noise"]
    mixed_fp = [row for row in r2_fp if subtype_by_row.get(row) not in {"voice_whistle", "handling_noise"}]

    postmortem = {
        "scope": {
            "analysis_only": True,
            "phase5_rerun_performed": False,
            "platform_noise_probe_rerun_performed": False,
            "springboard_touched": False,
            "detector_or_taxonomy_or_labels_or_classifier_changed": False,
        },
        "metrics": {
            "r2_auc": r2_metrics["auc"],
            "r3_auc": r3_metrics["auc"],
            "r2_macro_f1": r2_metrics["macro_f1"],
            "r3_macro_f1": r3_metrics["macro_f1"],
            "r2_confusion": r2_metrics["confusion_matrix"],
            "r3_confusion": r3_metrics["confusion_matrix"],
            "noise_fp_delta_r2_to_r3": int(r3_metrics["confusion_matrix"][1][0] - r2_metrics["confusion_matrix"][1][0]),
        },
        "part_a_r3_failure": {
            "tonal_penalty_too_narrow": (
                "The single multiplicative tonal_noise_penalty targets only one tonal axis, but accepted r2 residual FPs are mixed "
                "(handling_noise + voice_whistle + one ambiguous/null row). It cannot address diffuse handling clutter structure."
            ),
            "rows_not_separated_by_single_penalty": {
                "tonal_rows": tonal_fp,
                "diffuse_rows": diffuse_fp,
                "ambiguous_rows": mixed_fp,
                "r3_newly_reintroduced_fp_row": sorted(list(set(r3_metrics["false_positive_noise_to_platform_rows"]) - set(r2_fp))),
            },
            "redundancy_or_destabilization": (
                "r3 added a product of two already-present predictors (whistle_band_energy_fraction_post and tonal_peak_fraction_post_mean). "
                "In the same linear-logistic family this can add collinearity-like pressure and reweighting instability without adding a new "
                "orthogonal cue for diffuse clutter, consistent with AUC 0.65->0.64 and FP 6->7 regression."
            ),
        },
        "part_b_residual_cluster": {
            "accepted_r2_fp_rows": r2_fp,
            "accepted_r2_fn_rows": r2_fn,
            "tonal_dominant_fp_rows": tonal_fp,
            "diffuse_handling_fp_rows": diffuse_fp,
            "ambiguous_mixed_fp_rows": mixed_fp,
            "boundary_platform_fn_rows": r2_fn,
            "cluster_summary": (
                "Residual errors are a mixed cluster problem: tonal whistle-like noise + diffuse handling clutter + one ambiguous/null context row, "
                "with two boundary platform dives."
            ),
        },
        "decision": "R3_NEGATIVE_RESULT_CONFIRMED",
    }

    candidates = {
        "scope": {
            "analysis_only": True,
            "no_new_probe_run": True,
            "frozen_detector_taxonomy_labels_classifier": True,
            "springboard_unchanged": True,
        },
        "references": [
            {
                "id": "librosa_spectral_contrast",
                "title": "librosa.feature.spectral_contrast",
                "url": "https://librosa.org/doc/main/generated/librosa.feature.spectral_contrast.html",
                "key_point": "Peak-vs-valley contrast in sub-bands distinguishes narrow-band tonal structure from broad-band noise.",
            },
            {
                "id": "essentia_spectral_contrast_primary",
                "title": "Essentia SpectralContrast (with primary citations Jiang et al., ICME 2002; Akkermans et al., SMC 2009)",
                "url": "https://essentia.upf.edu/reference/std_SpectralContrast.html",
                "key_point": "Official implementation and direct references for octave-band spectral contrast descriptors.",
            },
            {
                "id": "librosa_onset_strength",
                "title": "librosa.onset.onset_strength",
                "url": "https://librosa.org/doc/main/generated/librosa.onset.onset_strength.html",
                "key_point": "Spectral-flux onset envelope captures transient density and local event activity.",
            },
            {
                "id": "librosa_tempogram",
                "title": "librosa.feature.tempogram",
                "url": "https://librosa.org/doc/main/generated/librosa.feature.tempogram.html",
                "key_point": "Local autocorrelation of onset envelope captures rhythmic/repetitive clutter versus isolated impulses.",
            },
        ],
        "part_c_design_principle": (
            "Use one small coherent bundle that adds orthogonal structure cues for mixed residual FPs, instead of another single tonal scalar."
        ),
        "final_candidate_bundle": {
            "bundle_id": "platform_noise_feature_bundle_r4_candidate",
            "bundle_size": 4,
            "features": [
                {
                    "name": "spectral_contrast_mean_post",
                    "what_it_is": "Mean octave-band spectral contrast over post-impact frames.",
                    "why_for_residual_cluster": "Separates tonal peak-vs-valley structure (voice/whistle) from broadband handling clutter in the mixed FP cluster.",
                    "derivable_from_current_windows": "yes",
                },
                {
                    "name": "spectral_contrast_low_high_slope_post",
                    "what_it_is": "Difference/slope between low-band and high-band spectral contrast averages.",
                    "why_for_residual_cluster": "Captures whether energy organization is low-band diffuse handling versus high-band whistle-like tonal concentration.",
                    "derivable_from_current_windows": "yes",
                },
                {
                    "name": "onset_tempogram_peak_ratio_post",
                    "what_it_is": "Ratio of strongest non-zero-lag tempogram peak to zero-lag energy in a short post-impact window.",
                    "why_for_residual_cluster": "Penalizes repetitive/onset-dense handling clutter that mimics platform impacts in scalar energy features.",
                    "derivable_from_current_windows": "yes",
                },
                {
                    "name": "onset_density_0_300ms_post",
                    "what_it_is": "Count or normalized density of onset-envelope local maxima in the first 300 ms after impact.",
                    "why_for_residual_cluster": "Distinguishes isolated platform impact signatures from diffuse multi-bump handling/noise bursts while preserving boundary dive recall.",
                    "derivable_from_current_windows": "yes",
                },
            ],
            "why_coherent_vs_r3": (
                "This bundle jointly addresses all three residual FP sub-clusters (tonal, diffuse, ambiguous) by combining spectral structure and local temporal pattern cues."
            ),
        },
        "decision": "FINAL_PLATFORM_NOISE_R4_FEATURE_BUNDLE_JUSTIFIED",
        "decision_rationale": (
            "A single narrow tonal penalty failed; residual errors are mixed-cluster. One last compact orthogonal bundle is justified before stopping iteration."
        ),
    }

    write_postmortem(postmortem)
    write_candidates(candidates)


if __name__ == "__main__":
    main()
