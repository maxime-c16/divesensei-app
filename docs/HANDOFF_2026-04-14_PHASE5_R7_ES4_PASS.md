# HANDOFF 2026-04-14 — Phase5 r7-es4 PASS

## Current state

- Phase 5 regime-aware execution now passes globally on `r7-es4`.
- Springboard remains pass and unchanged from accepted frozen setup.
- Platform/noise now passes using ES4 model-family upgrade (`xgboost_gbdt`) under unchanged detector/taxonomy/labels/splits/policy.
- Detector remains frozen proposal-generator only.

## Decisive artifacts

- `outputs/phase5_regime_aware_execution_r7_es4.json`
- `outputs/phase5_regime_aware_execution_r7_es4.md`
- `outputs/phase5_regime_aware_execution_r7_es4_comparison.json`
- `outputs/phase5_regime_aware_execution_r7_es4_comparison.md`
- `outputs/platform_noise_es4_holdout_predictions.jsonl`
- `outputs/platform_noise_es4_feature_attribution.json`

## Key metrics

- **Decision:** `PHASE5_R7_ES4_PASS`
- **Springboard:** AUC `0.7745`, macro F1 `0.5048`, FN `32`, FP `0`
- **Platform/noise:** AUC `0.7100`, macro F1 `0.6970`, accuracy `0.7000`, confusion `[[8,2],[4,6]]`, FN `2`, FP `4`
- **Catastrophic checks:** both pass

## What changed vs failing logistic regime run

- r4 logistic regime platform/noise: AUC `0.51`, macro F1 `0.3732`, FP `9`, FN `2`
- r7-es4 platform/noise: AUC `0.71`, macro F1 `0.6970`, FP `4`, FN `2`
- Transfer from ES4 benchmark to regime-aware context is clean (matching holdout outcome in comparison artifact).

## Residual hard rows

- FNs: `evaluation_insep_quick_9015_20260409_ui::det-0038`, `...::det-0042`
- FPs: `...::det-0062`, `...::det-0014`, `...::det-0022`, `...::det-0058`

## Closed in this cycle

- Springboard anchor crisis
- Platform/noise feature-only looping without family upgrade
- Threshold-only rescue attempts
- Detector-side intervention as primary path

## Next cycle (do not execute in this pass)

1. Robustness/uncertainty/reproducibility validation of r7-es4.
2. Model-family benchmarking beyond current winner.
3. Calibration and governance policy checks.
4. Broader generalization checks across additional held-out reviewed slices.

## Note

`outputs/detector_mental_model_meta_experiments_summary.json` is missing on this machine; freeze decisions above are grounded in present r7-es4 and ES4 artifacts.
