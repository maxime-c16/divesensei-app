# Phase 5 Final Status (r7-es4 freeze)

## Part A — Final status

- **Springboard track:** **PASS** (unchanged configuration from prior accepted pass)
- **Platform/noise track:** **PASS** with ES4 model-family upgrade (`xgboost_gbdt`)
- **Global Phase 5:** **PASS** (`PHASE5_R7_ES4_PASS`)
- **Detector behavior:** unchanged (validated legacy detector remains frozen proposal-generator)
- **Taxonomy:** unchanged (`springboard_dive`, `springboard_rebound_only`, `platform_dive`, `noise_or_other`)
- **Decisive change:** platform/noise model-family upgrade, not detector/taxonomy/label/split changes

## Part B — Best-known configuration (frozen)

- **Detector:** frozen validated legacy detector, proposal-generator only
- **Springboard feature family:** `probe_r1_only`
- **Platform/noise feature family:** accepted ES4 input representation (`platform_noise_feature_probe_r4` feature set retained)
- **Platform/noise model family:** `xgboost_gbdt`
- **Frozen split policy:** `outputs/phase5_regime_manifest_lists.json`
  - platform/noise scored holdout = 20 rows (10 `platform_dive`, 10 `noise_or_other`)
  - train/holdout overlap = 0
  - Champigny platform-only and ambiguity slices remain reporting-only
- **Catastrophic checks:** both pass
  - springboard all-dive-to-rebound catastrophe not triggered
  - platform holdout recall floor (0.75) passes (`0.80`)

Best achieved metrics (`outputs/phase5_regime_aware_execution_r7_es4.json`):

- **Springboard:** AUC `0.7745`, macro F1 `0.5048`, FN `32`, FP `0`
- **Platform/noise:** AUC `0.7100`, macro F1 `0.6970`, accuracy `0.7000`, confusion `[[8,2],[4,6]]`, FN `2`, FP `4`
- **Residual hard rows:**  
  FNs: `det-0038`, `det-0042`  
  FPs: `det-0062`, `det-0014`, `det-0022`, `det-0058`

## Part C — Why this pass happened

- Logistic-family regime baseline was the bottleneck on platform/noise (r4: AUC `0.51`, macro F1 `0.3732`, FP `9`).
- ES4 `xgboost_gbdt` captured tabular interaction structure better under identical frozen rows/features/policy.
- Row-level outcomes improved materially on the same scored holdout:
  - `noise_or_other -> platform_dive` FP reduced `9 -> 4`
  - `platform_dive -> noise_or_other` FN held `2 -> 2`
- Top attribution features were interaction-heavy temporal/spectral descriptors:
  `inter_peak_interval_cv`, `tail_half_life_ms`, `impact_peak_prominence_db`, `onset_density_0_300ms_post`, `spectral_entropy_post_mean`.

## Part D — Closed for this cycle

- springboard anchor crisis
- platform/noise feature-only looping (without family upgrade)
- threshold-only rescue attempts as primary path
- detector-side intervention as current priority

## Part E — Next-cycle study focus (not executed in this pass)

1. Robustness, uncertainty, and reproducibility checks.
2. Wider model-family benchmarking beyond the current winner.
3. Calibration and governance policy under deployment constraints.
4. Broader generalization checks across additional reviewed slices/sessions.
