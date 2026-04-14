# Phase 5 Pass Summary (r7-es4)

## Final status

- Springboard track: **PASS**
- Platform/noise track: **PASS** (`xgboost_gbdt`)
- Global Phase 5: **PASS** (`PHASE5_R7_ES4_PASS`)
- Detector and taxonomy remain frozen/unchanged.

## Frozen best-known configuration

- Springboard feature family: `probe_r1_only`
- Platform/noise feature family: accepted ES4 input representation
- Platform/noise model family: `xgboost_gbdt`
- Frozen split policy: `outputs/phase5_regime_manifest_lists.json`
- Holdout overlap: `0`
- Catastrophic checks: `PASS`, `PASS`

## Best metrics

- Springboard: AUC `0.7745`, macro F1 `0.5048`, FN `32`, FP `0`
- Platform/noise: AUC `0.7100`, macro F1 `0.6970`, accuracy `0.7000`, confusion `[[8,2],[4,6]]`, FN `2`, FP `4`

## Why pass happened

- Logistic-family bottleneck on platform/noise was resolved by ES4 XGBoost upgrade under the same frozen data/features policy.
- Holdout error pattern improved vs r4 logistic:
  - `noise_or_other -> platform_dive` FP `9 -> 4`
  - `platform_dive -> noise_or_other` FN `2 -> 2`
- Top attribution features: `inter_peak_interval_cv`, `tail_half_life_ms`, `impact_peak_prominence_db`, `onset_density_0_300ms_post`, `spectral_entropy_post_mean`.

## Residual hard rows

- FNs: `det-0038`, `det-0042`
- FPs: `det-0062`, `det-0014`, `det-0022`, `det-0058`

## Closed in this cycle

- springboard anchor crisis
- platform/noise feature-only looping
- threshold-only rescue attempts
- detector-side intervention as current priority

## Next-cycle study scope (not executed here)

- robustness / uncertainty / reproducibility
- model-family benchmarking beyond current winner
- calibration / governance
- broader generalization checks
