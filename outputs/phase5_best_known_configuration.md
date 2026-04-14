# Phase 5 Best-Known Configuration (r7-es4 freeze)

- **Global status:** `PHASE5_R7_ES4_PASS`
- **Frozen detector:** validated legacy detector, proposal-generator only (unchanged)
- **Taxonomy:** unchanged (`springboard_dive`, `springboard_rebound_only`, `platform_dive`, `noise_or_other`)
- **Springboard track:** unchanged accepted config (`probe_r1_only`)
- **Platform/noise representation:** unchanged accepted ES4 input feature set
- **Platform/noise model family:** `xgboost_gbdt` (only model-family change)
- **Frozen split policy:** `outputs/phase5_regime_manifest_lists.json`
  - platform/noise scored holdout: 20 rows (10 platform, 10 noise)
  - train/holdout overlap: 0
  - Champigny platform-only + ambiguity: reporting-only
- **Catastrophic checks:** PASS / PASS

## Best achieved metrics

- Springboard: AUC `0.7745`, macro F1 `0.5048`, FN `32`, FP `0`
- Platform/noise: AUC `0.7100`, macro F1 `0.6970`, accuracy `0.7000`, confusion `[[8,2],[4,6]]`, FN `2`, FP `4`

## Residual hard rows

- FNs: `det-0038`, `det-0042`
- FPs: `det-0062`, `det-0014`, `det-0022`, `det-0058`

## Why this passed

- Logistic-family bottleneck was removed by ES4 model-family upgrade to XGBoost GBDT.
- Row-level holdout improved materially on the same frozen scored slice (FP `9 -> 4`, FN `2 -> 2`).
- Attribution emphasizes interaction-heavy temporal/spectral features (`inter_peak_interval_cv`, `tail_half_life_ms`, `impact_peak_prominence_db`, `onset_density_0_300ms_post`, `spectral_entropy_post_mean`).
