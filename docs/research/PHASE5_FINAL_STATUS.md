# Phase 5 Final Status (Current Cycle Closure)

## Part A — Final status

- **Springboard track:** pass (stable, accepted, unchanged in this cycle)
- **Platform/noise track:** near-pass under current constraints; fails current joint recall/FP guardrail requirement
- **Global Phase 5:** near-pass / not fully passed

## Part B — Best-known configuration (frozen)

- **Detector line:** frozen validated detector as proposal generator only  
  (`frontend_region_pattern_exception`, `frontend_dense_pcen_pattern_exception`, `frontend_region_tail_imbalance_exception`)
- **Springboard feature family:** `probe_r1_only` (accepted best, no regression)
- **Platform/noise feature family (representation):** accepted `platform_noise_feature_probe_r4` bundle  
  (`spectral_contrast_mean_post`, `spectral_contrast_low_high_slope_post`, `onset_tempogram_peak_ratio_post`, `onset_density_0_300ms_post`)
- **Classifier family:** unchanged logistic model family
- **Validation policy:** frozen manifest/slice structure (platform/noise 20-row scored holdout: 10 platform, 10 noise)

Best observed platform/noise results with accepted r4 representation:

- **AUC:** 0.81
- **Macro F1:** 0.70
- **Accuracy:** 0.70
- **Confusion:** `[[7, 3], [3, 7]]`
- **Noise->Platform FP:** 3 (improved from 6 in accepted r2 run)
- **Platform recall:** 0.70 (down from 0.80 in accepted r2 run)

Operating-point analysis outcome:

- Threshold tuning changes precision/recall tradeoff, but no threshold satisfied all required constraints simultaneously:
  - platform recall >= 0.75
  - macro F1 >= 0.50
  - strict noise->platform FP improvement vs r4 default operating point

## Part C — Why this cycle stops here

- The remaining blocker is no longer weak representation signal (AUC is strong at 0.81).
- The blocker is decision-policy geometry under current constraints: improving recall reintroduces too many noise false positives; reducing false positives drops platform recall below floor.
- Additional tiny feature or threshold tweaks are unlikely to change this structural tradeoff enough to justify continued micro-iteration in this cycle.
- This is a constrained **near-pass**, not a broad failure.

## Part D — Next-cycle options (ranked; not for this pass)

1. **Stronger platform/noise model family:** move beyond current logistic family while keeping detector/taxonomy/labels frozen.
2. **More/cleaner platform-noise data:** expand and clean ambiguous handling/whistle/noise rows and boundary platform rows.
3. **Protocol/governance decision:** explicit policy on track-level acceptance vs strict global acceptance.
4. **Calibration/governed deployment mode:** if governance allows, adopt confidence-banded review policy rather than single hard threshold.
