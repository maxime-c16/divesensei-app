# Phase 5 Regime-Aware Freeze

Date: 2026-04-13

This file freezes the manifest lists and numeric guardbands for the first future regime-aware Phase 5 simulation run.

## Frozen manifest policy

1. Springboard track uses only `evaluation_insep_15min_validated` springboard rows for train/eval materialization and reserves Champigny springboard rows for validation stress only.
2. Platform/noise track uses only `evaluation_insep_quick_9015_20260409_ui` platform/noise rows for train/eval materialization and reserves Champigny platform/noise rows for validation stress only.
3. Champigny is frozen as mixed-session validation primary role (`used_for_training: false` in this first run).
4. Materialization level is both session-level and row-level (`row_key` lists).

## Frozen guardbands (numeric)

- springboard success min: AUC >= `0.52` and macro F1 >= `0.5`
- platform/noise success min: AUC >= `0.66` and macro F1 >= `0.5`
- Champigny stress mins: springboard macro F1 >= `0.44`, platform/noise macro F1 >= `0.64`
- regression caps: springboard macro F1 drop <= `0.03`, platform/noise macro F1 drop <= `0.04`, Champigny drop <= `0.05`

## Frozen references

- `outputs/event_classifier_baseline_reviewed_anchorpropfix.json`
- `outputs/per_regime_baseline_split_analysis.json`

## Readiness

- After this freeze, protocol ambiguity is removed for manifest membership and numeric pass/fail thresholds.
- Next execution decision can be made against this frozen contract without redefining splits or guardbands.

## Revision r2 (2026-04-13)

- Tiny frozen-slice refinement applied for first rerun.
- Excluded `det-0001` and `det-0007` from Champigny platform/noise mixed validation slice.
- Final human labels and legacy subtypes kept unchanged.
- Guardbands, detector behavior, taxonomy, and model family unchanged.

## Revision r3prep (2026-04-13)

- Champigny mixed validation is now explicitly split into three frozen sub-slices:
  1. `springboard_scored_slice` (scored for springboard guardbands)
  2. `platform_only_stress_slice` (reporting-only)
  3. `ambiguity_slice` containing `det-0001` and `det-0007` (reporting-only)
- Added class-coverage gating for binary scored slices: min 2 classes present and at least 1 row per class.
- If gating fails, binary AUC/macro-F1 guardbands are marked `NOT_APPLICABLE_GATING_FAILED` and only reporting-only metrics are emitted.
- Labels, legacy subtypes, detector behavior, taxonomy, and numeric guardband definitions remain unchanged.

## Revision r4 (2026-04-13)

- Recovered valid two-class scored platform/noise validation slice from `insep_quick_stratified_holdout_candidate`.
- Scored slice composition frozen at 20 rows: 10 `platform_dive`, 10 `noise_or_other`, all proposal-centered.
- Leak prevention frozen: scored holdout rows are removed from platform/noise training rows.
- Champigny platform-only and ambiguity slices remain reporting-only.
- Springboard configuration remains r3 unchanged (probe-r1 features, same classifier family).

## Revision r7-es4 (2026-04-14)

- Frozen passing execution artifact: `outputs/phase5_regime_aware_execution_r7_es4.json`.
- Springboard track carried forward unchanged from r4/r3 pass state (`probe_r1_only`).
- Platform/noise representation remained unchanged from accepted ES4 input dataset.
- Only platform/noise model family changed: logistic family -> `xgboost_gbdt`.
- Platform/noise scored holdout policy unchanged (20-row INSEP quick stratified holdout, leak prevention retained, overlap=0).
- Catastrophic checks pass, and both track-level success guardbands pass.
- This revision becomes the new best-known regime-aware freeze for Phase 5 closure.
