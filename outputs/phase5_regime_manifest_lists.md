# Phase 5 Regime Manifest Lists (Frozen)

- springboard train rows: `129`
- springboard validation rows: `104`
- platform/noise train rows: `53`
- platform/noise scored validation rows: `20`
- Champigny springboard scored slice rows: `104`
- Champigny platform-only stress slice rows: `8`
- Champigny ambiguity slice rows: `2`

## Policy summary

- springboard scored slice: Champigny springboard rows only (guardbands applicable)
- platform/noise scored slice: INSEP quick stratified row holdout (10 platform + 10 noise), removed from train
- Champigny platform-only and ambiguity slices remain reporting-only
- class-coverage gating retained for mixed reporting slices

## Revision r4 (platform/noise scored-slice recovery)

- adopted `insep_quick_stratified_holdout_candidate` as primary scored slice source
- leak prevention applied: all scored holdout rows removed from platform/noise training rows
- detector/taxonomy/labels/classifier family/springboard feature set unchanged

## Materialization

- includes session-level manifest roots and row-level `row_key` lists in JSON
