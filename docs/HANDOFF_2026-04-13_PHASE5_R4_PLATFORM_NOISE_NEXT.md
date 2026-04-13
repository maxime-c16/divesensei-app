# HANDOFF 2026-04-13 — Phase5 r4, Platform/Noise Next

## 1) Current project state

- Frozen validated detector remains unchanged.
- Event taxonomy remains unchanged: `springboard_dive`, `springboard_rebound_only`, `platform_dive`, `noise_or_other`.
- Springboard feature probe r1 transferred successfully into r3/r4.
- Platform/noise scored slice was recovered in r4 (valid two-class holdout with leak prevention).
- Platform/noise still failed as a **real scored modeling problem**.

## 2) Key r4 facts

### Springboard (pass)

- AUC: `0.7745098039215687`
- macro F1: `0.5047619047619047`
- FN count (`springboard_dive -> springboard_rebound_only`): `32`
- FP count (`springboard_rebound_only -> springboard_dive`): `0`
- Status: **PASS** (all springboard success guardbands pass)

### Platform/noise (scored fail)

- Holdout slice size: `20`
- Label balance: `10 platform_dive`, `10 noise_or_other`
- Anchor consistency: proposal_centered throughout
- Train/holdout overlap: `0` (leak prevention applied)
- Confusion matrix: `[[8, 2], [9, 1]]`
- AUC: `0.51`
- macro F1: `0.37321937321937326`
- Status: **FAIL** (fails platform/noise success guardbands)

### Global interpretation

- **Springboard pass + platform/noise scored failure**
- Global Phase 5 remains fail because platform/noise is not yet good enough.

## 3) What is closed

- Old detector-line experimentation
- Manifest propagation debugging
- Mixed-slice gating ambiguity as primary blocker
- Springboard anchor as primary issue
- Small springboard model-capacity tweak
- Springboard feature probe r2 (explicitly rejected)

## 4) What is active

- Platform/noise false-positive problem
- Especially `noise_or_other -> platform_dive`
- Next step is representation/feature diagnosis, not immediate rerun

## 5) Exact next action at home

1. Run platform/noise failure diagnosis on r4 scored-slice errors.
2. Design a compact platform/noise feature family targeting the false-positive mode.
3. Keep detector/taxonomy/labels/model family unchanged for that pass.
4. Do **not** rerun full Phase 5 before diagnosis + feature design pass is complete.

## 6) What not to touch

- Detector behavior
- Taxonomy
- Labels
- Springboard configuration (keep probe-r1)
- Broad new experiments outside platform/noise diagnosis/design
