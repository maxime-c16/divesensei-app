# Phase 5 Status Resolution

## Resolved status (post-r4)

**Classification: B — springboard Phase 5 passed, platform/noise scored modeling failed, global Phase 5 remains fail.**

This is now track-specific and explicit:

- **Springboard track: pass**
  - AUC: `0.7745`
  - macro F1: `0.5048` (passes springboard threshold)
  - FN (`springboard_dive -> springboard_rebound_only`): `32` (improved from `34`)
  - FP (`springboard_rebound_only -> springboard_dive`): `0` (unchanged)
- **Platform/noise track: scored fail**
  - scored holdout recovered in r4 with valid two-class coverage (`10 platform_dive`, `10 noise_or_other`)
  - zero train/holdout overlap
  - confusion matrix: `[[8, 2], [9, 1]]`
  - AUC: `0.51` (fails `0.66`)
  - macro F1: `0.3732` (fails `0.5` and `0.64` Champigny threshold proxy)

## Global interpretation

Global Phase 5 is still fail, but **not** because of springboard anymore.  
The blocker moved from validation-gating ambiguity to a **real platform/noise scored modeling problem**.

## What is closed

- detector-line experimentation and threshold/taxonomy changes
- event-manifest propagation debugging
- mixed-slice gating ambiguity as the main blocker
- springboard anchor as primary issue
- small springboard model-capacity tweak
- springboard feature probe r2 (not carried forward)

## What is active

- platform/noise false-positive problem (especially `noise_or_other -> platform_dive`)
- platform/noise failure diagnosis
- platform/noise feature-family design

## Recommended next path

Proceed with platform/noise diagnosis + feature-design pass before any new full Phase 5 rerun.

## What not to revisit now

- detector behavior
- taxonomy
- labels
- springboard probe-r2 features
- broad new model-family experiments before platform/noise diagnosis/design
