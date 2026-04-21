# Approve Review v2 Shadow Mode

`approve_review_v1` remains the active default.

`approve_review_v2_shadow` now points to the r24 guarded approve expansion candidate.

## Shadow Policy

- approve if `r9_score >= 0.92158`
- or approve if `r9_score >= 0.84` and `visual_late_fusion_logreg_c0.5 >= 0.55`
- suppress the expansion branch when reviewed subtype is `handling_noise`, `voice_whistle`, `non_dive_splash`, or `unknown_transient`
- otherwise `needs_review`
- if the visual score is missing, fall back to v1 and do not add a shadow approval

## Why It Changed

r23 exposed three unsafe v2-only approvals on the fresh CAO-SUN source:

- `det-0092`
- `det-0093`
- `det-0094`

All three were `voice_whistle` rows with high r9 and high visual scores. r24 blocks that failure family and keeps the 20 clean platform additions.

## Current r24 Evidence

- source count: `7`
- row count: `383`
- added approvals over v1: `20`
- approve precision: `1.0000`
- dangerous approvals: `0`
- suspicious added approvals: `0`

## Operational Status

- decision: `HARDENED_V2_READY_FOR_SHADOW_MODE`
- rollout: shadow mode only
- default replacement: not allowed yet

## Replacement Criteria

- visual_late_fusion_logreg_c0.5 generated serially in reviewed manifests for all eligible rows
- continued shadow telemetry shows zero suspicious added approvals
- source-aware dangerous approvals remain zero on new independent reviewed sources
- external approve precision remains near 1.0 and coverage remains above approve_review_v1
- human override rate for v2-added approvals remains zero or explicitly accepted
