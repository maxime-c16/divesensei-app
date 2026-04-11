# Final Project State Summary

Date: 2026-04-11

## What Works

- The validated detector is the region-descriptor tie-break branch at band `0.20`.
- It includes:
  - `frontend_region_pattern_exception`
  - `frontend_dense_pcen_pattern_exception`
  - `frontend_region_tail_imbalance_exception`
- It is stable on INSEP and Champigny.
- It preserves candidate count, replay coverage, and FP/min across the validated comparisons.

## What Does Not Work

- `frontend_short_region_tail_exception` is not a default branch.
- Event-level live integrations are not safe in the current peak-first detector.
- The following are closed:
  - event-level reranking
  - event-level score blending
  - representative selection
  - delayed cluster selection
  - cluster winner replacement
  - merge-stage event veto

## What Was Learned

- Event-level modeling is real and useful offline.
- It improves dive vs rebound separation.
- It improves cluster ranking offline.
- It does not fit cleanly into the current peak-first funnel.
- The failure is architectural, not feature-quality related.

## What To Do Next

- Keep the validated region-tie-break branch as the production reference.
- Keep `frontend_short_region_tail_exception` experimental only.
- Start any future event-aware work as a new architecture line, not as another local peak modification.
- For detector-side work, pivot to the remaining springboard-heavy below-threshold and suppression-collapse regimes.

