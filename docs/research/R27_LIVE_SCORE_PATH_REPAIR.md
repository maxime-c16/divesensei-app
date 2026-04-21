# R27 Live Score Path Repair

R27 repaired missing live pre-review score generation paths by adding bounded runtime score generation before manifest serialization.

## What changed

- Added runtime governed score generation (`governed_r9_score`) with cached local logistic model bootstrap from reviewed historical rows.
- Added runtime visual late-fusion generation (`visual_late_fusion_logreg_c0.5`) using bounded motion features + governed score fusion model.
- Wired enrichment into normal `evaluate-session` output path so scores appear in `ui_session_manifest.json` before review.
- Propagated enriched scores into proposal diagnostics and export artifacts.

## What did not change

- Detector selection behavior and taxonomy were not changed.
- No rollout promotion was performed; `approve_review_v1` remains the default policy.

## Runtime classification

- governed_r9_score: `live_runtime_score_working`
- visual_late_fusion_logreg_c0.5: `partially_working_but_not_reliable`

## Readiness

System is ready for the next runtime-only reevaluation pass, with the caveat that visual scoring still depends on successful source-video decode and a bounded bootstrap model.

## Final decisions

- `R27_LIVE_SCORE_PATH_REPAIR_PROGRESS`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
