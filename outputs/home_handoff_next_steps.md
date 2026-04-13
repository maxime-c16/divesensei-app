# Home Handoff — Next Steps (post-r4)

## Where the project sits now

- Springboard track is a **real pass** (r4 still holds r3 gain: FN `34 -> 32`, FP stays `0`, springboard guardbands pass).
- Platform/noise now has a **real scored failure** (not gating/not N/A): valid scored holdout, but AUC and macro F1 both fail.
- Global Phase 5 is still fail because platform/noise modeling is not yet good enough.

## First action at home

1. Run a focused **platform/noise failure diagnosis** on the r4 scored slice errors (especially `noise_or_other -> platform_dive`).
2. Design a **small platform/noise feature family** for that failure mode.
3. Keep detector/taxonomy/labels/model family unchanged during that diagnosis/design pass.
4. **Do not rerun full Phase 5 yet** before that diagnosis/design pass is complete.

## Do not revisit

- detector behavior or thresholds
- taxonomy
- reviewed labels
- springboard configuration (keep probe-r1)
- springboard probe-r2 feature set
- broad unrelated experiments
