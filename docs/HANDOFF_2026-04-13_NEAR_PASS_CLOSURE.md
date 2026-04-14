# Handoff — 2026-04-13 Near-Pass Closure

This cycle is closed as a **near-pass** with no further experiments in-pass.

## Fixed conclusions at closure

- Springboard is a real pass and remains frozen (`probe_r1_only`).
- Platform/noise improved substantially with accepted representation work.
- Accepted r4 platform/noise bundle produced strong ranking signal (`AUC=0.81`) and improved noise false positives (`6 -> 3` vs accepted r2 result).
- Under current constraints, platform recall and noise false-positive guardrails cannot be satisfied simultaneously by threshold tuning alone.
- Therefore, feature micro-iteration and threshold micro-iteration stop here.

## Best-known current state

- Detector behavior: frozen validated detector (proposal generator only).
- Taxonomy/labels: unchanged.
- Classifier family: unchanged logistic family.
- Platform/noise representation: accepted r4 4-feature bundle.
- Global status: **near-pass / not full pass**.

## Why no further in-cycle work

- The current gap is a constrained operating-policy tradeoff, not missing obvious signal.
- Additional tiny tweaks are not the highest-value next action.

## Next cycle (explicitly out of scope for this pass)

1. Evaluate stronger platform/noise model family.
2. Add cleaner/more platform-noise data for ambiguous residual clusters.
3. Decide governance policy for track-level acceptance vs global acceptance.
4. Optionally define calibrated review-band operating policy if governance permits.
