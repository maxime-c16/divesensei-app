# DiveSensei Detector Status

Date: 2026-04-11

This document freezes the current detector state, the validated research outcomes, and the lines of work that are closed.

## 1. Current Validated Detector

The production reference detector is the validated region-tie-break branch:

- `frontend_region_descriptor_enabled`
- `frontend_region_descriptor_pattern_tiebreak_band = 0.20`
- `frontend_region_pattern_exception`
- `frontend_dense_pcen_pattern_exception`
- `frontend_region_tail_imbalance_exception`

This is the last fully validated, stable, cross-session-safe detector.

Validated properties:

- stable on INSEP and Champigny
- stable replay coverage
- stable FP/min
- recovered `48.656411s`
- recovered `417.488888s`
- preserved `114.350347s` as an evaluation/accounting artifact recovery

## 2. Experimental Branches

The only remaining experimental branch to keep around is:

- `frontend_short_region_tail_exception`

Status:

- useful local movement on `145.714040s`
- no regression in the session where it first helped
- did not generalize on the springboard-heavy `insep_15min.mov` validation session

Interpretation:

- experimental only
- session-specific
- not promoted to default

## 3. Springboard Failure Analysis

The springboard-heavy session showed a different confuser regime from platform INSEP:

- board rebounds dominate the false-positive landscape
- voice / whistle / clap-like events are common
- many false negatives are close misses where the detector lands on the last rebound rather than the splash

The key result from that session:

- peak-first selection is structurally insufficient in rebound-heavy clusters
- event-level structure helps offline, but cannot be safely dropped into the live peak-first funnel

## 4. Rebound Discrimination Research

Offline event-level modeling was meaningful:

- dive vs rebound separation improved materially
- cluster ranking improved on average
- the event-level score could distinguish structured dive-like windows from rebound-like windows better than peak-only selection

However:

- event-level scoring alone did not create top-1 recoveries
- the live detector still needs a safe decision boundary

## 5. Event-Level Modeling Research

The offline research answered a narrow but important question:

- yes, event-level structure is real
- yes, it is better than peak-only ranking for some springboard clusters
- no, it is not directly compatible with the current detector funnel

The validated conclusion is architectural:

- event-level modeling improves dive vs rebound separation offline
- event-level modeling improves cluster ranking offline
- event-level modeling cannot be safely integrated as a local peak replacement inside the current detector

## 6. Integration Attempts And Failures

All of the following are closed:

- event-level reranking
- event-level score blending
- representative selection
- delayed cluster selection
- cluster winner replacement
- merge-stage event veto

Observed failure modes:

- candidate collapse
- replay degradation
- loss of accepted detections
- no measurable gain in live detector conditions

The repeated lesson:

- local cluster modifications are not safe injection points in the current peak-first funnel
- the funnel is tightly coupled to suppression, merge, and ranking dynamics

## 7. Final Architectural Conclusion

The limitation is architectural, not feature quality.

The current detector assumes:

- event equals best peak

That is wrong for springboard rebound-heavy sessions.

But the live funnel cannot be lightly changed by swapping winners or adding local vetoes because:

- suppression changes candidate survival
- merge changes the candidate set
- ranking changes who survives later gates

So the right conclusion is:

- event-level reasoning is valid
- the current peak-first detector is not the right architecture for direct event-level insertion
- a new event-aware pipeline would need a cleaner boundary than local peak modification

## 8. Recommended Future Directions

Use the validated branch as the default baseline.

Treat event-level work as a separate future architecture line, not a detector patch.

The next detector-side work that remains plausible is:

- springboard-heavy below-threshold misses
- pre-candidate capping / suppression
- rebound-heavy confuser analysis

Do not continue to stack failed event-integration heuristics.
Do not revisit solved or ambiguity cases as if they were unresolved detector misses.

