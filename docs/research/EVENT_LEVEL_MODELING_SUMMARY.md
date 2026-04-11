# Event-Level Modeling Summary

Date: 2026-04-11

## 1. Motivation

The springboard-heavy session exposed a specific failure mode:

- the detector often locked onto the last rebound peak instead of the dive splash
- peak-first selection was too narrow for multi-peak springboard structure

That made event-level modeling worth testing offline.

## 2. What Worked Offline

Offline event-level features improved the right problem:

- dive vs rebound separation improved materially
- cluster ranking improved on average
- event-level scores found better winners inside dense springboard clusters than peak-only selection

The best offline result was not a full winner replacement, but a better notion of event structure.

## 3. What Failed Live

Every live integration attempt hit one of two failure patterns:

- candidate collapse / replay collapse
- no meaningful FN improvement

The failed lines included:

- event-level reranking
- event-level score blending
- representative selection
- delayed cluster selection
- cluster winner replacement
- merge-stage event veto

Measured live failures included:

- candidate count collapse
- replay degradation
- loss of accepted detections
- no stable recall gain

## 4. Key Insight

Event-level modeling is not peak-level compatible in the current detector.

The detector funnel is tightly coupled:

- merge changes what survives
- suppression changes local competition
- ranking changes later acceptance

That means local event-based edits are not safe injection points.

## 5. Final Conclusion

Event-level modeling is promising, but it requires a different architecture.

It should not be treated as:

- a rerank patch
- a local veto patch
- a merge-time tie-break patch

The correct next step is a new event-aware decision boundary, not another incremental integration into the current peak-first funnel.

