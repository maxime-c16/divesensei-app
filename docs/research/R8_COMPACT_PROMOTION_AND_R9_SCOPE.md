# R8 Compact Promotion And R9 Scope

Date: 2026-04-16

## Frozen reference state

- Internal Phase 5 benchmark remains valid and solved.
- The corrected external platform/noise session remains the retained external reference.
- `es4_plus_noise_boundary_compact` is now the promoted governed next representation candidate for the platform/noise track.
- Coach-assist abstention mode is still not viable.

## Why the reference changed

The causal sequence is now clear:

1. Temporal centering materially improved external platform-positive performance.
2. Tightening nuisance subtypes alone did not move the residual noise boundary.
3. A bounded nuisance-aware representation change did move the residual core.

Accepted promoted result from `r8_compact`:

- Internal macro F1: `0.6970`
- Internal platform recall: `0.8000`
- External macro F1: `0.6955`
- External platform recall: `0.8387`
- External noise recall: `0.5405`
- External noise false positives: `19 -> 17`
- External platform false negatives: `12 -> 10`

## Workflow rule now frozen

The export workflow must be serial:

1. `export-event-review-support`
2. `export-event-reviewed-manifest`

Running these in parallel can allow stale support rows to leak into the reviewed manifest. The corrected external session and all future governed nuisance-bank sessions must follow the serial export order.

## What is still unresolved

The remaining problem is not broad platform detection. The bottleneck is the nuisance boundary:

- handling noise
- voice / whistle near the mic
- non-dive splash
- platform-context clutter that still looks dive-like to the scorer

The current product framing should also stay narrow:

- not “force a hard label on every sample”
- but “improve bounded automatic coverage while keeping unsafe rows review-required”

That product mode is still not viable yet, so the next cycle must harden generalization first.


## New reviewed nuisance source

A new mixed external session is now prepared and reviewed enough to count as the first real R9 augmentation-ready nuisance source:

- `evaluation_SNMT-16min_20260417-131944`
- mixed platform + springboard content
- candidate rows fully event-labeled
- nuisance subtypes confirmed: `voice_whistle`, `handling_noise`, `non_dive_splash`
- one false-negative row still needs an event label before final governed dataset assembly

SNMT should be used as augmentation, not as a replacement for the fixed corrected external reference.

## R9 scope

R9 is a nuisance-generalization cycle, not a broad retuning cycle.

It should:

- keep detector behavior frozen
- keep taxonomy frozen
- keep threshold policy frozen for comparability
- keep `r8_compact` as the promoted representation baseline
- add targeted reviewed nuisance coverage from additional external source units
- preserve the current internal and corrected external references as fixed comparisons

The next formal experiment is `r9_compact_nuisance_generalization`.

## April 18 Update

- `evaluation_Champigny-17-04-9min_20260418-065417` is now reviewed and exported.
- Three reviewed external nuisance sources are now available: `SNMT`, `IMG_8852`, and `Champigny-17-04`.
- Naive unweighted three-source augmentation improved the corrected external nuisance boundary but regressed the internal slice.
- A bounded source-weighted run corrected that problem.

Winning weighted scheme:

- `SNMT`: `1.0`
- `IMG_8852`: `1.0`
- `Champigny-17-04`: `0.3`

Weighted governed result vs `r8_compact`:

- internal macro F1: `0.6970 -> 0.6970`
- internal platform recall: `0.8000 -> 0.8000`
- corrected external macro F1: `0.6955 -> 0.8134`
- corrected external platform recall: `0.8387 -> 0.8871`
- corrected external noise recall: `0.5405 -> 0.7297`
- corrected external noise false positives: `17 -> 10`
- corrected external platform false negatives: `10 -> 7`

Current governed interpretation:

- `r8_compact` remains the promoted representation baseline
- `r9_compact_nuisance_generalization_weighted` is now the promoted next governed training protocol candidate
- coach-assist abstention behavior improved materially, but is still not strong enough to call product-ready

## Source-Balance Update

- The nuisance-bank question is no longer “do we need more reviewed nuisance data before any useful movement happens?” That is already answered yes.
- The key engineering lesson is now explicit: nuisance-bank expansion must be **source-balanced**, not naively merged.
- The next cycle should build on the weighted scheme above, not reopen broad feature sweeps or timing work.
