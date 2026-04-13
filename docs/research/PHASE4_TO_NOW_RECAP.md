# Phase 4 To Now Recap

Date: 2026-04-13

## Current State

The project is now in the event-window / event-review phase transition.

What is stable:

- the validated legacy detector remains frozen
- the review UI now supports event-review suggestions and human event labels
- reviewed event manifests can be exported separately from legacy detector labels
- the reviewed dataset summary pipeline is in place
- a simple offline event-classifier baseline exists

What is not yet stable:

- the full four-class event model is still not stable enough for Phase 5 simulation
- `springboard_dive` still collapses into `springboard_rebound_only` in the full reviewed baseline
- the current anchor/window policy is only partially sufficient

## Phase 4 Path

Phase 4 established the first usable offline baseline for the new taxonomy.

Observed results from the reviewed-label baseline:

- `platform_dive` vs `noise_or_other` became meaningfully learnable
- `springboard_dive` vs `springboard_rebound_only` remained weak under the original proposal-centered representation
- the four-class prototype remained unstable

That led to a stop decision rather than moving into Phase 5.

## Springboard-Specific Finding

The springboard failure analysis showed:

- proposal-centered windows were the wrong primary representation for springboard events
- the 3.0s window length was not the main problem
- `earliest_strong_peak_in_local_cluster` was the best springboard anchor tested
- a delayed-entry-centered proxy was weaker and less stable

The springboard-only experiment then confirmed:

- springboard became substantially more learnable when the anchor shifted to the earliest strong peak proxy
- the anchor choice mattered more than small window-length tweaks

## Manifest Policy Update

The manifest/export layer was updated so anchor policy is conditional by regime:

- springboard rows use `earliest_strong_peak_in_local_cluster`
- platform rows remain `proposal_centered`

This policy is explicit in the exported artifacts via:

- `event_anchor_strategy`
- `event_anchor_strategy_rationale`

However, when the full reviewed four-class baseline was rerun with the updated policy, the aggregate metrics did not materially change. That means the representation fix alone is not enough to justify simulation yet.

## UI / Workflow Status

The desktop review app now includes:

- event-review suggestions
- editable human event labels
- a refinement queue for high-value rows
- explicit handling for manual false-negative review rows
- a more compact review sidebar

The UI also had multiple real regressions that were fixed along the way:

- false-negative markers disappeared from the timeline and had to be reintroduced
- the refinement queue duplicated the old review list and had to be simplified
- the save path for event labels needed multiple passes to persist correctly
- false-negative rows required special handling so they behaved like first-class review items

These were workflow issues, not detector issues.

## Main Struggles

The main blockers encountered were:

1. `springboard_dive` remained hard to separate from `springboard_rebound_only`
2. session context mattered more than the initial manifest policy assumed
3. the review UI had several persistence and layout edge cases
4. the full four-class baseline still did not recover a stable decision boundary even after better springboard anchoring

## Current Conclusion

The project is still in a refinement phase.

Accepted conclusions:

- keep the validated detector frozen
- keep the new taxonomy and event-review workflow
- keep the conditional anchor policy in the manifest layer
- do not start Phase 5 simulation yet
- do not add model capacity yet
- do not broaden relabeling until the remaining confusion rows are prioritized again

## Ongoing Direction

The next useful work is still constrained:

- continue using the new review workflow
- keep the reviewed artifacts fresh
- inspect only targeted confusion-driving rows
- avoid another broad architecture jump until the springboard boundary is actually stable in the full four-class setting

