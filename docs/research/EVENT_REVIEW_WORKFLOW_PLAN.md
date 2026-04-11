# Event Review Workflow Plan

Date: 2026-04-11

## Purpose

This plan defines a small review-support layer for the event-window architecture.

The goal is reviewer efficiency and label-quality improvement.

This is not model training and not detector integration.

## Where The New Step Fits

Current flow:

1. `evaluate-session`
2. human review in the WebUI
3. `export-evaluation-review`
4. downstream analysis or replay

Proposed flow:

1. `evaluate-session`
2. `export-event-review-support`
3. human review in the WebUI with event hints visible
4. `export-evaluation-review`
5. downstream analysis or replay

The new step is a precompute layer after candidate detection and before review consumption.

It prepares event-level hints from the already-detected candidate set.

## New UI-Facing Concepts

The UI layer needs to distinguish:

- legacy detector label
- event-level review label
- machine-generated suggestion
- provenance / confidence of the suggestion
- event window bounds

Suggested UI concepts:

- `suggested_event_label`
- `suggested_event_label_confidence`
- `suggested_event_label_reason`
- `suggested_session_type_context`
- `has_preceding_rebound_context`
- `has_delayed_entry_candidate`
- `event_label_provenance_suggestion`
- `event_anchor_strategy`
- `event_window_start_seconds`
- `event_window_end_seconds`

## What Is A Suggestion Versus A Label

### Suggestions

Suggestions are machine-generated hints only.

Examples:

- `suggested_event_label`
- `suggested_event_label_reason`
- `suggested_session_type_context`
- `event_label_provenance_suggestion`

These fields should help the reviewer decide faster.

They are not final truth.

### Labels

Labels remain human-reviewed decisions.

Examples:

- legacy detector review label
- legacy non-dive subtype
- final event-window review label when that workflow is added later

The workflow must preserve legacy labels rather than overwriting them.

## Provenance Preservation

The support layer should preserve both:

- legacy review provenance
- new suggestion provenance

Suggested provenance categories:

- `direct_review`
- `session_type_inferred`
- `subtype_mapped`
- `uncertain`

Suggested UI behavior:

- show the suggestion provenance next to the machine hint
- keep the human-reviewed label separate
- keep uncertain rows visible instead of hiding them

## What The Reviewer Should Be Able To Confirm Quickly

The reviewer should be able to confirm:

- whether the candidate is a springboard dive
- whether the candidate is a springboard rebound-only event
- whether the candidate is a platform dive
- whether the candidate is noise or other clutter
- whether the proposed event window makes sense
- whether the suggestion should be accepted, rejected, or marked uncertain

The reviewer should not need to infer the detector lineage manually.

## Conservative Heuristic Rules

The first review-support pass should stay simple:

- if a dive occurs after a nearby board-rebound-labeled sound, suggest `springboard_dive`
- if a dive occurs with no nearby rebound-like context, keep the context flag visible and only suggest `platform_dive` when the session type itself supports that interpretation
- if a board-rebound event has no plausible delayed entry in the event window, suggest `springboard_rebound_only`
- if a negative event is whistle/handling/splash-like, map it toward `noise_or_other`

The output should remain a hint layer, not an automated labeling system.

## UI / Data Contract Impact

The session manifest should gain optional event-review support fields.

The review store should continue to persist only human decisions.

The UI should be able to read:

- legacy review data
- event-review support hints
- event window start/end markers

No existing review semantics should change.

## Workflow Safety Rules

- do not change detector thresholds
- do not change classifier logic
- do not introduce live event-level scoring into the detector
- do not silently convert suggestions into labels
- do not hide provenance
- do not turn `no_rebound_context_detected` into a class suggestion by itself

## Output Expectation

After the precompute step, the WebUI should have a richer artifact surface for review, but the final decision remains human-controlled.
