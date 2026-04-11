# Event Window Manifest Spec

Date: 2026-04-11

## Purpose

This spec defines the preview manifest layer for the next event-aware architecture.

The manifest layer reuses the existing evaluation/review/export workflow as the extraction backbone, but it does not rewrite legacy detector semantics.

Legacy labels remain preserved as legacy metadata.

New event-window labels are added separately.

## Scope

This spec covers:

- event-window manifest row shape
- label mapping rules
- provenance handling
- anchor and window metadata
- preview export behavior

This spec does not cover:

- model training
- detector changes
- threshold changes
- live integration

## 1. Manifest Source Inputs

The preview exporter should consume existing evaluated/reviewed session artifacts:

- `ui_session_manifest.json`
- `evaluation_review.json`
- `exports/evaluation-review/reviewed_candidates.jsonl`
- `exports/evaluation-review/false_negatives.jsonl`
- optional supporting proposal diagnostics where available

The exporter must treat these as source metadata, not as a replacement for event labels.

## 2. Manifest Row Contract

Each manifest row must contain the following minimum fields.

### 2.1 Session Identity

- `source_session_root`
- `source_session_id`
- `source_video_path`
- `session_type`
- `session_type_provenance`
- `session_type_confidence`

### 2.2 Legacy Review Metadata

- `legacy_candidate_id`
- `legacy_candidate_label`
- `legacy_non_dive_subtype`
- `is_false_negative_window`

### 2.3 Event Window Metadata

- `event_anchor_timestamp_seconds`
- `anchor_strategy`
- `event_window_start_seconds`
- `event_window_end_seconds`
- `event_label`
- `event_label_provenance`
- `uncertainty_flag`

### 2.4 Legacy Detector Metadata

- `proposal_timestamp_seconds`
- `proposal_frontend`
- `clip_probability`
- `audio_score`
- `combined_score`
- `audio_model_probability`
- `audio_clip_probability`
- `raw_proposal_score`
- `threshold_passed`

### 2.5 Compatibility Rule

Legacy metadata must remain intact.

The exporter may add new event-level fields, but it must not:

- overwrite legacy review labels in place
- reinterpret existing subtype strings as direct event truth
- drop uncertainty

## 3. Event Taxonomy

The first preview manifest must use the Phase 2 taxonomy exactly:

1. `springboard_dive`
2. `springboard_rebound_only`
3. `platform_dive`
4. `noise_or_other`

Operational meaning:

- `springboard_dive`: springboard-origin dive event window with rebound structure and delayed entry
- `springboard_rebound_only`: springboard-origin non-dive event dominated by rebound structure
- `platform_dive`: platform-origin dive event window with weak or diffuse onset and late entry
- `noise_or_other`: non-dive clutter or negative event window that is not a rebound-only springboard event

## 4. Provenance Rules

Each row must carry exactly one explicit provenance tag for the event label:

- `direct_review`
- `session_type_inferred`
- `subtype_mapped`
- `uncertain`

### 4.1 `direct_review`

Use when the event label is directly supported by human review without type conversion or subtype mapping.

### 4.2 `session_type_inferred`

Use when a reviewed `dive` label is mapped to `springboard_dive` or `platform_dive` from session type knowledge.

This must not be silently promoted to direct truth.

### 4.3 `subtype_mapped`

Use when a reviewed non-dive subtype is mapped into the event taxonomy.

Examples:

- `board_rebound` -> `springboard_rebound_only`
- `voice_whistle` -> `noise_or_other`
- `handling_noise` -> `noise_or_other`
- `non_dive_splash` -> `noise_or_other`

### 4.4 `uncertain`

Use when the row cannot be assigned confidently from current evidence.

`uncertain` rows must remain visible in counts and summaries.

## 5. Session-Type Rules

The exporter must attach a session type to each row.

Recommended initial session types:

- `springboard`
- `platform`
- `unknown`

Recommended provenance categories:

- `direct_review`
- `session_type_inferred`
- `uncertain`

Important rule:

- session type can be supported directly by the archived project docs for some sessions, or inferred from review context for others
- session type inference must not be disguised as direct review

Recommended session-type provenance categories:

- `direct_review`
- `session_type_inferred`
- `uncertain`

## 6. Anchor Strategy

The first preview exporter should support at least:

1. `proposal_centered`
2. `earliest_strong_peak_in_local_cluster`

Recommended primary anchor:

- `proposal_centered`

Recommended backup anchor:

- `earliest_strong_peak_in_local_cluster`

Rationale:

- `proposal_centered` matches the intended future architecture boundary
- `earliest_strong_peak_in_local_cluster` is a fallback for rebound-dominated springboard clusters

## 7. Window Strategy

The first preview exporter should emit asymmetric window boundaries.

Recommended primary window:

- `0.75s pre`
- `2.25s post`
- `3.0s total`

Recommended backup window:

- `1.0s pre`
- `3.0s post`
- `4.0s total`

## 8. Preview Export Behavior

The initial manifest export should be a preview, not a training set.

It should:

- cover a small authoritative set of local reviewed roots
- preserve provenance tags
- materialize rows reproducibly
- write a JSONL manifest
- write a markdown summary

The preview should not:

- train a model
- alter detector outputs
- rename legacy labels
- fabricate direct event truth

## 9. Required Output Artifacts

The preview export should write:

- `outputs/event_window_manifest_preview.jsonl`
- `outputs/event_window_manifest_preview_summary.md`

## 10. Acceptance Criteria

The preview export is acceptable if:

- rows can be materialized from the reviewed session roots
- every row has a valid event label and provenance tag
- legacy metadata is preserved
- uncertainty is explicit
- the preview summary reports counts by event label, provenance, and session type

## 11. Open Gaps

The preview export is not the final dataset.

Open gaps after Phase 3 preview:

- direct event-window review labels are still missing
- session-type certainty is still partly inferred
- the four-class taxonomy is still a prototype schema, not a validated dataset label set
