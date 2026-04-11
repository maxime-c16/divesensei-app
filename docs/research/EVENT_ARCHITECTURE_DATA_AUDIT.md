# Event Architecture Data Audit

Date: 2026-04-11

## Scope

This audit covers the local Phase 1 inventory only:

- reviewed session roots available locally
- springboard/platform coverage
- authoritative frozen-baseline outputs
- trained models present locally
- review/export label coverage
- whether current labels are sufficient for event-window training now

This audit does not define the event-window schema and does not start dataset building.

## Executive Answer

The local repo contains enough reviewed evidence to proceed to Phase 2 schema design now.

It does not contain a clean, fully direct, four-class event-window training set yet.

What exists now is sufficient for:

- inventory
- label-schema design
- anchor/window design
- planning a weakly supervised prototype

What does not exist yet is direct review evidence for:

- `springboard_dive` vs `platform_dive` at the event level
- `springboard_rebound_only` as a reviewed event-window class
- `noise_or_other` as a consistently reviewed event-window class
- event boundaries or delayed-entry timing inside a reviewed window

So:

- proceed directly to Phase 2
- do not begin Phase 3 dataset construction as if the four target classes were already directly labeled

## Inventory Summary

There are `47` local output roots with `evaluation_review.json`.

Those roots collapse to a much smaller set of underlying reviewed source sessions:

1. `insep_15min.mov`
2. `IMG_9015.MOV`
3. `Champigny.mov`
4. `IMG_8852.MOV`

Many other roots are replayed or derived runs whose `source_video_path` is a local review proxy such as `web/session_source_review.mp4` or a derived `session_audio.wav`. Those are useful for baseline simulation and diagnostics, but they are not new human-reviewed source sessions.

## Session Table

| Session Path / ID | Likely Type | Reviewed Counts | Artifact Completeness | Suitable For Event-Window Dataset |
|---|---|---:|---|---|
| `outputs/evaluation_insep_15min_raw` | `springboard` (directly documented) | 45 dive, 88 non-dive, 16 FN | Strong: direct source path, full review, full proposal artifacts, detections, raw/front-end/diagnostic JSONL | Yes |
| `outputs/evaluation_insep_15min_validated` | `springboard` (directly documented, replayed onto validated baseline) | 45 dive, 88 non-dive, 16 FN | Strong: validated baseline root with full review export and full proposal artifacts | Yes |
| `outputs/evaluation_insep_quick_9015_20260409_ui` | `platform` (inferred from repo narrative about earlier "platform INSEP"; not stated by path itself) | 36 dive, 35 non-dive, 7 FN | Mixed: full review/export, but this root does not contain `proposal_raw_peaks.jsonl` or `proposal_frontend_candidates.jsonl` | Maybe |
| `outputs/evaluation_champigny_20260406-labelling` | `springboard` (inferred from 68/70 non-dives labeled `board_rebound`; not explicitly named springboard in docs) | 41 dive, 70 non-dive, 3 FN | Mixed: full review/export and diagnostics, but missing `proposal_raw_peaks.jsonl` and `proposal_frontend_candidates.jsonl` in this root | Maybe |
| `outputs/evaluation_priority123_model_20260406-184900` | `unknown` | 2 dive, 2 non-dive, 1 unsure, 1 FN | Weak: tiny session, partial review value only, no proposal raw/front-end artifacts | No |

## Which Reviewed Sessions Exist Locally

### Direct reviewed source-session roots

These are the roots that matter most for event-architecture work:

1. [outputs/evaluation_insep_15min_raw](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_raw)
   - source video path in manifest: `/volumes/videos/insep quick/insep_15min.mov`
   - direct human review present
   - full proposal artifacts present

2. [outputs/evaluation_insep_quick_9015_20260409_ui](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_ui)
   - source video path in manifest: `/Volumes/Videos/Insep quick/IMG_9015.MOV`
   - direct human review present
   - review/export artifacts present
   - proposal artifacts are incomplete in this root

3. [outputs/evaluation_champigny_20260406-labelling](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260406-labelling)
   - source video path in manifest: `/Volumes/Videos/Champigny.mov`
   - direct human review present
   - review/export artifacts present
   - proposal artifacts are incomplete in this root

4. `IMG_8852` roots
   - [outputs/evaluation_priority123_model_20260406-184900](/Users/mcauchy/divesensei-app/outputs/evaluation_priority123_model_20260406-184900)
   - [outputs/evaluation_priority123_20260406-184700](/Users/mcauchy/divesensei-app/outputs/evaluation_priority123_20260406-184700)
   - [outputs/evaluation_priority123_champigny_model](/Users/mcauchy/divesensei-app/outputs/evaluation_priority123_champigny_model)
   - source video path: `/Users/mcauchy/Downloads/IMG_8852.MOV`
   - this is a small reviewed sample, not a primary session for the new architecture

### Replayed / derived reviewed roots

The repo also contains many replayed derivatives that reuse review data against alternative detector runs. These are useful for analysis, but they should not be mistaken for additional independent reviewed sessions.

Important derived groups:

- `insep_15min` replayed roots:
  - [outputs/evaluation_insep_15min_validated](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_validated)
  - [outputs/evaluation_insep_15min_experimental_145](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_experimental_145)
  - [outputs/evaluation_insep_15min_event_cluster_integration](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_event_cluster_integration)
  - [outputs/evaluation_insep_15min_event_cluster_veto](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_event_cluster_veto)

- `IMG_9015` replayed roots:
  - [outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020_validate](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020_validate)
  - [outputs/evaluation_insep_quick_9015_20260410_tailimbalance_417](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260410_tailimbalance_417)
  - [outputs/evaluation_insep_quick_9015_20260410_shortregion_145](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260410_shortregion_145)
  - plus other intermediate experiment roots

- `Champigny` replayed roots:
  - [outputs/evaluation_champigny_region_tiebreak_band020_validate](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_region_tiebreak_band020_validate)
  - [outputs/evaluation_champigny_20260410_tailimbalance_417](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260410_tailimbalance_417)
  - [outputs/evaluation_champigny_20260410_shortregion_145](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260410_shortregion_145)
  - plus other intermediate experiment roots

## Session Type Coverage

### Springboard

Directly documented:

- `insep_15min.mov`
  - documented in [docs/HANDOFF_2026-04-10_VALIDATED_BRANCH_AND_SPRINGBOARD_PIVOT.md](/Users/mcauchy/divesensei-app/docs/HANDOFF_2026-04-10_VALIDATED_BRANCH_AND_SPRINGBOARD_PIVOT.md) as "springboard-only"
  - review makeup explicitly includes `board_rebound: 68` and `voice_whistle: 20`

Inferred:

- `Champigny.mov`
  - not explicitly named springboard in the frozen handoff docs
  - inferred from review/export evidence: `68` of `70` non-dives are `board_rebound`
  - this is a strong inference, but still an inference

### Platform

Inferred:

- `IMG_9015.MOV`
  - the detector-status docs contrast the new springboard-heavy regime with earlier "platform INSEP"
  - the earlier INSEP session in the repo is `IMG_9015`
  - the file itself is not directly labeled "platform" in the local artifact names
  - treat this as an informed repo-level inference, not a direct session metadata fact

### Mixed

No session is directly documented as mixed in the local frozen docs.

### Unknown

- `IMG_8852.MOV`
  - no direct session-type documentation found locally
  - too small to anchor taxonomy decisions anyway

## Authoritative Frozen-Baseline References

These are the local roots that matter for the frozen validated baseline.

### Primary authoritative references

1. [outputs/evaluation_insep_15min_validated](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_validated)
   - full springboard validated baseline root
   - full proposal artifacts present
   - full review/export artifacts present
   - best local source for proposal-generator-plus-event simulation later

2. [outputs/evaluation_insep_quick_9015_20260410_tailimbalance_417](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260410_tailimbalance_417)
   - final frozen validated branch for the older INSEP session after tail-imbalance integration
   - proposal artifacts present
   - review/export artifacts present
   - reviewed coverage here is replayed and sparse (`15` dive, `1` non-dive), so this is a strong baseline reference but a weak training-label root

3. [outputs/evaluation_champigny_20260410_tailimbalance_417](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260410_tailimbalance_417)
   - frozen as part of the final validated branch story in the handoff docs
   - local artifact materialization is partial: `evaluation_review.json`, `detections.csv`, and manifest exist, but the export summary and proposal JSONL files are missing locally

### Complete fallback references for full local diagnostics

Because the final Champigny `tailimbalance_417` root is incomplete locally, the following earlier validated roots remain the best complete local references for full artifact analysis:

1. [outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020_validate](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020_validate)
2. [outputs/evaluation_champigny_region_tiebreak_band020_validate](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_region_tiebreak_band020_validate)

This does not change the frozen baseline definition. It only means the complete local artifacts for Champigny are better in the earlier validate root than in the later tail-imbalance root.

## Which Trained Models Exist Locally

Local model files under `.divesensei-runtime/models/`:

1. [.divesensei-runtime/models/audio_clip_model_insep_20260409.json](/Users/mcauchy/divesensei-app/.divesensei-runtime/models/audio_clip_model_insep_20260409.json)
   - training rows: `78`
   - positives: `43`
   - negatives: `35`

2. [.divesensei-runtime/models/audio_clip_model_champigny_20260406.json](/Users/mcauchy/divesensei-app/.divesensei-runtime/models/audio_clip_model_champigny_20260406.json)
   - training rows: `114`
   - positives: `44`
   - negatives: `70`

3. [.divesensei-runtime/models/audio_clip_model_validation.json](/Users/mcauchy/divesensei-app/.divesensei-runtime/models/audio_clip_model_validation.json)
   - tiny validation-only artifact
   - training rows: `4`
   - not useful as a serious event-model input source

These are existing clip-level detector models, not event-window models.

## What Labels Are Actually Available

### Direct review labels already present

From `evaluation_review.json` and `reviewed_candidates.jsonl`, the repo has direct reviewed labels for detected candidates:

- `dive`
- `non_dive`
- `unsure`

Direct non-dive subtypes observed locally:

- `board_rebound`
- `voice_whistle`
- `handling_noise`
- `non_dive_splash`
- `null` subtype in some roots

Example direct label coverage:

- `evaluation_insep_15min_validated`
  - `45` `dive`
  - `68` `non_dive / board_rebound`
  - `20` `non_dive / voice_whistle`

- `evaluation_insep_quick_9015_20260409_ui`
  - `36` `dive`
  - `23` `non_dive / handling_noise`
  - `5` `non_dive / board_rebound`
  - `5` `non_dive / voice_whistle`
  - `2` `non_dive / null`

- `evaluation_champigny_20260406-labelling`
  - `41` `dive`
  - `68` `non_dive / board_rebound`
  - `1` `non_dive / non_dive_splash`
  - `1` `non_dive / null`

### Direct false-negative review labels

From `falseNegatives` and `exports/evaluation-review/false_negatives.jsonl`, the repo has direct reviewed dive-miss windows:

- `evaluation_insep_15min_validated`: `16`
- `evaluation_insep_quick_9015_20260409_ui`: `7`
- `evaluation_champigny_20260406-labelling`: `3`

These false-negative rows already carry:

- reviewed dive timestamp
- review window start/end
- failure attribution fields in exported JSONL

This is important because missed dives can be treated as directly reviewed positive event neighborhoods later.

### Per-candidate metadata already present

`reviewed_candidates.jsonl` contains useful candidate-level metadata for later event-window work:

- source session and source path
- timestamp
- review window start/end
- review label and subtype
- confidence
- proposal frontend
- proposal stage
- clip-level features
- classifier probabilities when available

`candidate_diagnostics.jsonl` adds:

- selected/not-selected status
- pipeline stage
- failure type
- detailed score/features in some roots

`false_negative_neighborhoods.jsonl` exists for:

- `evaluation_insep_15min_validated`
- `evaluation_insep_quick_9015_20260409_ui`

and gives local frontend score traces around reviewed false negatives.

### What is not directly labeled

The following target classes are not directly reviewed today:

1. `springboard_dive`
   - current reviewed positives are generic `dive`
   - turning them into `springboard_dive` requires session-type inference or future relabeling

2. `platform_dive`
   - current reviewed positives are generic `dive`
   - turning them into `platform_dive` requires session-type inference or future relabeling

3. `springboard_rebound_only`
   - `non_dive / board_rebound` is the closest direct evidence
   - but this is still candidate-level, not reviewed event-window labeling

4. `noise_or_other`
   - partial direct evidence exists through `voice_whistle`, `handling_noise`, `non_dive_splash`, and some unlabeled `non_dive`
   - the unified event-window class would still require mapping/inference

Also missing:

- explicit event anchor labels
- explicit entry timestamps
- explicit board-contact timestamps
- explicit event start/end boundaries
- reviewed cluster-level annotations saying which peak belongs to the same dive event

## Are Current Labels Sufficient For Event-Window Training Now

### Short answer

Not for a clean direct four-class training set.

Partially yes for a weakly supervised prototype.

### What is sufficient now

Current labels are already sufficient to support:

- event-window schema design
- anchor-strategy design
- a weakly supervised prototype built from reviewed detections and reviewed false negatives
- a first offline binary or coarse multi-class feasibility pass, if the mapping rules are explicitly declared as inferred

Examples of immediately available weak supervision:

- reviewed `dive` candidates in springboard-only session -> likely `springboard_dive` by session-type inference
- reviewed `dive` candidates in likely platform INSEP session -> likely `platform_dive` by session-type inference
- reviewed `non_dive / board_rebound` -> likely `springboard_rebound_only` candidates, but still not guaranteed event-level purity
- reviewed `non_dive / voice_whistle` and `handling_noise` -> likely `noise_or_other`
- reviewed false negatives -> positive dive event windows, but still not directly typed as springboard/platform unless session type is known

### What is not sufficient now

Current labels are not sufficient for a clean claim that the repo already has direct reviewed labels for:

- all four target event classes
- event-level archetype boundaries
- archetype labels independent of session-level inference

So any Phase 3 dataset built immediately would need explicit label provenance such as:

- `direct_review`
- `session_type_inferred`
- `subtype_mapped`
- `uncertain`

## Additional Labeling Required

### Minimum additional labeling to make the four-class task defensible

1. Confirm session type for ambiguous reviewed sessions.
   - `IMG_9015` should be explicitly marked platform or not
   - `Champigny` should be explicitly marked springboard or not
   - right now both are evidence-backed inferences, not direct metadata facts

2. Add event-archetype labels for reviewed dive windows.
   - `springboard_dive`
   - `platform_dive`

3. Review a subset of `board_rebound` windows at the event level.
   - confirm they are truly `springboard_rebound_only`
   - confirm there is no valid delayed entry inside the intended event window

4. Normalize the negative event taxonomy.
   - map `voice_whistle`, `handling_noise`, `non_dive_splash`, and null-subtype `non_dive` into a controlled event-level negative taxonomy
   - decide whether the first prototype uses one bucket (`noise_or_other`) or preserves subtypes for analysis only

### Useful but optional extra labeling

1. Add approximate event anchor or entry timestamps for a subset of dives.
   - this would make anchor-strategy evaluation much more rigorous

2. Add event-window certainty grades.
   - direct
   - inferred
   - ambiguous

3. Review a small sample of multi-peak springboard clusters explicitly at the event level.
   - this would help test whether a proposal-centered window really covers the intended event

## Practical Recommendation After Phase 1

Proceed directly to Phase 2.

Reason:

- the repo has enough reviewed evidence to define the event-window task carefully
- the repo does not yet have enough direct four-class event labels to skip schema work and jump straight to dataset building

So the next step should be:

- define the label taxonomy
- define what counts as direct vs inferred labels
- define the starting anchor strategy and window length

What should not happen next:

- building a training manifest that silently treats session-type inference as direct truth
- claiming current review artifacts already provide a clean four-class event dataset
