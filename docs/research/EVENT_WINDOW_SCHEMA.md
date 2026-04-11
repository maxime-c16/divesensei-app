# Event Window Schema

Date: 2026-04-11

## Purpose

This document defines the Phase 2 event-window task for the next architecture line:

- validated detector as proposal generator
- event-window classifier as a separate offline layer

This document does not build the dataset and does not train a model.

It defines:

- the first-prototype taxonomy
- label provenance rules
- anchor strategy candidates
- window strategy
- evaluation targets
- Phase 2 non-goals

## 1. First-Prototype Event Taxonomy

The first prototype should use a four-class event taxonomy:

1. `springboard_dive`
2. `springboard_rebound_only`
3. `platform_dive`
4. `noise_or_other`

This is the right first taxonomy because it matches the frozen architectural conclusion:

- springboard and platform are different event archetypes
- rebound-only springboard events are a core confuser, not just generic noise
- non-dive clutter still needs a catch-all negative class

### 1.1 `springboard_dive`

Operational meaning:

- a springboard-origin event window that contains a real dive event
- expected structure may include:
  - board/contact onset
  - rebound cluster
  - delayed water entry

Important constraint:

- the class is event-level, not peak-level
- the classifier is allowed to see multiple peaks inside one window

Directly labelable now:

- not cleanly, not as a direct event archetype

What exists now:

- generic reviewed `dive` labels
- some reviewed dive sessions are strongly believed to be springboard

Initial labeling status:

- must initially rely on `session_type_inferred` for sessions believed to be springboard

### 1.2 `springboard_rebound_only`

Operational meaning:

- a springboard-origin event window dominated by board rebound structure
- no valid dive entry should be present in the intended event window

Why this class is separate:

- this is the core springboard confuser that broke the peak-first detector
- it should not be folded into generic noise

Directly labelable now:

- partially

What exists now:

- direct reviewed candidate subtype `non_dive / board_rebound`

Current limitation:

- these are candidate-level labels, not reviewed event-window labels
- some future windows may need confirmation that no delayed entry belongs to the same event

Initial labeling status:

- can start from `subtype_mapped`
- some rows may remain `uncertain` until event-window review exists

### 1.3 `platform_dive`

Operational meaning:

- a platform-origin dive event window
- expected structure may include:
  - weak or diffuse onset
  - less springboard-style rebound repetition
  - dominant late entry

Directly labelable now:

- not cleanly

What exists now:

- generic reviewed `dive` labels
- at least one reviewed INSEP session is believed to be platform-dominant from repo narrative

Current limitation:

- session type is inferred, not directly encoded in current review labels

Initial labeling status:

- must initially rely on `session_type_inferred`

### 1.4 `noise_or_other`

Operational meaning:

- a negative event window that is not a springboard rebound-only event and not a real dive
- examples may include:
  - `voice_whistle`
  - `handling_noise`
  - `non_dive_splash`
  - unlabeled generic `non_dive`

Why this class exists:

- the first prototype needs a practical negative bucket
- not all non-dive clutter needs subtype resolution in Phase 2

Directly labelable now:

- partially

What exists now:

- direct reviewed negative subtypes
- some generic `non_dive` rows with null subtype

Current limitation:

- this class is a controlled mapping from several existing negative labels

Initial labeling status:

- primarily `subtype_mapped`
- generic null-subtype negatives may be `uncertain` if they cannot be assigned confidently

## 2. Label Provenance Rules

Every future event-window row should carry an explicit provenance tag.

Recommended provenance categories:

1. `direct_review`
2. `session_type_inferred`
3. `subtype_mapped`
4. `uncertain`

### 2.1 `direct_review`

Meaning:

- the assigned event label is directly supported by existing human review without needing session-type conversion or subtype remapping

Examples:

- a future explicitly reviewed event-window label
- a directly reviewed false-negative dive window if later manually typed as springboard or platform

Important note:

- under current local artifacts, very few rows would qualify for the four-class taxonomy as `direct_review`

### 2.2 `session_type_inferred`

Meaning:

- the source review says `dive`, but the event archetype is assigned from the source session type

Examples:

- reviewed `dive` from a session believed to be springboard -> `springboard_dive`
- reviewed `dive` from a session believed to be platform -> `platform_dive`

Critical rule:

- this must never be silently treated as direct truth
- all reports later must separate `direct_review` metrics from inference-assisted metrics

### 2.3 `subtype_mapped`

Meaning:

- the event label is assigned by mapping an existing reviewed negative subtype into the event taxonomy

Examples:

- `non_dive / board_rebound` -> `springboard_rebound_only`
- `non_dive / voice_whistle` -> `noise_or_other`
- `non_dive / handling_noise` -> `noise_or_other`
- `non_dive / non_dive_splash` -> `noise_or_other`

Critical rule:

- mapped subtype labels are usable for prototype training and analysis
- but they still reflect a taxonomy mapping decision, not a direct event-window review

### 2.4 `uncertain`

Meaning:

- the row cannot be assigned cleanly to a target class from current evidence
- or the intended event window may contain ambiguous structure

Examples:

- generic `non_dive` with null subtype and no safe mapping
- board-rebound windows that may include possible delayed entry
- dive windows from sessions whose springboard/platform type remains unconfirmed

Usage rule:

- `uncertain` rows should not be silently dropped later without accounting
- they may be excluded from the first training split, but should remain visible in audit summaries

### 2.5 How Provenance Will Be Used Later

In Phase 3 and later:

- manifests should store both `event_label` and `label_provenance`
- summary stats should be reported by provenance category
- offline metrics should be reported at least two ways:
  - strict subset: `direct_review` only where possible
  - practical prototype subset: `direct_review + session_type_inferred + subtype_mapped`

This avoids the main failure mode of overclaiming from inferred labels.

## 3. Anchor Strategy Candidates

The first prototype should compare at least three anchor strategies:

1. proposal-centered
2. peak-centered
3. earliest strong peak in local cluster

### 3.1 Proposal-Centered

Definition:

- the event window is centered on the validated detector proposal timestamp
- in the hybrid architecture, the proposal generator provides the anchor candidate

Strengths:

- matches the intended future architecture exactly
- avoids re-entering peak-level coupling logic
- easy to simulate later with the validated detector unchanged
- naturally supports proposal-generator plus event-classifier evaluation

Weaknesses:

- if the proposal lands on the wrong peak inside a springboard cluster, the window must be long enough to capture the true event structure

Assessment:

- best primary choice for the first prototype

### 3.2 Peak-Centered

Definition:

- the event window is anchored on a raw or selected peak timestamp

Strengths:

- simple
- aligns with existing diagnostic artifacts built around peaks

Weaknesses:

- closest to the architecture that already hit its limit
- risks reintroducing the assumption that the peak is the event
- especially poor fit when rebounds dominate local competition

Assessment:

- useful as a control or ablation
- not recommended as the main prototype anchor

### 3.3 Earliest Strong Peak In Local Cluster

Definition:

- form a local cluster, then anchor the window on the earliest peak above a strength rule inside that cluster

Why it matters:

- springboard dive events often begin before the later splash-like winner
- this anchor has a better chance of capturing board/contact plus rebound progression

Strengths:

- more event-like than pure peak-centered anchoring
- may help when a later selected winner misses the true event onset

Weaknesses:

- requires a cluster definition and a strong-peak rule
- adds another layer of inference before classification
- less directly aligned with the lowest-risk future architecture than proposal-centered anchoring

Assessment:

- best backup anchor strategy

### 3.4 Phase 2 Recommendation

Primary anchor strategy:

- `proposal-centered`

Backup anchor strategy:

- `earliest_strong_peak_in_local_cluster`

Reason:

- the first prototype should test the intended new architecture boundary
- proposal-centered windows let the validated detector remain the proposal generator
- the earliest-strong-peak backup preserves a fallback if proposal-centered windows prove too late in rebound-heavy springboard cases

Peak-centered windows should remain an analysis control, not a recommended starting point.

## 4. Window Strategy

The first prototype should use asymmetric windows because both target regimes can require more post-anchor context than pre-anchor context.

Why asymmetry is needed:

- in springboard-heavy failures, the selected anchor may land on a rebound before the true splash
- in platform events, the onset may be weak while the decisive entry appears later

### 4.1 Primary Window Length

Recommended primary window:

- `3.0s total`
- `0.75s pre + 2.25s post`

Reasoning for springboard:

- gives enough pre-context to include local setup and some early board activity
- gives enough post-context to capture delayed entry after rebound-dominated onset

Reasoning for platform:

- platform events may have weak onset and stronger late-terminal entry
- longer post-context is more important than long pre-context

Why not shorter:

- the earlier event-level research window around peaks (`0.2s pre + 1.0s post`) was useful for offline analysis but too short and too peak-attached for the new architecture target

Why not longer by default:

- longer windows increase clutter and class overlap
- the first prototype should start with a compact but still event-aware window

### 4.2 Backup Window Length

Recommended backup window:

- `4.0s total`
- `1.0s pre + 3.0s post`

Why this is the backup:

- safer for long springboard chains where the proposal anchor lands early
- safer for platform windows if the decisive entry is substantially delayed

Why this is not primary:

- greater risk of mixing adjacent events and unrelated clutter
- higher chance of making the classification task noisier before learning whether the event-window idea is viable

### 4.3 Window Recommendation Summary

Primary:

- `3.0s total`
- `[-0.75s, +2.25s]` relative to anchor

Backup:

- `4.0s total`
- `[-1.0s, +3.0s]` relative to anchor

## 5. Event-Level Evaluation Targets

Success for the first prototype should be defined in three layers:

1. offline event classification quality
2. proposal-window ranking quality
3. future detector integration readiness

### 5.1 Offline Event Classification Quality

Question:

- can event windows separate the intended archetypes better than peak-first reasoning?

Minimum success condition:

- clear offline separation on the most important contrasts:
  - `springboard_dive` vs `springboard_rebound_only`
  - `platform_dive` vs `noise_or_other`

Recommended metrics:

- AUC for pairwise contrasts
- precision/recall for class decisions
- confusion matrix for the four-class prototype if label volume supports it

Reporting rule:

- metrics must be broken out by provenance where possible
- do not report inference-assisted metrics as if they were direct-review truth

### 5.2 Proposal-Window Ranking Quality

Question:

- given validated detector proposals, can the event classifier rank the proposal windows in a way that better surfaces true dive events and demotes rebound-only windows?

Minimum success condition:

- improves ranking inside proposal sets or dense local neighborhoods without modifying the detector funnel

Recommended measures:

- top-1 and top-k ranking within proposal neighborhoods
- dive-above-rebound ordering quality
- rebound suppression quality in proposal ranking terms
- springboard and platform sessions reported separately

### 5.3 Future Detector Integration Readiness

Question:

- is the event-window classifier strong enough offline to justify a later proposal-generator-plus-classifier simulation?

Readiness condition:

- offline event classification is materially better than peak-only heuristics on the key contrasts
- proposal-window ranking shows meaningful signal on frozen baseline sessions
- gains are visible without changing the validated detector behavior

Important limit:

- Phase 2 does not declare the architecture integrated
- it only defines what later evidence would justify simulation work in Phase 5

## 6. Explicit Non-Goals For Phase 2

Phase 2 does not assume:

- that current labels already form a clean four-class reviewed dataset
- that session-type inference is direct truth
- that the first prototype will work equally well on springboard and platform without data balancing
- that event windows should replace the current detector funnel directly

Phase 2 does not claim:

- live detector improvement
- integration readiness by itself
- a final model family choice
- a final anchor or window choice validated by experiment

Phase 2 is not doing:

- dataset construction
- feature extraction
- event-window extraction
- model training
- threshold tuning
- detector modification

## 7. Phase 2 Recommendation

Use the following configuration as the first prototype definition:

- taxonomy:
  - `springboard_dive`
  - `springboard_rebound_only`
  - `platform_dive`
  - `noise_or_other`
- primary anchor:
  - `proposal-centered`
- backup anchor:
  - `earliest strong peak in local cluster`
- primary window:
  - `3.0s total`
  - `0.75s pre + 2.25s post`
- backup window:
  - `4.0s total`
  - `1.0s pre + 3.0s post`
- mandatory label provenance:
  - `direct_review`
  - `session_type_inferred`
  - `subtype_mapped`
  - `uncertain`

This is the lowest-risk way to move from the frozen validated detector toward an event-aware architecture without repeating the failed live-integration pattern.
