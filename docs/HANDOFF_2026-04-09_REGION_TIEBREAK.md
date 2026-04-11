# Region Tie-Break Handoff

Date: 2026-04-09

Purpose: transfer the exact current detector and research state to another Mac that has the missing reviewed-session and replay/export assets.

This document is a state capture only. No new experiments are proposed here.

## 1. Current Best Detector Configuration

Current best-known configuration to resume on the Mac:

- Region descriptor: `ENABLED`
- Integration mode: `PATTERN TIE-BREAK ONLY`
- Tie-break band: `0.20`
- Region descriptor parameters:
  - `frontend_region_descriptor_weight = 2.0`
  - `frontend_region_descriptor_max_bonus = 0.6`
  - `frontend_region_descriptor_pre_seconds = 0.2`
  - `frontend_region_descriptor_post_seconds = 0.8`

Critical constraints:

- The region descriptor must NOT participate in threshold promotion.
- The region descriptor must NOT participate in `proposal_evidence_boost`.
- The region descriptor must NOT act before thresholding.
- It acts ONLY as a late pattern tie-break for already-formed near-threshold rows.

## 2. Current Implementation Summary

Files modified:

- [src/divesensei/detection/audio_detector.py](/home/macauchy/divesensei-app/src/divesensei/detection/audio_detector.py)
- [src/divesensei/detection/config.py](/home/macauchy/divesensei-app/src/divesensei/detection/config.py)
- [src/divesensei/app/session_pipeline.py](/home/macauchy/divesensei-app/src/divesensei/app/session_pipeline.py)

Where the logic lives:

- Region descriptor config fields:
  - [src/divesensei/detection/config.py#L137](/home/macauchy/divesensei-app/src/divesensei/detection/config.py#L137)
- CLI/config wiring:
  - [src/divesensei/app/session_pipeline.py#L154](/home/macauchy/divesensei-app/src/divesensei/app/session_pipeline.py#L154)
  - [src/divesensei/app/session_pipeline.py#L370](/home/macauchy/divesensei-app/src/divesensei/app/session_pipeline.py#L370)
- Region descriptor envelope + feature computation:
  - [src/divesensei/detection/audio_detector.py#L2048](/home/macauchy/divesensei-app/src/divesensei/detection/audio_detector.py#L2048)
  - [src/divesensei/detection/audio_detector.py#L2093](/home/macauchy/divesensei-app/src/divesensei/detection/audio_detector.py#L2093)
- Tie-break application point:
  - [src/divesensei/detection/audio_detector.py#L618](/home/macauchy/divesensei-app/src/divesensei/detection/audio_detector.py#L618)
- Proposal ranking path no longer includes region bonus in `proposal_evidence_boost`:
  - [src/divesensei/detection/audio_detector.py#L2665](/home/macauchy/divesensei-app/src/divesensei/detection/audio_detector.py#L2665)

Exact behavior:

- A region window is built around each raw peak using roughly `peak - 0.2s` to `peak + 0.8s`.
- The normalized envelope is computed as:
  - `0.65 * flux / EMA(flux, 0.35s) + 0.35 * rms / EMA(rms, 0.35s)`
- The fixed Family A linear descriptor uses:
  - `decay_slope`
  - `early_energy`
  - `mid_energy`
  - `late_energy`
  - `late_over_early`
  - `duration_above_1p10`
- The descriptor is mapped to a bounded region bonus:
  - `frontend_region_descriptor_bonus`

Tie-break activates only when all of these are true:

- `frontend_region_descriptor_bonus > 0`
- `threshold_passed == true`
- `timestamp_allowed == true`
- `hf_allowed == true`
- `score_allowed == true`
- `early_peak_allowed == false`
- `audio_pattern_score < audio_pattern_min_score`
- `audio_pattern_score >= audio_pattern_min_score - frontend_region_descriptor_pattern_tiebreak_band`

Current band:

- `frontend_region_descriptor_pattern_tiebreak_band = 0.20`

Application behavior:

- If eligible, the detector adds:
  - `frontend_region_pattern_tiebreak_bonus = frontend_region_descriptor_bonus`
- This bonus is added only to `audio_pattern_score`.
- It does not affect threshold crossing.
- It does not affect `proposal_evidence_boost`.

Operational consequence:

- Rejection stage can change from `weak_pattern_score` to `accepted`.
- Rejection stage cannot change from `below_threshold` to `accepted` through this mechanism.

## 3. Key Experiment Results

### A. Before Region Descriptor

Observed state:

- Weak INSEP misses mostly formed no usable proposal.
- `114.350347s` had no usable rescue path.
- In the earlier rescue attempts, the target neighborhood moved from `sustained_noise_reject` to `weak_pattern_score`, but still did not form a usable proposal.

Conclusion:

- Peak-based rescue heuristics were exhausted.
- No usable rescue path existed for the weak INSEP misses under the old representation.

### B. Broad Additive Region Integration

This integration is now rejected.

INSEP:

- Baseline:
  - `candidate_count = 78`
  - `frontend_candidate_count = 130`
  - `final_proposal_count = 78`
  - `evidence_threshold_promoted_count = 0`
- Broad additive region integration:
  - `candidate_count = 88`
  - `frontend_candidate_count = 175`
  - `final_proposal_count = 88`
  - `evidence_threshold_promoted_count = 749`

Champigny:

- Baseline:
  - `final_proposal_count = 111`
  - `frontend_candidate_count = 240`
  - heuristic accepted pre-candidates `494`
  - pcen accepted pre-candidates `402`
- Broad additive region integration:
  - `final_proposal_count = 101`
  - `frontend_candidate_count = 240`
  - `evidence_threshold_promoted_count = 6344`
  - heuristic accepted pre-candidates `1206`
  - pcen accepted pre-candidates `668`

Conclusion:

- Broad additive integration caused major funnel inflation.
- It destabilized upstream acceptance behavior.
- It is NOT usable.

### C. Tie-Break Integration

This is the successful direction.

Mechanism:

- No threshold promotion.
- No pre-threshold rescue.
- Region score is used only as a late pattern tie-break.

INSEP with tie-break band `0.35`:

- `candidate_count = 85`
- `frontend_candidate_count = 138`
- `final_proposal_count = 85`
- `evidence_threshold_promoted_count = 0`

Champigny with tie-break band `0.35`:

- `candidate_count = 111`
- `frontend_candidate_count = 240`
- `final_proposal_count = 111`
- `evidence_threshold_promoted_count = 0`

Local improvements:

- `114.350347s`
  - rescued via nearby heuristic row at `114.176s`
  - NOT rescued via the PCEN row at `114.352s`
- `398.422697s`
  - partial rescue via nearby heuristic row at `397.952s`
- `157.576774s`
  - no effect

Conclusion:

- Tie-break integration preserves Champigny stability.
- It keeps the region representation useful locally.

### D. Band Refinement: 0.35 -> 0.20

This is the current best state.

INSEP:

- Tie-break band `0.35`:
  - `final_proposal_count = 85`
  - `frontend_candidate_count = 138`
- Tie-break band `0.20`:
  - `final_proposal_count = 84`
  - `frontend_candidate_count = 137`

Champigny:

- Tie-break band `0.35`:
  - `final_proposal_count = 111`
  - `frontend_candidate_count = 240`
  - tie-break activations `138`
  - tie-break accepted rows `60`
- Tie-break band `0.20`:
  - `final_proposal_count = 111`
  - `frontend_candidate_count = 240`
  - tie-break activations `70`
  - tie-break accepted rows `44`

Preserved at `0.20`:

- `114.176s` rescue still survives.
- `397.952s` rescue still survives.

Removed at `0.20`:

- non-converting tie-break activity such as the wasted `398.368s` activation.

## FINAL STATUS UPDATE

- validated branch confirmation: the region-descriptor tie-break branch at `0.20` is still the official working detector
- experimental branch status: `frontend_short_region_tail_exception` remains experimental and session-specific
- event-level integration failure summary: reranking, score blending, representative selection, delayed cluster selection, cluster winner replacement, and merge-stage veto all failed to integrate safely
- stop decision: stop this family; do not keep adding local event-level cluster modifications to the peak-first detector

