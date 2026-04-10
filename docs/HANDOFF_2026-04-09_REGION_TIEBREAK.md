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

Final state:

- Region descriptor enabled
- Pattern tie-break only
- Tie-break band `0.20`

This is the BEST CURRENT CONFIGURATION.

## 4. What This Mechanism Can And Cannot Do

Works for:

- near-threshold rows
- already threshold-passed candidates
- weak-but-structured neighborhoods where baseline thresholding already formed a candidate and only pattern rejection remains

Does NOT work for:

- below-threshold rows
- fundamentally weak proposal evidence
- upstream failures where `threshold_passed == false`
- cases like `157.576774s`

This is NOT a general recall solution.

## 5. Known Important Examples

- `114.350347s`
  - BEST SUCCESS CASE
  - rescued via nearby heuristic row at `114.176s`
  - not rescued via the PCEN row itself

- `398.422697s`
  - PARTIAL SUCCESS
  - rescued via nearby heuristic row at `397.952s`

- `157.576774s`
  - NO EFFECT
  - still too weak upstream

## 6. Limitations Of Current Machine

This machine does NOT have:

- reviewed session artifacts:
  - `outputs/evaluation_insep_quick_9015_20260409_ui`
- usable reviewed export metrics
- full replay/export capability for the target reviewed INSEP session
- trained model:
  - `.divesensei-runtime/models/audio_clip_model_insep_20260409.json`

Therefore this machine cannot truthfully measure:

- false negatives
- false_negative_nearby_frontend_candidate_count
- false_negative_nearby_final_proposal_count
- replay coverage
- reviewed FP/min

All results here are:

- proposal-stage only
- NOT final detector evaluation

## 7. Exact Artifacts To Transfer Or Reuse

Primary outputs to reuse:

- [outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020](/home/macauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020)
- [outputs/evaluation_champigny_8897_20260409_region_tiebreak_band020](/home/macauchy/divesensei-app/outputs/evaluation_champigny_8897_20260409_region_tiebreak_band020)

Important files inside each run directory:

- [proposal_diagnostics_summary.json](/home/macauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020/proposal_diagnostics_summary.json)
- [proposal_raw_peaks.jsonl](/home/macauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020/proposal_raw_peaks.jsonl)
- [proposal_frontend_candidates.jsonl](/home/macauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020/proposal_frontend_candidates.jsonl)

Champigny equivalents:

- [proposal_diagnostics_summary.json](/home/macauchy/divesensei-app/outputs/evaluation_champigny_8897_20260409_region_tiebreak_band020/proposal_diagnostics_summary.json)
- [proposal_raw_peaks.jsonl](/home/macauchy/divesensei-app/outputs/evaluation_champigny_8897_20260409_region_tiebreak_band020/proposal_raw_peaks.jsonl)
- [proposal_frontend_candidates.jsonl](/home/macauchy/divesensei-app/outputs/evaluation_champigny_8897_20260409_region_tiebreak_band020/proposal_frontend_candidates.jsonl)

Also useful as comparison references:

- [outputs/evaluation_insep_quick_9015_20260409_region_baseline](/home/macauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_baseline)
- [outputs/evaluation_insep_quick_9015_20260409_region_linear](/home/macauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_linear)
- [outputs/evaluation_insep_quick_9015_20260409_region_tiebreak](/home/macauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak)
- [outputs/evaluation_champigny_8897_20260409_region_baseline](/home/macauchy/divesensei-app/outputs/evaluation_champigny_8897_20260409_region_baseline)
- [outputs/evaluation_champigny_8897_20260409_region_linear](/home/macauchy/divesensei-app/outputs/evaluation_champigny_8897_20260409_region_linear)
- [outputs/evaluation_champigny_8897_20260409_region_tiebreak](/home/macauchy/divesensei-app/outputs/evaluation_champigny_8897_20260409_region_tiebreak)

## 8. What The Mac Should Do Next

The Mac agent must resume from the current frozen detector state.

Required next steps:

1. Re-run `evaluate-session` with:
   - region descriptor enabled
   - tie-break integration only
   - tie-break band `0.20`
2. Run:
   - `replay-evaluation-review`
   - `export-evaluation-review`
3. Measure:
   - false negatives
   - false_negative_nearby_frontend_candidate_count
   - false_negative_nearby_final_proposal_count
   - replay coverage
   - reviewed FP/min
4. Compare against:
   - retrained baseline
   - persistence-only branch

The Mac should use the current best config:

```bash
--frontend-region-descriptor-enabled \
--frontend-region-descriptor-weight 2.0 \
--frontend-region-descriptor-max-bonus 0.6 \
--frontend-region-descriptor-pre-seconds 0.2 \
--frontend-region-descriptor-post-seconds 0.8 \
--frontend-region-descriptor-pattern-tiebreak-band 0.20
```

## 9. Do Not Continue Exploration Here

This branch is now frozen for evaluation.

Do NOT continue tuning on this machine.

Do NOT add new features.

Do NOT modify the region descriptor further.

Do NOT restart heuristic exploration on this machine.

The next step is evaluation on the Mac with full replay/export capability.

## 10. Post-Validation Follow-Up: True Unresolved INSEP Misses

After evaluation cleanup, the active unresolved INSEP detector-side misses were:

- `23.778071s`
- `48.656411s`
- `145.714040s`
- `157.576774s`
- `417.488888s`

The cleaned boundary also established:

- `114.350347s` is already detector-side solved and should not be optimized further
- `398.422697s` has nearby proposal formation but remains an ambiguous downstream classifier case and should not be forced through acceptance

### One bounded follow-up experiment that worked

Implemented a new opt-in weak-pattern escape hatch for threshold-passed, region-supported, splash-like moderate events:

New config flags:

- `--frontend-region-pattern-exception-enabled`
- `--frontend-region-pattern-exception-min-score`
- `--frontend-region-pattern-exception-min-prominence`
- `--frontend-region-pattern-exception-min-post-flux-ratio`
- `--frontend-region-pattern-exception-min-post-rms-ratio`
- `--frontend-region-pattern-exception-min-bonus`

Tested configuration:

```bash
--frontend-region-descriptor-enabled \
--frontend-region-descriptor-weight 2.0 \
--frontend-region-descriptor-max-bonus 0.6 \
--frontend-region-descriptor-pre-seconds 0.2 \
--frontend-region-descriptor-post-seconds 0.8 \
--frontend-region-descriptor-pattern-tiebreak-band 0.20 \
--frontend-region-pattern-exception-enabled \
--frontend-region-pattern-exception-min-score 6.0 \
--frontend-region-pattern-exception-min-prominence 6.0 \
--frontend-region-pattern-exception-min-post-flux-ratio 1.5 \
--frontend-region-pattern-exception-min-post-rms-ratio 1.6 \
--frontend-region-pattern-exception-min-bonus 0.25
```

### Measured INSEP result

Baseline validated region tie-break run:

- candidate count: `19`
- false negatives: `7`
- nearby frontend candidates: `2`
- nearby final proposals: `2`
- practical `±1.0s` accepted detections: `1`
- practical `±1.0s` unresolved: `5`
- reviewed FP/min: `0.13769931976536035`
- replay coverage: `0.22535211267605634`

Bounded region-pattern exception run:

- candidate count: `22`
- false negatives: `7`
- nearby frontend candidates: `3`
- nearby final proposals: `3`
- practical `±1.0s` accepted detections: `2`
- practical `±1.0s` unresolved: `4`
- reviewed FP/min: `0.13769931976536035`
- replay coverage: `0.22535211267605634`

Exact recovered unresolved case:

- `48.656411s` -> nearby accepted detection at `48.096s`
- offset: `-0.5604108979997378s`
- frontend: `pcen_multiband`
- practical resolution bucket at `±1.0s`: `nearby_accepted_detection`

This is a real bounded detector-side improvement on a previously unresolved miss.

### Champigny sanity check

Run: `outputs/evaluation_champigny_region_exception_48`

Measured result:

- candidate count: `14` (unchanged)
- false negatives: `3` (unchanged)
- nearby frontend candidates: `0` (unchanged)
- nearby final proposals: `0` (unchanged)
- practical accepted detections at `±1.0s`: `0`
- reviewed FP/min: `0.01660957126544171` (unchanged)
- replay coverage: `0.12612612612612611` (unchanged)

So the exception improved INSEP without measurable Champigny regression.

### Current best branch after this follow-up

Preferred current experimental path is now:

1. validated region descriptor tie-break baseline
2. plus the bounded region-pattern exception above

### Remaining true unresolved detector-side targets after the new run

At `±1.0s`, these remain unresolved in INSEP:

- `23.778071s`
- `145.714040s`
- `157.576774s`
- `417.488888s`

And this remains proposal-formed but downstream ambiguous:

- `398.422697s`

### Recommendation

Continue only if targeting one of the four remaining unresolved cases above.

Do not spend more work on:

- `114.350347s`
- `48.656411s`
- `398.422697s`

The next highest-value unresolved target is still likely `417.488888s` if staying in the same heuristic family, but the recent bounded exception already shows the family is not exhausted yet.
