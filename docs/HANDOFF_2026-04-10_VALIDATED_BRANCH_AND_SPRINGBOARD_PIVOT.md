# Validated Branch And Springboard Pivot Handoff

Date: 2026-04-10

Purpose: freeze the current Mac research state, document the validated default branch versus the experimental 145 branch, and hand off the next pivot on the school machine without redoing detector exploration.

## 1. Current Validated Default Branch

This remains the best validated default branch.

Enabled components:

- `frontend_region_descriptor_enabled`
- `frontend_region_descriptor_pattern_tiebreak_band = 0.20`
- `frontend_region_pattern_exception`
- `frontend_dense_pcen_pattern_exception`
- `frontend_region_tail_imbalance_exception`

Not part of the validated default:

- `frontend_short_region_tail_exception`

Why this remains the validated default:

- validated on INSEP and Champigny
- FP/min stayed stable on the validated recovery passes
- replay coverage stayed stable on the validated recovery passes
- recovered accepted detections:
  - `48.656411s`
  - `417.488888s`
- preserved evaluation/accounting recovery:
  - `114.350347s`
- did not regress Champigny

Operational interpretation:

- keep this branch frozen as the safest default for detector-side work
- use it as the baseline for any new family of research

## 2. Current Experimental-Only Branch

This is the validated default branch plus:

- `frontend_short_region_tail_exception`

Status:

- useful local movement on `145.714040s`
- no regression on the sessions where it was tested
- but it did not generalize on the new springboard-heavy session

Decision:

- keep as experimental only
- do not promote to default
- do not stack more 145-style exceptions as the main path forward

## 3. Current Miss Taxonomy

### Solved / recovered detector-side

- `114.350347s`
  - detector-side recovery already present
  - remains only an evaluation/accounting artifact when listed as a manual false negative
- `48.656411s`
  - recovered accepted detection
- `417.488888s`
  - recovered accepted detection

### Proposal formed but still downstream ambiguous

- `23.778071s`
- `398.422697s`
- `145.714040s` on the experimental 145 branch only
  - nearby accepted proposal but not accepted detection

These are not active proposal-formation targets anymore.

### Current unresolved detector-side targets after the last bounded-pattern cycle

On the older INSEP session, the remaining unresolved target at the end of the line was:

- `157.576774s`

However, the new springboard-heavy session changed the priority of future work. See Section 5.

## 4. Cross-Session Validation On The Springboard-Heavy Session

Session:

- source video: `/volumes/videos/insep quick/insep_15min.mov`
- raw reviewed session: [outputs/evaluation_insep_15min_raw](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_raw)

This session is springboard-only and is dominated by:

- board rebound
- voice / whistle / clap-like confusers
- close misses where the detector lands on the last board rebound instead of the splash

Review makeup:

- reviewed dives: `45`
- reviewed non-dives: `88`
- manual false negatives: `16`
- non-dive subtypes:
  - `board_rebound: 68`
  - `voice_whistle: 20`

### Branch A: validated default

Output:

- [outputs/evaluation_insep_15min_validated](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_validated)

### Branch B: experimental 145 branch

Output:

- [outputs/evaluation_insep_15min_experimental_145](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_experimental_145)

### Result

The two branches were effectively identical on this session.

Validated vs experimental:

- false negatives: `16` vs `16`
- practical `±1.0s` accepted detections: `1` vs `1`
- practical `±1.0s` accepted proposals but not detections: `0` vs `0`
- practical `±1.0s` unresolved: `15` vs `15`
- nearby frontend candidates: `13` vs `13`
- nearby final proposals: `13` vs `13`
- candidate count: `133` vs `133`
- replay coverage: `1.0` vs `1.0`
- median delta: `0.0` vs `0.0`
- review-level FP/min derived from labeled session: `6.163364169609245` vs `6.163364169609245`

Decision:

- the 145 branch did not generalize on the new springboard-heavy session
- it should remain experimental only

## 5. Why The 145 Branch Is Not Promoted

It helped locally on the earlier INSEP session:

- `145.714040s` moved from `no_proposal_generated` to a nearby accepted proposal / ambiguity case

But on the new springboard-heavy validation session:

- it recovered no new accepted detections
- it reduced no unresolved count
- it produced no new proposal-only gain
- it did not improve nearby-candidate counts

So the right interpretation is:

- it is a useful subtype-specific experiment
- it is not a general enough improvement to replace the validated default branch

## 6. Next Pivot For The School Machine

Pivot away from the 145 short-region subtype family.

Why:

- session-specific gain only
- no cross-session generalization on the springboard-heavy session

Keep the validated default branch frozen.

Why:

- currently the safest and best validated default

New active research family:

- springboard-heavy miss regime
- below-threshold misses
- pre-candidate capping / suppression
- rebound-heavy confusers

Explicit constraints for the next school-machine phase:

- do not continue stacking 145-style exceptions
- do not revisit solved cases as proposal problems:
  - `114.350347s`
  - `48.656411s`
  - `417.488888s`
- do not revisit ambiguity cases as proposal problems:
  - `23.778071s`
  - `398.422697s`
  - `145.714040s` on the experimental branch
- start from the validated default branch, not the experimental short-region branch

## 7. Required Models And Artifacts Available For Transfer

Machine-readable manifest:

- [docs/HANDOFF_2026-04-10_ARTIFACT_MANIFEST.json](/Users/mcauchy/divesensei-app/docs/HANDOFF_2026-04-10_ARTIFACT_MANIFEST.json)

Required model files:

- [.divesensei-runtime/models/audio_clip_model_insep_20260409.json](/Users/mcauchy/divesensei-app/.divesensei-runtime/models/audio_clip_model_insep_20260409.json)
- [.divesensei-runtime/models/audio_clip_model_champigny_20260406.json](/Users/mcauchy/divesensei-app/.divesensei-runtime/models/audio_clip_model_champigny_20260406.json)

Key reviewed sessions:

- [outputs/evaluation_insep_quick_9015_20260409_ui](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_ui)
- [outputs/evaluation_champigny_20260406-labelling](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260406-labelling)

Validated branch evaluation roots:

- [outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020_validate](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_region_tiebreak_band020_validate)
- [outputs/evaluation_champigny_region_tiebreak_band020_validate](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_region_tiebreak_band020_validate)
- [outputs/evaluation_insep_quick_9015_20260410_tailimbalance_417](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260410_tailimbalance_417)
- [outputs/evaluation_champigny_20260410_tailimbalance_417](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260410_tailimbalance_417)

Experimental 145 branch roots:

- [outputs/evaluation_insep_quick_9015_20260410_shortregion_145](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260410_shortregion_145)
- [outputs/evaluation_champigny_20260410_shortregion_145](/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260410_shortregion_145)

New springboard-heavy validation roots:

- [outputs/evaluation_insep_15min_raw](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_raw)
- [outputs/evaluation_insep_15min_validated](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_validated)
- [outputs/evaluation_insep_15min_experimental_145](/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_experimental_145)

Important note:

- these models and outputs are intentionally not tracked by git
- transfer them from the listed local paths to the school machine separately

## 8. Exact Commands To Resume From The Validated Branch

Validated default branch evaluation command skeleton:

```bash
PYTHONPATH=src python3 -m divesensei.cli evaluate-session \
  <reviewable-video-path> \
  --prepared-audio-path <cached-audio-wav> \
  --profile long-session \
  --detector-id audio_v2_pcen_classifier \
  --output-dir <target-output-dir> \
  --skip-review-proxy \
  --debug \
  --json \
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
  --frontend-region-pattern-exception-min-bonus 0.25 \
  --frontend-dense-pcen-pattern-exception-enabled \
  --frontend-dense-pcen-pattern-exception-min-score 7.0 \
  --frontend-dense-pcen-pattern-exception-min-prominence 7.0 \
  --frontend-dense-pcen-pattern-exception-min-post-flux-ratio 1.8 \
  --frontend-dense-pcen-pattern-exception-min-post-rms-ratio 2.3 \
  --frontend-dense-pcen-pattern-exception-min-nearby-peaks 15 \
  --frontend-dense-pcen-pattern-exception-max-flatness 0.5 \
  --frontend-region-tail-imbalance-exception-enabled \
  --frontend-region-tail-imbalance-exception-min-score 6.0 \
  --frontend-region-tail-imbalance-exception-min-prominence 5.8 \
  --frontend-region-tail-imbalance-exception-min-post-flux-ratio 1.9 \
  --frontend-region-tail-imbalance-exception-min-post-rms-ratio 1.2 \
  --frontend-region-tail-imbalance-exception-max-post-rms-ratio 1.35 \
  --frontend-region-tail-imbalance-exception-min-bonus 0.25 \
  --frontend-region-tail-imbalance-exception-min-late-over-early 2.8 \
  --frontend-region-tail-imbalance-exception-min-duration-above-1p10 0.14 \
  --frontend-region-tail-imbalance-exception-min-time-to-peak 0.18 \
  --frontend-region-tail-imbalance-exception-max-time-to-peak 0.24 \
  --frontend-region-tail-imbalance-exception-max-flatness 0.55
```

Replay/export skeleton:

```bash
PYTHONPATH=src python3 -m divesensei.cli replay-evaluation-review \
  <reviewed-session-source> \
  <new-run-target>

PYTHONPATH=src python3 -m divesensei.cli export-evaluation-review \
  <new-run-target>
```

## 9. Bottom-Line Decision

Default branch on the school machine:

- validated region tie-break + validated bounded exceptions

Do not promote:

- `frontend_short_region_tail_exception`

Next school-machine work:

- start from the validated default branch
- treat the short-region branch as reference only
- begin a new family aimed at:
  - below-threshold misses
  - pre-candidate capping / suppression
  - springboard-heavy confuser regime
