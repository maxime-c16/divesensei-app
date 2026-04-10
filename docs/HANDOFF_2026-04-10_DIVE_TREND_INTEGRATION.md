# Dive Trend Integration Handoff

Date: 2026-04-10

Purpose: freeze the current validated detector baseline, document the bounded `dive_trend_score` experiment, and hand off the next Mac-side replay/export step without changing the validated branch.

## 1. Current Validated Default Branch

Keep this branch frozen.

Enabled components:

- `frontend_region_descriptor_enabled`
- `frontend_region_descriptor_pattern_tiebreak_band = 0.20`
- `frontend_region_pattern_exception`
- `frontend_dense_pcen_pattern_exception`
- `frontend_region_tail_imbalance_exception`

Not part of the validated default:

- `frontend_short_region_tail_exception`
- `frontend_dive_trend_enabled`

Operational rule:

- this branch remains the baseline reference
- do not modify it
- do not stack new exceptions on top of it unless they are explicitly isolated and evaluated

## 2. Current Experiment

Experiment name:

- `dive_trend_score`

Integration mode:

- post-threshold only
- rank-only / late selection influence
- bounded
- opt-in

Flags used:

- `frontend_dive_trend_enabled = true`
- `frontend_dive_trend_weight = 0.20`
- `frontend_dive_trend_max_bonus = 0.12`

What it uses:

- long-window trend features over approximately `[-1.0s, +2.0s]`
- `flatness_slope`
- `centroid_slope`
- `hf_lf_slope`
- `time_to_peak`
- `cluster_density`

What it does not do:

- it does not participate in threshold promotion
- it does not change classifier acceptance
- it does not create a new acceptance path
- it does not affect pre-threshold behavior
- it does not add a new gate or exception

## 3. Code State

Files modified for the experiment:

- `src/divesensei/detection/audio_detector.py`
- `src/divesensei/detection/config.py`
- `src/divesensei/app/session_pipeline.py`

Behavior summary:

- long-window trend features are computed on candidate proposals only
- a bounded bonus is derived from the trend score
- the bonus is used only in proposal ranking
- the baseline candidate count and threshold promotion path remain unchanged

## 4. Measured Result

Output artifact:

- `outputs/analysis_dive_trend_integration.json`
- `outputs/analysis_dive_trend_integration.md`

Classification:

- `C - No effect`

Measured structural summary:

- validated baseline false negatives: `16`
- baseline reviewed FP/min: `6.163321002800342`
- baseline replay coverage: `1.0`
- baseline frontend candidates: `240`
- baseline final proposals: `133`
- experiment frontend candidates: `240`
- experiment final proposals: `134`
- experiment threshold promotions: `0`
- nearby FN candidate gain: `0`
- nearby FN proposal gain: `0`
- FN rank improvements: `0`
- FN top-1 changes: `0`

Interpretation:

- the representation is valid offline
- the first bounded rank-only integration did not produce measurable detector-stage improvement on the springboard session
- the experiment remains non-promoted

## 5. Replay / Export Limitation On This Machine

Important limitation:

- the new run replay/export path on this machine produced zero reviewed decisions
- because of that, new-run FP/min and replay coverage are not measurable here

Baseline review/export values that should be treated as authoritative for comparison:

- validated INSEP reviewed candidates: `133`
- validated INSEP false negatives: `16`
- validated INSEP reviewed FP/min: `6.163321002800342`
- validated INSEP replay coverage: `1.0`

## 6. What The Mac Should Do Next

The Mac should be used for the final replay/export judgment of this branch.

Recommended next steps:

1. Re-run `evaluate-session` with the validated baseline and the `dive_trend_score` flags enabled.
2. Run `replay-evaluation-review`.
3. Run `export-evaluation-review`.
4. Compare:
   - false negatives
   - practical unresolved counts at `±1.0s`
   - nearby frontend candidates
   - nearby final proposals
   - reviewed FP/min
   - replay coverage

Decision rule:

- keep the experiment non-promoted unless the Mac-side replay/export shows a real improvement
- do not treat the current local structural result as detector improvement

## 7. Frozen Baseline Reminder

Do not change the validated branch while evaluating this experiment.

The validated branch remains the reference baseline for:

- INSEP
- Champigny
- future replay/export comparisons

## 8. Post-Check Note

This cluster-selection experiment is safe and inert. It produced no final winner changes, no suppression changes, and no recall gain. The cluster-selection line looks close to exhausted unless a less bounded or different structural approach is justified later.

