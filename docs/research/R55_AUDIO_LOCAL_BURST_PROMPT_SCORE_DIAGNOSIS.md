# R55 Audio-Local Burst Prompt/Score Diagnosis

- remote VLM rerun: `False`
- best candidate: `lower_score_any_ge_0.70` at `8.0` FPS
- converted: `34/41`
- estimated union recall: `0.9514`
- added recovered over r54: `21`
- false unmatched controls: `5`

## CAO Finding
Best score-only rule converts 18/25 CAO A-V- targets. CAO improves only when thresholds are substantially relaxed, indicating prompt/model weakness remains.

## SNMT / INSEP Finding
Best score-only rule converts 16/16 SNMT/INSEP A-V- targets and preserves positive controls. These sessions are more score-threshold recoverable than CAO.

- decision: `R55_BURST_SCORE_AGGREGATION_GAIN`
- next recipe: `VISUAL_RECOVERY_RECIPE_AUDIO_LOCAL_BURST_WITH_SCORE_AGGREGATION`
