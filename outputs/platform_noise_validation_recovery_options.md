# Platform/Noise Validation Recovery Options

- resolved status classification: **B**
- status statement: Springboard Phase5 passed; platform/noise is blocked by validation design/gating, so global Phase5 is not yet pass-complete.

| path | feasibility_now | risk | work | scientific_cleanliness | comparability |
|---|---|---|---|---|---|
| `A` | `high` | `medium` | `low` | `high_for_honesty_medium_for_global_closure` | `True` |
| `B` | `high` | `medium` | `medium` | `medium_high` | `partial` |
| `C` | `medium` | `low_medium` | `medium_high` | `high_if_row_selection_is_predeclared` | `False` |
| `D` | `medium` | `high_if_used_to_mask_blocker` | `medium` | `medium` | `partial` |

- **Path A (track_level_acceptance)**: Accept springboard as passed; mark platform/noise validation-blocked until valid scored slice exists. Honest status framing now, but does not unblock global Phase5 pass/fail.
- **Path B (build_new_scored_slice_from_existing_reviewed_data)**: Materialize a valid two-class platform/noise scored validation slice from existing reviewed manifests. Best immediate unblocker; requires explicit provenance and split-freeze update.
- **Path C (minimal_additional_human_review_for_platform_noise_slice)**: Add smallest targeted human review set to recover a clean two-class mixed validation slice. Current reviewed Champigny set lacks non-ambiguous noise depth; would require new reviewed rows (not yet materialized).
- **Path D (protocol_revision_for_track_scored_vs_reporting_only)**: Revise global Phase5 pass/fail rule when one track is scored and another is reporting-only. Useful only after explicit governance decision; should not replace missing platform/noise scored validation evidence.

- **Best next path: B** — It is the fastest scientifically defensible route to restore a valid two-class platform/noise scored slice using already reviewed data, without relabeling or detector/model changes.
- Recommended first candidate scored slice: `insep_quick_stratified_holdout_candidate` (fallback `champigny_plus_insep15min_validated_noise`)
