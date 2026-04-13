# Platform/Noise Candidate Scored Slices

- source reviewed sessions scanned: `['evaluation_champigny_20260406-labelling', 'evaluation_insep_15min_validated', 'evaluation_insep_quick_9015_20260409_ui']`

| candidate_id | sessions | row_count | label_counts | acceptable | classification |
|---|---|---:|---|---|---|
| `champigny_current_platform_plus_ambiguity` | `['evaluation_champigny_20260406-labelling']` | `10` | `{'noise_or_other': 2, 'platform_dive': 8}` | `False` | `mixed_ambiguous` |
| `champigny_platform_only_stress` | `['evaluation_champigny_20260406-labelling']` | `8` | `{'platform_dive': 8}` | `False` | `clean_but_single_class` |
| `champigny_plus_insep15min_validated_noise` | `['evaluation_champigny_20260406-labelling', 'evaluation_insep_15min_validated']` | `28` | `{'platform_dive': 8, 'noise_or_other': 20}` | `True` | `mixed_cross_regime` |
| `insep_quick_stratified_holdout_candidate` | `['evaluation_insep_quick_9015_20260409_ui']` | `73` | `{'noise_or_other': 30, 'platform_dive': 43}` | `True` | `clean_anchor_session` |
| `all_reviewed_platform_noise_pool` | `['evaluation_champigny_20260406-labelling', 'evaluation_insep_15min_validated', 'evaluation_insep_quick_9015_20260409_ui']` | `103` | `{'noise_or_other': 52, 'platform_dive': 51}` | `True` | `broad_mixed_pool` |

- **champigny_current_platform_plus_ambiguity**: Contains only 10 rows and includes known ambiguity rows det-0001/det-0007; too fragile for scored guardbands.
- **champigny_platform_only_stress**: Fails two-class coverage gate (platform-only).
- **champigny_plus_insep15min_validated_noise**: Immediately satisfies two-class coverage with existing reviewed data and no relabeling, but mixes noise from springboard-root session.
- **insep_quick_stratified_holdout_candidate**: Strong two-class reviewed coverage with proposal-centered anchors; requires explicit train/validation repartition to avoid leakage.
- **all_reviewed_platform_noise_pool**: Max coverage but highest heterogeneity/drift from frozen regime intent.
