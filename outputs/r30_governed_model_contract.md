# R30 Governed Model Contract

- Model id: `r9_compact_nuisance_generalization_weighted`
- Model family: `xgboost.XGBClassifier`
- Model artifact: `/Users/mcauchy/divesensei-app/.divesensei-runtime/models/r9_compact_nuisance_weighted/xgboost_model.json`
- Feature count: `21`
- Runtime threshold: `0.92158`

## Feature Ordering

1. `audio_score`
2. `audio_clip_probability`
3. `event_anchor_timestamp_seconds`
4. `is_false_negative_window`
5. `impact_peak_to_window_rms_ratio`
6. `impact_peak_prominence_db`
7. `transient_peak_count`
8. `inter_peak_interval_cv`
9. `post_impact_early_to_late_rms_ratio`
10. `tail_half_life_ms`
11. `spectral_flatness_post_mean`
12. `tonal_peak_fraction_post_mean`
13. `whistle_band_energy_fraction_post`
14. `spectral_entropy_post_mean`
15. `spectral_contrast_mean_post`
16. `spectral_contrast_low_high_slope_post`
17. `onset_tempogram_peak_ratio_post`
18. `onset_density_0_300ms_post`
19. `dominant_frequency_hz_post_std`
20. `spectral_rolloff_90_post_mean`
21. `zero_crossing_rate_post_mean`
