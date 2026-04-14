# Platform/Noise ES4 Feature Attribution

## Global importance (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | `inter_peak_interval_cv` | 2.445173 |
| 2 | `tail_half_life_ms` | 2.196725 |
| 3 | `impact_peak_prominence_db` | 2.144932 |
| 4 | `onset_density_0_300ms_post` | 1.958612 |
| 5 | `spectral_entropy_post_mean` | 1.929296 |
| 6 | `onset_tempogram_peak_ratio_post` | 1.846910 |
| 7 | `impact_peak_to_window_rms_ratio` | 1.758446 |
| 8 | `whistle_band_energy_fraction_post` | 1.738183 |
| 9 | `audio_clip_probability` | 1.555050 |
| 10 | `audio_score` | 1.402292 |
| 11 | `spectral_contrast_low_high_slope_post` | 1.370361 |
| 12 | `spectral_flatness_post_mean` | 1.195887 |
| 13 | `tonal_peak_fraction_post_mean` | 1.176167 |
| 14 | `post_impact_early_to_late_rms_ratio` | 1.150145 |
| 15 | `spectral_contrast_mean_post` | 1.093512 |

## Local attribution for holdout errors

- `evaluation_insep_quick_9015_20260409_ui::det-0038` (platform_dive -> noise_or_other, p=0.2096): inter_peak_interval_cv=-1.1214, audio_score=-0.4431, tail_half_life_ms=-0.2884, tonal_peak_fraction_post_mean=-0.2350, onset_density_0_300ms_post=-0.1754
- `evaluation_insep_quick_9015_20260409_ui::det-0042` (platform_dive -> noise_or_other, p=0.3473): inter_peak_interval_cv=-1.1667, audio_score=0.2461, tail_half_life_ms=-0.2143, impact_peak_prominence_db=0.1941, whistle_band_energy_fraction_post=0.1912
- `evaluation_insep_quick_9015_20260409_ui::det-0062` (noise_or_other -> platform_dive, p=0.5585): inter_peak_interval_cv=0.7057, audio_score=-0.4858, tail_half_life_ms=-0.2723, tonal_peak_fraction_post_mean=-0.2604, transient_peak_count=-0.2400
- `evaluation_insep_quick_9015_20260409_ui::det-0014` (noise_or_other -> platform_dive, p=0.9641): inter_peak_interval_cv=0.7174, tail_half_life_ms=0.5965, onset_density_0_300ms_post=0.2465, whistle_band_energy_fraction_post=0.2311, audio_score=0.2203
- `evaluation_insep_quick_9015_20260409_ui::det-0022` (noise_or_other -> platform_dive, p=0.5249): inter_peak_interval_cv=0.6934, whistle_band_energy_fraction_post=-0.6044, spectral_flatness_post_mean=-0.4562, tail_half_life_ms=-0.3192, onset_tempogram_peak_ratio_post=0.2051
- `evaluation_insep_quick_9015_20260409_ui::det-0058` (noise_or_other -> platform_dive, p=0.5972): inter_peak_interval_cv=0.7572, audio_score=-0.4858, tail_half_life_ms=-0.2722, transient_peak_count=-0.2400, whistle_band_energy_fraction_post=0.2354
