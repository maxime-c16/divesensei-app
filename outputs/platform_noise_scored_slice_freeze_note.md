# Platform/Noise Scored Slice Freeze Note (r4)

- source candidate: `insep_quick_stratified_holdout_candidate`
- scored slice policy: stratified holdout from INSEP quick reviewed platform/noise rows
- scored slice counts:
  - platform_dive: `10`
  - noise_or_other: `10`
  - total: `20`
- anchor consistency: proposal_centered for all scored slice rows
- leak prevention: scored holdout rows removed from platform/noise train rows
- Champigny platform-only and ambiguity slices remain reporting-only
- unchanged: detector behavior, taxonomy, labels, classifier family, springboard probe-r1 features
