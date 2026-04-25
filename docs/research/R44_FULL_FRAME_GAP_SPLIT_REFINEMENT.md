# R44 Full-Frame Gap Split Refinement

- r43 best reference: `split_internal_gap_3s` with union recall `0.8558` and unmatched visual `4`.
- best r44 candidate: `gap_split_3.0s_cleanup_merge_0.5s`
- best r44 candidate union recall: `0.8558`
- best r44 candidate unmatched visual: `4`
- no clear gain vs r43 best: `True`
- interpretation: nearby split values sit on the same performance plateau.
- interpretation: conservative cleanup merge slightly reduces proposal count but does not materially change burden or recall.
- interpretation: interval geometry remains the next lever; prefilter is still premature.
