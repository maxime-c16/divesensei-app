# R47 Audio-Anchored Visual Window Verification

This pass reframes the visual task around candidate-window verification rather than exact splash/contact localization.

- audio candidates: `110`
- best cached-prompt aggregation: `max_score`
- best cached-prompt precision: `0.7241`
- best cached-prompt recall: `0.6462`
- nuisance rejection: `0.6444`

The cached baseline frame prompt is too permissive for verification. The product framing is correct, but the available cached prompt evidence does not yet support switching visual's primary role to verifier.
