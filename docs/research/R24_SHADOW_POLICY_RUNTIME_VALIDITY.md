# R24 Shadow Policy Runtime Validity

r24 remains the best reviewed-data shadow candidate, but it is not a live runtime-valid approve policy.

- classification: `mixed_runtime_and_reviewed_leakage`
- active/default policy remains: `approve_review_v1`
- r24 use allowed now: reviewed-data shadow analytics and post-review telemetry
- r24 use not allowed now: live automatic approve expansion before review

## Critical Distinction

The r24 guard depends on `subtype`. In the current app this subtype comes from `evaluation_review.json`, meaning it is human-reviewed metadata. That makes the policy safe as an audit/shadow report after review, but invalid as a pre-review automation policy.

## Before Any Rollout

- Emit governed r9_score for every candidate before review, distinct from detector audio_model_probability.
- Generate visual_late_fusion_logreg_c0.5 before review for every candidate eligible for shadow policy evaluation.
- Replace human-reviewed subtype suppression with a runtime nuisance-subtype predictor or nuisance-risk score, or restrict r24 to post-review analytics only.
- Re-run source-aware shadow evaluation using only pre-review/runtime-available fields.
- Keep approve_review_v1 as default until the runtime-only policy has zero dangerous/suspicious additions on independent sources.
