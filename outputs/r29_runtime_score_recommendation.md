# R29 Runtime Score Recommendation

- Classification: `runtime_scorer_should_load_exact_governed_offline_model`
- Blocked by: `model_identity_mismatch, feature_vector_mismatch, score_calibration_mismatch`
- Dangerous r28 row offline dangerous: `False`

## Recommendation

Replace the runtime bootstrapped governed_r9_score proxy with exact governed r9 model loading and exact feature extraction, then rerun r28/r29 before using runtime scores for live approval decisions.

## Decisions

- `R29_RUNTIME_OFFLINE_EQUIVALENCE_NOT_CONFIRMED`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
