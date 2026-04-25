# R31 Exact Runtime Approve Boundary

R31 verifies the post-r30 state after exact governed runtime parity.

## Result

- Existing r31 follow-up was found and rerun: `benchmarks/r31_runtime_only_approve_reevaluation_after_r30_parity.py`.
- Canonical outputs were written as `outputs/r31_exact_runtime_approve_reevaluation.*` and `outputs/r31_boundary_truth_diagnosis.*`.
- Runtime/offline parity is no longer the blocker.
- No bounded runtime-only approve candidate safely improves over `approve_review_v1`.
- The current blocker is a true governed-model nuisance boundary issue: `evaluation_r30_exact_scorepath_champigny_proxy::det-0007`, label `noise_or_other`, subtype `non_dive_splash`, r9 `0.9423382878`, visual `0.9901774245`.

## Product State

`approve_review_v1` remains the configured/default policy. Do not expand approval or discuss rollout from this result. The next pass should focus on the high-score nuisance boundary, not plumbing or threshold search.

## Decisions

- `R31_EXACT_RUNTIME_REEVALUATION_NO_CLEAR_GAIN`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`