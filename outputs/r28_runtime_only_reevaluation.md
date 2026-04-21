# R28 Runtime-Only Approve Reevaluation

## Best Runtime-Only Candidate

- Policy: `ultra_conservative_r9_score_gate::0.95`
- Description: approve if governed_r9_score >= 0.95
- Approve count: 2
- Approve coverage: 0.0290
- Approve precision: 1.0
- Dangerous approvals: 0
- Coverage delta vs v1: -0.0290

## Candidate Comparison

| policy_id | approve_count | approve_coverage | approve_precision | dangerous_approvals | coverage_delta_vs_v1 |
| --- | --- | --- | --- | --- | --- |
| ultra_conservative_r9_score_gate::0.95 | 2 | 0.0290 | 1.0000 | 0 | -0.0290 |
| ultra_conservative_r9_score_gate::0.97 | 2 | 0.0290 | 1.0000 | 0 | -0.0290 |
| ultra_conservative_r9_score_gate::0.99 | 1 | 0.0145 | 1.0000 | 0 | -0.0435 |
| r9_score_gate::0.86 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| r9_score_gate::0.84 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| runtime_or_visual_gate::r9_0.84::visual_0.55 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| runtime_or_visual_gate::r9_0.84::visual_0.70 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| runtime_or_visual_gate::r9_0.84::visual_0.85 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| runtime_or_visual_gate::r9_0.84::visual_0.95 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| runtime_or_visual_gate::r9_0.86::visual_0.55 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| runtime_or_visual_gate::r9_0.86::visual_0.70 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |
| runtime_or_visual_gate::r9_0.86::visual_0.85 | 11 | 0.1594 | 0.9091 | 1 | 0.1014 |

## Visual Reliability

- Visual present: 69/69
- Visual missing: 0
- Presence rate: 1.0000
- Fallback: visual-gated candidate branches do not approve if visual score is missing; v1 r9-score approvals still apply.

## Conclusion

Runtime-only widened approval does not safely improve over v1 on this repaired-score benchmark.

Critical finding: the repaired runtime score implementation is not yet calibrated like the governed offline r9 reference. The live runtime v1 threshold approved one nuisance row, so widened runtime approval must remain blocked until score-path calibration is reconciled.

Final decisions:

- `R28_RUNTIME_ONLY_REEVALUATION_NO_CLEAR_GAIN`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
