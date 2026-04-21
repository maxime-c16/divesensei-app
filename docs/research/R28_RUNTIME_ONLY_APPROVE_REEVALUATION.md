# R28 Runtime-Only Approve Reevaluation

R28 is the first approve-lane reevaluation using the repaired r27 live score path.

The policy benchmark used only fields that can exist before review:

- `governed_r9_score`
- `visual_late_fusion_logreg_c0.5`

It did not use reviewed subtype or any persisted human-review metadata as policy input.

## Result

Best runtime-only candidate: `ultra_conservative_r9_score_gate::0.95`.

- v1 coverage: 0.0580
- best coverage: 0.0290
- best precision: 1.0
- best dangerous approvals: 0

Critical runtime finding: `approve_review_v1` itself produced 1 dangerous approval on the repaired runtime score path. This means r27 made the score path executable, but the bootstrapped runtime scorer is not yet product-equivalent to the governed offline r9 model.

## Decision

- `R28_RUNTIME_ONLY_REEVALUATION_NO_CLEAR_GAIN`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
