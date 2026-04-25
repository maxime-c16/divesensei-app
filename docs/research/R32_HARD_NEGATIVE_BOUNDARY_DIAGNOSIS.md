# R32 Hard-Negative Boundary Diagnosis

R32 confirms that the post-r31 blocker is a true nuisance-boundary problem, not runtime parity.

Primary hard negative:

- `evaluation_r30_exact_scorepath_champigny_proxy::det-0007`
- reviewed label: `noise_or_other`
- subtype: `non_dive_splash`
- note: `shammy thrown`
- exact governed r9 score: `0.9423382878`
- visual late-fusion score: `0.9901774245`

Conclusion: the current visual late-fusion score is not sufficient as a runtime veto. The next bounded move is a hard-negative visual/splash morphology probe centered on this bank, without reviewed subtype leakage and without threshold tuning as the main mechanism.

Decision:

- `R32_HARD_NEGATIVE_DIAGNOSIS_COMPLETE`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
