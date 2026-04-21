# R29 Runtime/Offline Score Equivalence

R29 compared repaired runtime `governed_r9_score` values against the governed offline r9 reference on rows replayed into r27 runtime sessions.

- Matched rows: `69`
- Pearson correlation: `0.24607682444167173`
- Spearman correlation: `0.2626987824258512`
- Threshold agreement: `0.8695652173913043`
- Runtime dangerous approvals: `1`
- Offline dangerous approvals: `1`

The runtime scorer is not equivalent to the governed offline reference. It should be replaced with exact governed model loading and exact feature extraction before any live approve decision uses widened runtime scoring.

Decisions:

- `R29_RUNTIME_OFFLINE_EQUIVALENCE_NOT_CONFIRMED`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
