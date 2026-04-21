# R28 Runtime-Only Baseline

Default live policy: `approve_review_v1`.

## Runtime Sessions Used

| session_id | platform_noise_rows_used | governed_present_count | visual_present_count | visual_missing_count |
| --- | --- | --- | --- | --- |
| evaluation_r27_scorepath_insep_quick_v2 | 59 | 59 | 59 | 0 |
| evaluation_r27_scorepath_champigny_proxy | 10 | 10 | 10 | 0 |

## Availability

- Rows: 69
- Labels: `{"noise_or_other": 28, "platform_dive": 41}`
- Governed r9 present: 69/69
- Governed r9 nonzero: 69/69
- Visual score present: 69/69
- Visual score missing: 0

## approve_review_v1

- Approve count: 4
- Approve coverage: 0.0580
- Approve precision: 0.75
- Dangerous approvals: 1

Dangerous v1 rows:

```json
[
  {
    "session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "detection_id": "det-0039",
    "event_label": "noise_or_other",
    "governed_r9_score": 0.9421256462285259,
    "visual_late_fusion_logreg_c0.5": 0.9991066586590726,
    "audio_score": 17.266101837158203,
    "timestamp_seconds": 222.048
  }
]
```

Pre-r27, this could not be evaluated as a real live governed-score path because manifests had no governed r9 score. R28 confirms the score is now present and nontrivial on the included runtime sessions.
