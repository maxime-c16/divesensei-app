# R29 Runtime/Offline Score Alignment

- Runtime rows: `69`
- Matched rows: `69`
- Unmatched rows: `0`
- Labels: `{"noise_or_other": 28, "platform_dive": 41}`
- Sources: `{"evaluation_champigny_20260406-labelling": 10, "evaluation_insep_quick_9015_20260409_ui": 59}`

## Dangerous R28 Row

```json
[
  {
    "row_key": "evaluation_insep_quick_9015_20260409_ui::det-0037",
    "runtime_session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "runtime_detection_id": "det-0039",
    "label": "noise_or_other",
    "runtime_governed_r9_score": 0.9421256462285259,
    "offline_governed_r9_score": 0.239240825176239,
    "runtime_approved_v1": true,
    "offline_approved_v1": false,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9991066586590726,
    "score_delta_runtime_minus_offline": 0.7028848210522869
  }
]
```
