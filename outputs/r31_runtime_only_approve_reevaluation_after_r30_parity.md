# R31 Runtime-Only Approve Reevaluation After R30 Parity

R31 reruns the runtime-only approve benchmark after exact governed runtime/offline parity and governed 3-second runtime window scoring.

- rows: `69`
- labels: `{"noise_or_other": 28, "platform_dive": 41}`
- visual present: `69/69`

## Baseline v1

- approve count: `5`
- approve coverage: `0.0725`
- approve precision: `0.8`
- dangerous approvals: `1`
- dangerous rows: `[{"detection_id": "det-0007", "label": "noise_or_other", "r9": 0.9423382878303528, "session_id": "evaluation_r30_exact_scorepath_champigny_proxy", "subtype": "non_dive_splash", "timestamp_seconds": 516.72, "visual": 0.9901774245153963}]`

## Candidate Comparison

| policy | approvals | coverage | precision | dangerous | delta vs v1 |
|---|---:|---:|---:|---:|---:|
| `r9_score_gate::0.99` | 0 | 0.0000 | n/a | 0 | -0.0725 |
| `r9_score_gate::0.97` | 0 | 0.0000 | n/a | 0 | -0.0725 |
| `r9_score_gate::0.95` | 0 | 0.0000 | n/a | 0 | -0.0725 |
| `r9_score_gate::0.8` | 16 | 0.2319 | 0.9375 | 1 | 0.1594 |
| `r9_score_gate::0.84` | 13 | 0.1884 | 0.9231 | 1 | 0.1159 |
| `runtime_or_visual_gate::r9_0.84::visual_0.55` | 13 | 0.1884 | 0.9231 | 1 | 0.1159 |
| `runtime_or_visual_gate::r9_0.84::visual_0.70` | 13 | 0.1884 | 0.9231 | 1 | 0.1159 |
| `runtime_or_visual_gate::r9_0.84::visual_0.85` | 13 | 0.1884 | 0.9231 | 1 | 0.1159 |
| `runtime_or_visual_gate::r9_0.84::visual_0.95` | 13 | 0.1884 | 0.9231 | 1 | 0.1159 |
| `runtime_or_visual_gate::r9_0.84::visual_0.99` | 13 | 0.1884 | 0.9231 | 1 | 0.1159 |
| `r9_score_gate::0.86` | 12 | 0.1739 | 0.9167 | 1 | 0.1014 |
| `runtime_or_visual_gate::r9_0.86::visual_0.55` | 12 | 0.1739 | 0.9167 | 1 | 0.1014 |
| `runtime_or_visual_gate::r9_0.86::visual_0.70` | 12 | 0.1739 | 0.9167 | 1 | 0.1014 |
| `runtime_or_visual_gate::r9_0.86::visual_0.85` | 12 | 0.1739 | 0.9167 | 1 | 0.1014 |
| `runtime_or_visual_gate::r9_0.86::visual_0.95` | 12 | 0.1739 | 0.9167 | 1 | 0.1014 |

## Best Candidate

- policy: `approve_review_v1`
- approve count: `5`
- approve coverage: `0.0725`
- approve precision: `0.8`
- dangerous approvals: `1`

## Decisions

- `R31_RUNTIME_ONLY_APPROVE_REEVALUATION_NO_CLEAR_GAIN`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`