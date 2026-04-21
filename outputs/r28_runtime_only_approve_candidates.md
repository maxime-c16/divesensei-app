# R28 Runtime-Only Approve Candidates

Policy inputs allowed:

- `governed_r9_score`
- `visual_late_fusion_logreg_c0.5`

Explicitly excluded:

- reviewed subtype
- persisted human review metadata as a policy input
- event label as a policy input

Candidate count: 39

## Practical Candidate Rows

| policy_id | family | approve_count | approve_coverage | approve_precision | dangerous_approvals | approve_count_delta_vs_v1 |
| --- | --- | --- | --- | --- | --- | --- |
| ultra_conservative_r9_score_gate::0.95 | ultra_conservative_r9_threshold | 2 | 0.0290 | 1.0000 | 0 | -2 |
| ultra_conservative_r9_score_gate::0.97 | ultra_conservative_r9_threshold | 2 | 0.0290 | 1.0000 | 0 | -2 |
| ultra_conservative_r9_score_gate::0.99 | ultra_conservative_r9_threshold | 1 | 0.0145 | 1.0000 | 0 | -3 |
| r9_score_gate::0.86 | r9_threshold | 11 | 0.1594 | 0.9091 | 1 | 7 |
| r9_score_gate::0.84 | r9_threshold | 11 | 0.1594 | 0.9091 | 1 | 7 |
| runtime_or_visual_gate::r9_0.84::visual_0.55 | runtime_audio_visual_or_gate | 11 | 0.1594 | 0.9091 | 1 | 7 |
| runtime_or_visual_gate::r9_0.84::visual_0.70 | runtime_audio_visual_or_gate | 11 | 0.1594 | 0.9091 | 1 | 7 |
| runtime_or_visual_gate::r9_0.84::visual_0.85 | runtime_audio_visual_or_gate | 11 | 0.1594 | 0.9091 | 1 | 7 |
| runtime_or_visual_gate::r9_0.84::visual_0.95 | runtime_audio_visual_or_gate | 11 | 0.1594 | 0.9091 | 1 | 7 |
| runtime_or_visual_gate::r9_0.86::visual_0.55 | runtime_audio_visual_or_gate | 11 | 0.1594 | 0.9091 | 1 | 7 |
| runtime_or_visual_gate::r9_0.86::visual_0.70 | runtime_audio_visual_or_gate | 11 | 0.1594 | 0.9091 | 1 | 7 |
| runtime_or_visual_gate::r9_0.86::visual_0.85 | runtime_audio_visual_or_gate | 11 | 0.1594 | 0.9091 | 1 | 7 |
