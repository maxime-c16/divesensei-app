# R52 Visual Recovery Remote Scoring Expansion

This pass expands the fixed visual recovery recipe across additional reviewed sessions. It does not change approve policy, clip defaults, prompt, ROI, FPS, or interval geometry.

- remote health: `None`
- scored priority sessions: `['evaluation_CAO-1st-15min_20260421-072906', 'evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927', 'evaluation_SNMT-WED-8:4:26_20260419-142758']`
- skipped priority sessions: `[{'session_id': 'evaluation_champigny_20260406-labelling', 'reason': 'priority_session_not_scored_or_remote_artifact_missing'}, {'session_id': 'evaluation_insep_quick_9015_20260409_ui', 'reason': 'priority_session_not_scored_or_remote_artifact_missing'}]`
- evaluated sessions: `5`
- aggregate audio recall: `0.7986`
- aggregate union recall: `0.8333`
- aggregate recovered anchors: `10`
- conclusion: `generalizes_to_multiple_sources`
- next bottleneck: `algorithm_quality`

Final policy/product state remains unchanged: approve_review_v1 stays default and audio-anchor clip extraction remains the product default.
