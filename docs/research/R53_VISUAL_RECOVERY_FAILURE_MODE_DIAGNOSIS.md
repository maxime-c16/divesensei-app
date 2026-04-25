# R53 Visual Recovery Failure Mode Diagnosis

This pass diagnoses the fixed r52 visual recovery recipe without changing prompt, ROI, FPS, geometry, product defaults, or approve policy.

- aggregate category summary: `{'audio_matched_visual_matched': {'count': 70, 'percentage_of_anchors': 0.2431}, 'audio_matched_visual_missed': {'count': 160, 'percentage_of_anchors': 0.5556}, 'audio_missed_visual_recovered': {'count': 17, 'percentage_of_anchors': 0.059}, 'audio_missed_visual_also_missed': {'count': 41, 'percentage_of_anchors': 0.1424}}`
- main cause: `low_temporal_sampling_density_and_prompt_false_negatives_near_audio_misses`
- recommended next lever: `NEXT_LEVER_AUDIO_LOCAL_VISUAL_BURST`
- product defaults: unchanged
- visual recovery remains research-only and retained for detection.
