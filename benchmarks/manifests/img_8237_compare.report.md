# DiveSensei Detector Validation

Manifest: `/home/mcauchy/divesensei-app/benchmarks/manifests/img_8237_compare.json`

## Detector Summary

### `audio_v1_heuristic`

- Existing cases: 1
- Pass rate: 0.0
- Precision: 1.0
- Recall: 0.25
- Mean runtime seconds: 0.742

### `audio_v2_pcen_classifier`

- Existing cases: 1
- Pass rate: 1.0
- Precision: 1.0
- Recall: 1.0
- Mean runtime seconds: 0.509

### `audio_v2_hybrid_video`

- Existing cases: 1
- Pass rate: 1.0
- Precision: 1.0
- Recall: 1.0
- Mean runtime seconds: 0.478

## Cases

### FAIL - `IMG_8237.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.7424604892730713`
- Predicted timestamps: `[5.088]`
- Notes: Pool-session hard case with close late dives and horn/clapping negative.

### PASS - `IMG_8237.MOV` [audio_v2_pcen_classifier]

- Detected events: `4`
- Runtime seconds: `0.5089535713195801`
- Predicted timestamps: `[2.16, 4.8, 19.248, 19.712]`
- Notes: Pool-session hard case with close late dives and horn/clapping negative.

### PASS - `IMG_8237.MOV` [audio_v2_hybrid_video]

- Detected events: `4`
- Runtime seconds: `0.47844552993774414`
- Predicted timestamps: `[2.16, 4.8, 19.248, 19.712]`
- Notes: Pool-session hard case with close late dives and horn/clapping negative.
