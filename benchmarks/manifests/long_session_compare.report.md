# DiveSensei Detector Validation

Manifest: `/home/mcauchy/divesensei-app/benchmarks/manifests/long_session_compare.json`

## Detector Summary

### `audio_v1_heuristic`

- Existing cases: 1
- Pass rate: 1.0
- Precision: n/a
- Recall: n/a
- Mean runtime seconds: 5.476

### `audio_v2_pcen_classifier`

- Existing cases: 1
- Pass rate: 1.0
- Precision: n/a
- Recall: n/a
- Mean runtime seconds: 9.552

## Cases

### PASS - `IMG_8281.MOV` [audio_v1_heuristic]

- Detected events: `51`
- Runtime seconds: `5.475554466247559`
- Predicted timestamps: `[4.192, 10.032, 14.736, 20.688, 54.096, 65.344, 70.736, 78.592, 98.048, 114.224, 129.952, 140.688, 147.504, 159.856, 170.352, 181.92, 191.136, 200.336, 215.648, 227.76, 243.936, 247.904, 271.952, 295.184, 317.312, 327.072, 332.304, 338.048, 360.32, 369.552, 374.512, 415.856, 424.288, 440.528, 449.44, 473.296, 483.632, 508.512, 513.36, 546.48, 553.008, 573.136, 582.016, 592.864, 605.264, 622.512, 628.0, 661.696, 676.992, 688.176, 708.608]`
- Notes: Real long session gate. Expected roughly 40 to 50 splash-driven dive events.

### PASS - `IMG_8281.MOV` [audio_v2_pcen_classifier]

- Detected events: `53`
- Runtime seconds: `9.551607608795166`
- Predicted timestamps: `[3.152, 4.192, 20.688, 21.44, 40.912, 49.104, 63.68, 65.344, 73.648, 103.392, 114.224, 129.904, 138.192, 138.784, 140.768, 147.504, 196.576, 200.336, 204.336, 215.648, 216.544, 243.936, 247.904, 248.608, 258.736, 270.16, 298.192, 326.896, 332.192, 338.048, 338.912, 369.488, 384.016, 400.48, 424.4, 425.216, 449.44, 459.856, 462.672, 483.632, 484.496, 508.672, 529.296, 539.408, 545.776, 546.48, 550.832, 553.008, 562.752, 576.304, 661.696, 662.304, 687.216]`
- Notes: Real long session gate. Expected roughly 40 to 50 splash-driven dive events.
