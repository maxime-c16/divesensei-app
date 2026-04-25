# R45 Fragment Scoring And Event-Center Geometry

- reference: `split_internal_gap_3s` (union recall `0.8558`, unmatched visual `4`).
- best candidate: `fragment_center_highest_score`
- best center strategy: `highest_score_frame`
- best support strategy: `none`
- best union recall: `0.8462`
- best recovered anchors: `7`
- best unmatched visual: `4`
- best false visual/min: `0.271`
- meaningful gain vs reference: `False`
- interpretation: fragment/event-center geometry was tested with bounded support/centering variants under fixed full-frame controls.
- interpretation: interval geometry remains the primary next lever.
- interpretation: prefilter is still premature.

## Decisions

- `R45_FRAGMENT_GEOMETRY_NO_CLEAR_GAIN`
- `R45_INTERVAL_GEOMETRY_REMAINS_PRIMARY`
