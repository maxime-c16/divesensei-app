# R46 Multi-Event Fragment Sub-Segmentation

- reference: `split_internal_gap_3s` (union recall `0.8558`, recovered `8`, unmatched `4`).
- best candidate: `reference_interval_anchor_passthrough`
- strategy: `single_reference`
- best union recall: `0.8558`
- best recovered anchors: `8`
- best unmatched visual: `4`
- best false visual/min: `0.271`
- meaningful gain vs reference: `False`
- interpretation: bounded multi-event sub-segmentation was tested under fixed full-frame controls.
- interpretation: interval geometry remains the primary next lever.
- interpretation: prefilter is still premature.

## Decisions

- `R46_SUBSEGMENTATION_NO_CLEAR_GAIN`
- `R46_INTERVAL_GEOMETRY_REMAINS_PRIMARY`
