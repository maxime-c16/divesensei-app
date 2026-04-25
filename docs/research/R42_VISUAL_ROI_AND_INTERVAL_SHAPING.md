# R42 Visual ROI And Interval Shaping

## ROI Comparison

- full-frame recovered anchors: `5`
- full-frame union recall: `0.8269`
- full-frame unmatched visual proposals: `1`
- center-pool recovered anchors: `3`
- center-pool union recall: `0.8077`
- center-pool unmatched visual proposals: `2`

## Interval Shaping

- best interval variant: `tighter_grouping_plus_cap12`
- best interval variant union recall: `0.8558`
- best interval variant recovered anchors: `8`
- best interval variant unmatched visual proposals: `16`
- best interval variant max interval seconds: `9.5`

## Interpretation

- ROI value confirmed: `False`
- interval shaping primary bottleneck: `True`
- conclusion: `full_frame` is the better ROI on this session; `center_pool` is not the main lever.
- conclusion: interval shaping changes recall and interval width materially, but the current tighter variants buy recall by adding too many extra review items.
