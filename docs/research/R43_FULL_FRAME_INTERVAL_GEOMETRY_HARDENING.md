# R43 Full-Frame Interval Geometry Hardening

## Control

- union recall: `0.8269`
- recovered anchors: `5`
- unmatched visual proposals: `1`
- interval max seconds: `72.5`

## Best Bounded Variant

- label: `split_internal_gap_3s`
- union recall: `0.8558`
- recovered anchors: `8`
- unmatched visual proposals: `4`
- false visual proposals / min: `0.271`
- interval max seconds: `24.5`

## Interpretation

- internal-gap splitting on the better `full_frame` ROI is the useful geometry lever.
- hard caps improve recall further, but they increase unmatched burden too sharply.
- bounded recommendation set: `split_internal_gap_4s, split_internal_gap_3s, split_gap_3s_plus_margin_valley_0.25`
- prefilter work is still premature because geometry-only hardening still buys meaningful utility.
