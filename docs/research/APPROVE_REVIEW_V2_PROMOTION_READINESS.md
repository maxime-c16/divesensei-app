# Approve Review v2 Promotion Readiness

`approve_review_v2_candidate` is the r18 approve-side expansion candidate.

## Candidate Logic

- approve if `r9_score >= 0.92158`
- or approve if `r9_score >= 0.70` and `visual_late_fusion_logreg_c0.5 >= 0.55`
- otherwise keep the row in `Needs review`

## r19 Decision

- generalization decision: `R19_APPROVE_V2_NOT_YET_PROMOTABLE`
- rollout decision: `APPROVE_REVIEW_V1_REMAINS_DEFAULT`

## Product Guidance

- `approve_review_v1` remains the visible default.
- `approve_review_v2_candidate` can be prepared for flagged shadow-mode evaluation only if the visual guard score is available in manifests.
- Do not introduce an auto-excluded lane.
- Do not replace v1 silently.

## Evidence

| validation | rows | v1 precision | v1 coverage | v1 danger | v2 precision | v2 coverage | v2 danger | delta coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_model_on_internal_official_holdout` | 20 | -1.0000 | 0.0000 | 0 | 1.0000 | 0.1000 | 0 | +0.1000 |
| `full_model_on_corrected_external_holdout` | 99 | 1.0000 | 0.1717 | 0 | 1.0000 | 0.2323 | 0 | +0.0606 |
| `full_model_on_source_unit_snmt` | 76 | 1.0000 | 0.2368 | 0 | 1.0000 | 0.4868 | 0 | +0.2500 |
| `full_model_on_source_unit_img_8852` | 11 | 1.0000 | 0.0909 | 0 | 1.0000 | 0.1818 | 0 | +0.0909 |
| `full_model_on_source_unit_champigny_1704` | 31 | -1.0000 | 0.0000 | 0 | -1.0000 | 0.0000 | 0 | +0.0000 |
| `leave_one_source_out_snmt` | 76 | -1.0000 | 0.0000 | 0 | -1.0000 | 0.0000 | 0 | +0.0000 |
| `leave_one_source_out_img_8852` | 11 | -1.0000 | 0.0000 | 0 | 0.0000 | 0.1818 | 2 | +0.1818 |
| `leave_one_source_out_champigny_1704` | 31 | -1.0000 | 0.0000 | 0 | -1.0000 | 0.0000 | 0 | +0.0000 |
