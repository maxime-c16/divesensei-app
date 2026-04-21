# Approve/Review v1 Product Mode

`approve_review_v1` reframes the current product workflow as a review tool with a trusted fast lane.

The app should not present the current system as a full autonomous triage tool. The governed evidence supports:

- `Auto-approved`: a narrow set of high-confidence platform-dive rows.
- `Needs review`: all other rows.

The governed evidence does not support:

- a trusted `Auto-excluded` lane.
- high-coverage automation claims.
- symmetric three-queue triage as a product promise.

## Policy Definition

- policy id: `approve_review_v1`
- model reference: `r9_compact_nuisance_generalization_weighted`
- score field: `scores.audio_model_probability`
- approve threshold: `0.92158`
- source experiment: `r17_high_precision_approve_coverage_benchmark`

Rows with `r9_score >= 0.92158` can be shown as auto-approved. Rows below the threshold remain review-required.

## Governed Metrics

The accepted r17 best-safe policy was `score_gate::r9_score::0.922`, concretely `r9_score >= 0.92158`.

| metric | value |
|---|---:|
| external approve precision | `1.0000` |
| external approve count | `17` |
| external approve coverage | `0.1717` |
| dangerous external auto-approves | `0` |
| dangerous internal auto-approves | `0` |

This improved coverage over the r16 best-safe approve point from `0.0909` to `0.1717`, while preserving zero dangerous approve errors.

## App Behavior

The review UI should:

- expose `Auto-approved` as a high-confidence lane.
- keep `Needs review` as the main workflow.
- avoid implying that low-score rows are safely auto-excluded.
- keep all non-approved rows visible and reviewable.
- show policy metadata where practical: policy id, model reference, raw score, threshold, session/source context.

## Observability

For product evaluation and later overrides, each auto-approved row should be attributable to:

- `policy_id`
- `model_ref`
- raw score
- threshold
- session/source id
- later human review decision, if any

Existing review decisions remain authoritative. A human override should be interpreted as a policy outcome observation, not as a model-label rewrite.
