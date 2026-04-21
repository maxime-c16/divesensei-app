# Approve Review v2 Promotion Evidence Status

- shadow evidence decision: `R22_SHADOW_EVIDENCE_STILL_CLEAN`
- rollout decision: `V2_REQUIRES_MORE_SHADOW_ACCUMULATION`

The hardened v2 shadow policy remains clean on the accumulated shadow-scored bank, but it is not ready to replace v1.

## Current Evidence

- eligible source count: `5`
- unique eligible rows: `221`
- shadow-only added approvals: `20`
- suspicious added approvals: `0`
- fresh/less-central shadow-only additions: `15`

## Required Before Limited Rollout

- Review and shadow-score the new CAO source.
- Add at least one more independent reviewed source with the visual shadow score path.
- Keep suspicious v2-only approvals at zero.
- Keep human override rate for v2-only approvals at zero or explicitly accepted.
- Keep approve_review_v1 as default until these checks pass.
