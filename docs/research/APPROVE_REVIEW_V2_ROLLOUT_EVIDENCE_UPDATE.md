# Approve Review v2 Rollout Evidence Update

- shadow evidence decision: `R23_SHADOW_EVIDENCE_REGRESSED`
- rollout decision: `V2_REQUIRES_MORE_SHADOW_ACCUMULATION`

The hardened v2 shadow policy is not ready to replace v1. The expanded bank now exposes suspicious v2-only approvals on a fresh CAO source.

## Current Evidence

- eligible source count: `7`
- unique eligible rows: `383`
- shadow-only added approvals: `23`
- suspicious added approvals: `3`
- fresh/less-central shadow-only additions: `18`

## Required Before Limited Rollout

- Treat CAO-SUN voice_whistle rows as a hard blocker for the current v2 rule.
- Run the next bounded policy hardening pass against source-aware voice_whistle false approvals before any rollout decision.
- Keep suspicious v2-only approvals at zero in any replacement candidate.
- Keep human override rate for v2-only approvals at zero or explicitly accepted.
- Keep approve_review_v1 as default until these checks pass.
