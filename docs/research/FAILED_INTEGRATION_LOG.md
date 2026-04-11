# Failed Integration Log

Date: 2026-04-11

This log records the event-level integration attempts that were tested and closed.

## 1. Event-Level Reranking

- Type: rerank
- Expected effect: better springboard cluster winner choice
- Actual effect: structural improvement offline only; no safe live gain
- Classification: `C`
- Reason for failure: did not improve live FN outcomes without destabilizing the funnel

## 2. Event-Level Score Blending

- Type: score blend
- Expected effect: smooth event structure into existing ranking
- Actual effect: no safe live gain
- Classification: `C`
- Reason for failure: blended scores still interacted poorly with peak-first survival dynamics

## 3. Representative Selection

- Type: representative selection
- Expected effect: choose cluster representative based on event structure
- Actual effect: candidate collapse in live detector conditions
- Classification: `D`
- Reason for failure: local cluster replacement disrupted downstream survival and replay

## 4. Delayed Cluster Selection

- Type: delayed selection
- Expected effect: let springboard clusters settle before winner choice
- Actual effect: no stable improvement in live conditions
- Classification: `C`
- Reason for failure: did not overcome peak-first collapse reliably

## 5. Cluster Winner Replacement

- Type: winner replacement
- Expected effect: replace rebound-dominated winners with dive-like event winners
- Actual effect: catastrophic regression
- Classification: `D`
- Reason for failure: candidate count and replay coverage collapsed

## 6. Merge-Stage Event Veto

- Type: veto
- Expected effect: block obviously rebound-dominated winners only
- Actual effect: candidate collapse and replay degradation
- Classification: `D`
- Reason for failure: merge-stage is too coupled to the funnel to be used as a safe injection point

## 7. Short-Region Tail Exception

- Type: narrow exception
- Expected effect: rescue 145-style short-region misses
- Actual effect: useful locally, but did not generalize to the springboard-heavy session
- Classification: `B`
- Reason for failure: subtype-specific gain only; not cross-session-safe

## 8. Validated Region Tie-Break

- Type: late pattern tie-break
- Expected effect: rescue weak near-threshold rows without threshold promotion
- Actual effect: validated and stable
- Classification: `A`
- Reason for success: stayed in the safe late-stage pattern tie-break boundary

