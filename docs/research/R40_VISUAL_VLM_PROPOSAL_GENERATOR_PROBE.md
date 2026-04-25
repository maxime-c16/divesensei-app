# R40 Visual VLM Proposal Generator Probe

## Status

- status: `implemented_but_vlm_runtime_blocked`
- optional stack ready: `False`
- approval policy changed: `false`
- taxonomy changed: `false`

## Current Audio Baseline And Visual Artifact Baseline

| metric | value |
|---|---:|
| reviewed dive anchors | 255 |
| audio proposals | 485 |
| visual proposals currently available | 0 |
| audio matched anchors | 225 |
| visual matched anchors | 0 |
| union matched anchors | 225 |
| audio recall @2s | 0.8823529411764706 |
| visual recall @2s | 0.0 |
| union recall @2s | 0.8823529411764706 |

## Candidate Strategy Status

| strategy | implementation | benchmark status |
|---|---|---|
| `full-session` | `implemented` | `requires_paligemma_runtime` |
| `audio-gated` | `implemented` | `availability_smoke_complete` |
| `roi-aware` | `implemented` | `requires_paligemma_runtime` |

## Recommendation

- adopt now: `False`
- best mode for first real run: audio-gated visual sweep is the lowest-cost first real VLM run because it covers current audio proposals and reviewed FN neighborhoods
- next phase: run real PaliGemma inference on Compete echo/rebound session with audio-gated/full-frame and center_pool ROI variants after installing optional stack and accepting model license

## Final Decision

`R40_VISUAL_VLM_PROPOSAL_PATH_IMPLEMENTED_RESEARCH_ONLY`

`APPROVE_REVIEW_V1_REMAINS_DEFAULT`
