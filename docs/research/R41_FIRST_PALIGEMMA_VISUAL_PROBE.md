# R41 First PaliGemma Visual Probe Results

- status: `model_access_ready_but_local_model_load_blocked`
- primary session: `outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957`
- attempted mode: `audio-gated / full_frame / 1 FPS`
- frame predictions produced: `0`
- visual proposals produced: `0`
- small comparison executed: `false`

## Access Result

Hugging Face login and gated model processor access now work. The failure is no longer token/license access.

## Runtime Result

The run downloaded the model weights, then died while loading checkpoint shards before frame inference. The machine is CPU-only with 8 GiB RAM, so local PaliGemma2 3B inference is not operationally tolerable here.

## Metrics

| metric | value |
|---|---:|
| visual proposals | 0 |
| visual matched anchors @2s | n/a |
| recovered false negatives | n/a |
| false visual proposals/min | n/a |
| visual-only proposals | 0 |
| overlap proposals | 0 |
| timing delta | n/a |

## Decision

- results justify further probing: `true`, but not on this CPU-only 8 GiB machine.
- CPU-only operationally tolerable: `false`.
- next run: same audio-gated full-frame 1 FPS benchmark on a GPU or larger-RAM machine before widening modes.
- `approve_review_v1` remains default.


# R41 Failure Analysis

- status: `local_resource_blocked_before_frame_predictions`
- observed failure mode: PaliGemma access is fixed, but local CPU/RAM could not load the 3B checkpoint shards.

No qualitative visual FP/FN analysis is possible because no frame predictions were produced.
