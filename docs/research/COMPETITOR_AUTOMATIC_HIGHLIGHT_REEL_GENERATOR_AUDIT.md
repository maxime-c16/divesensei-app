# Competitor Audit: Pitt CIC Automatic Highlight Reel Generator

Audit date: 2026-04-22

Repository: https://github.com/pitt-cic/automatic-highlight-reel-generator

Local clone inspected at commit `2f29573` (`docs: add MIT license and AWS disclaimers to README`).

## Executive Summary

This competitor is not solving the same product problem as DiveSensei.

Their system is a cloud GPU, post-session highlight reel generator. It uploads full practice videos to S3, runs a large zero-shot vision-language model frame-by-frame, groups positive frames into intervals, and returns a merged highlight video. It is designed for a facility workflow where long footage is already recorded and can be processed after the fact.

DiveSensei is targeting a harder product shape: phone-first, coach/diver-friendly capture and review, with on-device or near-device constraints, live-ish review ergonomics, and governed labels/policies.

The competitor's main technical advantage is that a VLM can detect visible diver-in-air frames without custom training. Their main weakness is compute cost, latency, boundary coarseness, lack of public evaluation rigor, and poor fit for phone-only/on-device usage.

## What They Actually Built

The repo implements a three-stage pipeline:

1. Downsample source video to a low FPS stream.
2. Run PaliGemma on each downsampled frame with a yes/no prompt.
3. Group positive frames into intervals, add buffers, cut those intervals from the original video, and merge the clips.

Key implementation files:

- `/tmp/automatic-highlight-reel-generator/video-processing/main.py`
- `/tmp/automatic-highlight-reel-generator/video-processing/downsample_videos.py`
- `/tmp/automatic-highlight-reel-generator/video-processing/run_inference_and_postprocess.py`
- `/tmp/automatic-highlight-reel-generator/video-processing/clipping_and_merging.py`
- `/tmp/automatic-highlight-reel-generator/video-processing/config.yaml`
- `/tmp/automatic-highlight-reel-generator/lib/highlight-processor-stack.ts`

## Detection Pipeline

### Input

The input is a full video file uploaded to S3 under a `videos/` prefix. The Lambda trigger validates only basic file properties: extension, minimum size, and content type if available. It does not inspect the video semantically before launching ECS.

Relevant code:

- `lambda/handler.py`: validates supported extensions and size, then starts ECS.
- `video-processing/main.py`: downloads the S3 object into a temp directory and executes the pipeline.

### Downsampling

The pipeline uses FFmpeg to produce a low-FPS proxy:

- configured target FPS: `4`
- command uses `ffmpeg -filter:v fps=4`
- source-to-proxy timestamp mapping is generated with `stride = round(orig_fps / target_fps)`

This is efficient, but it is not a precise temporal contract. On non-integer FPS sources, variable frame-rate videos, or phone videos with timestamp irregularities, this can drift. They compensate with clip buffers.

Relevant code:

- `video-processing/config.yaml`: `downsampling.target_fps: 4`
- `video-processing/downsample_videos.py`: `generate_timestamp_mapping`

### Frame Classifier

They load:

- model: `google/paligemma2-3b-mix-224`
- class: `PaliGemmaForConditionalGeneration`
- precision: `torch.float16`
- device: CUDA if available

Default prompt:

```text
<image> Is there a person in the air jumping into the water? Answer with 'yes' or 'no'.
```

Each downsampled frame is:

- converted BGR to RGB
- optionally horizontally cropped using fractional config bounds
- resized to `224x224`
- sent to PaliGemma with the prompt
- decoded as generated text

The predicted label is assigned with:

```python
pred_label = "yes" if "yes" in answer else "no"
```

This is simple and recall-oriented, but brittle. It is not constrained decoding over a calibrated binary head. A generated phrase containing "yes" anywhere is enough.

Their confidence score is computed as the average probability of generated tokens. That is not the same as a calibrated event probability.

Relevant code:

- `video-processing/run_inference_and_postprocess.py`: `load_model`, `_run_inference_on_video`, `process_batch`
- `video-processing/config.yaml`: `inference.model_id`, `batch_size`, crop fields

### Postprocessing

The system thresholds frame-level positives and converts them to clips:

- confidence threshold: `0.845`
- group positives if gaps are <= `2.5s`
- add `1.5s` before the first positive frame
- add `3.0s` after the last positive frame
- merge intervals separated by <= `3.5s`

This means the core detector only needs to see a diver in a few frames. Clip quality depends heavily on the buffers.

Relevant code:

- `video-processing/run_inference_and_postprocess.py`: `postprocess_predictions`, `merge_intervals`
- `video-processing/config.yaml`: `post_processing`

### Output

The predicted intervals are written to CSV, then FFmpeg extracts those segments from the original high-resolution source and concatenates them into a highlight reel.

Relevant code:

- `video-processing/clipping_and_merging.py`

## Compute and Deployment Tradeoffs

The architecture is AWS-first:

- S3 upload triggers Lambda.
- Lambda launches ECS.
- ECS runs a CUDA Docker container.
- EC2 capacity is `g4dn.2xlarge`.
- Task allocates 8 vCPUs, roughly 30 GiB memory, and 1 GPU.
- Docker image bakes/downloads the Hugging Face PaliGemma model.

The README estimates:

- 15 min video -> roughly 10 min processing
- 2h06 video -> roughly 90 min processing
- g4dn.2xlarge cost roughly $0.752/hour
- 2h video run roughly $2.11 including storage/transfer assumptions
- monthly floor roughly $33.40 if infrastructure such as NAT gateway remains up

This is not phone-first and not on-device. It is viable for a centralized team/facility workflow, but not as a low-friction phone camera tool for coaches and divers.

Relevant code/docs:

- `lib/highlight-processor-stack.ts`: EC2/ECS/GPU resources
- `video-processing/Dockerfile`: CUDA base image and PaliGemma download
- `README.md`: performance and cost section

## 97 Percent Dive Recovery Claim

The repo itself does not contain a reproducible evaluation harness, labeled dataset, ground-truth manifest, precision/recall report, or boundary-quality script.

The 97% number appears in Pitt Digital's public success story, which says the solution can find 97% of dives in a 2.5-hour practice session. Based on repo evidence, this should be interpreted as event recall on one or more practice videos, not as a full product-grade metric.

Important missing details:

- denominator: exact number of dives unknown
- false positives unknown
- clip boundary error unknown
- whether platform and springboard are mixed unknown
- whether camera was fixed/facility-mounted unknown
- whether the prompt/config was tuned on that same video unknown
- whether the metric counts a dive as recovered if any part of it appears in a buffered clip unknown

This makes the claim useful as a proof-of-possibility, not as a governed benchmark.

Public source:

- https://www.digital.pitt.edu/news/success-stories/dive-revolutionizing-coaching-automated-highlight-reels

## Strengths

- Strong recall strategy for visually obvious diver-in-air frames.
- No labeled training set required.
- Promptable and sport-adaptable.
- Handles cases where the jump/approach is not visible by focusing on falling/diver-in-air evidence.
- Very simple postprocessing creates clips from sparse positives.
- Avoids the audio nuisance problem entirely by using visual evidence.
- Cloud GPU path can process long videos faster than real-time on their reported setup.

## Weaknesses

- Large VLM requires cloud GPU; not phone-first.
- Latency is minutes to hours, not immediate review.
- No public evaluation harness.
- Confidence is token-generation confidence, not calibrated event probability.
- Binary "yes" substring parsing is fragile.
- Frame-only model ignores motion continuity.
- Boundary accuracy is mostly heuristic buffers.
- 4 FPS downsampling can miss fast events or cause timestamp imprecision.
- Variable frame-rate phone footage may break the stride-based timestamp map.
- False positives are likely accepted as a tradeoff because the solution prioritizes recall.
- The product returns a merged highlight reel, not a structured review/labeling workflow.
- It does not classify platform vs springboard, rebound, handling noise, whistle, or non-dive splash.

## Direct Comparison to DiveSensei

| Dimension | Pitt CIC repo | DiveSensei direction |
|---|---|---|
| Primary input | Existing long video uploaded to S3 | Phone camera review workflow |
| Runtime | AWS ECS GPU | Local/on-device/near-device target |
| Core detector | PaliGemma frame VLM | Audio + governed event model, moving toward runtime-valid visual support |
| Evaluation | No repo-local governed eval | Reviewed manifests, source-aware holdouts, policy audits |
| Output | Merged highlight reel | Review queue, approved/review decisions, labels |
| Strength | Visual recall from sparse frames | Product workflow, labels, on-device ambition |
| Weakness | Compute/latency/eval rigor | Current visual detector not yet strong enough |

## Implications for r31 and Ongoing Roadmap

The competitor changes the roadmap in one important way: we should add a visual candidate generator track, not just a visual approve verifier.

Recommended next track:

### r32_visual_vlm_candidate_generator_probe

Purpose:

- Test whether a small visual model or VLM-like frame classifier can recover dives from phone/facility footage with high recall.
- Benchmark it as a candidate generator, not as the final product policy.

Required constraints:

- Keep detector/taxonomy governance intact.
- Do not replace the current audio/event pipeline blindly.
- Evaluate with our reviewed manifests.
- Report recall, false proposal rate, and boundary error separately.

Candidate variants:

- cloud upper bound: PaliGemma/Qwen2.5-VL/InternVL frame prompt probe on downsampled frames
- local/mobile candidate: MobileNet/EfficientNet/ViT-small frame classifier or embedding probe
- hybrid: low-cost motion/splash proposals plus visual verifier

### r33_phone_first_visual_runtime_feasibility

Purpose:

- Determine if the visual candidate generator can run on-device or near-device.

Report:

- FPS on Mac/iPhone-class hardware
- memory footprint
- battery/thermal risk
- model size
- latency to first candidate

### r34_multimodal_candidate_fusion

Purpose:

- Combine high-recall visual proposals with our governed audio/runtime scoring.
- Use visual to reduce false negatives and audio/event policy to manage approve/review safety.

Key metric:

- Not "97% found" alone.
- We need:
  - dive event recall
  - false clip/proposal rate
  - boundary start/end error
  - review burden
  - safe approve precision

## Product Takeaway

The competitor validates that visual evidence is strong for dive recovery. It does not validate that a cloud VLM highlight reel is the right product.

For DiveSensei, the opportunity is to absorb the useful lesson without copying the architecture:

- Use visual models for high-recall candidate generation.
- Keep our review workflow and governed evaluation discipline.
- Optimize for phone-first capture and coach/diver UX.
- Treat cloud VLMs as an upper-bound benchmark or fallback, not the core product.

