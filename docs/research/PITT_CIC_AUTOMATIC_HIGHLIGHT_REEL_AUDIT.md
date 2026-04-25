# Pitt CIC Automatic Highlight Reel Generator Audit

## Executive Summary

The Pitt CIC project is a high-recall visual highlight generator, not a coach-safe event classifier.

Their reported 97% result is best interpreted as dive-recovery performance on a 2.5-hour practice video: did the generated intervals catch almost all dive repetitions. It is not equivalent to our current governed metrics for `platform_dive` vs `noise_or_other`, dangerous approve errors, source-aware validation, or runtime/offline parity.

The useful technical lesson for DiveSensei is clear:

- Add a visual high-recall proposal generator.
- Use it to reduce missed dives and improve review coverage.
- Do not use it as an auto-approve policy until it passes our governed safety monitor.

## Sources Audited

- GitHub repo: `https://github.com/pitt-cic/automatic-highlight-reel-generator`
- Pitt Digital success story: `https://www.digital.pitt.edu/news/success-stories/dive-revolutionizing-coaching-automated-highlight-reels`
- AWS Innovation Islands project page: `https://www.innovation-islands.com/point/automatic-highlight-reel-generator/`
- Hugging Face model card: `https://huggingface.co/google/paligemma2-3b-mix-224`

Local clone inspected at commit:

- `2f29573`

Repo files audited:

- `README.md`
- `video-processing/config.yaml`
- `video-processing/main.py`
- `video-processing/downsample_videos.py`
- `video-processing/run_inference_and_postprocess.py`
- `video-processing/clipping_and_merging.py`
- `video-processing/Dockerfile`
- `video-processing/requirements.txt`
- `lambda/handler.py`
- `lib/highlight-processor-stack.ts`

## What They Built

They built an AWS-backed video processing pipeline:

1. User uploads a long practice video to S3.
2. S3 upload triggers Lambda.
3. Lambda starts an ECS GPU task.
4. ECS runs a Dockerized Python pipeline on a GPU EC2 instance.
5. The Python pipeline downsamples video, runs PaliGemma frame inference, clusters positive frames into event intervals, clips the original high-resolution video, and uploads a highlight reel.

The AWS project page describes the deployed architecture as S3, Lambda, ECS, GPU EC2, ECR, and CDK. The repo implements the same pattern in `lambda/handler.py` and `lib/highlight-processor-stack.ts`.

## How The Detection Works

### Model

The detector uses:

- `google/paligemma2-3b-mix-224`
- `PaliGemmaForConditionalGeneration`
- `AutoProcessor`
- PyTorch `float16`
- CUDA when available

The model is not trained on their diving footage in this repo. It is used as a zero-shot / prompt-driven vision-language model.

The Hugging Face model card says PaliGemma 2 is an image-and-text to text model, supports prompt-conditioned image understanding, and requires accepting Google license terms before downloading the gated files.

### Prompt

Default prompt in `video-processing/config.yaml`:

```text
<image> Is there a person in the air jumping into the water? Answer with 'yes' or 'no'.
```

This is important. They are not trying to recognize the splash acoustically or classify dive types. They are detecting the visually salient falling/airborne phase.

### Frame Sampling

`video-processing/downsample_videos.py` uses FFmpeg to convert the source video to a lower frame rate. Default:

- `target_fps: 4`

It also creates a CSV mapping from downsampled inference frame index back to original timestamps. That lets them cut clips from the original high-resolution source after running inference on cheap low-FPS frames.

### Visual Preprocessing

`run_inference_and_postprocess.py`:

- opens the downsampled video with OpenCV
- converts BGR to RGB
- applies horizontal crop configured by `crop_width_start` and `crop_width_end`
- resizes each frame to `224x224`
- batches frames, default batch size `16`

Default crop is full width.

### Inference

For each frame batch:

- model receives the same text prompt for every frame
- `model.generate(...)` emits a text answer
- answer is parsed as positive if the decoded answer contains `yes`
- confidence is computed from the average softmax probability of generated tokens

That confidence is not a calibrated event probability. It is a token-generation confidence proxy.

### Postprocessing

`postprocess_predictions` applies:

- predicted label must be `yes`
- confidence must be at least `0.845`
- nearby yes timestamps are grouped if gap is no more than `2.5s`
- each group gets `1.5s` start buffer and `3.0s` end buffer
- adjacent intervals are merged if gap is no more than `3.5s`

This is a recall-friendly interval generator. It tolerates imperfect frame decisions because positive frames are clustered and padded.

### Clip Generation

`clipping_and_merging.py`:

- extracts each interval from the original video using FFmpeg
- re-encodes clips as H.264/AAC
- concatenates clips into one highlight reel

## Why The 97% Claim Is Plausible

The Pitt Digital article states the solution found 97% of dives in 2.5 hours of practice footage. That result is plausible because their problem setup favors recall:

- visual airborne/diver-entry evidence is more direct than splash audio in echo-heavy pools
- the prompt asks a broad semantic question
- inference runs across many frames per dive at 4 FPS
- one or a few positive frames can recover an entire dive interval
- intervals are buffered and merged
- false positives are less damaging for a highlight reel than false negatives

This is not the same as precise event classification. A highlight reel can tolerate extra seconds and some false-positive clips. Our approve/review product cannot tolerate dangerous nuisance auto-approvals.

## Key Differences From DiveSensei

| Dimension | Pitt CIC Highlight Reel | DiveSensei Current System |
|---|---|---|
| Primary signal | Video VLM | Audio event detector + governed scorer |
| Primary goal | Recover dive clips | Review workflow and safe high-confidence approval |
| Output | Highlight intervals / merged reel | Candidate queue, event labels, approved/needs-review |
| Metric implied by 97% | Dive recovery / recall | Precision, dangerous approvals, source-aware safety |
| Model | Zero-shot PaliGemma | Governed XGBoost r9 audio scorer, exact runtime parity |
| Infrastructure | AWS GPU batch pipeline | Local desktop review flow, phone/coach-oriented direction |
| Timing tolerance | Buffers and interval merging | Event anchors, review rows, false-negative accounting |
| Taxonomy | Event present vs not | `platform_dive`, `springboard_dive`, rebound, noise subtypes |
| Human loop | Review generated clips | Human-reviewed labels are first-class governance artifacts |

## Subtle Differences That Matter

1. Their confidence is generated-token confidence, not calibrated classification probability.
2. Their positive parser is string-based: answer contains `yes`.
3. Their event windows are forgiving; ours are audited row-by-row.
4. Their architecture assumes GPU/cloud batch processing.
5. Their result can merge close activity into one useful clip; our product needs per-event accountability.
6. Their visual approach likely ignores board rebound audio entirely, which is useful for echo sessions but insufficient for nuisance taxonomy.
7. Their VLM may fail if the diver is too small, occluded, off-camera, or visually ambiguous.

## What We Should Copy

Copy the detection idea, not the product architecture.

The strongest adaptation for DiveSensei is:

- visual high-recall proposal generation from video frames
- source/provenance tags for visual proposals
- merged audio + visual candidate queue
- human review as the arbiter
- governed evaluation before any product policy change

This directly targets our current hard problem: missed dives and echo/rebound sessions where audio can be unreliable.

## What We Should Not Copy Directly

Do not copy:

- cloud-only GPU product dependency
- highlight-only success metric
- token-confidence threshold as product confidence
- prompt-only output as an auto-approval decision
- no-taxonomy event labeling
- no source-aware validation / hard-negative safety monitor

## Proposed DiveSensei Implementation

### Phase V1: Offline Visual Proposal Prototype

Add a new optional workflow:

```text
evaluate-session --with-visual-vlm-proposals
```

Outputs:

- `visual_frame_predictions.jsonl`
- `visual_event_intervals.json`
- `visual_proposals.jsonl`
- merged `proposal_diagnostics.jsonl` rows with provenance `visual_vlm_paligemma2`

Runtime behavior:

1. Downsample video to 2-4 FPS.
2. Run PaliGemma2 frame inference with a dive-entry prompt.
3. Cluster positive frames into event intervals.
4. Create review candidates from visual intervals.
5. Merge visual candidates with existing audio candidates by timestamp.
6. Preserve provenance: `audio`, `visual_vlm`, or `audio_visual_overlap`.

Initial prompts to test:

```text
<image> Is a diver airborne above a swimming pool or entering the water? Answer yes or no.
```

```text
<image> Is there a person in the air jumping or diving into the pool? Answer yes or no.
```

```text
<image> Is this frame part of a diving attempt into water? Answer yes or no.
```

### Phase V2: Governed Visual Proposal Benchmark

Use reviewed sessions:

- echo/rebound `Compete 16:11:2025` crop
- platform-heavy CAO sessions
- INSEP and Champigny references
- known false-negative neighborhoods

Evaluate:

- dive recall gain
- false visual proposals per minute
- recovered false negatives
- visual-only vs audio-only vs union proposals
- timing delta against reviewed event anchors
- source-family generalization

Success criteria should not be approve precision yet. It should be proposal recall with acceptable review burden.

### Phase V3: Mobile Path

PaliGemma2 is a strong offline teacher, not the likely final on-device model.

A realistic phone path:

1. Use PaliGemma2 to generate visual proposal pseudo-labels and hard negatives.
2. Human-review the visual proposals.
3. Train a smaller on-device visual proposal model:
   - MobileNetV3 / EfficientNet-Lite frame classifier
   - small temporal CNN over sampled frame embeddings
   - lightweight object/pose model if visibility supports it
4. Keep audio as a cheap continuous proposal source.
5. Use visual model only on sampled frames or candidate windows to save battery.

### Phase V4: Product Integration

Do not change `approve_review_v1`.

Use visual proposals as:

- recall assistance
- false-negative discovery
- review queue enrichment
- candidate provenance
- later visual verifier evidence

Only consider product approval logic after:

- exact runtime parity for the visual path
- source-aware validation
- `make approve-safety-monitor` stays clean
- no dangerous added approvals on independent banks

## Relevant External Artifacts Needed

To implement their technique we need:

- Hugging Face account and accepted `google/paligemma2-3b-mix-224` license
- Hugging Face token for model download
- `transformers`
- `torch`
- `opencv-python-headless`
- `Pillow`
- `pandas`
- FFmpeg
- preferably CUDA GPU for first runs

Their Dockerfile bakes the model into the image to avoid runtime download. For our local prototype, a cached model directory under external storage is preferable.

## Recommended Next DiveSensei Step

Do not interrupt the current approve safety monitor track.

The bounded next research pass should be:

```text
r40_visual_vlm_proposal_generator_probe
```

Scope:

- no approve-policy changes
- no auto-exclude
- no taxonomy change
- add visual proposals as review-support only
- benchmark on hard FN and echo/rebound sessions

Main question:

Can PaliGemma-style visual proposals recover missed dive attempts in sessions where audio is weak or misleading, without exploding review burden?

## Bottom Line

They achieved strong recovery because they solved an easier and different problem: visually find broad dive intervals in fixed practice footage and generate clips. That is exactly the kind of high-recall signal we should add as a proposal generator.

For DiveSensei, the right architecture is not audio vs video. It is:

```text
audio proposals + visual VLM proposals -> unified review queue -> governed labels -> exact-runtime scorer -> approve/review policy
```

The visual VLM should first improve recall and review coverage. It should not become a trusted approval lane until it survives the same governance that blocked earlier unsafe shadow policies.
