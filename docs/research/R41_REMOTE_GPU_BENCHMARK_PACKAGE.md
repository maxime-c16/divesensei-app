# R41 Remote GPU Benchmark Package

This package runs the first true PaliGemma2-3B visual proposal probe on hosted GPU infrastructure.

It is research-only. It does not change `approve_review_v1`, taxonomy, auto-approval, auto-exclusion, or the default review workflow.

## Minimal Repo Subset

Required:

- `src/`
- `benchmarks/r41_remote_gpu_runner.py`
- `outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957/ui_session_manifest.json`
- `outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957/session_pipeline_report.json`
- `outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957/evaluation_review.json`
- `outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957/proposal_diagnostics.jsonl`
- `outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957/proposal_diagnostics_summary.json`
- `outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957/web/session_source_review.mp4`
- optional review exports:
  - `exports/evaluation-review/reviewed_candidates.jsonl`
  - `exports/evaluation-review/false_negatives.jsonl`

Excluded:

- full `outputs/` tree
- local virtual environments
- Node app files
- historical benchmark outputs
- raw detector peak dumps
- `session_audio.wav`
- local Hugging Face cache
- approval-policy artifacts unrelated to visual proposal generation

## Assumptions

- The hosted runtime has a CUDA GPU.
- The Hugging Face token is supplied as `HF_TOKEN` and has access to `google/paligemma2-3b-mix-224`.
- The review proxy video is sufficient for visual proposal probing.
- The runner uses local notebook disk/cache, not direct cloud-drive streaming.
- CUDA uses bf16 when supported, otherwise fp16. This matters because Kaggle often provides T4 GPUs, which are fp16-friendly but not native-bf16 hardware.

## Build Package Locally

```bash
PYTHONPATH=src python3 benchmarks/r41_prepare_remote_gpu_package.py
```

This creates:

- `outputs/r41_remote_gpu_package/`
- `outputs/r41_remote_gpu_package`

## Local Kaggle CLI Credentials

Required credentials:

- Kaggle API credential for local dataset upload.
- Hugging Face token with accepted PaliGemma access, stored in Kaggle Notebook Secrets as `HF_TOKEN`.

Optional credentials:

- None for the runner itself. The Kaggle notebook only needs `HF_TOKEN`; Kaggle CLI auth is only needed locally to create/update the private dataset.

Install Kaggle CLI locally in a dedicated venv:

```bash
python3 -m venv .venv-kaggle
.venv-kaggle/bin/python -m pip install --upgrade pip kaggle
export KAGGLE_BIN=.venv-kaggle/bin/kaggle
```

Preferred local authentication for dataset upload: legacy `kaggle.json`.

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
scripts/r41_kaggle_cli_check.sh
```

Alternative local authentication: OAuth login, if enabled by your Kaggle CLI/account.

```bash
KAGGLE_ENABLE_OAUTH=1 .venv-kaggle/bin/kaggle auth login
scripts/r41_kaggle_cli_check.sh
```

Note: in local testing, the newer `KGAT_...` API token shown by Kaggle was not accepted by the Kaggle CLI dataset endpoints through `KAGGLE_API_TOKEN`. Use `kaggle.json` or OAuth login for the local dataset upload path.

The check is expected to print the Kaggle CLI version and return a private dataset listing without an authentication error.

## Kaggle Dataset Upload

Dataset:

- slug: `maximecauchy/divesensei-r41-remote-gpu-package-v3`
- local metadata: `kaggle/r41_remote_gpu_package/dataset-metadata.json`
- local package file: `kaggle/r41_remote_gpu_package/r41_remote_gpu_package`

Stage the dataset folder locally:

```bash
scripts/r41_stage_kaggle_dataset.sh
```

First-time private dataset creation:

```bash
.venv-kaggle/bin/kaggle datasets create \
  -p kaggle/r41_remote_gpu_package \
  --dir-mode zip \
  --private
```

Subsequent private dataset updates:

```bash
.venv-kaggle/bin/kaggle datasets version \
  -p kaggle/r41_remote_gpu_package \
  --dir-mode zip \
  -m "Update r41 remote GPU package"
```

Validate the uploaded dataset:

```bash
.venv-kaggle/bin/kaggle datasets status maximecauchy/divesensei-r41-remote-gpu-package-v3
```

## Kaggle

Create/update the private Kaggle Dataset with the CLI flow above, then open:

- `notebooks/r41_kaggle_runner.ipynb`

Enable a GPU accelerator and add a Kaggle secret:

- `HF_TOKEN`

Attach this Dataset to the notebook:

- `maximecauchy/divesensei-r41-remote-gpu-package-v3`

The notebook runs:

```bash
python benchmarks/r41_remote_gpu_runner.py \
  --cache-dir /kaggle/working/hf-cache \
  --output-root /kaggle/working/r41_remote_gpu_results \
  --prompt-id diving_attempt \
  --decision-rule yes_no_first_token_margin \
  --smoke-max-frames 1 \
  --full-fps 1.0
```

## Colab

Open:

- `notebooks/r41_colab_runner.ipynb`

Enable GPU runtime and add a Colab secret:

- `HF_TOKEN`

Upload `outputs/r41_remote_gpu_package` when prompted.

The notebook runs:

```bash
python benchmarks/r41_remote_gpu_runner.py \
  --cache-dir /content/hf-cache \
  --output-root /content/r41_remote_gpu_results \
  --prompt-id diving_attempt \
  --decision-rule yes_no_first_token_margin \
  --smoke-max-frames 1 \
  --full-fps 1.0
```

## Output Bundle

Both notebooks produce:

- `r41_remote_gpu_results_bundle.zip`

The bundle contains:

- `r41_remote_preflight.json`
- `r41_remote_gpu_run_summary.json`
- `r41_remote_gpu_run_summary.md`
- `smoke_one_frame/visual_frame_predictions.jsonl`
- `smoke_one_frame/visual_event_intervals.json`
- `smoke_one_frame/visual_proposals.jsonl`
- `smoke_one_frame/merged_proposal_diagnostics.jsonl`
- `smoke_one_frame/visual_vlm_proposal_summary.json`
- `audio_gated_full_frame_1fps/visual_frame_predictions.jsonl`
- `audio_gated_full_frame_1fps/visual_event_intervals.json`
- `audio_gated_full_frame_1fps/visual_proposals.jsonl`
- `audio_gated_full_frame_1fps/merged_proposal_diagnostics.jsonl`
- `audio_gated_full_frame_1fps/visual_vlm_proposal_summary.json`

If the one-frame smoke fails, the runner does not execute the full benchmark.

## Validation Signals

Local Kaggle CLI auth:

```bash
scripts/r41_kaggle_cli_check.sh
```

Notebook GPU:

```python
import torch, subprocess
subprocess.run(["nvidia-smi"], check=False)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Notebook Hugging Face access is checked by:

```bash
python -m divesensei.cli visual-vlm-preflight \
  --model-id google/paligemma2-3b-mix-224 \
  --model-cache-dir /kaggle/working/hf-cache \
  --check-processor
```

Expected successful fields:

- `cuda_available: true`
- `hf_token_present: true`
- `hf_whoami.name` populated
- `model_info_accessible: true`
- `processor_load_status: ok`
- `can_proceed: true`

## Operator Checklist

1. Build/stage local package:

   ```bash
   scripts/r41_stage_kaggle_dataset.sh
   ```

2. Validate Kaggle CLI auth:

   ```bash
   export KAGGLE_BIN=.venv-kaggle/bin/kaggle
   scripts/r41_kaggle_cli_check.sh
   ```

3. Create private Dataset once:

   ```bash
   .venv-kaggle/bin/kaggle datasets create -p kaggle/r41_remote_gpu_package --dir-mode zip --private
   ```

4. For later updates:

   ```bash
   .venv-kaggle/bin/kaggle datasets version -p kaggle/r41_remote_gpu_package --dir-mode zip -m "Update r41 remote GPU package"
   ```

5. In Kaggle, open `notebooks/r41_kaggle_runner.ipynb`.

6. Attach Dataset `maximecauchy/divesensei-r41-remote-gpu-package-v3`.

7. Add Kaggle Secret `HF_TOKEN`.

8. Enable GPU, preferably T4 x2 if offered.

9. Run all notebook cells.

10. Download `/kaggle/working/r41_remote_gpu_results_bundle.zip`.
