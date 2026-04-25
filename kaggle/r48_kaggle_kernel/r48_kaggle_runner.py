from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DATASET_SLUGS = [
    "r41-remote-gpu-package-v3",
    "divesensei-r41-remote-gpu-package-v3",
]
PACKAGED_MODEL_NAME = "models--google--paligemma2-3b-mix-224"
SESSION_NAME = "evaluation_insep_plateform_mixed_sound"
WINDOW_OFFSETS_SECONDS = (-2.0, -1.0, 0.0, 1.0, 2.0)
PROMPTS = [
    {
        "prompt_id": "baseline_frame_prompt",
        "prompt": "answer en is this frame part of a diving attempt into water?\n",
    },
    {
        "prompt_id": "window_dive_evidence_prompt",
        "prompt": "answer en does this frame show visual evidence of a real dive attempt into the water?\n",
    },
    {
        "prompt_id": "anti_clutter_prompt",
        "prompt": "answer en does this frame show a real dive attempt, not poolside activity, standing, walking, talking, or unrelated splash?\n",
    },
]


def _find_package_root() -> Path:
    candidates: list[Path] = []
    for slug in DATASET_SLUGS:
        candidates.extend(
            [
                Path("/kaggle/input") / slug / "r41_remote_gpu_package",
                Path("/kaggle/input") / slug,
            ]
        )
    candidates.extend(Path("/kaggle/input").glob("**/r41_remote_gpu_package"))
    candidates.extend(path.parent for path in Path("/kaggle/input").glob("**/REMOTE_PACKAGE_MANIFEST.json"))
    for candidate in candidates:
        if (candidate / "REMOTE_PACKAGE_MANIFEST.json").exists():
            return candidate
    raise FileNotFoundError("Attach the private Kaggle Dataset maximecauchy/r41-remote-gpu-package-v3 first.")


def _prepare_cache(package_root: Path) -> Path:
    work_cache = Path("/kaggle/working/hf-cache")
    packaged_cache = package_root / "hf-cache" / PACKAGED_MODEL_NAME
    if packaged_cache.exists():
        target = work_cache / PACKAGED_MODEL_NAME
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying packaged model cache from {packaged_cache} to {target}")
        shutil.copytree(packaged_cache, target)
    work_cache.mkdir(parents=True, exist_ok=True)
    return work_cache


def _resolve_model_id(package_root: Path) -> str:
    local_model_root = package_root / "local-model"
    if local_model_root.exists():
        candidates = sorted(path for path in local_model_root.iterdir() if path.is_dir())
        if candidates:
            print(f"Using packaged local model directory: {candidates[0]}")
            return str(candidates[0])
    return "google/paligemma2-3b-mix-224"


def _install_dependencies(package_root: Path) -> None:
    # Avoid pinning numpy on Kaggle. The prior numpy downgrade produced noisy resolver conflicts
    # and is unnecessary for this narrow inference-only runner.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "transformers==4.53.3",
            "accelerate",
            "sentencepiece",
            "huggingface-hub",
            "opencv-python-headless",
            "Pillow",
        ],
        cwd=package_root,
        check=True,
    )


def _maybe_load_hf_secret() -> str:
    existing = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if existing:
        return "environment"
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            return "kaggle_secret"
    except Exception as exc:
        print(f"HF_TOKEN secret unavailable or not needed: {exc!r}")
    return "missing"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _zip_outputs(output_root: Path, bundle_path: Path) -> None:
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent))


def _crop_full_frame(frame: Any) -> Any:
    from PIL import Image

    rgb = frame[:, :, ::-1]
    return Image.fromarray(rgb)


def _build_window_requests(session_root: Path) -> list[dict[str, Any]]:
    reviewed = _read_jsonl(session_root / "exports/evaluation-review/reviewed_candidates.jsonl")
    requests: list[dict[str, Any]] = []
    for row in reviewed:
        label = row.get("review_label")
        if label == "dive":
            candidate_label = "true_dive_candidate"
        elif label == "non_dive":
            candidate_label = "nuisance_non_dive_candidate"
        else:
            candidate_label = "ambiguous"
        anchor = float(row["timestamp_seconds"])
        for offset in WINDOW_OFFSETS_SECONDS:
            target = max(0.0, anchor + offset)
            requests.append(
                {
                    "proposal_id": row["proposal_id"],
                    "source_candidate_id": row.get("source_candidate_id"),
                    "anchor_timestamp_seconds": anchor,
                    "review_label": label,
                    "candidate_label": candidate_label,
                    "offset_seconds": offset,
                    "target_timestamp_seconds": round(target, 3),
                }
            )
    return requests


def _run_inference(package_root: Path, output_root: Path, model_id: str) -> dict[str, Any]:
    import cv2  # type: ignore
    import torch  # type: ignore
    import torch._dynamo  # type: ignore

    torch._dynamo.config.suppress_errors = True

    from divesensei.workflows.visual_vlm_proposals import PaliGemmaBackend, VisualProposalConfig

    session_root = package_root / "outputs" / SESSION_NAME
    video_path = session_root / "web/session_source_review.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    requests = _build_window_requests(session_root)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not decode video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    duration_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    config = VisualProposalConfig(
        backend="paligemma",
        model_id=model_id,
        model_cache_dir=os.environ.get("HF_HOME", ""),
        fps=1.0,
        roi_mode="full_frame",
        prompt_ids=tuple(),
        decision_rules=("yes_no_first_token_margin",),
        max_new_tokens=5,
    )
    backend = PaliGemmaBackend(config)
    rows: list[dict[str, Any]] = []
    started = time.time()

    for idx, req in enumerate(requests, start=1):
        frame_idx = int(round(float(req["target_timestamp_seconds"]) * source_fps))
        if duration_frames > 0:
            frame_idx = max(0, min(duration_frames - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            for prompt in PROMPTS:
                rows.append(
                    {
                        **req,
                        "prompt_id": prompt["prompt_id"],
                        "prompt": prompt["prompt"],
                        "available": False,
                        "error": "frame_decode_failed",
                    }
                )
            continue
        actual_timestamp = frame_idx / source_fps if source_fps > 0 else float(req["target_timestamp_seconds"])
        image = _crop_full_frame(frame)
        images = [image] * len(PROMPTS)
        model_prompts = ["<image> " + prompt["prompt"] for prompt in PROMPTS]
        predictions = backend.predict_batch(images, model_prompts)
        for prompt, prediction in zip(PROMPTS, predictions, strict=True):
            margin = prediction.get("yes_no_first_token_margin")
            yes_prob = prediction.get("yes_first_token_probability")
            no_prob = prediction.get("no_first_token_probability")
            rows.append(
                {
                    **req,
                    "prompt_id": prompt["prompt_id"],
                    "prompt": prompt["prompt"],
                    "model_prompt": "<image> " + prompt["prompt"],
                    "available": True,
                    "frame_index": frame_idx,
                    "frame_timestamp_seconds": round(actual_timestamp, 3),
                    "nearest_delta_seconds": round(actual_timestamp - float(req["target_timestamp_seconds"]), 3),
                    "raw_response": prediction.get("raw_response"),
                    "token_probability_mean": prediction.get("token_probability_mean"),
                    "token_probabilities": prediction.get("token_probabilities"),
                    "yes_first_token_probability": yes_prob,
                    "no_first_token_probability": no_prob,
                    "yes_no_first_token_margin": margin,
                    "score": float(yes_prob if yes_prob is not None else 0.0),
                    "is_positive": bool(margin is not None and float(margin) > 0.0),
                }
            )
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(requests)} audio-window frames")

    cap.release()
    prediction_path = output_root / "r48_audio_window_frame_predictions.jsonl"
    _write_jsonl(prediction_path, rows)
    summary = {
        "benchmark_id": "r48_remote_audio_window_prompt_ablation",
        "session_root": str(session_root),
        "video_path": str(video_path),
        "model_id": model_id,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "candidate_count": len({row["proposal_id"] for row in requests}),
        "window_request_count": len(requests),
        "prediction_rows": len(rows),
        "prompt_ids": [prompt["prompt_id"] for prompt in PROMPTS],
        "elapsed_seconds": round(time.time() - started, 3),
        "prediction_path": str(prediction_path),
    }
    _write_json(output_root / "r48_remote_audio_window_run_summary.json", summary)
    return summary


def main() -> int:
    package_root = _find_package_root()
    output_root = Path("/kaggle/working/r48_remote_audio_window_prompt_ablation")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print("Package root:", package_root)
    print("r48 runner: exact r47 audio-window prompt ablation")
    print(json.dumps(json.loads((package_root / "REMOTE_PACKAGE_MANIFEST.json").read_text()), indent=2)[:4000])

    _install_dependencies(package_root)
    os.environ["PYTHONPATH"] = str(package_root / "src")
    if str(package_root / "src") not in sys.path:
        sys.path.insert(0, str(package_root / "src"))
    os.environ["HF_HOME"] = str(_prepare_cache(package_root))
    hf_secret_source = _maybe_load_hf_secret()
    model_id = _resolve_model_id(package_root)

    import torch  # type: ignore
    import torch._dynamo  # type: ignore

    torch._dynamo.config.suppress_errors = True

    health = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "hf_secret_source": hf_secret_source,
        "model_id": model_id,
        "hf_home": os.environ["HF_HOME"],
    }
    _write_json(output_root / "r48_remote_run_health.json", health)
    print(json.dumps(health, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for r48 remote prompt ablation.")

    summary = _run_inference(package_root, output_root, model_id)
    bundle_path = Path("/kaggle/working/r48_remote_audio_window_prompt_ablation_bundle.zip")
    _zip_outputs(output_root, bundle_path)
    print(json.dumps({"health": health, "summary": summary, "bundle_path": str(bundle_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
