from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from divesensei.io.media_io import probe_media_duration_seconds
from divesensei.workflows.evaluation_session_support import load_jsonl, read_json, write_json, write_jsonl


DEFAULT_MODEL_ID = "google/paligemma2-3b-mix-224"
DEFAULT_PROMPTS = {
    "airborne_entry": "<image> Is a diver airborne above a swimming pool or entering the water? Answer yes or no.",
    "jumping_or_diving": "<image> Is there a person in the air jumping or diving into the pool? Answer yes or no.",
    "diving_attempt": "<image> Is this frame part of a diving attempt into water? Answer yes or no.",
    "pool_entry": "<image> Is a person entering the water from a diving board or platform? Answer yes or no.",
}


def _resolve_visual_source_video(session_dir: Path, manifest: dict[str, Any]) -> tuple[Path, str, str]:
    manifest_path = Path(str(manifest.get("session", {}).get("source_video_path") or "")).expanduser()
    if manifest_path.exists():
        return manifest_path, str(manifest_path), "manifest_source_video_path"

    review_proxy = session_dir / "web" / "session_source_review.mp4"
    if review_proxy.exists():
        return review_proxy, str(manifest_path), "review_proxy_fallback"

    return manifest_path, str(manifest_path), "missing"


@dataclass(frozen=True)
class VisualProposalConfig:
    mode: str = "full-session"
    backend: str = "paligemma"
    model_id: str = DEFAULT_MODEL_ID
    model_cache_dir: str = ""
    fps: float = 1.0
    resolution: int = 224
    roi_mode: str = "full_frame"
    custom_roi: str = ""
    batch_size: int = 4
    max_new_tokens: int = 5
    confidence_threshold: float = 0.845
    grouping_threshold_seconds: float = 2.5
    buffer_start_seconds: float = 1.5
    buffer_end_seconds: float = 3.0
    merge_gap_seconds: float = 3.5
    audio_gate_pre_seconds: float = 4.0
    audio_gate_post_seconds: float = 4.0
    prompt_ids: tuple[str, ...] = ("airborne_entry", "jumping_or_diving", "diving_attempt")
    decision_rules: tuple[str, ...] = ("naive_contains_yes", "strict_yes_no", "yes_no_first_token_margin")
    max_frames: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei visual-vlm-proposals",
        description="Generate optional visual VLM proposal artifacts for an evaluated session. Research-only; no approval policy changes.",
    )
    parser.add_argument("session_path", help="Evaluation session directory, ui_session_manifest.json, or session_pipeline_report.json")
    parser.add_argument("--output-dir", default="", help="Defaults to <session>/exports/visual-vlm-proposals")
    parser.add_argument("--mode", choices=["full-session", "audio-gated", "oracle-gated"], default="full-session")
    parser.add_argument("--backend", choices=["paligemma", "motion-proxy", "availability-check"], default="paligemma")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-cache-dir", default=os.environ.get("DIVESENSEI_VLM_CACHE_DIR", ""))
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--roi-mode", choices=["full_frame", "center_pool", "lower_water", "custom"], default="full_frame")
    parser.add_argument("--custom-roi", default="", help="Custom ROI as left,top,right,bottom fractions, e.g. 0.1,0.1,0.9,0.85")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.845)
    parser.add_argument("--grouping-threshold-seconds", type=float, default=2.5)
    parser.add_argument("--buffer-start-seconds", type=float, default=1.5)
    parser.add_argument("--buffer-end-seconds", type=float, default=3.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=3.5)
    parser.add_argument("--audio-gate-pre-seconds", type=float, default=4.0)
    parser.add_argument("--audio-gate-post-seconds", type=float, default=4.0)
    parser.add_argument("--prompt-id", action="append", choices=sorted(DEFAULT_PROMPTS), dest="prompt_ids")
    parser.add_argument("--decision-rule", action="append", choices=["naive_contains_yes", "strict_yes_no", "yes_no_first_token_margin"], dest="decision_rules")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug cap. 0 means no cap.")
    return parser


def build_preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei visual-vlm-preflight",
        description="Check whether the optional visual VLM proposal runtime can run on this machine.",
    )
    parser.add_argument("--backend", choices=["paligemma", "motion-proxy", "availability-check"], default="paligemma")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-cache-dir", default=os.environ.get("DIVESENSEI_VLM_CACHE_DIR", ""))
    parser.add_argument("--check-processor", action="store_true", help="Try loading the processor. May require model license/token access.")
    parser.add_argument("--check-model-load", action="store_true", help="Try loading full model weights. Heavy; not recommended on CPU-only machines.")
    return parser


def visual_vlm_preflight(
    *,
    backend: str = "paligemma",
    model_id: str = DEFAULT_MODEL_ID,
    model_cache_dir: str = "",
    check_processor: bool = False,
    check_model_load: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": backend,
        "model_id": model_id,
        "cache_dir": model_cache_dir or os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or str(Path.home() / ".cache" / "huggingface"),
        "dependencies": {
            "torch": importlib.util.find_spec("torch") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
            "PIL": importlib.util.find_spec("PIL") is not None,
            "cv2": importlib.util.find_spec("cv2") is not None,
            "huggingface_hub": importlib.util.find_spec("huggingface_hub") is not None,
        },
        "can_proceed": False,
        "warnings": [],
        "errors": [],
    }
    try:
        disk = shutil.disk_usage(model_cache_dir or str(Path.home() / ".cache" / "huggingface"))
    except Exception:
        disk = shutil.disk_usage(Path.home())
    result["resource_status"] = {
        "cache_disk_free_gib": round(disk.free / (1024**3), 3),
        "cache_disk_total_gib": round(disk.total / (1024**3), 3),
    }
    try:
        mem_output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        result["resource_status"]["system_ram_gib"] = round(int(mem_output) / (1024**3), 3)
    except Exception:
        result["resource_status"]["system_ram_gib"] = None
    if backend in {"motion-proxy", "availability-check"}:
        result["can_proceed"] = result["dependencies"]["PIL"] and result["dependencies"]["cv2"] if backend == "motion-proxy" else True
        return result
    if not all(result["dependencies"][name] for name in ["torch", "transformers", "PIL", "cv2"]):
        result["errors"].append("missing_required_visual_vlm_dependencies")
        return result
    try:
        import torch  # type: ignore

        result["torch_version"] = str(torch.__version__)
        result["device"] = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if result["device"] == "cpu":
            result["warnings"].append("no_gpu_available_real_paligemma_runs_will_be_slow")
            ram = result["resource_status"].get("system_ram_gib")
            if ram is not None and float(ram) < 16.0:
                result["warnings"].append("cpu_ram_below_recommended_for_paligemma2_3b_local_load")
        if float(result["resource_status"].get("cache_disk_free_gib") or 0.0) < 8.0:
            result["warnings"].append("cache_disk_free_space_below_recommended_for_paligemma2_3b")
    except Exception as exc:
        result["errors"].append(f"torch_import_failed: {exc}")
        return result
    try:
        import transformers  # type: ignore

        result["transformers_version"] = str(transformers.__version__)
    except Exception as exc:
        result["errors"].append(f"transformers_import_failed: {exc}")
        return result
    try:
        from huggingface_hub import HfApi, get_token  # type: ignore

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or get_token()
        result["hf_token_present"] = bool(token)
        if token and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
            result["hf_token_source"] = "huggingface_cache"
        elif token:
            result["hf_token_source"] = "environment"
        else:
            result["hf_token_source"] = "none"
        try:
            whoami = HfApi(token=token).whoami() if token else None
            result["hf_whoami"] = {
                "name": whoami.get("name") if isinstance(whoami, dict) else None,
                "orgs": [org.get("name") for org in whoami.get("orgs", [])] if isinstance(whoami, dict) else [],
            } if whoami else None
        except Exception as exc:
            result["hf_whoami_error"] = str(exc)
        try:
            info = HfApi(token=token).model_info(model_id)
            result["model_info_accessible"] = True
            result["model_private"] = bool(getattr(info, "private", False))
            result["model_gated"] = str(getattr(info, "gated", "unknown"))
        except Exception as exc:
            result["model_info_accessible"] = False
            result["model_access_error"] = str(exc)
            result["errors"].append("model_info_not_accessible_or_license_not_accepted")
    except Exception as exc:
        result["warnings"].append(f"huggingface_hub_check_skipped: {exc}")
    if check_processor:
        try:
            from transformers import AutoProcessor  # type: ignore

            AutoProcessor.from_pretrained(model_id, cache_dir=model_cache_dir or None)
            result["processor_load_status"] = "ok"
        except Exception as exc:
            result["processor_load_status"] = "failed"
            result["processor_load_error"] = str(exc)
            result["errors"].append("processor_load_failed")
    if check_model_load:
        try:
            _ = PaliGemmaBackend(
                VisualProposalConfig(
                    backend="paligemma",
                    model_id=model_id,
                    model_cache_dir=model_cache_dir,
                    max_frames=1,
                )
            )
            result["model_load_status"] = "ok"
        except Exception as exc:
            result["model_load_status"] = "failed"
            result["model_load_error"] = str(exc)
            result["errors"].append("model_load_failed")
    result["can_proceed"] = not result["errors"] or (result["errors"] == ["model_info_not_accessible_or_license_not_accepted"] and bool(result.get("processor_load_status") == "ok"))
    return result


def _resolve_session_path(raw: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(raw).expanduser().resolve()
    if path.is_dir():
        session_dir = path
    elif path.name in {"ui_session_manifest.json", "session_pipeline_report.json"}:
        session_dir = path.parent
    else:
        raise FileNotFoundError(f"Could not resolve evaluation session from {path}")
    manifest = read_json(session_dir / "ui_session_manifest.json")
    report = read_json(session_dir / "session_pipeline_report.json")
    return session_dir, manifest, report


def _roi_bounds(width: int, height: int, mode: str, custom: str) -> tuple[int, int, int, int]:
    if mode == "full_frame":
        return 0, 0, width, height
    if mode == "center_pool":
        return int(width * 0.08), int(height * 0.05), int(width * 0.94), int(height * 0.88)
    if mode == "lower_water":
        return int(width * 0.05), int(height * 0.35), int(width * 0.98), int(height * 0.98)
    if mode == "custom":
        try:
            left, top, right, bottom = [float(part.strip()) for part in custom.split(",")]
            return int(width * left), int(height * top), int(width * right), int(height * bottom)
        except Exception as exc:
            raise ValueError(f"Invalid --custom-roi value {custom!r}; expected left,top,right,bottom fractions") from exc
    raise ValueError(f"Unsupported ROI mode: {mode}")


def _parse_response(text: str, rule: str) -> bool:
    normalized = " ".join(str(text or "").lower().strip().replace(".", " ").replace(",", " ").split())
    if rule == "naive_contains_yes":
        return "yes" in normalized
    if rule == "strict_yes_no":
        tokens = normalized.split()
        if not tokens:
            return False
        first = tokens[-1] if len(tokens) > 1 and tokens[-1] in {"yes", "no"} else tokens[0]
        return first == "yes" or normalized == "yes"
    if rule == "yes_no_first_token_margin":
        return False
    raise ValueError(f"Unsupported decision rule: {rule}")


def _merge_intervals(intervals: list[dict[str, Any]], max_gap: float) -> list[dict[str, Any]]:
    if not intervals:
        return []
    merged = [dict(item) for item in sorted(intervals, key=lambda row: float(row["start_seconds"]))]
    out = [merged[0]]
    for current in merged[1:]:
        previous = out[-1]
        if float(current["start_seconds"]) - float(previous["end_seconds"]) <= max_gap:
            previous["end_seconds"] = max(float(previous["end_seconds"]), float(current["end_seconds"]))
            previous["positive_frame_count"] = int(previous.get("positive_frame_count", 0)) + int(current.get("positive_frame_count", 0))
            previous["max_score"] = max(float(previous.get("max_score", 0.0)), float(current.get("max_score", 0.0)))
            previous["merged_interval_count"] = int(previous.get("merged_interval_count", 1)) + int(current.get("merged_interval_count", 1))
        else:
            out.append(current)
    return out


def frame_predictions_to_intervals(
    rows: Sequence[dict[str, Any]],
    *,
    config: VisualProposalConfig,
    decision_rule: str,
    prompt_id: str,
    duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    positives = [
        row
        for row in rows
        if row.get("prompt_id") == prompt_id
        and row.get("decision_rule") == decision_rule
        and bool(row.get("is_positive"))
        and float(row.get("score", 0.0) or 0.0) >= config.confidence_threshold
    ]
    positives.sort(key=lambda row: float(row["timestamp_seconds"]))
    groups: list[list[dict[str, Any]]] = []
    for row in positives:
        if not groups or float(row["timestamp_seconds"]) - float(groups[-1][-1]["timestamp_seconds"]) > config.grouping_threshold_seconds:
            groups.append([row])
        else:
            groups[-1].append(row)
    intervals: list[dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        start = max(0.0, float(group[0]["timestamp_seconds"]) - config.buffer_start_seconds)
        end = float(group[-1]["timestamp_seconds"]) + config.buffer_end_seconds
        if duration_seconds is not None and duration_seconds > 0:
            end = min(duration_seconds, end)
        intervals.append(
            {
                "visual_interval_id": f"vis-int-{idx:04d}",
                "start_seconds": round(start, 3),
                "end_seconds": round(max(start + 0.1, end), 3),
                "anchor_timestamp_seconds": round(float(group[int(len(group) / 2)]["timestamp_seconds"]), 3),
                "first_positive_timestamp_seconds": round(float(group[0]["timestamp_seconds"]), 3),
                "last_positive_timestamp_seconds": round(float(group[-1]["timestamp_seconds"]), 3),
                "positive_frame_count": len(group),
                "max_score": max(float(row.get("score", 0.0) or 0.0) for row in group),
                "prompt_id": prompt_id,
                "decision_rule": decision_rule,
                "mode": config.mode,
                "roi_mode": config.roi_mode,
                "merged_interval_count": 1,
            }
        )
    return _merge_intervals(intervals, config.merge_gap_seconds)


def intervals_to_proposals(intervals: Sequence[dict[str, Any]], *, session_id: str, source_video_path: str) -> list[dict[str, Any]]:
    proposals = []
    for idx, interval in enumerate(intervals, start=1):
        proposals.append(
            {
                "proposal_id": f"vis-prop-{idx:04d}",
                "session_id": session_id,
                "source_video_path": source_video_path,
                "timestamp": float(interval["anchor_timestamp_seconds"]),
                "start_seconds": float(interval["start_seconds"]),
                "end_seconds": float(interval["end_seconds"]),
                "proposal_frontend": "visual_vlm_paligemma2",
                "proposal_provenance": "visual_vlm_paligemma2",
                "raw_proposal_score": float(interval.get("max_score", 0.0) or 0.0),
                "visual_interval_id": interval.get("visual_interval_id"),
                "prompt_id": interval.get("prompt_id"),
                "decision_rule": interval.get("decision_rule"),
                "roi_mode": interval.get("roi_mode"),
                "mode": interval.get("mode"),
                "positive_frame_count": interval.get("positive_frame_count"),
                "pipeline_selected": False,
                "pipeline_stage": "visual_proposal_only",
            }
        )
    return proposals


def _sampling_windows(manifest: dict[str, Any], session_dir: Path, mode: str, pre: float, post: float) -> list[tuple[float, float, str]]:
    duration = float(manifest.get("session", {}).get("session_duration_seconds") or 0.0)
    if mode == "full-session":
        return [(0.0, duration if duration > 0 else math.inf, "full_session")]
    windows: list[tuple[float, float, str]] = []
    for detection in manifest.get("detections", []):
        ts = float(detection.get("timestamp_seconds") or 0.0)
        windows.append((max(0.0, ts - pre), ts + post, str(detection.get("id") or "audio_detection")))
    review_path = session_dir / "evaluation_review.json"
    if mode == "oracle-gated" and review_path.exists():
        review = read_json(review_path)
        for item in review.get("falseNegatives", []):
            ts = float(item.get("timestampSeconds") or item.get("timestamp_seconds") or 0.0)
            windows.append((max(0.0, ts - pre), ts + post, str(item.get("id") or "false_negative")))
    if duration > 0:
        windows = [(start, min(duration, end), reason) for start, end, reason in windows if start < duration and end > 0]
    return windows


def _timestamp_allowed(timestamp: float, windows: Sequence[tuple[float, float, str]]) -> tuple[bool, str | None]:
    for start, end, reason in windows:
        if start <= timestamp <= end:
            return True, reason
    return False, None


def _motion_proxy_score(image: Any) -> float:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    # Deliberately weak visual proxy for smoke tests only. It is not a VLM result.
    edge_x = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
    edge_y = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    contrast = float(arr.std())
    score = (0.45 * edge_x + 0.35 * edge_y + 0.20 * contrast) / 80.0
    return float(max(0.0, min(1.0, score)))


class PaliGemmaBackend:
    def __init__(self, config: VisualProposalConfig):
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PaliGemma backend requires torch and transformers. Install them in the active environment or use --backend availability-check."
            ) from exc
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(config.model_id, cache_dir=config.model_cache_dir or None)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            cuda_dtype = os.environ.get("DIVESENSEI_VLM_CUDA_DTYPE", "bfloat16").strip().lower()
            if cuda_dtype in {"bfloat16", "bf16"} and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                dtype = torch.bfloat16
            elif cuda_dtype in {"float32", "fp32"}:
                dtype = torch.float32
            else:
                dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            cpu_dtype = os.environ.get("DIVESENSEI_VLM_CPU_DTYPE", "float32").strip().lower()
            dtype = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(cpu_dtype, torch.float32)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            config.model_id,
            cache_dir=config.model_cache_dir or None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = config.max_new_tokens

    def predict_batch(self, images: Sequence[Any], prompts: Sequence[str]) -> list[dict[str, Any]]:
        import torch.nn.functional as F  # type: ignore

        with self.torch.inference_mode():
            inputs = self.processor(images=list(images), text=list(prompts), return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
            answers = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            scores = outputs.scores
            generated_tokens = outputs.sequences
            first_step_logits = scores[0] if scores else None
            tokenizer = self.processor.tokenizer
            yes_ids = tokenizer.encode("yes", add_special_tokens=False) + tokenizer.encode(" yes", add_special_tokens=False)
            no_ids = tokenizer.encode("no", add_special_tokens=False) + tokenizer.encode(" no", add_special_tokens=False)
            rows: list[dict[str, Any]] = []
            for i, answer in enumerate(answers):
                token_probs = [
                    F.softmax(step_logits, dim=-1)[i, token_id.item()].item()
                    for step_logits, token_id in zip(scores, generated_tokens[i, -len(scores):], strict=False)
                ]
                yes_prob = None
                no_prob = None
                margin = None
                if first_step_logits is not None:
                    probs = F.softmax(first_step_logits[i], dim=-1)
                    yes_values = [probs[token_id].item() for token_id in set(yes_ids) if 0 <= token_id < probs.shape[0]]
                    no_values = [probs[token_id].item() for token_id in set(no_ids) if 0 <= token_id < probs.shape[0]]
                    yes_prob = float(max(yes_values)) if yes_values else None
                    no_prob = float(max(no_values)) if no_values else None
                    if yes_prob is not None and no_prob is not None:
                        margin = float(yes_prob - no_prob)
                rows.append(
                    {
                        "raw_response": str(answer).strip(),
                        "token_probability_mean": float(sum(token_probs) / len(token_probs)) if token_probs else 0.0,
                        "token_probabilities": [float(value) for value in token_probs],
                        "yes_first_token_probability": yes_prob,
                        "no_first_token_probability": no_prob,
                        "yes_no_first_token_margin": margin,
                    }
                )
            return rows


def run_visual_proposal_generation(session_path: str | Path, output_dir: str | Path | None, config: VisualProposalConfig) -> tuple[dict[str, Any], Path]:
    started = time.time()
    session_dir, manifest, _report = _resolve_session_path(session_path)
    output_root = Path(output_dir).expanduser().resolve() if output_dir else session_dir / "exports" / "visual-vlm-proposals"
    output_root.mkdir(parents=True, exist_ok=True)
    frame_path = output_root / "visual_frame_predictions.jsonl"
    interval_path = output_root / "visual_event_intervals.json"
    proposal_path = output_root / "visual_proposals.jsonl"
    merged_path = output_root / "merged_proposal_diagnostics.jsonl"
    summary_path = output_root / "visual_vlm_proposal_summary.json"

    source_video_path, requested_source_video_path, source_video_resolution = _resolve_visual_source_video(session_dir, manifest)
    manifest_duration = manifest.get("session", {}).get("session_duration_seconds")
    if source_video_resolution == "review_proxy_fallback" and manifest_duration:
        duration = manifest_duration
    else:
        duration = probe_media_duration_seconds(source_video_path) if source_video_path.exists() else manifest_duration
    summary: dict[str, Any] = {
        "status": "started",
        "session_id": session_dir.name,
        "source_video_path": str(source_video_path),
        "requested_source_video_path": requested_source_video_path,
        "source_video_resolution": source_video_resolution,
        "config": {
            "mode": config.mode,
            "backend": config.backend,
            "model_id": config.model_id,
            "fps": config.fps,
            "resolution": config.resolution,
            "roi_mode": config.roi_mode,
            "prompt_ids": list(config.prompt_ids),
            "decision_rules": list(config.decision_rules),
        },
        "artifacts": {
            "visual_frame_predictions": str(frame_path),
            "visual_event_intervals": str(interval_path),
            "visual_proposals": str(proposal_path),
            "merged_proposal_diagnostics": str(merged_path),
            "summary": str(summary_path),
        },
    }
    if not source_video_path.exists():
        summary.update({"status": "skipped", "reason": "source_video_missing"})
        write_json(summary_path, summary)
        write_json(interval_path, {"intervals": [], "summary": summary})
        write_jsonl(proposal_path, [])
        write_jsonl(merged_path, _merged_proposal_rows(session_dir, []))
        write_jsonl(frame_path, [])
        return summary, summary_path
    if config.backend == "availability-check":
        summary.update({"status": "skipped", "reason": "availability_check_only"})
        write_json(summary_path, summary)
        write_json(interval_path, {"intervals": [], "summary": summary})
        write_jsonl(proposal_path, [])
        write_jsonl(merged_path, _merged_proposal_rows(session_dir, []))
        write_jsonl(frame_path, [])
        return summary, summary_path

    backend: PaliGemmaBackend | None = None
    if config.backend == "paligemma":
        try:
            backend = PaliGemmaBackend(config)
        except Exception as exc:
            summary.update(
                {
                    "status": "skipped",
                    "reason": "paligemma_backend_unavailable",
                    "error": str(exc),
                    "model_access_note": "Accept the Google PaliGemma license on Hugging Face and provide a valid token/cache before running VLM inference.",
                }
            )
            write_json(summary_path, summary)
            write_json(interval_path, {"intervals": [], "summary": summary})
            write_jsonl(proposal_path, [])
            write_jsonl(merged_path, _merged_proposal_rows(session_dir, []))
            write_jsonl(frame_path, [])
            return summary, summary_path

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        summary.update({"status": "skipped", "reason": "video_decode_failed"})
        write_json(summary_path, summary)
        write_json(interval_path, {"intervals": [], "summary": summary})
        write_jsonl(proposal_path, [])
        write_jsonl(merged_path, _merged_proposal_rows(session_dir, []))
        write_jsonl(frame_path, [])
        return summary, summary_path
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(round(source_fps / max(0.1, config.fps))))
    windows = _sampling_windows(manifest, session_dir, config.mode, config.audio_gate_pre_seconds, config.audio_gate_post_seconds)

    frame_rows: list[dict[str, Any]] = []
    from PIL import Image

    batch_images: list[Any] = []
    batch_meta: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal batch_images, batch_meta
        if not batch_images:
            return
        for prompt_id in config.prompt_ids:
            prompt = DEFAULT_PROMPTS[prompt_id]
            if config.backend == "motion-proxy":
                predictions = [{"raw_response": "yes" if _motion_proxy_score(image) >= config.confidence_threshold else "no", "token_probability_mean": _motion_proxy_score(image), "token_probabilities": []} for image in batch_images]
            else:
                assert backend is not None
                predictions = backend.predict_batch(batch_images, [prompt] * len(batch_images))
            for meta, prediction in zip(batch_meta, predictions, strict=True):
                raw_response = str(prediction.get("raw_response") or "")
                for rule in config.decision_rules:
                    if rule == "yes_no_first_token_margin":
                        margin = prediction.get("yes_no_first_token_margin")
                        yes_prob = prediction.get("yes_first_token_probability")
                        positive = margin is not None and float(margin) > 0.0
                        score = float(yes_prob if yes_prob is not None else 0.0)
                    else:
                        positive = _parse_response(raw_response, rule)
                        score = float(prediction.get("token_probability_mean", 0.0) or 0.0)
                    frame_rows.append(
                        {
                            **meta,
                            "model_id": config.model_id,
                            "backend": config.backend,
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "decision_rule": rule,
                            "raw_response": raw_response,
                            "score": score,
                            "yes_first_token_probability": prediction.get("yes_first_token_probability"),
                            "no_first_token_probability": prediction.get("no_first_token_probability"),
                            "yes_no_first_token_margin": prediction.get("yes_no_first_token_margin"),
                            "token_probabilities": prediction.get("token_probabilities", []),
                            "is_positive": bool(positive),
                        }
                    )
        batch_images = []
        batch_meta = []

    frame_idx = 0
    sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue
        timestamp = frame_idx / source_fps if source_fps > 0 else 0.0
        allowed, window_reason = _timestamp_allowed(timestamp, windows)
        if allowed:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            left, top, right, bottom = _roi_bounds(w, h, config.roi_mode, config.custom_roi)
            crop = rgb[max(0, top): min(h, bottom), max(0, left): min(w, right)]
            image = Image.fromarray(crop).resize((config.resolution, config.resolution))
            batch_images.append(image)
            batch_meta.append(
                {
                    "session_id": session_dir.name,
                    "source_video_path": str(source_video_path),
                    "mode": config.mode,
                    "roi_mode": config.roi_mode,
                    "frame_index": int(frame_idx),
                    "timestamp_seconds": round(timestamp, 3),
                    "window_reason": window_reason,
                    "roi_bounds_pixels": [int(left), int(top), int(right), int(bottom)],
                    "resolution": int(config.resolution),
                }
            )
            sampled += 1
            if len(batch_images) >= config.batch_size:
                flush()
            if config.max_frames and sampled >= config.max_frames:
                break
        frame_idx += 1
    flush()
    cap.release()

    intervals: list[dict[str, Any]] = []
    for prompt_id in config.prompt_ids:
        for rule in config.decision_rules:
            intervals.extend(
                frame_predictions_to_intervals(
                    frame_rows,
                    config=config,
                    decision_rule=rule,
                    prompt_id=prompt_id,
                    duration_seconds=float(duration or 0.0),
                )
            )
    proposals = intervals_to_proposals(intervals, session_id=session_dir.name, source_video_path=str(source_video_path))
    write_jsonl(frame_path, frame_rows)
    write_json(interval_path, {"intervals": intervals, "summary": summary})
    write_jsonl(proposal_path, proposals)
    merged_rows = _merged_proposal_rows(session_dir, proposals)
    write_jsonl(merged_path, merged_rows)
    summary.update(
        {
            "status": "complete",
            "source_fps": source_fps,
            "total_frames": total_frames,
            "sampled_frame_count": sampled,
            "prediction_row_count": len(frame_rows),
            "interval_count": len(intervals),
            "proposal_count": len(proposals),
            "merged_proposal_count": len(merged_rows),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    write_json(summary_path, summary)
    return summary, summary_path


def _merged_proposal_rows(session_dir: Path, visual_rows: Sequence[dict[str, Any]], tolerance_seconds: float = 2.0) -> list[dict[str, Any]]:
    audio_rows = load_jsonl(session_dir / "proposal_diagnostics.jsonl")
    merged: list[dict[str, Any]] = []
    visual_used: set[int] = set()
    for audio in audio_rows:
        row = dict(audio)
        row["proposal_provenance"] = "audio"
        audio_ts = float(row.get("timestamp", 0.0) or 0.0)
        overlaps = [
            (idx, abs(audio_ts - float(visual.get("timestamp", 0.0) or 0.0)), visual)
            for idx, visual in enumerate(visual_rows)
            if abs(audio_ts - float(visual.get("timestamp", 0.0) or 0.0)) <= tolerance_seconds
        ]
        if overlaps:
            idx, delta, visual = sorted(overlaps, key=lambda item: item[1])[0]
            visual_used.add(idx)
            row["proposal_provenance"] = "audio_visual_overlap"
            row["visual_vlm_match_delta_seconds"] = float(delta)
            row["visual_vlm_proposal_id"] = visual.get("proposal_id")
            row["visual_vlm_score"] = visual.get("raw_proposal_score")
            row["visual_vlm_prompt_id"] = visual.get("prompt_id")
            row["visual_vlm_decision_rule"] = visual.get("decision_rule")
            row["visual_vlm_roi_mode"] = visual.get("roi_mode")
        merged.append(row)
    for idx, visual in enumerate(visual_rows):
        if idx in visual_used:
            continue
        row = dict(visual)
        row["proposal_provenance"] = "visual_vlm_paligemma2"
        row["pipeline_stage"] = row.get("pipeline_stage") or "visual_proposal_only"
        merged.append(row)
    merged.sort(key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VisualProposalConfig(
        mode=args.mode,
        backend=args.backend,
        model_id=args.model_id,
        model_cache_dir=args.model_cache_dir,
        fps=float(args.fps),
        resolution=int(args.resolution),
        roi_mode=args.roi_mode,
        custom_roi=args.custom_roi,
        batch_size=int(args.batch_size),
        confidence_threshold=float(args.confidence_threshold),
        grouping_threshold_seconds=float(args.grouping_threshold_seconds),
        buffer_start_seconds=float(args.buffer_start_seconds),
        buffer_end_seconds=float(args.buffer_end_seconds),
        merge_gap_seconds=float(args.merge_gap_seconds),
        audio_gate_pre_seconds=float(args.audio_gate_pre_seconds),
        audio_gate_post_seconds=float(args.audio_gate_post_seconds),
        prompt_ids=tuple(args.prompt_ids or VisualProposalConfig().prompt_ids),
        decision_rules=tuple(args.decision_rules or VisualProposalConfig().decision_rules),
        max_frames=int(args.max_frames),
    )
    summary, path = run_visual_proposal_generation(args.session_path, args.output_dir or None, config)
    print(json.dumps({"summary_path": str(path), **summary}, indent=2))
    return 0


def preflight_main(argv: Sequence[str] | None = None) -> int:
    args = build_preflight_parser().parse_args(argv)
    result = visual_vlm_preflight(
        backend=args.backend,
        model_id=args.model_id,
        model_cache_dir=args.model_cache_dir,
        check_processor=bool(args.check_processor),
        check_model_load=bool(args.check_model_load),
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("can_proceed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
