from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_NAME = "evaluation_Compete-16-11-2025-first-10min_20260422-154957"
DEFAULT_MODEL_ID = "google/paligemma2-3b-mix-224"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    return {
        "command": " ".join(args),
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_json_stdout(run: dict[str, Any]) -> dict[str, Any]:
    text = str(run.get("stdout") or "").strip()
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def import_status() -> dict[str, Any]:
    status: dict[str, Any] = {}
    for module in ("torch", "transformers", "PIL", "cv2", "huggingface_hub"):
        try:
            imported = __import__(module)
            status[module] = {"ok": True, "version": getattr(imported, "__version__", None)}
        except Exception as exc:
            status[module] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if status.get("torch", {}).get("ok"):
        import torch  # type: ignore

        status["torch"]["cuda_available"] = bool(torch.cuda.is_available())
        status["torch"]["cuda_device_count"] = int(torch.cuda.device_count())
        status["torch"]["cuda_devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": list(torch.cuda.get_device_capability(idx)),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return status


def ensure_gpu(imports: dict[str, Any], allow_cpu: bool) -> None:
    cuda_available = bool(imports.get("torch", {}).get("cuda_available"))
    if cuda_available or allow_cpu:
        return
    raise RuntimeError("CUDA GPU is required for this remote benchmark. CPU-only execution is intentionally blocked.")


def zip_outputs(output_root: Path, bundle_path: Path) -> None:
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent))


def resolve_session_root(requested: str) -> Path:
    requested_path = Path(requested)
    if requested_path.exists():
        return requested_path.resolve()

    packaged_outputs = ROOT / "outputs"
    if not packaged_outputs.exists():
        raise FileNotFoundError(f"Session root not found and packaged outputs directory is missing: {requested}")

    candidates = sorted(
        path
        for path in packaged_outputs.glob("evaluation_*")
        if path.is_dir() and (path / "web/session_source_review.mp4").exists()
    )
    if len(candidates) == 1:
        return candidates[0].resolve()

    candidate_names = [path.name for path in candidates]
    raise FileNotFoundError(
        "Session root not found. "
        f"requested={requested!r}; packaged evaluation candidates={candidate_names}"
    )


def render_summary(path: Path, payload: dict[str, Any]) -> None:
    smoke = payload.get("smoke_summary", {})
    full = payload.get("full_summary", {})
    preflight = payload.get("preflight", {})
    imports = payload.get("imports", {})
    smoke_run = payload.get("smoke_run") or {}
    full_run = payload.get("full_run") or {}
    lines = [
        "# R41 Remote PaliGemma GPU Run",
        "",
        "## Runtime",
        f"- CUDA available: `{imports.get('torch', {}).get('cuda_available')}`",
        f"- CUDA devices: `{imports.get('torch', {}).get('cuda_devices')}`",
        f"- model: `{payload.get('model_id')}`",
        f"- cache dir: `{payload.get('cache_dir')}`",
        f"- HF token present: `{preflight.get('hf_token_present')}`",
        f"- processor load: `{preflight.get('processor_load_status')}`",
        f"- can proceed: `{preflight.get('can_proceed')}`",
        "",
        "## One-Frame Smoke",
        f"- return code: `{smoke_run.get('returncode')}`",
        f"- elapsed seconds: `{smoke_run.get('elapsed_seconds')}`",
        f"- status: `{smoke.get('status')}`",
        f"- frame predictions: `{smoke.get('frame_prediction_count')}`",
        "",
        "## Audio-Gated 1 FPS Benchmark",
        f"- return code: `{full_run.get('returncode')}`",
        f"- elapsed seconds: `{full_run.get('elapsed_seconds')}`",
        f"- status: `{full.get('status')}`",
        f"- visual proposals: `{full.get('visual_proposal_count')}`",
        f"- visual intervals: `{full.get('visual_interval_count')}`",
        "",
        "## Bundle",
        f"- path: `{payload.get('bundle_path')}`",
    ]
    write_md(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the r41 PaliGemma remote GPU benchmark package.")
    parser.add_argument("--session-root", default=str(ROOT / "outputs" / DEFAULT_SESSION_NAME))
    parser.add_argument("--output-root", default=str(ROOT / "outputs/r41_remote_gpu_results"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", str(ROOT / ".hf-cache")))
    parser.add_argument("--prompt-id", default="diving_attempt")
    parser.add_argument("--decision-rule", default="yes_no_first_token_margin")
    parser.add_argument("--roi-mode", default="full_frame", choices=["full_frame", "center_pool", "lower_water", "custom"])
    parser.add_argument("--smoke-max-frames", type=int, default=1)
    parser.add_argument("--full-fps", type=float, default=1.0)
    parser.add_argument("--allow-cpu", action="store_true", help="Debug only. The governed remote run should use CUDA.")
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args()

    session_root = resolve_session_root(args.session_root)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    imports = import_status()
    ensure_gpu(imports, args.allow_cpu)

    python = sys.executable
    preflight_run = run_cmd(
        [
            python,
            "-m",
            "divesensei.cli",
            "visual-vlm-preflight",
            "--model-id",
            args.model_id,
            "--model-cache-dir",
            str(cache_dir),
            "--check-processor",
        ],
        timeout=20 * 60,
    )
    preflight = parse_json_stdout(preflight_run)
    write_json(output_root / "r41_remote_preflight.json", {"imports": imports, "run": preflight_run, "preflight": preflight})

    smoke_dir = output_root / "smoke_one_frame"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_run = run_cmd(
        [
            python,
            "-m",
            "divesensei.cli",
            "visual-vlm-proposals",
            str(session_root),
            "--backend",
            "paligemma",
            "--mode",
            "audio-gated",
            "--roi-mode",
            args.roi_mode,
            "--fps",
            str(args.full_fps),
            "--model-id",
            args.model_id,
            "--model-cache-dir",
            str(cache_dir),
            "--max-frames",
            str(args.smoke_max_frames),
            "--prompt-id",
            args.prompt_id,
            "--decision-rule",
            args.decision_rule,
            "--output-dir",
            str(smoke_dir),
        ],
        timeout=90 * 60,
    )
    smoke_summary = parse_json_stdout(smoke_run)
    if not smoke_summary and (smoke_dir / "visual_vlm_proposal_summary.json").exists():
        smoke_summary = json.loads((smoke_dir / "visual_vlm_proposal_summary.json").read_text())

    full_run: dict[str, Any] | None = None
    full_summary: dict[str, Any] = {}
    if not args.skip_full and smoke_run["returncode"] == 0 and smoke_summary.get("status") == "complete":
        roi_label = args.roi_mode
        fps_label = str(args.full_fps).replace(".", "p")
        full_dir = output_root / f"audio_gated_{roi_label}_{fps_label}fps"
        if full_dir.exists():
            shutil.rmtree(full_dir)
        full_run = run_cmd(
            [
                python,
                "-m",
                "divesensei.cli",
                "visual-vlm-proposals",
                str(session_root),
                "--backend",
                "paligemma",
                "--mode",
                "audio-gated",
                "--roi-mode",
                args.roi_mode,
                "--fps",
                str(args.full_fps),
                "--model-id",
                args.model_id,
                "--model-cache-dir",
                str(cache_dir),
                "--prompt-id",
                args.prompt_id,
                "--decision-rule",
                args.decision_rule,
                "--output-dir",
                str(full_dir),
            ],
            timeout=6 * 60 * 60,
        )
        full_summary = parse_json_stdout(full_run)
        if not full_summary and (full_dir / "visual_vlm_proposal_summary.json").exists():
            full_summary = json.loads((full_dir / "visual_vlm_proposal_summary.json").read_text())

    bundle_path = output_root.parent / "r41_remote_gpu_results_bundle.zip"
    result = {
        "benchmark_id": "r41_paligemma_remote_gpu_first_run",
        "session_root": str(session_root),
        "model_id": args.model_id,
        "cache_dir": str(cache_dir),
        "imports": imports,
        "preflight": preflight,
        "preflight_run": preflight_run,
        "smoke_run": smoke_run,
        "smoke_summary": smoke_summary,
        "full_run": full_run,
        "full_summary": full_summary,
        "bundle_path": str(bundle_path),
        "decision_scope": "research_only_visual_proposals_no_policy_change",
    }
    write_json(output_root / "r41_remote_gpu_run_summary.json", result)
    render_summary(output_root / "r41_remote_gpu_run_summary.md", result)
    zip_outputs(output_root, bundle_path)
    print(json.dumps(result, indent=2))
    return 0 if smoke_run["returncode"] == 0 else int(smoke_run["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
