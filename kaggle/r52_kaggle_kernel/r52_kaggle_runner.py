from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_PACKAGE_SLUGS = ["r41-remote-gpu-package-v3"]
SESSION_PACKAGE_SLUGS = ["divesensei-r52-visual-recovery-sessions"]
PACKAGED_MODEL_NAME = "models--google--paligemma2-3b-mix-224"
SESSIONS = [
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
]


def _find_with_manifest(slugs: list[str], marker: str) -> Path:
    candidates: list[Path] = []
    for slug in slugs:
        candidates.extend([Path("/kaggle/input") / slug, Path("/kaggle/input/datasets/maximecauchy") / slug])
    candidates.extend(Path("/kaggle/input").glob(f"**/{marker}"))
    for candidate in candidates:
        root = candidate if candidate.is_dir() else candidate.parent
        if (root / marker).exists():
            return root
    raise FileNotFoundError(f"Could not find package marker {marker}")


def _find_session_package() -> Path:
    candidates = []
    for slug in SESSION_PACKAGE_SLUGS:
        candidates.extend([Path("/kaggle/input") / slug, Path("/kaggle/input/datasets/maximecauchy") / slug])
    candidates.extend(Path("/kaggle/input").glob("**/r52_visual_recovery_sessions"))
    for root in candidates:
        if (root / "outputs").exists():
            return root
        nested = root / "r52_visual_recovery_sessions"
        if (nested / "outputs").exists():
            return nested
    raise FileNotFoundError("Could not find r52 visual recovery session package")


def _prepare_cache(package_root: Path) -> Path:
    work_cache = Path("/kaggle/working/hf-cache")
    packaged_cache = package_root / "hf-cache" / PACKAGED_MODEL_NAME
    if packaged_cache.exists():
        target = work_cache / PACKAGED_MODEL_NAME
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(packaged_cache, target)
    work_cache.mkdir(parents=True, exist_ok=True)
    return work_cache


def _resolve_model_id(package_root: Path) -> str:
    local_model_root = package_root / "local-model"
    if local_model_root.exists():
        candidates = sorted(path for path in local_model_root.iterdir() if path.is_dir())
        if candidates:
            return str(candidates[0])
    return "google/paligemma2-3b-mix-224"


def _install_dependencies(package_root: Path) -> None:
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


def _run(cmd: list[str], cwd: Path) -> dict[str, Any]:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-8000:], file=sys.stderr)
    return {"command": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-8000:]}


def _zip_outputs(output_root: Path, bundle_path: Path) -> None:
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent))


def main() -> int:
    base_root = _find_with_manifest(BASE_PACKAGE_SLUGS, "REMOTE_PACKAGE_MANIFEST.json")
    session_root = _find_session_package()
    output_root = Path("/kaggle/working/r52_visual_recovery_scoring")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _install_dependencies(base_root)
    os.environ["PYTHONPATH"] = str(base_root / "src")
    if str(base_root / "src") not in sys.path:
        sys.path.insert(0, str(base_root / "src"))
    os.environ["HF_HOME"] = str(_prepare_cache(base_root))
    model_id = _resolve_model_id(base_root)

    import torch  # type: ignore
    import torch._dynamo  # type: ignore

    torch._dynamo.config.suppress_errors = True
    health = {
        "base_root": str(base_root),
        "session_root": str(session_root),
        "model_id": model_id,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "sessions": SESSIONS,
    }
    (output_root / "r52_remote_scoring_health.json").write_text(json.dumps(health, indent=2))
    print(json.dumps(health, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    runs = []
    for session in SESSIONS:
        source = session_root / "outputs" / session
        session_output = output_root / session / "audio_gated_full_frame_1p0fps"
        cmd = [
            sys.executable,
            "-m",
            "divesensei.cli",
            "visual-vlm-proposals",
            str(source),
            "--backend",
            "paligemma",
            "--mode",
            "audio-gated",
            "--roi-mode",
            "full_frame",
            "--fps",
            "1.0",
            "--model-id",
            model_id,
            "--model-cache-dir",
            os.environ["HF_HOME"],
            "--prompt-id",
            "diving_attempt",
            "--decision-rule",
            "yes_no_first_token_margin",
            "--output-dir",
            str(session_output),
        ]
        run = _run(cmd, base_root)
        run["session_id"] = session
        run["output_dir"] = str(session_output)
        runs.append(run)
        if run["returncode"] != 0:
            break

    summary = {"benchmark_id": "r52_visual_recovery_remote_scoring", "health": health, "runs": runs}
    (output_root / "r52_remote_scoring_summary.json").write_text(json.dumps(summary, indent=2))
    _zip_outputs(output_root, Path("/kaggle/working/r52_visual_recovery_scoring_bundle.zip"))
    if any(run["returncode"] != 0 for run in runs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
