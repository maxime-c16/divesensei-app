from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_SLUGS = [
    "r41-remote-gpu-package-v3",
    "divesensei-r41-remote-gpu-package-v3",
]
PACKAGED_MODEL_NAME = "models--google--paligemma2-3b-mix-224"


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


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    package_root = _find_package_root()
    print("Package root:", package_root)
    print("r42 kernel revision: dataset-ready-rerun")
    manifest = json.loads((package_root / "REMOTE_PACKAGE_MANIFEST.json").read_text())
    print(json.dumps(manifest, indent=2)[:4000])

    _run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], cwd=package_root)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch",
            "transformers==4.53.3",
            "accelerate",
            "sentencepiece",
            "huggingface-hub",
            "opencv-python-headless",
            "Pillow",
        ],
        cwd=package_root,
    )

    os.environ["HF_HOME"] = str(_prepare_cache(package_root))
    model_id = _resolve_model_id(package_root)

    _run([sys.executable, "-c", "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"], cwd=package_root)

    cmd = [
        sys.executable,
        "benchmarks/r41_remote_gpu_runner.py",
        "--cache-dir",
        os.environ["HF_HOME"],
        "--model-id",
        model_id,
        "--output-root",
        "/kaggle/working/r42_visual_full_frame_control",
        "--prompt-id",
        "diving_attempt",
        "--decision-rule",
        "yes_no_first_token_margin",
        "--roi-mode",
        "full_frame",
        "--smoke-max-frames",
        "1",
        "--full-fps",
        "1.0",
    ]
    _run(cmd, cwd=package_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
