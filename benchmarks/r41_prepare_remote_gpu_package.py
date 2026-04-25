from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = ROOT / "outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957"
DEFAULT_PACKAGE_ROOT = ROOT / "outputs/r41_remote_gpu_package"
DEFAULT_ARCHIVE = ROOT / "outputs/r41_remote_gpu_package.tar.gz"

SESSION_KEEP_FILES = (
    "ui_session_manifest.json",
    "session_pipeline_report.json",
    "evaluation_review.json",
    "proposal_diagnostics.jsonl",
    "proposal_diagnostics_summary.json",
    "web/session_source_review.mp4",
    "exports/evaluation-review/reviewed_candidates.jsonl",
    "exports/evaluation-review/false_negatives.jsonl",
)


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree(src: Path, dst: Path, *, hardlink_files: bool = False) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    copy_function = os.link if hardlink_files else shutil.copy2
    shutil.copytree(src, dst, ignore=ignore, copy_function=copy_function)


def build_package(
    session_root: Path,
    package_root: Path,
    archive_path: Path | None,
    *,
    model_cache_root: Path | None = None,
    model_snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    _copy_tree(ROOT / "src", package_root / "src")
    (package_root / "benchmarks").mkdir(parents=True, exist_ok=True)
    _copy_file(ROOT / "benchmarks/r41_remote_gpu_runner.py", package_root / "benchmarks/r41_remote_gpu_runner.py")

    session_dst = package_root / "outputs" / session_root.name
    copied_session_files: list[str] = []
    missing_session_files: list[str] = []
    for rel in SESSION_KEEP_FILES:
        if _copy_file(session_root / rel, session_dst / rel):
            copied_session_files.append(rel)
        else:
            missing_session_files.append(rel)

    copied_model_cache = False
    if model_cache_root is not None and model_cache_root.exists():
        cache_dst = package_root / "hf-cache" / model_cache_root.name
        _copy_tree(model_cache_root, cache_dst, hardlink_files=True)
        copied_model_cache = True

    copied_local_model_dir = False
    local_model_dst = Path()
    if model_snapshot_root is not None and model_snapshot_root.exists():
        model_name = model_snapshot_root.parent.parent.name.replace("models--", "")
        local_model_dst = package_root / "local-model" / model_name
        _copy_tree(model_snapshot_root, local_model_dst, hardlink_files=True)
        copied_local_model_dir = True

    manifest = {
        "package_id": "r41_paligemma_remote_gpu_package",
        "session_root_name": session_root.name,
        "entrypoint": "benchmarks/r41_remote_gpu_runner.py",
        "copied_session_files": copied_session_files,
        "missing_session_files": missing_session_files,
        "excluded_by_design": [
            "full outputs/ tree",
            "local virtualenvs",
            "node_modules",
            "historical benchmark outputs",
            "proposal_raw_peaks.jsonl",
            "proposal_transient_peaks.jsonl",
            "session_audio.wav",
            "Hugging Face model cache",
        ],
        "assumptions": [
            "Runner executes from the package root with PYTHONPATH=src.",
            "The review proxy video is sufficient for visual proposal probing.",
            "HF_TOKEN is supplied by Kaggle/Colab secret or environment.",
            "GPU execution is required; CPU execution is intentionally rejected by default.",
        ],
        "packaged_model_cache": copied_model_cache,
        "packaged_model_cache_root": str((package_root / "hf-cache" / model_cache_root.name).relative_to(package_root)) if copied_model_cache and model_cache_root is not None else "",
        "packaged_local_model_dir": copied_local_model_dir,
        "packaged_local_model_root": str(local_model_dst.relative_to(package_root)) if copied_local_model_dir else "",
    }
    (package_root / "REMOTE_PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest["package_root"] = str(package_root)
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            archive_path.unlink()
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(package_root, arcname=package_root.name)
        manifest["archive_path"] = str(archive_path)
        manifest["archive_size_bytes"] = archive_path.stat().st_size
    else:
        manifest["archive_path"] = ""
        manifest["archive_size_bytes"] = 0
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the minimal r41 remote GPU package for Kaggle/Colab.")
    parser.add_argument("--session-root", default=str(DEFAULT_SESSION))
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--archive-path", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--model-cache-root", default="", help="Optional local cached model directory to include in the remote package.")
    parser.add_argument("--model-snapshot-root", default="", help="Optional local snapshot directory to include once as a standalone local model directory.")
    parser.add_argument("--skip-archive", action="store_true", help="Build the package directory only; skip the tar.gz archive.")
    args = parser.parse_args()

    model_cache_root = Path(args.model_cache_root).expanduser() if args.model_cache_root else None
    model_snapshot_root = Path(args.model_snapshot_root).expanduser() if args.model_snapshot_root else None
    archive_path = None if args.skip_archive else Path(args.archive_path)
    manifest = build_package(
        Path(args.session_root),
        Path(args.package_root),
        archive_path,
        model_cache_root=model_cache_root,
        model_snapshot_root=model_snapshot_root,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
