#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${ROOT}/kaggle/r41_remote_gpu_package"
PACKAGE_ROOT="${ROOT}/outputs/r41_remote_gpu_package"
SESSION_ROOT="${SESSION_ROOT:-}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-}"
MODEL_SNAPSHOT_ROOT="${MODEL_SNAPSHOT_ROOT:-}"

cd "${ROOT}"
ARGS=()
if [[ -n "${SESSION_ROOT}" ]]; then
  ARGS+=(--session-root "${SESSION_ROOT}")
fi
if [[ -n "${MODEL_CACHE_ROOT}" ]]; then
  ARGS+=(--model-cache-root "${MODEL_CACHE_ROOT}")
fi
if [[ -n "${MODEL_SNAPSHOT_ROOT}" ]]; then
  ARGS+=(--model-snapshot-root "${MODEL_SNAPSHOT_ROOT}")
fi
if [[ ${#ARGS[@]} -gt 0 ]]; then
  PYTHONPATH=src python3 benchmarks/r41_prepare_remote_gpu_package.py --skip-archive "${ARGS[@]}"
else
  PYTHONPATH=src python3 benchmarks/r41_prepare_remote_gpu_package.py --skip-archive
fi

mkdir -p "${DATASET_DIR}"
rm -rf "${DATASET_DIR}/r41_remote_gpu_package"
python3 - <<'PY'
import os
import shutil
from pathlib import Path

src = Path("outputs/r41_remote_gpu_package")
dst = Path("kaggle/r41_remote_gpu_package/r41_remote_gpu_package")

if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(
    src,
    dst,
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    copy_function=os.link,
)
PY

python3 - <<'PY'
import json
from pathlib import Path

metadata_path = Path("kaggle/r41_remote_gpu_package/dataset-metadata.json")
metadata = json.loads(metadata_path.read_text())
required = {"title", "id", "licenses"}
missing = sorted(required - set(metadata))
if missing:
    raise SystemExit(f"Missing required Kaggle dataset metadata keys: {missing}")
print(json.dumps({
    "dataset_id": metadata["id"],
    "dataset_dir": str(metadata_path.parent),
    "package_dir": str(metadata_path.parent / "r41_remote_gpu_package"),
}, indent=2))
PY
