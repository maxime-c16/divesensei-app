from __future__ import annotations

import importlib.util
import shutil


def _missing_python_module(name: str) -> bool:
    return importlib.util.find_spec(name) is None


def missing_runtime_dependencies() -> list[str]:
    missing: list[str] = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if _missing_python_module("cv2"):
        missing.append("opencv-python")
    if _missing_python_module("numpy"):
        missing.append("numpy")
    return missing


def format_missing_dependencies_message(missing: list[str]) -> str:
    joined = ", ".join(missing)
    return (
        "Missing runtime dependencies: "
        f"{joined}. Install the Python package(s) and ensure ffmpeg is on PATH."
    )
