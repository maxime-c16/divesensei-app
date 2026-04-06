from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np


def _temp_media_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")


def _resolve_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    for fallback in (f"/usr/bin/{name}", f"/usr/local/bin/{name}"):
        if Path(fallback).exists():
            return fallback
    return name


def _review_scale_filter(max_dimension: int, target_fps: float | None = None) -> str:
    filters: list[str] = []
    if target_fps is not None and target_fps > 0:
        filters.append(f"fps={float(target_fps):.3f}")
    filters.append(
        f"scale='if(gt(iw,ih),min({max_dimension},iw),-2)':'if(gt(iw,ih),-2,min({max_dimension},ih))'"
    )
    filters.append("format=yuv420p")
    return ",".join(filters)


def probe_media_duration_seconds(video_path: str | Path) -> float | None:
    cmd = [
        _resolve_binary("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def extract_audio_wav_ffmpeg(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int,
    ffmpeg_threads: int = 0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = _temp_media_output_path(output_path)
    if temp_output_path.exists():
        temp_output_path.unlink()
    cmd = [
        _resolve_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        str(max(0, int(ffmpeg_threads))),
        "-i",
        str(video_path),
        "-map",
        "0:a:0?",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temp_output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if temp_output_path.exists():
            temp_output_path.unlink()
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio extraction failed for {output_path.name}: {stderr}")
    os.replace(temp_output_path, output_path)


def decode_audio_mono_s16le(
    video_path: str | Path,
    sample_rate: int,
    timeout_seconds: float,
    ffmpeg_threads: int = 0,
    progress_callback: Callable[[dict[str, float | int | str]], None] | None = None,
    progress_interval_seconds: float = 15.0,
) -> np.ndarray:
    cmd = [
        _resolve_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-threads",
        str(max(0, int(ffmpeg_threads))),
        "-i",
        str(video_path),
        "-map",
        "0:a:0?",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    started_at = time.monotonic()
    deadline = started_at + float(timeout_seconds)
    heartbeat_interval = max(1.0, float(progress_interval_seconds))
    last_heartbeat_at = started_at
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if progress_callback:
        progress_callback(
            {
                "event": "audio_decode_started",
                "video_path": str(video_path),
                "sample_rate": int(sample_rate),
                "timeout_seconds": float(timeout_seconds),
                "ffmpeg_threads": int(max(0, int(ffmpeg_threads))),
            }
        )

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Audio decode timed out after {float(timeout_seconds):.1f}s")
        try:
            stdout, stderr = process.communicate(timeout=min(heartbeat_interval, remaining))
            break
        except subprocess.TimeoutExpired:
            if progress_callback:
                now = time.monotonic()
                if now - last_heartbeat_at >= heartbeat_interval:
                    progress_callback(
                        {
                            "event": "audio_decode_progress",
                            "video_path": str(video_path),
                            "elapsed_seconds": round(now - started_at, 3),
                            "timeout_seconds": float(timeout_seconds),
                        }
                    )
                    last_heartbeat_at = now
            continue

    if process.returncode != 0:
        stderr = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Failed to decode audio with ffmpeg: {stderr or 'unknown error'}")

    if progress_callback:
        progress_callback(
            {
                "event": "audio_decode_complete",
                "video_path": str(video_path),
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
                "sample_count": int(len(stdout) // np.dtype(np.int16).itemsize),
            }
        )

    samples = np.frombuffer(stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        raise RuntimeError("No decodable audio track found in video")

    return samples / 32768.0


def extract_clip_ffmpeg(
    video_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    preset: str = "ultrafast",
    ffmpeg_threads: int = 0,
    max_dimension: int = 960,
    target_fps: float | None = 30.0,
) -> None:
    start_time = max(0.0, float(start_time))
    end_time = max(start_time + 0.25, float(end_time))
    duration = end_time - start_time
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = _temp_media_output_path(output_path)
    if temp_output_path.exists():
        temp_output_path.unlink()
    cmd = [
        _resolve_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        str(max(0, int(ffmpeg_threads))),
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        _review_scale_filter(max_dimension=max_dimension, target_fps=target_fps),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        "24",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(temp_output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if temp_output_path.exists():
            temp_output_path.unlink()
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg extraction failed for {Path(output_path).name}: {stderr}")
    os.replace(temp_output_path, output_path)


def generate_review_proxy_ffmpeg(
    video_path: str | Path,
    output_path: str | Path,
    preset: str = "ultrafast",
    ffmpeg_threads: int = 0,
    max_dimension: int = 720,
    target_fps: float = 24.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = _temp_media_output_path(output_path)
    if temp_output_path.exists():
        temp_output_path.unlink()
    cmd = [
        _resolve_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        str(max(0, int(ffmpeg_threads))),
        "-i",
        str(video_path),
        "-vf",
        _review_scale_filter(max_dimension=max_dimension, target_fps=target_fps),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        "24",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(temp_output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if temp_output_path.exists():
            temp_output_path.unlink()
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg review proxy failed for {output_path.name}: {stderr}")
    os.replace(temp_output_path, output_path)
