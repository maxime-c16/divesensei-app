from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def decode_audio_mono_s16le(
    video_path: str | Path,
    sample_rate: int,
    timeout_seconds: float,
    ffmpeg_threads: int = 1,
) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        str(max(1, int(ffmpeg_threads))),
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
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Audio decode timed out after {float(timeout_seconds):.1f}s") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Failed to decode audio with ffmpeg: {stderr or 'unknown error'}")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        raise RuntimeError("No decodable audio track found in video")

    return samples / 32768.0


def extract_clip_ffmpeg(
    video_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    preset: str = "ultrafast",
    ffmpeg_threads: int = 1,
) -> None:
    start_time = max(0.0, float(start_time))
    end_time = max(start_time + 0.25, float(end_time))
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        str(max(1, int(ffmpeg_threads))),
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg extraction failed for {Path(output_path).name}: {stderr}")

