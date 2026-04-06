#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from divesensei.io.media_io import _resolve_binary
from divesensei.workflows.evaluation_session_support import NON_DIVE_SUBTYPES, normalize_non_dive_subtype


@dataclass
class AudioLabelRecord:
    id: str
    source_video_path: str
    source_file: str
    timestamp_seconds: float
    clip_start_seconds: float
    clip_duration_seconds: float
    label: str
    subtype: str | None
    notes: str
    audio_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei label-audio",
        description="Save a labeled audio clip around a timestamp for later classifier training.",
    )
    parser.add_argument("video_path", help="Path to the source session video")
    parser.add_argument("timestamp_seconds", type=float, help="Anchor timestamp in seconds")
    parser.add_argument("--label", choices=["dive", "non-dive"], required=True, help="Training label")
    parser.add_argument("--subtype", choices=list(NON_DIVE_SUBTYPES), default="", help="Optional hard-negative subtype metadata")
    parser.add_argument("--notes", default="", help="Optional note about the sound event")
    parser.add_argument("--pre-seconds", type=float, default=2.0, help="Seconds to include before the timestamp")
    parser.add_argument("--post-seconds", type=float, default=2.0, help="Seconds to include after the timestamp")
    parser.add_argument(
        "--dataset-root",
        default=".divesensei-runtime/audio-labels",
        help="Directory where labeled clips and metadata are stored",
    )
    return parser


def extract_audio_clip(video_path: Path, output_path: Path, start_time: float, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio extraction failed for {output_path.name}: {stderr}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))

    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    dataset_root = Path(args.dataset_root).resolve()
    source_stem = video_path.stem.lower()
    timestamp_slug = f"{args.timestamp_seconds:08.3f}".replace(".", "p")
    record_id = f"{source_stem}_{timestamp_slug}_{args.label}"
    clip_start = max(0.0, float(args.timestamp_seconds) - float(args.pre_seconds))
    clip_duration = float(args.pre_seconds) + float(args.post_seconds)

    clip_dir = dataset_root / source_stem / args.label
    audio_path = clip_dir / f"{record_id}.wav"
    metadata_path = clip_dir / f"{record_id}.json"
    index_path = dataset_root / "labels.jsonl"

    try:
        extract_audio_clip(video_path, audio_path, clip_start, clip_duration)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    record = AudioLabelRecord(
        id=record_id,
        source_video_path=str(video_path),
        source_file=video_path.name,
        timestamp_seconds=float(args.timestamp_seconds),
        clip_start_seconds=clip_start,
        clip_duration_seconds=clip_duration,
        label=args.label,
        subtype=normalize_non_dive_subtype(args.subtype) if args.label == "non-dive" else None,
        notes=args.notes,
        audio_path=str(audio_path),
    )

    metadata_path.write_text(json.dumps(asdict(record), indent=2))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")

    print(json.dumps({"id": record.id, "audio_path": str(audio_path), "metadata_path": str(metadata_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
