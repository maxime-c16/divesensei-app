#!/usr/bin/env python3
"""
Run the audio-first dive detector on a single session video and extract clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from divesensei.detection.audio_detector import AudioVisualDiveDetector
from divesensei.detection.config import DetectionConfig, SplashEvent, extract_dive_around_splash
from divesensei.io.logging_utils import StructuredLogger, build_candidate_debug_summary
from divesensei.io.media_io import extract_clip_ffmpeg
from divesensei.metadata.ui_contract import build_ui_session_manifest, write_ui_session_manifest
from divesensei.profiles import apply_named_profile


def apply_profile_overrides(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "audio_min_score": args.audio_min_score,
        "audio_pattern_min_score": args.audio_pattern_min_score,
        "audio_decode_timeout_seconds": args.audio_decode_timeout_seconds,
    }
    merged = apply_named_profile(defaults, args.profile)
    args.audio_min_score = float(merged["audio_min_score"])
    args.audio_pattern_min_score = float(merged["audio_pattern_min_score"])
    args.audio_decode_timeout_seconds = float(merged["audio_decode_timeout_seconds"])
    if args.quality == "balanced" and args.ffmpeg_preset == "ultrafast":
        args.ffmpeg_preset = "medium"
    if args.quality == "fast" and args.ffmpeg_preset == "ultrafast":
        args.ffmpeg_preset = "ultrafast"
    return args


def build_config(args: argparse.Namespace) -> DetectionConfig:
    return DetectionConfig(
        method="audio_visual",
        splash_zone_top_norm=args.bbox[0],
        splash_zone_bottom_norm=args.bbox[1],
        splash_zone_left_norm=args.bbox[2],
        splash_zone_right_norm=args.bbox[3],
        pre_splash_duration=args.pre_duration,
        post_splash_duration=args.post_duration,
        audio_peak_threshold=args.audio_peak_threshold,
        audio_peak_min_separation_seconds=args.audio_peak_separation,
        audio_ignore_before_seconds=args.audio_ignore_before_seconds,
        audio_min_score=args.audio_min_score,
        audio_min_hf_ratio=args.audio_min_hf_ratio,
        audio_early_peak_score=args.audio_early_peak_score,
        audio_early_peak_max_seconds=args.audio_early_peak_max_seconds,
        audio_early_peak_max_hf_ratio=args.audio_early_peak_max_hf_ratio,
        audio_early_peak_max_centroid_hz=args.audio_early_peak_max_centroid_hz,
        audio_early_peak_max_flatness=args.audio_early_peak_max_flatness,
        audio_pattern_min_score=args.audio_pattern_min_score,
        audio_noise_max_peak_count=args.audio_noise_max_peak_count,
        audio_noise_max_top_ratio=args.audio_noise_max_top_ratio,
        audio_long_session_seconds=args.audio_long_session_seconds,
        audio_long_session_max_candidates=args.audio_long_session_max_candidates,
        audio_model_path=args.audio_model_path,
        audio_model_min_probability=args.audio_model_min_probability,
        audio_decode_timeout_seconds=args.audio_decode_timeout_seconds,
        ffmpeg_threads=args.ffmpeg_threads,
        opencv_threads=args.opencv_threads,
        audio_only_pre_seconds=args.audio_only_pre_seconds,
        audio_only_post_seconds=args.audio_only_post_seconds,
        audio_visual_skip_video_verification=args.skip_video_verification,
        audio_visual_verify_pre_seconds=args.audio_verify_pre,
        audio_visual_verify_post_seconds=args.audio_verify_post,
        audio_visual_verify_target_fps=args.audio_visual_verify_target_fps,
        audio_visual_max_proposals=args.audio_visual_max_proposals,
        audio_visual_min_video_score=args.audio_visual_min_video_score,
        audio_visual_hard_video_floor=args.audio_visual_hard_video_floor,
        audio_visual_audio_rescue_score=args.audio_visual_audio_rescue_score,
        audio_visual_rescue_splash_ratio=args.audio_visual_rescue_splash_ratio,
        audio_visual_min_combined_score=args.audio_visual_min_combined_score,
        audio_visual_merge_seconds=args.audio_visual_merge_seconds,
        audio_visual_max_verify_width=args.audio_visual_max_verify_width,
        enable_debug_plots=False,
    )


def candidate_to_event(config: DetectionConfig, candidate) -> SplashEvent:
    return SplashEvent(
        frame_idx=candidate.frame_idx,
        timestamp=candidate.timestamp,
        score=candidate.audio_score,
        filtered_score=candidate.combined_score,
        confidence=candidate.confidence,
        zone_info={
            "top_norm": config.splash_zone_top_norm,
            "bottom_norm": config.splash_zone_bottom_norm,
            "left_norm": config.splash_zone_left_norm,
            "right_norm": config.splash_zone_right_norm,
            "method": config.method,
            "segment_start_time": candidate.start_time,
            "segment_end_time": candidate.end_time,
            "video_score": candidate.video_score,
            "audio_score": candidate.audio_score,
            "details": candidate.details,
        },
        detection_method="audio_visual",
    )

def extract_candidate_clip_ffmpeg_path(video_path: Path, candidate, output_dir: Path, dive_number: int, preset: str, ffmpeg_threads: int) -> str:
    confidence_suffix = f"_{candidate.confidence}" if candidate.confidence != "high" else ""
    output_filename = f"dive_splash_{dive_number + 1}_t{candidate.timestamp:.1f}s{confidence_suffix}.mp4"
    output_path = output_dir / output_filename
    extract_clip_ffmpeg(
        video_path=video_path,
        output_path=output_path,
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        preset=preset,
        ffmpeg_threads=ffmpeg_threads,
    )
    return str(output_path)


def default_output_dir(video_path: str) -> Path:
    stem = Path(video_path).stem
    return Path.cwd() / "outputs" / stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei detect",
        description="Detect dives from a session video and export one clip per detected dive.",
    )
    parser.add_argument("video_path", help="Path to the input session video")
    parser.add_argument(
        "--output-dir",
        help="Directory for clips and reports. Default: ./outputs/<video-name>",
    )
    parser.add_argument(
        "--profile",
        choices=["reviewed", "long-session"],
        default="reviewed",
        help="Detection profile. Use 'long-session' for multi-minute training sessions.",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced"],
        default="fast",
        help="Clip export quality preset. 'fast' is recommended for bulk extraction.",
    )
    parser.add_argument("--pre-duration", type=float, default=6.0, help="Seconds to keep before the detected splash")
    parser.add_argument("--post-duration", type=float, default=2.0, help="Seconds to keep after the detected splash")
    parser.add_argument("--detect-only", action="store_true", help="Run detection only and skip clip extraction")
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON")
    parser.add_argument("--debug", action="store_true", help="Keep structured logs and debug summary files")

    internal = parser.add_argument_group("advanced")
    internal.add_argument("--no-extract", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--use-opencv-extraction", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--ffmpeg-preset", default="ultrafast", help=argparse.SUPPRESS)
    internal.add_argument("--ffmpeg-threads", type=int, default=1, help="Bound FFmpeg worker threads")
    internal.add_argument("--opencv-threads", type=int, default=1, help="Bound OpenCV worker threads")
    internal.add_argument("--skip-video-verification", action="store_true", default=True, help=argparse.SUPPRESS)
    internal.add_argument("--with-video-verification", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--bbox", nargs=4, type=float, default=[0.72, 0.95, 0.0, 1.0], metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), help=argparse.SUPPRESS)
    internal.add_argument("--audio-peak-threshold", type=float, default=4.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-peak-separation", type=float, default=4.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-ignore-before-seconds", type=float, default=0.35, help=argparse.SUPPRESS)
    internal.add_argument("--audio-min-score", type=float, default=4.5, help=argparse.SUPPRESS)
    internal.add_argument("--audio-min-hf-ratio", type=float, default=0.115, help=argparse.SUPPRESS)
    internal.add_argument("--audio-early-peak-score", type=float, default=4.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-early-peak-max-seconds", type=float, default=0.8, help=argparse.SUPPRESS)
    internal.add_argument("--audio-early-peak-max-hf-ratio", type=float, default=0.6, help=argparse.SUPPRESS)
    internal.add_argument("--audio-early-peak-max-centroid-hz", type=float, default=2200.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-early-peak-max-flatness", type=float, default=0.45, help=argparse.SUPPRESS)
    internal.add_argument("--audio-pattern-min-score", type=float, default=0.4, help=argparse.SUPPRESS)
    internal.add_argument("--audio-noise-max-peak-count", type=int, default=5, help=argparse.SUPPRESS)
    internal.add_argument("--audio-noise-max-top-ratio", type=float, default=1.8, help=argparse.SUPPRESS)
    internal.add_argument("--audio-long-session-seconds", type=float, default=120.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-long-session-max-candidates", type=int, default=120, help=argparse.SUPPRESS)
    internal.add_argument("--audio-model-path", default="", help=argparse.SUPPRESS)
    internal.add_argument("--audio-model-min-probability", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-decode-timeout-seconds", type=float, default=180.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-only-pre-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-only-post-seconds", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-verify-pre", type=float, default=3.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-verify-post", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-verify-target-fps", type=float, default=12.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-max-proposals", type=int, default=4, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-min-video-score", type=float, default=0.8, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-hard-video-floor", type=float, default=0.2, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-audio-rescue-score", type=float, default=4.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-rescue-splash-ratio", type=float, default=1.35, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-min-combined-score", type=float, default=3.8, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-merge-seconds", type=float, default=2.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-visual-max-verify-width", type=int, default=640, help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.output_dir:
        args.output_dir = str(default_output_dir(args.video_path))
    if args.detect_only:
        args.no_extract = True
    if args.with_video_verification:
        args.skip_video_verification = False
    return apply_profile_overrides(args)


def print_human_summary(summary: dict) -> None:
    print(f"Video: {summary['video_path']}")
    print(f"Detected dives: {summary['candidate_count']}")
    print(f"Clips written: {summary['extracted_count']}")
    print(f"Detection time: {summary['detector_seconds']:.2f}s")
    print(f"Extraction time: {summary['extract_seconds']:.2f}s")
    print(f"Total runtime: {summary['total_runtime_seconds']:.2f}s")
    print(f"Peak RSS: {summary['peak_rss_kb']} KB")
    print(f"UI manifest: {summary['ui_manifest_path']}")
    print(f"Detections CSV: {summary['detections_csv']}")
    print(f"Report: {summary['report_path']}")


def write_candidates_csv(output_dir: Path, candidates: Sequence[Any]) -> Path:
    csv_path = output_dir / "detections.csv"
    fieldnames = [
        "index",
        "timestamp",
        "start_time",
        "end_time",
        "confidence",
        "audio_score",
        "video_score",
        "combined_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, candidate in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "timestamp": candidate.timestamp,
                    "start_time": candidate.start_time,
                    "end_time": candidate.end_time,
                    "confidence": candidate.confidence,
                    "audio_score": candidate.audio_score,
                    "video_score": candidate.video_score,
                    "combined_score": candidate.combined_score,
                }
            )
    return csv_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(output_dir / "session_pipeline.log.jsonl")

    config = build_config(args)
    detector = AudioVisualDiveDetector(config)
    logger.log("session_start", video_path=video_path, output_dir=output_dir, profile=args.profile, config=config.__dict__)

    run_start = time.time()
    start = run_start
    candidates = detector.detect(str(video_path))
    detect_seconds = time.time() - start
    logger.log(
        "detection_complete",
        detector_seconds=detect_seconds,
        candidate_count=len(candidates),
        debug_summary=build_candidate_debug_summary(candidates),
    )

    report = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "profile": args.profile,
        "detector_seconds": detect_seconds,
        "config": config.__dict__,
        "candidate_count": len(candidates),
        "session_estimated_duration_seconds": max((candidate.end_time for candidate in candidates), default=0.0),
        "debug_summary": build_candidate_debug_summary(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }

    extracted = []
    extraction_errors = []
    extract_seconds = 0.0
    if not args.no_extract:
        extract_start = time.time()
        for idx, candidate in enumerate(candidates):
            try:
                if args.use_opencv_extraction:
                    event = candidate_to_event(config, candidate)
                    output_path = extract_dive_around_splash(str(video_path), event, config, str(output_dir), idx)
                else:
                    output_path = extract_candidate_clip_ffmpeg_path(
                        video_path,
                        candidate,
                        output_dir,
                        idx,
                        args.ffmpeg_preset,
                        args.ffmpeg_threads,
                    )
                extracted.append(output_path)
                logger.log("clip_extracted", index=idx, timestamp=candidate.timestamp, output_path=output_path)
            except Exception as exc:
                extraction_errors.append(
                    {
                        "index": idx,
                        "timestamp": candidate.timestamp,
                        "error": str(exc),
                    }
                )
                logger.log("clip_extract_error", index=idx, timestamp=candidate.timestamp, error=str(exc))
        extract_seconds = time.time() - extract_start
        report["extracted_paths"] = extracted
        report["extraction_error_count"] = len(extraction_errors)
        report["extraction_errors"] = extraction_errors
    else:
        report["extraction_error_count"] = 0
        report["extraction_errors"] = []

    csv_path = write_candidates_csv(output_dir, candidates)
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    total_runtime_seconds = time.time() - run_start
    report["detections_csv"] = str(csv_path)
    report["extract_seconds"] = extract_seconds
    report["total_runtime_seconds"] = total_runtime_seconds
    report["peak_rss_kb"] = peak_rss_kb
    report_path = output_dir / "session_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "session_debug_summary.json").write_text(json.dumps(build_candidate_debug_summary(candidates), indent=2))
    ui_manifest = build_ui_session_manifest(
        video_path=video_path,
        output_dir=output_dir,
        profile=args.profile,
        report=report,
        candidates=candidates,
        extracted_paths=extracted,
    )
    ui_manifest_path = write_ui_session_manifest(output_dir / "ui_session_manifest.json", ui_manifest)
    logger.log(
        "session_complete",
        report_path=report_path,
        ui_manifest_path=ui_manifest_path,
        detections_csv=csv_path,
        candidate_count=len(candidates),
        extracted_count=len(extracted),
        extraction_error_count=len(extraction_errors),
        extract_seconds=extract_seconds,
        total_runtime_seconds=total_runtime_seconds,
        peak_rss_kb=peak_rss_kb,
    )

    summary = {
        "video_path": str(video_path),
        "candidate_count": len(candidates),
        "detector_seconds": detect_seconds,
        "extract_seconds": extract_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "peak_rss_kb": peak_rss_kb,
        "report_path": str(report_path),
        "ui_manifest_path": str(ui_manifest_path),
        "detections_csv": str(csv_path),
        "extracted_count": len(extracted),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_human_summary(summary)
        if args.debug:
            print(f"Debug summary: {output_dir / 'session_debug_summary.json'}")
            print(f"Run log: {output_dir / 'session_pipeline.log.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
