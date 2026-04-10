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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from divesensei.io.logging_utils import StructuredLogger, build_candidate_debug_summary
from divesensei.io.media_io import extract_clip_ffmpeg, generate_review_proxy_ffmpeg, probe_media_duration_seconds
from divesensei.metadata.ui_contract import build_ui_session_manifest, write_ui_session_manifest
from divesensei.profiles import apply_named_profile


def apply_profile_overrides(args: argparse.Namespace, explicit_flags: set[str] | None = None) -> argparse.Namespace:
    explicit_flags = explicit_flags or set()
    defaults = {
        "audio_peak_threshold": args.audio_peak_threshold,
        "audio_min_score": args.audio_min_score,
        "audio_pattern_min_score": args.audio_pattern_min_score,
        "audio_peak_separation": args.audio_peak_separation,
        "audio_visual_merge_seconds": args.audio_visual_merge_seconds,
        "audio_clip_model_min_probability": args.audio_clip_model_min_probability,
        "audio_decode_timeout_seconds": args.audio_decode_timeout_seconds,
    }
    merged = apply_named_profile(defaults, args.profile, args.detector_id)
    if "--audio-peak-threshold" not in explicit_flags:
        args.audio_peak_threshold = float(merged["audio_peak_threshold"])
    if "--audio-min-score" not in explicit_flags:
        args.audio_min_score = float(merged["audio_min_score"])
    if "--audio-pattern-min-score" not in explicit_flags:
        args.audio_pattern_min_score = float(merged["audio_pattern_min_score"])
    if "--audio-peak-separation" not in explicit_flags:
        args.audio_peak_separation = float(merged["audio_peak_separation"])
    if "--audio-visual-merge-seconds" not in explicit_flags:
        args.audio_visual_merge_seconds = float(merged["audio_visual_merge_seconds"])
    if "--audio-clip-model-min-probability" not in explicit_flags:
        args.audio_clip_model_min_probability = float(merged["audio_clip_model_min_probability"])
    if "--audio-decode-timeout-seconds" not in explicit_flags:
        args.audio_decode_timeout_seconds = float(merged["audio_decode_timeout_seconds"])
    if args.quality == "balanced" and args.ffmpeg_preset == "ultrafast":
        args.ffmpeg_preset = "medium"
    if args.quality == "fast" and args.ffmpeg_preset == "ultrafast":
        args.ffmpeg_preset = "ultrafast"
    return args


def build_config(args: argparse.Namespace):
    from divesensei.detection.config import DetectionConfig

    return DetectionConfig(
        detector_id=args.detector_id,
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
        audio_noise_diversity_bucket_seconds=args.audio_noise_diversity_bucket_seconds,
        pre_candidate_soft_ratio=args.pre_candidate_soft_ratio,
        pre_candidate_max_extra_candidates=args.pre_candidate_max_extra_candidates,
        pre_candidate_local_window_seconds=args.pre_candidate_local_window_seconds,
        pre_candidate_local_ratio=args.pre_candidate_local_ratio,
        pre_candidate_tail_ratio=args.pre_candidate_tail_ratio,
        pre_candidate_tail_boost=args.pre_candidate_tail_boost,
        pre_candidate_rank_tail_weight=args.pre_candidate_rank_tail_weight,
        pre_candidate_rank_asymmetry_weight=args.pre_candidate_rank_asymmetry_weight,
        pre_candidate_rank_broadband_weight=args.pre_candidate_rank_broadband_weight,
        pre_candidate_rank_decay_weight=args.pre_candidate_rank_decay_weight,
        pre_candidate_rank_promotion_bonus=args.pre_candidate_rank_promotion_bonus,
        pre_candidate_rank_promotion_min_score=args.pre_candidate_rank_promotion_min_score,
        pre_candidate_rank_promotion_min_dive_likeness=args.pre_candidate_rank_promotion_min_dive_likeness,
        pre_candidate_rank_promotion_min_prominence=args.pre_candidate_rank_promotion_min_prominence,
        pre_candidate_rank_promotion_min_nearby_peaks=args.pre_candidate_rank_promotion_min_nearby_peaks,
        pre_candidate_protect_survivor_window_seconds=args.pre_candidate_protect_survivor_window_seconds,
        pre_candidate_protect_survivor_max_per_bucket=args.pre_candidate_protect_survivor_max_per_bucket,
        pre_candidate_protect_survivor_min_score=args.pre_candidate_protect_survivor_min_score,
        pre_candidate_protect_survivor_min_dive_likeness=args.pre_candidate_protect_survivor_min_dive_likeness,
        pre_candidate_protect_survivor_min_prominence=args.pre_candidate_protect_survivor_min_prominence,
        pre_candidate_protect_survivor_min_tail_ratio=args.pre_candidate_protect_survivor_min_tail_ratio,
        pre_candidate_local_rescue_window_seconds=args.pre_candidate_local_rescue_window_seconds,
        pre_candidate_local_rescue_max_per_bucket=args.pre_candidate_local_rescue_max_per_bucket,
        pre_candidate_local_rescue_max_per_session=args.pre_candidate_local_rescue_max_per_session,
        pre_candidate_local_rescue_anchor_min_rank_score=args.pre_candidate_local_rescue_anchor_min_rank_score,
        pre_candidate_local_rescue_min_score=args.pre_candidate_local_rescue_min_score,
        pre_candidate_local_rescue_min_dive_likeness=args.pre_candidate_local_rescue_min_dive_likeness,
        pre_candidate_local_rescue_min_prominence=args.pre_candidate_local_rescue_min_prominence,
        pre_candidate_local_rescue_min_tail_persistence_score=args.pre_candidate_local_rescue_min_tail_persistence_score,
        pre_candidate_local_rescue_min_cluster_support_score=args.pre_candidate_local_rescue_min_cluster_support_score,
        pre_candidate_tail_persistence_weight=args.pre_candidate_tail_persistence_weight,
        pre_candidate_tail_persistence_short_seconds=args.pre_candidate_tail_persistence_short_seconds,
        pre_candidate_tail_persistence_medium_seconds=args.pre_candidate_tail_persistence_medium_seconds,
        pre_candidate_tail_persistence_long_seconds=args.pre_candidate_tail_persistence_long_seconds,
        pre_candidate_cluster_support_weight=args.pre_candidate_cluster_support_weight,
        pre_candidate_cluster_support_window_seconds=args.pre_candidate_cluster_support_window_seconds,
        pre_candidate_cluster_support_min_peak_ratio=args.pre_candidate_cluster_support_min_peak_ratio,
        pre_candidate_consolidation_weight=args.pre_candidate_consolidation_weight,
        pre_candidate_consolidation_window_seconds=args.pre_candidate_consolidation_window_seconds,
        pre_candidate_consolidation_top_peaks=args.pre_candidate_consolidation_top_peaks,
        pre_candidate_consolidation_min_score=args.pre_candidate_consolidation_min_score,
        pre_candidate_consolidation_min_cluster_size=args.pre_candidate_consolidation_min_cluster_size,
        pre_candidate_consolidation_merge_gap_seconds=args.pre_candidate_consolidation_merge_gap_seconds,
        pre_candidate_consolidation_max_bonus=args.pre_candidate_consolidation_max_bonus,
        pre_candidate_consolidation_centering_weight=args.pre_candidate_consolidation_centering_weight,
        pre_candidate_consolidation_group_by_peak_timestamps=args.pre_candidate_consolidation_group_by_peak_timestamps,
        pre_candidate_overlap_agreement_weight=args.pre_candidate_overlap_agreement_weight,
        pre_candidate_overlap_window_seconds=args.pre_candidate_overlap_window_seconds,
        pre_candidate_overlap_min_pcen_persistence=args.pre_candidate_overlap_min_pcen_persistence,
        pre_candidate_overlap_min_total_score=args.pre_candidate_overlap_min_total_score,
        pre_candidate_overlap_pcen_center_weight=args.pre_candidate_overlap_pcen_center_weight,
        frontend_persistence_integral_weight=args.frontend_persistence_integral_weight,
        frontend_persistence_integral_start_seconds=args.frontend_persistence_integral_start_seconds,
        frontend_persistence_integral_end_seconds=args.frontend_persistence_integral_end_seconds,
        frontend_persistence_integral_pre_seconds=args.frontend_persistence_integral_pre_seconds,
        frontend_persistence_integral_pcen_weight=args.frontend_persistence_integral_pcen_weight,
        frontend_persistence_integral_max_bonus=args.frontend_persistence_integral_max_bonus,
        frontend_sustained_noise_exception_enabled=args.frontend_sustained_noise_exception_enabled,
        frontend_sustained_noise_exception_min_bonus=args.frontend_sustained_noise_exception_min_bonus,
        frontend_sustained_noise_exception_min_flux_ratio=args.frontend_sustained_noise_exception_min_flux_ratio,
        frontend_sustained_noise_exception_min_post_flux_ratio=args.frontend_sustained_noise_exception_min_post_flux_ratio,
        frontend_sustained_noise_exception_min_post_rms_ratio=args.frontend_sustained_noise_exception_min_post_rms_ratio,
        frontend_sustained_noise_exception_min_prominence=args.frontend_sustained_noise_exception_min_prominence,
        frontend_sustained_noise_exception_min_pcen_ratio=args.frontend_sustained_noise_exception_min_pcen_ratio,
        frontend_pattern_persistence_bonus_weight=args.frontend_pattern_persistence_bonus_weight,
        frontend_pattern_persistence_bonus_max=args.frontend_pattern_persistence_bonus_max,
        frontend_pattern_persistence_bonus_min_bonus=args.frontend_pattern_persistence_bonus_min_bonus,
        frontend_pattern_persistence_bonus_min_post_flux_ratio=args.frontend_pattern_persistence_bonus_min_post_flux_ratio,
        frontend_pattern_persistence_bonus_min_post_rms_ratio=args.frontend_pattern_persistence_bonus_min_post_rms_ratio,
        frontend_pattern_persistence_bonus_min_prominence=args.frontend_pattern_persistence_bonus_min_prominence,
        frontend_region_descriptor_enabled=args.frontend_region_descriptor_enabled,
        frontend_region_descriptor_weight=args.frontend_region_descriptor_weight,
        frontend_region_descriptor_max_bonus=args.frontend_region_descriptor_max_bonus,
        frontend_region_descriptor_pre_seconds=args.frontend_region_descriptor_pre_seconds,
        frontend_region_descriptor_post_seconds=args.frontend_region_descriptor_post_seconds,
        frontend_region_descriptor_pattern_tiebreak_band=args.frontend_region_descriptor_pattern_tiebreak_band,
        frontend_dive_trend_enabled=args.frontend_dive_trend_enabled,
        frontend_dive_trend_weight=args.frontend_dive_trend_weight,
        frontend_dive_trend_max_bonus=args.frontend_dive_trend_max_bonus,
        pre_candidate_cluster_delay_enabled=args.pre_candidate_cluster_delay_enabled,
        pre_candidate_cluster_delay_seconds=args.pre_candidate_cluster_delay_seconds,
        pre_candidate_cluster_delay_min_cluster_size=args.pre_candidate_cluster_delay_min_cluster_size,
        pre_candidate_cluster_representative_weight=args.pre_candidate_cluster_representative_weight,
        frontend_region_pattern_exception_enabled=args.frontend_region_pattern_exception_enabled,
        frontend_region_pattern_exception_min_score=args.frontend_region_pattern_exception_min_score,
        frontend_region_pattern_exception_min_prominence=args.frontend_region_pattern_exception_min_prominence,
        frontend_region_pattern_exception_min_post_flux_ratio=args.frontend_region_pattern_exception_min_post_flux_ratio,
        frontend_region_pattern_exception_min_post_rms_ratio=args.frontend_region_pattern_exception_min_post_rms_ratio,
        frontend_region_pattern_exception_min_bonus=args.frontend_region_pattern_exception_min_bonus,
        frontend_dense_pcen_pattern_exception_enabled=args.frontend_dense_pcen_pattern_exception_enabled,
        frontend_dense_pcen_pattern_exception_min_score=args.frontend_dense_pcen_pattern_exception_min_score,
        frontend_dense_pcen_pattern_exception_min_prominence=args.frontend_dense_pcen_pattern_exception_min_prominence,
        frontend_dense_pcen_pattern_exception_min_post_flux_ratio=args.frontend_dense_pcen_pattern_exception_min_post_flux_ratio,
        frontend_dense_pcen_pattern_exception_min_post_rms_ratio=args.frontend_dense_pcen_pattern_exception_min_post_rms_ratio,
        frontend_dense_pcen_pattern_exception_min_nearby_peaks=args.frontend_dense_pcen_pattern_exception_min_nearby_peaks,
        frontend_dense_pcen_pattern_exception_max_flatness=args.frontend_dense_pcen_pattern_exception_max_flatness,
        frontend_region_tail_imbalance_exception_enabled=args.frontend_region_tail_imbalance_exception_enabled,
        frontend_region_tail_imbalance_exception_min_score=args.frontend_region_tail_imbalance_exception_min_score,
        frontend_region_tail_imbalance_exception_min_prominence=args.frontend_region_tail_imbalance_exception_min_prominence,
        frontend_region_tail_imbalance_exception_min_post_flux_ratio=args.frontend_region_tail_imbalance_exception_min_post_flux_ratio,
        frontend_region_tail_imbalance_exception_min_post_rms_ratio=args.frontend_region_tail_imbalance_exception_min_post_rms_ratio,
        frontend_region_tail_imbalance_exception_max_post_rms_ratio=args.frontend_region_tail_imbalance_exception_max_post_rms_ratio,
        frontend_region_tail_imbalance_exception_min_bonus=args.frontend_region_tail_imbalance_exception_min_bonus,
        frontend_region_tail_imbalance_exception_min_late_over_early=args.frontend_region_tail_imbalance_exception_min_late_over_early,
        frontend_region_tail_imbalance_exception_min_duration_above_1p10=args.frontend_region_tail_imbalance_exception_min_duration_above_1p10,
        frontend_region_tail_imbalance_exception_min_time_to_peak=args.frontend_region_tail_imbalance_exception_min_time_to_peak,
        frontend_region_tail_imbalance_exception_max_time_to_peak=args.frontend_region_tail_imbalance_exception_max_time_to_peak,
        frontend_region_tail_imbalance_exception_max_flatness=args.frontend_region_tail_imbalance_exception_max_flatness,
        frontend_short_region_tail_exception_enabled=args.frontend_short_region_tail_exception_enabled,
        frontend_short_region_tail_exception_min_score=args.frontend_short_region_tail_exception_min_score,
        frontend_short_region_tail_exception_min_prominence=args.frontend_short_region_tail_exception_min_prominence,
        frontend_short_region_tail_exception_min_post_flux_ratio=args.frontend_short_region_tail_exception_min_post_flux_ratio,
        frontend_short_region_tail_exception_min_post_rms_ratio=args.frontend_short_region_tail_exception_min_post_rms_ratio,
        frontend_short_region_tail_exception_min_bonus=args.frontend_short_region_tail_exception_min_bonus,
        frontend_short_region_tail_exception_max_bonus=args.frontend_short_region_tail_exception_max_bonus,
        frontend_short_region_tail_exception_min_late_over_early=args.frontend_short_region_tail_exception_min_late_over_early,
        frontend_short_region_tail_exception_min_duration_above_1p10=args.frontend_short_region_tail_exception_min_duration_above_1p10,
        frontend_short_region_tail_exception_max_duration_above_1p10=args.frontend_short_region_tail_exception_max_duration_above_1p10,
        frontend_short_region_tail_exception_min_nearby_peaks=args.frontend_short_region_tail_exception_min_nearby_peaks,
        frontend_short_region_tail_exception_max_nearby_peaks=args.frontend_short_region_tail_exception_max_nearby_peaks,
        frontend_short_region_tail_exception_min_time_to_peak=args.frontend_short_region_tail_exception_min_time_to_peak,
        frontend_short_region_tail_exception_max_time_to_peak=args.frontend_short_region_tail_exception_max_time_to_peak,
        frontend_short_region_tail_exception_max_flatness=args.frontend_short_region_tail_exception_max_flatness,
        audio_long_session_seconds=args.audio_long_session_seconds,
        audio_long_session_max_candidates=args.audio_long_session_max_candidates,
        audio_model_path=args.audio_model_path,
        audio_model_min_probability=args.audio_model_min_probability,
        audio_clip_model_path=args.audio_clip_model_path,
        audio_clip_model_min_probability=args.audio_clip_model_min_probability,
        audio_clip_classifier_window_seconds=args.audio_clip_classifier_window_seconds,
        audio_clip_classifier_ambiguity_low=args.audio_clip_classifier_ambiguity_low,
        audio_clip_classifier_ambiguity_high=args.audio_clip_classifier_ambiguity_high,
        audio_pcen_threshold=args.audio_pcen_threshold,
        audio_pcen_merge_weight=args.audio_pcen_merge_weight,
        audio_duplicate_suppress_window_seconds=args.audio_duplicate_suppress_window_seconds,
        audio_duplicate_leader_min_score=args.audio_duplicate_leader_min_score,
        audio_duplicate_leader_min_prominence=args.audio_duplicate_leader_min_prominence,
        audio_duplicate_follower_max_score_ratio=args.audio_duplicate_follower_max_score_ratio,
        audio_decode_timeout_seconds=args.audio_decode_timeout_seconds,
        audio_decode_progress_interval_seconds=args.audio_decode_progress_interval_seconds,
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


def candidate_to_event(config, candidate):
    from divesensei.detection.config import SplashEvent

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
        "--detector-id",
        choices=["audio_v1_heuristic", "audio_v2_pcen_classifier", "audio_v2_hybrid_video"],
        default="audio_v1_heuristic",
        help="Detector strategy. Use the v2 options for PCEN-driven proposals and classifier filtering.",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced"],
        default="fast",
        help="Clip export quality preset. 'fast' is recommended for bulk extraction.",
    )
    parser.add_argument("--session-name", default="", help="Optional human-readable name for this session run")
    parser.add_argument("--pre-duration", type=float, default=6.0, help="Seconds to keep before the detected splash")
    parser.add_argument("--post-duration", type=float, default=3.0, help="Seconds to keep after the detected splash")
    parser.add_argument("--detect-only", action="store_true", help="Run detection only and skip clip extraction")
    parser.add_argument("--review-only", action="store_true", help="Prepare the review queue and session proxy without extracting per-dive clips")
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON")
    parser.add_argument("--debug", action="store_true", help="Keep structured logs and debug summary files")

    internal = parser.add_argument_group("advanced")
    internal.add_argument("--no-extract", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--skip-review-proxy", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--use-opencv-extraction", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--ffmpeg-preset", default="ultrafast", help=argparse.SUPPRESS)
    internal.add_argument("--ffmpeg-threads", type=int, default=0, help="FFmpeg worker threads; 0 lets FFmpeg choose")
    internal.add_argument("--opencv-threads", type=int, default=0, help="OpenCV worker threads; 0 lets OpenCV choose")
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
    internal.add_argument("--audio-noise-diversity-bucket-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-soft-ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-max-extra-candidates", type=int, default=2, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-window-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-tail-ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-tail-boost", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-tail-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-asymmetry-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-broadband-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-decay-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-promotion-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-promotion-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-promotion-min-dive-likeness", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-promotion-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-rank-promotion-min-nearby-peaks", type=int, default=0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-protect-survivor-window-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-protect-survivor-max-per-bucket", type=int, default=1, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-protect-survivor-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-protect-survivor-min-dive-likeness", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-protect-survivor-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-protect-survivor-min-tail-ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-window-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-max-per-bucket", type=int, default=1, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-max-per-session", type=int, default=2, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-anchor-min-rank-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-min-dive-likeness", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-min-tail-persistence-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-local-rescue-min-cluster-support-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-tail-persistence-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-tail-persistence-short-seconds", type=float, default=0.5, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-tail-persistence-medium-seconds", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-tail-persistence-long-seconds", type=float, default=2.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-cluster-support-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-cluster-support-window-seconds", type=float, default=1.5, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-cluster-support-min-peak-ratio", type=float, default=0.55, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-window-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-top-peaks", type=int, default=3, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-min-cluster-size", type=int, default=2, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-merge-gap-seconds", type=float, default=0.12, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-max-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-centering-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-consolidation-group-by-peak-timestamps", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-overlap-agreement-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-overlap-window-seconds", type=float, default=0.12, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-overlap-min-pcen-persistence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-overlap-min-total-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-overlap-pcen-center-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-persistence-integral-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-persistence-integral-start-seconds", type=float, default=0.15, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-persistence-integral-end-seconds", type=float, default=0.8, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-persistence-integral-pre-seconds", type=float, default=0.4, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-persistence-integral-pcen-weight", type=float, default=0.6, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-persistence-integral-max-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-sustained-noise-exception-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--frontend-sustained-noise-exception-min-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-sustained-noise-exception-min-flux-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-sustained-noise-exception-min-post-flux-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-sustained-noise-exception-min-post-rms-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-sustained-noise-exception-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-sustained-noise-exception-min-pcen-ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-pattern-persistence-bonus-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-pattern-persistence-bonus-max", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-pattern-persistence-bonus-min-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-pattern-persistence-bonus-min-post-flux-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-pattern-persistence-bonus-min-post-rms-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-pattern-persistence-bonus-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-descriptor-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-descriptor-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-descriptor-max-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-descriptor-pre-seconds", type=float, default=0.2, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-descriptor-post-seconds", type=float, default=0.8, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-descriptor-pattern-tiebreak-band", type=float, default=0.35, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dive-trend-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dive-trend-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dive-trend-max-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-cluster-delay-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-cluster-delay-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-cluster-delay-min-cluster-size", type=int, default=2, help=argparse.SUPPRESS)
    internal.add_argument("--pre-candidate-cluster-representative-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-pattern-exception-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-pattern-exception-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-pattern-exception-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-pattern-exception-min-post-flux-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-pattern-exception-min-post-rms-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-pattern-exception-min-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dense-pcen-pattern-exception-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dense-pcen-pattern-exception-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dense-pcen-pattern-exception-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dense-pcen-pattern-exception-min-post-flux-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dense-pcen-pattern-exception-min-post-rms-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dense-pcen-pattern-exception-min-nearby-peaks", type=int, default=0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-dense-pcen-pattern-exception-max-flatness", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-post-flux-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-post-rms-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-max-post-rms-ratio", type=float, default=999.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-late-over-early", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-duration-above-1p10", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-min-time-to-peak", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-max-time-to-peak", type=float, default=999.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-region-tail-imbalance-exception-max-flatness", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-enabled", action="store_true", help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-score", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-prominence", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-post-flux-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-post-rms-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-bonus", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-max-bonus", type=float, default=999.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-late-over-early", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-duration-above-1p10", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-max-duration-above-1p10", type=float, default=999.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-nearby-peaks", type=int, default=0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-max-nearby-peaks", type=int, default=999, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-min-time-to-peak", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-max-time-to-peak", type=float, default=999.0, help=argparse.SUPPRESS)
    internal.add_argument("--frontend-short-region-tail-exception-max-flatness", type=float, default=1.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-long-session-seconds", type=float, default=120.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-long-session-max-candidates", type=int, default=120, help=argparse.SUPPRESS)
    internal.add_argument("--audio-model-path", default="", help=argparse.SUPPRESS)
    internal.add_argument("--audio-model-min-probability", type=float, default=0.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-clip-model-path", default="", help=argparse.SUPPRESS)
    internal.add_argument("--audio-clip-model-min-probability", type=float, default=0.5, help=argparse.SUPPRESS)
    internal.add_argument("--audio-clip-classifier-window-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-clip-classifier-ambiguity-low", type=float, default=0.35, help=argparse.SUPPRESS)
    internal.add_argument("--audio-clip-classifier-ambiguity-high", type=float, default=0.65, help=argparse.SUPPRESS)
    internal.add_argument("--audio-pcen-threshold", type=float, default=2.4, help=argparse.SUPPRESS)
    internal.add_argument("--audio-pcen-merge-weight", type=float, default=0.65, help=argparse.SUPPRESS)
    internal.add_argument("--audio-duplicate-suppress-window-seconds", type=float, default=0.9, help=argparse.SUPPRESS)
    internal.add_argument("--audio-duplicate-leader-min-score", type=float, default=12.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-duplicate-leader-min-prominence", type=float, default=10.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-duplicate-follower-max-score-ratio", type=float, default=0.55, help=argparse.SUPPRESS)
    internal.add_argument("--audio-decode-timeout-seconds", type=float, default=3600.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-decode-progress-interval-seconds", type=float, default=15.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-only-pre-seconds", type=float, default=6.0, help=argparse.SUPPRESS)
    internal.add_argument("--audio-only-post-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
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
    raw_argv = list(argv or [])
    args = parser.parse_args(raw_argv)
    if not args.output_dir:
        args.output_dir = str(default_output_dir(args.video_path))
    if args.detect_only:
        args.no_extract = True
        args.skip_review_proxy = True
    if args.review_only:
        args.no_extract = True
    if args.with_video_verification:
        args.skip_video_verification = False
    if args.detector_id == "audio_v2_hybrid_video" and "--skip-video-verification" not in raw_argv:
        args.skip_video_verification = False
    explicit_flags = {item for item in raw_argv if item.startswith("--")}
    return apply_profile_overrides(args, explicit_flags)


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


def write_session_outputs(
    *,
    video_path: Path,
    output_dir: Path,
    profile: str,
    report: dict[str, Any],
    candidates: Sequence[Any],
    extracted_paths: Sequence[str],
    status_override: str | None = None,
    session_mode: str = "standard",
    source_audio_path: str | None = None,
    review_proxy_path: str | None = None,
    evaluation_review_path: str | None = None,
) -> tuple[Path, Path]:
    report_path = output_dir / "session_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "session_debug_summary.json").write_text(json.dumps(build_candidate_debug_summary(candidates), indent=2))
    ui_manifest = build_ui_session_manifest(
        video_path=video_path,
        output_dir=output_dir,
        profile=profile,
        report=report,
        candidates=candidates,
        extracted_paths=extracted_paths,
        status_override=status_override,
        session_mode=session_mode,
        source_audio_path=source_audio_path,
        review_proxy_path=review_proxy_path,
        evaluation_review_path=evaluation_review_path,
    )
    ui_manifest_path = write_ui_session_manifest(output_dir / "ui_session_manifest.json", ui_manifest)
    return report_path, ui_manifest_path


def write_incremental_session_outputs(
    *,
    video_path: Path,
    output_dir: Path,
    profile: str,
    report: dict[str, Any],
    candidates: Sequence[Any],
    extracted_paths: Sequence[str],
    extraction_errors: Sequence[dict[str, Any]],
    status_override: str,
    session_mode: str = "standard",
    source_audio_path: str | None = None,
    review_proxy_path: str | None = None,
    evaluation_review_path: str | None = None,
) -> tuple[Path, Path]:
    report["extracted_paths"] = list(extracted_paths)
    report["extraction_error_count"] = len(extraction_errors)
    report["extraction_errors"] = list(extraction_errors)
    return write_session_outputs(
        video_path=video_path,
        output_dir=output_dir,
        profile=profile,
        report=report,
        candidates=candidates,
        extracted_paths=extracted_paths,
        status_override=status_override,
        session_mode=session_mode,
        source_audio_path=source_audio_path,
        review_proxy_path=review_proxy_path,
        evaluation_review_path=evaluation_review_path,
    )


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
    from divesensei.detection.audio_detector import AudioVisualDiveDetector

    detector = AudioVisualDiveDetector(
        config,
        progress_callback=lambda payload: logger.log(
            str(payload["event"]),
            **{key: value for key, value in payload.items() if key != "event"},
        ),
    )
    logger.log("session_start", video_path=video_path, output_dir=output_dir, profile=args.profile, config=config.__dict__)

    run_start = time.time()
    start = run_start
    logger.log("detection_start", video_path=video_path, profile=args.profile)
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
        "detector_id": args.detector_id,
        "session_created_at": datetime.now(timezone.utc).isoformat(),
        "session_name": (args.session_name.strip() if getattr(args, "session_name", "") else "") or f"{video_path.stem} · {output_dir.name.replace('.tmp_ui_run_', '').replace('_', ' ')}",
        "detector_seconds": detect_seconds,
        "config": config.__dict__,
        "candidate_count": len(candidates),
        "session_estimated_duration_seconds": probe_media_duration_seconds(video_path) or max((candidate.end_time for candidate in candidates), default=0.0),
        "debug_summary": build_candidate_debug_summary(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }

    extracted = []
    extraction_errors = []
    extract_seconds = 0.0
    review_proxy_error = None
    csv_path = write_candidates_csv(output_dir, candidates)
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report["detections_csv"] = str(csv_path)
    report["extract_seconds"] = extract_seconds
    report["peak_rss_kb"] = peak_rss_kb
    report["manifest_ready_seconds"] = time.time() - run_start
    report["total_runtime_seconds"] = report["manifest_ready_seconds"]
    report["review_proxy_status"] = "skipped" if args.skip_review_proxy else "pending"
    report["review_proxy_path"] = str(output_dir / "web" / "session_source_review.mp4")
    report_path, ui_manifest_path = write_incremental_session_outputs(
        video_path=video_path,
        output_dir=output_dir,
        profile=args.profile,
        report=report,
        candidates=candidates,
        extracted_paths=extracted,
        extraction_errors=extraction_errors,
        status_override="ready_proxy_pending" if not args.skip_review_proxy else "complete",
    )
    logger.log(
        "review_ready",
        report_path=report_path,
        ui_manifest_path=ui_manifest_path,
        detections_csv=csv_path,
        candidate_count=len(candidates),
        extracted_count=len(extracted),
        extraction_error_count=len(extraction_errors),
        manifest_ready_seconds=report["manifest_ready_seconds"],
        peak_rss_kb=peak_rss_kb,
    )

    if not args.no_extract:
        from divesensei.detection.config import extract_dive_around_splash

        extract_start = time.time()
        logger.log("clip_extraction_start", candidate_count=len(candidates))
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
                report["extract_seconds"] = time.time() - extract_start
                report_path, ui_manifest_path = write_incremental_session_outputs(
                    video_path=video_path,
                    output_dir=output_dir,
                    profile=args.profile,
                    report=report,
                    candidates=candidates,
                    extracted_paths=extracted,
                    extraction_errors=extraction_errors,
                    status_override="ready_proxy_pending",
                )
                logger.log(
                    "clip_extracted",
                    index=idx,
                    completed_clips=len(extracted),
                    total_clips=len(candidates),
                    timestamp=candidate.timestamp,
                    output_path=output_path,
                    ui_manifest_path=ui_manifest_path,
                )
            except Exception as exc:
                extraction_errors.append(
                    {
                        "index": idx,
                        "timestamp": candidate.timestamp,
                        "error": str(exc),
                    }
                )
                report["extract_seconds"] = time.time() - extract_start
                write_incremental_session_outputs(
                    video_path=video_path,
                    output_dir=output_dir,
                    profile=args.profile,
                    report=report,
                    candidates=candidates,
                    extracted_paths=extracted,
                    extraction_errors=extraction_errors,
                    status_override="ready_proxy_pending",
                )
                logger.log("clip_extract_error", index=idx, timestamp=candidate.timestamp, error=str(exc))
        extract_seconds = time.time() - extract_start
    else:
        report["extraction_error_count"] = 0
        report["extraction_errors"] = []

    review_proxy_path = output_dir / "web" / "session_source_review.mp4"
    if not args.skip_review_proxy:
        logger.log("review_proxy_start", output_path=review_proxy_path)
        try:
            generate_review_proxy_ffmpeg(
                video_path=video_path,
                output_path=review_proxy_path,
                preset="ultrafast" if args.quality == "fast" else "veryfast",
                ffmpeg_threads=args.ffmpeg_threads,
            )
            report["review_proxy_path"] = str(review_proxy_path)
            report["review_proxy_status"] = "ready"
        except Exception as exc:
            review_proxy_error = str(exc)
            report["review_proxy_error"] = review_proxy_error
            report["review_proxy_status"] = "failed"
            logger.log("review_proxy_error", output_path=review_proxy_path, error=review_proxy_error)

    total_runtime_seconds = time.time() - run_start
    report["total_runtime_seconds"] = total_runtime_seconds
    report["manifest_ready_seconds"] = report.get("manifest_ready_seconds", total_runtime_seconds)
    report_path, ui_manifest_path = write_session_outputs(
        video_path=video_path,
        output_dir=output_dir,
        profile=args.profile,
        report=report,
        candidates=candidates,
        extracted_paths=extracted,
        status_override=(
            "complete"
            if args.skip_review_proxy or review_proxy_error is None
            else "complete_proxy_error"
        ),
    )
    logger.log(
        "session_complete",
        report_path=report_path,
        ui_manifest_path=ui_manifest_path,
        detections_csv=csv_path,
        candidate_count=len(candidates),
        extracted_count=len(extracted),
        extraction_error_count=len(extraction_errors),
        review_proxy_path=str(review_proxy_path) if review_proxy_path.exists() else None,
        review_proxy_error=review_proxy_error,
        review_proxy_status=report.get("review_proxy_status"),
        manifest_ready_seconds=report.get("manifest_ready_seconds"),
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
        "manifest_ready_seconds": report.get("manifest_ready_seconds"),
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
