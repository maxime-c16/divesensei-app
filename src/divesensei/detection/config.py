from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2


@dataclass
class SplashEvent:
    frame_idx: int
    timestamp: float
    score: float
    filtered_score: float
    confidence: str
    zone_info: Dict[str, Any]
    detection_method: str


@dataclass
class DetectionConfig:
    detector_id: str = "audio_v1_heuristic"
    method: str = "audio_visual"
    splash_zone_top_norm: float = 0.72
    splash_zone_bottom_norm: float = 0.95
    splash_zone_left_norm: float = 0.0
    splash_zone_right_norm: float = 1.0
    spatial_gaussian_kernel: Tuple[int, int] = (5, 5)
    temporal_gaussian_sigma: float = 1.5
    temporal_window_size: int = 15
    base_threshold: float = 12.0
    adaptive_threshold_factor: float = 1.2
    min_threshold: float = 10.0
    max_threshold: float = 25.0
    min_extraction_score: float = 15.0
    auto_extract_threshold: bool = False
    high_confidence_threshold: float = 20.0
    medium_confidence_threshold: float = 12.0
    allow_close_dives: bool = False
    min_peak_prominence: float = 5.0
    min_peak_distance: int = 30
    peak_width_range: Tuple[int, int] = (3, 20)
    min_sustained_frames: int = 3
    cooldown_frames: int = 60
    pre_splash_duration: float = 6.0
    post_splash_duration: float = 2.0
    audio_sample_rate: int = 16000
    audio_frame_length: int = 1024
    audio_hop_length: int = 256
    audio_high_freq_cutoff_hz: float = 1800.0
    audio_peak_threshold: float = 4.0
    audio_peak_min_separation_seconds: float = 4.0
    audio_backtrack_frames: int = 20
    audio_ignore_before_seconds: float = 0.35
    audio_min_score: float = 4.5
    audio_min_hf_ratio: float = 0.115
    audio_early_peak_score: float = 4.0
    audio_early_peak_max_seconds: float = 0.8
    audio_early_peak_max_hf_ratio: float = 0.6
    audio_early_peak_max_centroid_hz: float = 2200.0
    audio_early_peak_max_flatness: float = 0.45
    audio_pattern_min_score: float = 0.4
    audio_noise_max_peak_count: int = 5
    audio_noise_max_top_ratio: float = 1.8
    audio_long_session_seconds: float = 120.0
    audio_long_session_max_candidates: int = 120
    audio_model_path: str = ""
    audio_model_min_probability: float = 0.0
    audio_clip_model_path: str = ""
    audio_clip_model_min_probability: float = 0.5
    audio_clip_classifier_window_seconds: float = 3.0
    audio_clip_classifier_ambiguity_low: float = 0.35
    audio_clip_classifier_ambiguity_high: float = 0.65
    audio_pcen_threshold: float = 2.4
    audio_pcen_merge_weight: float = 0.65
    audio_duplicate_suppress_window_seconds: float = 0.9
    audio_duplicate_leader_min_score: float = 12.0
    audio_duplicate_leader_min_prominence: float = 10.0
    audio_duplicate_follower_max_score_ratio: float = 0.55
    audio_decode_timeout_seconds: float = 20.0
    ffmpeg_threads: int = 1
    opencv_threads: int = 1
    audio_only_pre_seconds: float = 3.0
    audio_only_post_seconds: float = 1.0
    audio_visual_verify_pre_seconds: float = 3.0
    audio_visual_verify_post_seconds: float = 1.0
    audio_visual_verify_target_fps: float = 12.0
    audio_visual_max_proposals: int = 4
    audio_visual_skip_video_verification: bool = False
    audio_visual_min_video_score: float = 0.8
    audio_visual_hard_video_floor: float = 0.2
    audio_visual_audio_rescue_score: float = 4.0
    audio_visual_rescue_splash_ratio: float = 1.35
    audio_visual_min_combined_score: float = 3.8
    audio_visual_merge_seconds: float = 2.0
    audio_visual_max_verify_width: int = 640
    audio_priority_weight: float = 0.85
    diver_zone_top_norm: float = 0.15
    diver_zone_bottom_norm: float = 0.72
    diver_zone_left_norm: float = 0.0
    diver_zone_right_norm: float = 1.0
    enable_debug_plots: bool = False
    save_debug_frames: bool = False
    debug_output_dir: str = "debug"


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def extract_dive_around_splash(
    video_path: str,
    splash_event: SplashEvent,
    config: DetectionConfig,
    output_dir: str,
    dive_number: int,
    progress_callback=None,
) -> str:
    video_fps = get_video_fps(video_path)
    segment_start_time = splash_event.zone_info.get("segment_start_time")
    segment_end_time = splash_event.zone_info.get("segment_end_time")
    if segment_start_time is not None and segment_end_time is not None:
        start_frame = max(0, int(segment_start_time * video_fps))
        end_frame = max(start_frame + 1, int(segment_end_time * video_fps))
    else:
        pre_frames = int(config.pre_splash_duration * video_fps)
        post_frames = int(config.post_splash_duration * video_fps)
        start_frame = max(0, splash_event.frame_idx - pre_frames)
        end_frame = splash_event.frame_idx + post_frames

    confidence_suffix = f"_{splash_event.confidence}" if splash_event.confidence != "high" else ""
    output_filename = f"dive_splash_{dive_number + 1}_t{splash_event.timestamp:.1f}s{confidence_suffix}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if current_frame == splash_event.frame_idx:
            cv2.putText(frame, "SPLASH DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Score: {splash_event.filtered_score:.1f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        zone_top = int(config.splash_zone_top_norm * height)
        zone_bottom = int(config.splash_zone_bottom_norm * height)
        zone_left = int(config.splash_zone_left_norm * width)
        zone_right = int(config.splash_zone_right_norm * width)
        cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), (255, 0, 0), 2)
        cv2.putText(frame, "Splash Zone", (zone_left + 10, zone_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    if progress_callback:
        progress_callback(f"Extracted {output_filename}")
    return output_path
