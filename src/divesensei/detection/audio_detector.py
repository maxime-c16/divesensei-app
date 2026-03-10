#!/usr/bin/env python3
"""
Audio-led dive proposal detection with lightweight video verification.

This module is designed for long training-session videos where the most
reliable event anchor is the water-entry sound. It generates candidate dive
endings from audio, then verifies and refines each candidate using cheap video
signals inside the diver and splash regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from divesensei.detection.audio_model import AudioCandidateModel
from divesensei.io.media_io import decode_audio_mono_s16le
from divesensei.io.runtime import configure_runtime


@dataclass
class AudioCandidate:
    timestamp: float
    audio_score: float
    spectral_flux: float
    rms: float
    hf_ratio: float
    spectral_centroid_hz: float
    spectral_flatness: float
    post_flux_ratio: float
    post_rms_ratio: float
    local_prominence: float
    nearby_peaks_8s: int


@dataclass
class VerifiedDiveCandidate:
    frame_idx: int
    timestamp: float
    audio_score: float
    video_score: float
    combined_score: float
    start_time: float
    end_time: float
    confidence: str
    details: Dict[str, Any]


class AudioVisualDiveDetector:
    """
    Fast detector for long untrimmed training videos.

    Pipeline:
    1. Decode mono audio and find sharp transient proposals.
    2. Verify proposals using motion in the diver corridor and splash ROI.
    3. Refine clip boundaries around takeoff and splash decay.
    """

    def __init__(self, config: Any):
        self.config = config
        configure_runtime(int(getattr(config, "opencv_threads", 1)))
        self.audio_candidate_model = self._load_audio_candidate_model()

    def detect(self, video_path: str) -> List[VerifiedDiveCandidate]:
        signal, sample_rate = self._extract_audio_signal(video_path)
        proposals = self._propose_from_audio(signal, sample_rate)
        if not proposals:
            return []
        if bool(getattr(self.config, "audio_visual_skip_video_verification", False)):
            return self._promote_audio_only(proposals)
        return self._verify_with_video(video_path, proposals)

    def _extract_audio_signal(self, video_path: str) -> Tuple[np.ndarray, int]:
        sample_rate = int(getattr(self.config, "audio_sample_rate", 16000))
        timeout_seconds = float(getattr(self.config, "audio_decode_timeout_seconds", 20.0))
        ffmpeg_threads = int(getattr(self.config, "ffmpeg_threads", 1))
        samples = decode_audio_mono_s16le(
            video_path=video_path,
            sample_rate=sample_rate,
            timeout_seconds=timeout_seconds,
            ffmpeg_threads=ffmpeg_threads,
        )
        return samples, sample_rate

    def _promote_audio_only(self, proposals: Sequence[AudioCandidate]) -> List[VerifiedDiveCandidate]:
        pre_seconds = float(getattr(self.config, "audio_only_pre_seconds", 3.0))
        post_seconds = float(getattr(self.config, "audio_only_post_seconds", 1.0))
        promoted: List[VerifiedDiveCandidate] = []
        for proposal in proposals:
            confidence = "high" if proposal.audio_score >= 7.5 else "medium" if proposal.audio_score >= 3.8 else "low"
            promoted.append(
                VerifiedDiveCandidate(
                    frame_idx=0,
                    timestamp=proposal.timestamp,
                    audio_score=proposal.audio_score,
                    video_score=0.0,
                    combined_score=proposal.audio_score,
                    start_time=max(0.0, proposal.timestamp - pre_seconds),
                    end_time=max(proposal.timestamp + post_seconds, proposal.timestamp + 0.5),
                    confidence=confidence,
                    details={
                        "audio_score": proposal.audio_score,
                        "spectral_flux": proposal.spectral_flux,
                        "rms": proposal.rms,
                        "hf_ratio": proposal.hf_ratio,
                        "spectral_centroid_hz": proposal.spectral_centroid_hz,
                        "spectral_flatness": proposal.spectral_flatness,
                        "post_flux_ratio": proposal.post_flux_ratio,
                        "post_rms_ratio": proposal.post_rms_ratio,
                        "local_prominence": proposal.local_prominence,
                        "nearby_peaks_8s": proposal.nearby_peaks_8s,
                        "audio_model_probability": self._audio_model_probability(proposal),
                        "video_score": 0.0,
                        "audio_only": True,
                        "detector": "audio_visual_audio_only",
                    },
                )
            )
        return self._deduplicate(promoted)

    def _propose_from_audio(self, signal: np.ndarray, sample_rate: int) -> List[AudioCandidate]:
        frame_length = int(getattr(self.config, "audio_frame_length", 1024))
        hop_length = int(getattr(self.config, "audio_hop_length", 256))
        if signal.size < frame_length:
            return []

        frames = self._frame_audio(signal, frame_length, hop_length)
        if frames.size == 0:
            return []

        window = np.hanning(frame_length).astype(np.float32)
        windowed = frames * window[None, :]
        spectrum = np.abs(np.fft.rfft(windowed, axis=1))
        flux = np.maximum(0.0, spectrum[1:] - spectrum[:-1]).sum(axis=1)
        flux = np.concatenate([[0.0], flux])
        rms = np.sqrt(np.mean(windowed ** 2, axis=1))

        freqs = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
        hf_mask = freqs >= float(getattr(self.config, "audio_high_freq_cutoff_hz", 1800.0))
        hf_energy = spectrum[:, hf_mask].sum(axis=1)
        total_energy = spectrum.sum(axis=1) + 1e-8
        hf_ratio = hf_energy / total_energy
        spectral_centroid_hz = (spectrum * freqs[None, :]).sum(axis=1) / total_energy
        spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-8), axis=1)) / (np.mean(spectrum, axis=1) + 1e-8)

        flux_z = self._robust_zscore(flux)
        rms_z = self._robust_zscore(rms)
        hf_z = self._robust_zscore(hf_ratio)
        score = 0.6 * flux_z + 0.25 * hf_z + 0.15 * rms_z

        min_separation_seconds = float(getattr(self.config, "audio_peak_min_separation_seconds", 1.2))
        min_distance_frames = max(1, int(min_separation_seconds * sample_rate / hop_length))
        threshold = max(
            float(getattr(self.config, "audio_peak_threshold", 4.0)),
            float(np.median(score) + 2.0 * self._mad(score)),
        )

        peaks = self._find_peaks(score, threshold=threshold, min_distance=min_distance_frames)
        proposals: List[AudioCandidate] = []
        min_timestamp = float(getattr(self.config, "audio_ignore_before_seconds", 0.35))
        min_audio_score = float(getattr(self.config, "audio_min_score", 4.5))
        min_hf_ratio = float(getattr(self.config, "audio_min_hf_ratio", 0.115))
        early_peak_score = float(getattr(self.config, "audio_early_peak_score", 4.0))
        early_peak_max_seconds = float(getattr(self.config, "audio_early_peak_max_seconds", 0.8))
        early_peak_max_hf_ratio = float(getattr(self.config, "audio_early_peak_max_hf_ratio", 0.6))
        early_peak_max_centroid_hz = float(getattr(self.config, "audio_early_peak_max_centroid_hz", 2200.0))
        early_peak_max_flatness = float(getattr(self.config, "audio_early_peak_max_flatness", 0.45))
        for peak_idx in peaks:
            backtracked_idx = self._backtrack_onset(score, peak_idx)
            timestamp = backtracked_idx * hop_length / sample_rate
            peak_score = float(score[peak_idx])
            peak_hf_ratio = float(hf_ratio[peak_idx])
            peak_centroid_hz = float(spectral_centroid_hz[peak_idx])
            peak_flatness = float(spectral_flatness[peak_idx])
            post_flux_ratio = self._forward_ratio(flux, peak_idx, 10)
            post_rms_ratio = self._forward_ratio(rms, peak_idx, 10)
            local_prominence = self._local_prominence(score, peak_idx, 30, 3)
            nearby_peaks_8s = self._count_nearby_peaks(peaks, peak_idx, int(8.0 * sample_rate / hop_length))
            early_peak_allowed = (
                timestamp <= early_peak_max_seconds
                and peak_score >= early_peak_score
                and peak_hf_ratio <= early_peak_max_hf_ratio
                and peak_centroid_hz <= early_peak_max_centroid_hz
                and peak_flatness <= early_peak_max_flatness
            )
            if timestamp < min_timestamp and not early_peak_allowed:
                continue
            if peak_hf_ratio < min_hf_ratio:
                continue
            if peak_score < min_audio_score and not early_peak_allowed:
                continue
            audio_pattern_score = self._audio_pattern_score(
                post_flux_ratio=post_flux_ratio,
                post_rms_ratio=post_rms_ratio,
                local_prominence=local_prominence,
                spectral_flatness=peak_flatness,
                spectral_centroid_hz=peak_centroid_hz,
                hf_ratio=peak_hf_ratio,
                nearby_peaks_8s=nearby_peaks_8s,
            )
            min_pattern_score = float(getattr(self.config, "audio_pattern_min_score", 0.4))
            if not early_peak_allowed and audio_pattern_score < min_pattern_score:
                continue
            proposal = AudioCandidate(
                timestamp=timestamp,
                audio_score=peak_score,
                spectral_flux=float(flux[peak_idx]),
                rms=float(rms[peak_idx]),
                hf_ratio=peak_hf_ratio,
                spectral_centroid_hz=peak_centroid_hz,
                spectral_flatness=peak_flatness,
                post_flux_ratio=post_flux_ratio,
                post_rms_ratio=post_rms_ratio,
                local_prominence=local_prominence,
                nearby_peaks_8s=nearby_peaks_8s,
            )
            audio_model_min_probability = float(getattr(self.config, "audio_model_min_probability", 0.0))
            if self.audio_candidate_model is not None and not early_peak_allowed:
                if self._audio_model_probability(proposal) < audio_model_min_probability:
                    continue
            proposals.append(proposal)
        duration_seconds = float(signal.size) / float(sample_rate)
        return self._filter_noisy_audio_files(proposals, duration_seconds)

    def _filter_noisy_audio_files(self, proposals: Sequence[AudioCandidate], duration_seconds: float) -> List[AudioCandidate]:
        if len(proposals) <= 1:
            return list(proposals)

        noisy_peak_count = int(getattr(self.config, "audio_noise_max_peak_count", 5))
        noisy_peak_ratio = float(getattr(self.config, "audio_noise_max_top_ratio", 1.8))
        long_session_seconds = float(getattr(self.config, "audio_long_session_seconds", 120.0))
        long_session_max_candidates = int(getattr(self.config, "audio_long_session_max_candidates", 120))
        ranked = sorted(proposals, key=lambda p: p.audio_score, reverse=True)
        if len(ranked) > noisy_peak_count and ranked[1].audio_score > 0:
            top_ratio = ranked[0].audio_score / ranked[1].audio_score
            if top_ratio < noisy_peak_ratio:
                if duration_seconds >= long_session_seconds:
                    return ranked[:max(1, long_session_max_candidates)]
                return []

        return sorted(proposals, key=lambda p: p.timestamp)

    def _verify_with_video(self, video_path: str, proposals: Sequence[AudioCandidate]) -> List[VerifiedDiveCandidate]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        target_verify_fps = float(getattr(self.config, "audio_visual_verify_target_fps", 12.0))
        effective_fps = fps
        frame_step = 1
        if target_verify_fps > 0 and fps > target_verify_fps:
            frame_step = max(1, int(round(fps / target_verify_fps)))
            effective_fps = fps / frame_step
        max_verify_width = int(getattr(self.config, "audio_visual_max_verify_width", 640))
        if max_verify_width > 0 and width > max_verify_width:
            verify_scale = max_verify_width / float(width)
        else:
            verify_scale = 1.0
        verify_width = max(1, int(round(width * verify_scale)))
        verify_height = max(1, int(round(height * verify_scale)))

        splash_top = int(self.config.splash_zone_top_norm * verify_height)
        splash_bottom = int(self.config.splash_zone_bottom_norm * verify_height)
        splash_left = int(self.config.splash_zone_left_norm * verify_width)
        splash_right = int(self.config.splash_zone_right_norm * verify_width)

        diver_top = int(max(0.0, getattr(self.config, "diver_zone_top_norm", 0.15)) * verify_height)
        diver_bottom = int(min(self.config.splash_zone_top_norm, getattr(self.config, "diver_zone_bottom_norm", 0.72)) * verify_height)
        diver_left = int(max(0.0, getattr(self.config, "diver_zone_left_norm", self.config.splash_zone_left_norm)) * verify_width)
        diver_right = int(min(1.0, getattr(self.config, "diver_zone_right_norm", self.config.splash_zone_right_norm)) * verify_width)

        max_proposals = int(getattr(self.config, "audio_visual_max_proposals", 4))
        ranked_proposals = sorted(proposals, key=lambda p: p.audio_score, reverse=True)
        selected_proposals = ranked_proposals[: max(1, max_proposals)]
        verified: List[VerifiedDiveCandidate] = []
        for proposal in sorted(selected_proposals, key=lambda p: p.timestamp):
            event_frame = min(total_frames - 1, max(0, int(round(proposal.timestamp * fps))))
            pre_seconds = float(getattr(self.config, "audio_visual_verify_pre_seconds", 3.0))
            post_seconds = float(getattr(self.config, "audio_visual_verify_post_seconds", 1.0))
            start_frame = max(0, int(event_frame - pre_seconds * fps))
            end_frame = min(total_frames - 1, int(event_frame + post_seconds * fps))
            local_splash_motion, local_diver_motion = self._measure_motion_window(
                cap=cap,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_step=frame_step,
                target_width=verify_width,
                target_height=verify_height,
                splash_roi=(splash_top, splash_bottom, splash_left, splash_right),
                diver_roi=(diver_top, diver_bottom, diver_left, diver_right),
            )
            if len(local_splash_motion) < 3:
                continue

            event_local_idx = min(len(local_splash_motion) - 1, max(1, int(round((event_frame - start_frame) / frame_step))))

            splash_peak = float(
                np.max(local_splash_motion[max(0, event_local_idx - 4): min(len(local_splash_motion), event_local_idx + 6)])
            )
            pre_diver_peak = float(
                np.max(local_diver_motion[max(0, event_local_idx - int(1.5 * effective_fps)): max(1, event_local_idx)])
            )
            pre_splash_baseline = float(np.median(local_splash_motion[:max(2, event_local_idx)]))
            post_splash_baseline = float(np.median(local_splash_motion[min(len(local_splash_motion) - 1, event_local_idx):]))

            video_score = (
                0.55 * self._safe_ratio(splash_peak, pre_splash_baseline + 1e-6)
                + 0.30 * self._safe_ratio(pre_diver_peak, float(np.median(local_diver_motion[:max(2, event_local_idx)]) + 1e-6))
                + 0.15 * self._safe_ratio(splash_peak, post_splash_baseline + 1e-6)
            )

            min_video_score = float(getattr(self.config, "audio_visual_min_video_score", 0.8))
            hard_video_floor = float(getattr(self.config, "audio_visual_hard_video_floor", 0.2))
            audio_rescue_score = float(getattr(self.config, "audio_visual_audio_rescue_score", 4.0))
            rescue_splash_ratio = float(getattr(self.config, "audio_visual_rescue_splash_ratio", 1.35))
            splash_ratio = self._safe_ratio(splash_peak, pre_splash_baseline + 1e-6)

            video_gate_passed = video_score >= min_video_score
            audio_rescue_passed = (
                proposal.audio_score >= audio_rescue_score
                and video_score >= hard_video_floor
                and splash_ratio >= rescue_splash_ratio
            )
            if not (video_gate_passed or audio_rescue_passed):
                continue

            takeoff_idx = self._estimate_takeoff_index(local_diver_motion, event_local_idx, effective_fps)
            clip_end_idx = self._estimate_end_index(local_splash_motion, event_local_idx, effective_fps)
            refined_start_time = start_frame / fps + (takeoff_idx / effective_fps)
            refined_end_time = start_frame / fps + (clip_end_idx / effective_fps)

            audio_weight = float(getattr(self.config, "audio_priority_weight", 0.85))
            audio_weight = min(0.95, max(0.5, audio_weight))
            combined_score = audio_weight * proposal.audio_score + (1.0 - audio_weight) * video_score
            confidence = "high" if combined_score >= 7.5 else "medium" if combined_score >= 3.8 else "low"
            min_combined_score = float(getattr(self.config, "audio_visual_min_combined_score", 3.8))
            if combined_score < min_combined_score:
                continue

            verified.append(
                VerifiedDiveCandidate(
                    frame_idx=event_frame,
                    timestamp=proposal.timestamp,
                    audio_score=proposal.audio_score,
                    video_score=float(video_score),
                    combined_score=float(combined_score),
                    start_time=max(0.0, refined_start_time),
                    end_time=max(refined_start_time + 0.5, refined_end_time),
                    confidence=confidence,
                    details={
                        "audio_score": proposal.audio_score,
                        "spectral_flux": proposal.spectral_flux,
                        "rms": proposal.rms,
                        "hf_ratio": proposal.hf_ratio,
                        "video_score": float(video_score),
                        "video_gate_passed": video_gate_passed,
                        "audio_rescue_passed": audio_rescue_passed,
                        "splash_ratio": float(splash_ratio),
                        "splash_peak": splash_peak,
                        "pre_diver_peak": pre_diver_peak,
                        "detector": "audio_visual",
                    },
                )
            )

        cap.release()
        return self._deduplicate(verified)

    def _measure_motion_window(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        frame_step: int,
        target_width: int,
        target_height: int,
        splash_roi: Tuple[int, int, int, int],
        diver_roi: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = max(0, 1 + (max(0, end_frame - start_frame) // max(1, frame_step)))
        splash_motion = np.zeros(frame_count, dtype=np.float32)
        diver_motion = np.zeros(frame_count, dtype=np.float32)
        prev_splash_region: Optional[np.ndarray] = None
        prev_diver_region: Optional[np.ndarray] = None
        source_idx = start_frame
        out_idx = 0
        splash_top, splash_bottom, splash_left, splash_right = splash_roi
        diver_top, diver_bottom, diver_left, diver_right = diver_roi
        while source_idx <= end_frame and out_idx < frame_count:
            ok = cap.grab()
            if not ok:
                break
            if (source_idx - start_frame) % max(1, frame_step) != 0:
                source_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] != target_width or gray.shape[0] != target_height:
                gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

            splash_region = gray[splash_top:splash_bottom, splash_left:splash_right]
            if splash_region.size != 0:
                splash_region = cv2.GaussianBlur(splash_region, (5, 5), 0)
                if prev_splash_region is not None:
                    splash_motion[out_idx] = float(np.mean(cv2.absdiff(splash_region, prev_splash_region)))
                prev_splash_region = splash_region

            diver_region = gray[diver_top:diver_bottom, diver_left:diver_right]
            if diver_region.size != 0:
                diver_region = cv2.GaussianBlur(diver_region, (5, 5), 0)
                if prev_diver_region is not None:
                    diver_motion[out_idx] = float(np.mean(cv2.absdiff(diver_region, prev_diver_region)))
                prev_diver_region = diver_region

            source_idx += 1
            out_idx += 1

        return splash_motion[:out_idx], diver_motion[:out_idx]

    def _estimate_takeoff_index(self, diver_motion: np.ndarray, event_idx: int, fps: float) -> int:
        if event_idx <= 1:
            return 0
        search_start = max(0, event_idx - int(2.8 * fps))
        search_end = max(search_start + 1, event_idx - int(0.2 * fps))
        window = diver_motion[search_start:search_end]
        if window.size == 0:
            return max(0, event_idx - int(1.8 * fps))

        baseline = float(np.median(window))
        threshold = baseline + max(2.0, float(np.std(window) * 1.5))
        above = np.where(window >= threshold)[0]
        if above.size == 0:
            return max(0, event_idx - int(1.8 * fps))
        first_idx = int(above[0]) + search_start
        return max(0, first_idx - int(0.35 * fps))

    def _estimate_end_index(self, splash_motion: np.ndarray, event_idx: int, fps: float) -> int:
        if event_idx >= len(splash_motion) - 1:
            return len(splash_motion) - 1
        tail = splash_motion[event_idx:]
        if tail.size == 0:
            return event_idx
        baseline = float(np.median(tail[-max(3, min(len(tail), int(0.4 * fps))):]))
        threshold = baseline + max(1.5, float(np.std(tail) * 0.8))
        for offset in range(min(len(tail) - 1, int(1.5 * fps))):
            if tail[offset] <= threshold:
                return min(len(splash_motion) - 1, event_idx + offset + int(0.15 * fps))
        return min(len(splash_motion) - 1, event_idx + int(0.8 * fps))

    def _deduplicate(self, candidates: Sequence[VerifiedDiveCandidate]) -> List[VerifiedDiveCandidate]:
        if not candidates:
            return []
        sorted_candidates = sorted(candidates, key=lambda c: c.timestamp)
        merged: List[VerifiedDiveCandidate] = []
        merge_window = float(getattr(self.config, "audio_visual_merge_seconds", 2.0))
        for candidate in sorted_candidates:
            if not merged or candidate.timestamp - merged[-1].timestamp > merge_window:
                merged.append(candidate)
                continue
            if candidate.combined_score > merged[-1].combined_score:
                merged[-1] = candidate
        return merged

    def _frame_audio(self, signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        if signal.size < frame_length:
            return np.empty((0, frame_length), dtype=np.float32)
        windowed = np.lib.stride_tricks.sliding_window_view(signal, frame_length)
        return windowed[::hop_length]

    def _find_peaks(self, values: np.ndarray, threshold: float, min_distance: int) -> List[int]:
        peaks: List[int] = []
        last_peak = -min_distance
        for idx in range(1, len(values) - 1):
            if values[idx] < threshold:
                continue
            if values[idx] < values[idx - 1] or values[idx] < values[idx + 1]:
                continue
            if idx - last_peak < min_distance:
                if peaks and values[idx] > values[peaks[-1]]:
                    peaks[-1] = idx
                    last_peak = idx
                continue
            peaks.append(idx)
            last_peak = idx
        return peaks

    def _backtrack_onset(self, score: np.ndarray, peak_idx: int) -> int:
        floor = max(0, peak_idx - int(getattr(self.config, "audio_backtrack_frames", 20)))
        local = score[floor : peak_idx + 1]
        if local.size == 0:
            return peak_idx
        return floor + int(np.argmin(local))

    def _robust_zscore(self, values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = self._mad(values)
        return (values - median) / max(mad * 1.4826, 1e-6)

    def _mad(self, values: np.ndarray) -> float:
        median = np.median(values)
        return float(np.median(np.abs(values - median)) + 1e-6)

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        return float(numerator / max(denominator, 1e-6))

    def _forward_ratio(self, values: np.ndarray, peak_idx: int, window: int) -> float:
        pre = values[max(0, peak_idx - window):peak_idx]
        post = values[peak_idx + 1:min(len(values), peak_idx + window + 1)]
        if pre.size == 0 or post.size == 0:
            return 0.0
        return float(np.mean(post) / (np.mean(pre) + 1e-6))

    def _local_prominence(self, values: np.ndarray, peak_idx: int, radius: int, guard: int) -> float:
        left = values[max(0, peak_idx - radius):max(0, peak_idx - guard)]
        right = values[min(len(values), peak_idx + guard + 1):min(len(values), peak_idx + radius + 1)]
        if left.size or right.size:
            background = np.concatenate([left, right])
        else:
            background = values[max(0, peak_idx - radius):min(len(values), peak_idx + radius + 1)]
        return float(values[peak_idx] - np.median(background))

    def _count_nearby_peaks(self, peaks: Sequence[int], peak_idx: int, max_distance: int) -> int:
        return sum(1 for other in peaks if abs(other - peak_idx) <= max_distance)

    def _audio_pattern_score(
        self,
        post_flux_ratio: float,
        post_rms_ratio: float,
        local_prominence: float,
        spectral_flatness: float,
        spectral_centroid_hz: float,
        hf_ratio: float,
        nearby_peaks_8s: int,
    ) -> float:
        score = (
            1.0 * max(post_flux_ratio - 1.0, 0.0)
            + 0.35 * max(post_rms_ratio - 1.0, 0.0)
            + 0.15 * max(local_prominence - 4.0, 0.0)
            - 1.2 * max(spectral_flatness - 0.35, 0.0)
            - 0.4 * max(spectral_centroid_hz - 1800.0, 0.0) / 1000.0
            - 0.6 * max(hf_ratio - 0.45, 0.0)
            - 0.35 * max(float(nearby_peaks_8s) - 1.0, 0.0)
        )
        return float(score)

    def _load_audio_candidate_model(self) -> Optional[AudioCandidateModel]:
        model_path = str(getattr(self.config, "audio_model_path", "") or "").strip()
        if not model_path:
            return None
        candidate = Path(model_path)
        if not candidate.exists():
            return None
        try:
            return AudioCandidateModel.load(candidate)
        except Exception:
            return None

    def _audio_model_probability(self, proposal: AudioCandidate) -> float:
        if self.audio_candidate_model is None:
            return 0.0
        return self.audio_candidate_model.predict_probability(
            {
                "audio_score": proposal.audio_score,
                "spectral_flux": proposal.spectral_flux,
                "rms": proposal.rms,
                "hf_ratio": proposal.hf_ratio,
                "spectral_centroid_hz": proposal.spectral_centroid_hz,
                "spectral_flatness": proposal.spectral_flatness,
                "post_flux_ratio": proposal.post_flux_ratio,
                "post_rms_ratio": proposal.post_rms_ratio,
                "local_prominence": proposal.local_prominence,
                "nearby_peaks_8s": float(proposal.nearby_peaks_8s),
            }
        )
