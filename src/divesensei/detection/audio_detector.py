#!/usr/bin/env python3
"""
Audio-led dive proposal detection with optional classifier and video verification.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from divesensei.detection.audio_clip_model import AudioClipModel
from divesensei.detection.audio_features import compute_multiband_pcen_features, extract_clip_feature_map, frame_audio, load_wav_mono_float32
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
    def __init__(self, config: Any, progress_callback: Callable[[dict[str, Any]], None] | None = None):
        self.config = config
        self.progress_callback = progress_callback
        configure_runtime(int(getattr(config, "opencv_threads", 1)))
        self.audio_candidate_model = self._load_audio_candidate_model()
        self.audio_clip_model = self._load_audio_clip_model()

    def inspect_audio_proposals(self, video_path: str) -> List[Dict[str, Any]]:
        signal, sample_rate = self._extract_audio_signal(video_path)
        return self.inspect_audio_proposals_from_signal(signal, sample_rate, source_path=video_path)

    def inspect_audio_proposals_from_audio_file(self, audio_path: str) -> List[Dict[str, Any]]:
        signal, sample_rate = load_wav_mono_float32(audio_path)
        return self.inspect_audio_proposals_from_signal(signal, sample_rate, source_path=audio_path)

    def inspect_audio_proposal_pipeline_from_audio_file(self, audio_path: str) -> Dict[str, Any]:
        signal, sample_rate = load_wav_mono_float32(audio_path)
        return self.inspect_audio_proposal_pipeline_from_signal(signal, sample_rate, source_path=audio_path)

    def inspect_audio_proposals_from_signal(self, signal: np.ndarray, sample_rate: int, *, source_path: str) -> List[Dict[str, Any]]:
        pipeline = self.inspect_audio_proposal_pipeline_from_signal(signal, sample_rate, source_path=source_path)
        return list(pipeline.get("final_proposals", []))

    def inspect_audio_proposal_pipeline_from_signal(self, signal: np.ndarray, sample_rate: int, *, source_path: str) -> Dict[str, Any]:
        detector_id = str(getattr(self.config, "detector_id", "audio_v1_heuristic") or "audio_v1_heuristic")
        source_file = Path(source_path).name
        heuristic_trace = self._inspect_heuristic_frontend_trace(signal, sample_rate)
        pcen_trace = self._inspect_pcen_frontend_trace(signal, sample_rate)
        heuristic_proposals = list(heuristic_trace["frontend_candidates"])
        pcen_proposals = list(pcen_trace["frontend_candidates"])
        frontend_candidates = [*heuristic_proposals, *pcen_proposals]
        merged_candidates, merge_events = self._merge_audio_candidates_with_events(heuristic_proposals, pcen_proposals)
        rebound_candidates, rebound_events = self._suppress_rebound_precursors_with_events(merged_candidates)
        deduped_candidates, duplicate_events = self._suppress_dominant_duplicate_followers_with_events(rebound_candidates)
        scored_candidates = self._score_audio_candidates(signal, sample_rate, deduped_candidates)
        return {
            "transient_peaks": [
                {
                    "source_video_path": str(source_path),
                    "source_file": source_file,
                    "detector_id": detector_id,
                    **row,
                }
                for row in [*heuristic_trace["transient_peaks"], *pcen_trace["transient_peaks"]]
            ],
            "raw_peaks": [
                {
                    "source_video_path": str(source_path),
                    "source_file": source_file,
                    "detector_id": detector_id,
                    **row,
                }
                for row in [*heuristic_trace["frontend_score_peaks"], *pcen_trace["frontend_score_peaks"]]
            ],
            "frontend_candidates": [
                self._serialize_proposal_candidate(
                    proposal,
                    source_path=source_path,
                    source_file=source_file,
                    detector_id=detector_id,
                    pipeline_stage="frontend_candidate",
                )
                for proposal in frontend_candidates
            ],
            "frontend_stage_summaries": [
                {
                    "source_video_path": str(source_path),
                    "source_file": source_file,
                    "detector_id": detector_id,
                    **heuristic_trace["frontend_stage_summary"],
                },
                {
                    "source_video_path": str(source_path),
                    "source_file": source_file,
                    "detector_id": detector_id,
                    **pcen_trace["frontend_stage_summary"],
                },
            ],
            "merged_candidates": [
                self._serialize_proposal_candidate(
                    proposal,
                    source_path=source_path,
                    source_file=source_file,
                    detector_id=detector_id,
                    pipeline_stage="merged_candidate",
                )
                for proposal in merged_candidates
            ],
            "suppression_events": [
                {
                    "source_video_path": str(source_path),
                    "source_file": source_file,
                    "detector_id": detector_id,
                    **event,
                }
                for event in [*merge_events, *rebound_events, *duplicate_events]
            ],
            "final_proposals": [
                self._serialize_proposal_candidate(
                    proposal,
                    source_path=source_path,
                    source_file=source_file,
                    detector_id=detector_id,
                    pipeline_stage=str(self._proposal_details(proposal).get("audio_clip_bucket", "unclassified")),
                )
                for proposal in scored_candidates
            ],
        }

    def inspect_false_negative_neighborhoods_from_audio_file(
        self,
        audio_path: str,
        timestamps_seconds: Sequence[float],
        *,
        window_seconds: float = 2.5,
        trace_stride_frames: int = 4,
    ) -> List[Dict[str, Any]]:
        signal, sample_rate = load_wav_mono_float32(audio_path)
        return self.inspect_false_negative_neighborhoods_from_signal(
            signal,
            sample_rate,
            timestamps_seconds,
            source_path=audio_path,
            window_seconds=window_seconds,
            trace_stride_frames=trace_stride_frames,
        )

    def inspect_false_negative_neighborhoods_from_signal(
        self,
        signal: np.ndarray,
        sample_rate: int,
        timestamps_seconds: Sequence[float],
        *,
        source_path: str,
        window_seconds: float = 2.5,
        trace_stride_frames: int = 4,
    ) -> List[Dict[str, Any]]:
        heuristic_trace = self._inspect_heuristic_frontend_trace(signal, sample_rate)
        pcen_trace = self._inspect_pcen_frontend_trace(signal, sample_rate)
        return [
            self._build_false_negative_neighborhood(
                float(timestamp),
                [heuristic_trace, pcen_trace],
                source_path=source_path,
                window_seconds=window_seconds,
                trace_stride_frames=trace_stride_frames,
            )
            for timestamp in timestamps_seconds
        ]

    def detect(self, video_path: str) -> List[VerifiedDiveCandidate]:
        signal, sample_rate = self._extract_audio_signal(video_path)
        return self.detect_from_signal(signal, sample_rate, video_path=video_path)

    def detect_from_audio_file(self, audio_path: str, *, video_path: str | None = None) -> List[VerifiedDiveCandidate]:
        signal, sample_rate = load_wav_mono_float32(audio_path)
        return self.detect_from_signal(signal, sample_rate, video_path=video_path)

    def detect_from_signal(
        self,
        signal: np.ndarray,
        sample_rate: int,
        *,
        video_path: str | None = None,
    ) -> List[VerifiedDiveCandidate]:
        detector_id = str(getattr(self.config, "detector_id", "audio_v1_heuristic") or "audio_v1_heuristic")

        if detector_id == "audio_v1_heuristic":
            proposals = self._propose_from_audio_heuristic(signal, sample_rate)
            if not proposals:
                return []
            if bool(getattr(self.config, "audio_visual_skip_video_verification", False)):
                return self._promote_audio_only(proposals)
            if not video_path:
                raise RuntimeError("Video verification requires a source video path.")
            return self._verify_with_video(video_path, proposals)

        proposals = self._merge_audio_candidates(
            self._propose_from_audio_heuristic(signal, sample_rate),
            self._propose_from_audio_pcen(signal, sample_rate),
        )
        if not proposals:
            return []

        accepted, ambiguous = self._classify_audio_candidates(signal, sample_rate, proposals)
        accepted = self._suppress_rebound_precursors(accepted)
        ambiguous = self._suppress_rebound_precursors(ambiguous)
        accepted = self._suppress_dominant_duplicate_followers(accepted)
        ambiguous = self._suppress_dominant_duplicate_followers(ambiguous)
        if detector_id == "audio_v2_pcen_classifier" or bool(getattr(self.config, "audio_visual_skip_video_verification", False)):
            return self._promote_audio_only(accepted)

        if detector_id == "audio_v2_hybrid_video":
            if not video_path:
                raise RuntimeError("Hybrid video verification requires a source video path.")
            promoted = self._promote_audio_only(accepted)
            verified = self._verify_with_video(video_path, ambiguous)
            return self._deduplicate([*promoted, *verified])

        if not video_path:
            raise RuntimeError("Video verification requires a source video path.")
        promoted = self._promote_audio_only(accepted)
        verified = self._verify_with_video(video_path, proposals)
        return self._deduplicate([*promoted, *verified])

    def _extract_audio_signal(self, video_path: str) -> Tuple[np.ndarray, int]:
        sample_rate = int(getattr(self.config, "audio_sample_rate", 16000))
        timeout_seconds = float(getattr(self.config, "audio_decode_timeout_seconds", 20.0))
        ffmpeg_threads = int(getattr(self.config, "ffmpeg_threads", 1))
        samples = decode_audio_mono_s16le(
            video_path=video_path,
            sample_rate=sample_rate,
            timeout_seconds=timeout_seconds,
            ffmpeg_threads=ffmpeg_threads,
            progress_callback=self.progress_callback,
            progress_interval_seconds=float(getattr(self.config, "audio_decode_progress_interval_seconds", 15.0)),
        )
        return samples, sample_rate

    def _promote_audio_only(self, proposals: Sequence[AudioCandidate]) -> List[VerifiedDiveCandidate]:
        pre_seconds = float(getattr(self.config, "audio_only_pre_seconds", 3.0))
        post_seconds = float(getattr(self.config, "audio_only_post_seconds", 1.0))
        promoted: List[VerifiedDiveCandidate] = []
        for proposal in proposals:
            confidence = "high" if proposal.audio_score >= 7.5 else "medium" if proposal.audio_score >= 3.8 else "low"
            details = self._proposal_details(proposal)
            details["video_score"] = 0.0
            details["audio_only"] = True
            details["detector"] = str(getattr(self.config, "detector_id", "audio_v1_heuristic"))
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
                    details=details,
                )
            )
        return self._deduplicate(promoted)

    def _propose_from_audio_heuristic(self, signal: np.ndarray, sample_rate: int) -> List[AudioCandidate]:
        proposals, _ = self._inspect_heuristic_frontend(signal, sample_rate)
        return proposals

    def _inspect_heuristic_frontend(self, signal: np.ndarray, sample_rate: int) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        trace = self._inspect_heuristic_frontend_trace(signal, sample_rate)
        return list(trace["frontend_candidates"]), list(trace["frontend_score_peaks"])

    def _inspect_heuristic_frontend_trace(self, signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        features = self._compute_audio_base_features(signal, sample_rate)
        score = 0.6 * self._robust_zscore(features["flux"]) + 0.25 * self._robust_zscore(features["hf_ratio"]) + 0.15 * self._robust_zscore(features["rms"])
        threshold = max(
            float(getattr(self.config, "audio_peak_threshold", 4.0)),
            float(np.median(score) + 2.0 * self._mad(score)),
        )
        transient_peaks = self._collect_signal_peaks(
            features["flux"],
            sample_rate,
            frontend_name="heuristic",
            signal_name="spectral_flux",
        )
        proposals, score_peak_rows = self._analyze_scored_frontend_peaks(features, sample_rate, score, threshold, frontend_name="heuristic")
        return {
            "frontend_name": "heuristic",
            "features": features,
            "score": score,
            "threshold": float(threshold),
            "frontend_candidates": proposals,
            "frontend_score_peaks": score_peak_rows,
            "transient_peaks": transient_peaks,
            "frontend_stage_summary": self._build_frontend_stage_summary(
                frontend_name="heuristic",
                transient_peaks=transient_peaks,
                frontend_score_peaks=score_peak_rows,
            ),
        }

    def _propose_from_audio_pcen(self, signal: np.ndarray, sample_rate: int) -> List[AudioCandidate]:
        proposals, _ = self._inspect_pcen_frontend(signal, sample_rate)
        return proposals

    def _inspect_pcen_frontend(self, signal: np.ndarray, sample_rate: int) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        trace = self._inspect_pcen_frontend_trace(signal, sample_rate)
        return list(trace["frontend_candidates"]), list(trace["frontend_score_peaks"])

    def _inspect_pcen_frontend_trace(self, signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        features = self._compute_audio_base_features(signal, sample_rate)
        pcen_features = compute_multiband_pcen_features(
            signal,
            sample_rate,
            int(getattr(self.config, "audio_frame_length", 1024)),
            int(getattr(self.config, "audio_hop_length", 256)),
        )
        onset = pcen_features["pcen_onset"]
        onset_sum = onset.mean(axis=1) if onset.size else np.empty(0, dtype=np.float32)
        onset_peak = onset.max(axis=1) if onset.size else np.empty(0, dtype=np.float32)
        if onset_sum.size == 0:
            return {
                "frontend_name": "pcen_multiband",
                "features": features,
                "score": np.empty(0, dtype=np.float32),
                "threshold": float(getattr(self.config, "audio_pcen_threshold", 2.4)),
                "frontend_candidates": [],
                "frontend_score_peaks": [],
                "transient_peaks": [],
                "frontend_stage_summary": self._build_frontend_stage_summary(
                    frontend_name="pcen_multiband",
                    transient_peaks=[],
                    frontend_score_peaks=[],
                ),
            }
        pcen_score = 0.6 * self._robust_zscore(onset_sum) + 0.4 * self._robust_zscore(onset_peak)
        heuristic_score = 0.6 * self._robust_zscore(features["flux"]) + 0.25 * self._robust_zscore(features["hf_ratio"]) + 0.15 * self._robust_zscore(features["rms"])
        merge_weight = float(getattr(self.config, "audio_pcen_merge_weight", 0.65))
        score = merge_weight * pcen_score + (1.0 - merge_weight) * heuristic_score
        threshold = max(
            float(getattr(self.config, "audio_pcen_threshold", 2.4)),
            float(np.median(score) + 1.25 * self._mad(score)),
        )
        transient_peaks = self._collect_signal_peaks(
            onset_peak,
            sample_rate,
            frontend_name="pcen_multiband",
            signal_name="pcen_onset_peak",
        )
        proposals, score_peak_rows = self._analyze_scored_frontend_peaks(
            features,
            sample_rate,
            score,
            threshold,
            frontend_name="pcen_multiband",
            onset_sum=onset_sum,
            onset_peak=onset_peak,
        )
        return {
            "frontend_name": "pcen_multiband",
            "features": features,
            "score": score,
            "threshold": float(threshold),
            "frontend_candidates": proposals,
            "frontend_score_peaks": score_peak_rows,
            "transient_peaks": transient_peaks,
            "frontend_stage_summary": self._build_frontend_stage_summary(
                frontend_name="pcen_multiband",
                transient_peaks=transient_peaks,
                frontend_score_peaks=score_peak_rows,
            ),
        }

    def _proposals_from_scored_peaks(
        self,
        features: Dict[str, np.ndarray],
        sample_rate: int,
        score: np.ndarray,
        threshold: float,
        *,
        frontend_name: str,
        onset_sum: np.ndarray | None = None,
        onset_peak: np.ndarray | None = None,
    ) -> List[AudioCandidate]:
        proposals, _ = self._analyze_scored_frontend_peaks(
            features,
            sample_rate,
            score,
            threshold,
            frontend_name=frontend_name,
            onset_sum=onset_sum,
            onset_peak=onset_peak,
        )
        return proposals

    def _analyze_scored_frontend_peaks(
        self,
        features: Dict[str, np.ndarray],
        sample_rate: int,
        score: np.ndarray,
        threshold: float,
        *,
        frontend_name: str,
        onset_sum: np.ndarray | None = None,
        onset_peak: np.ndarray | None = None,
    ) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        hop_length = int(getattr(self.config, "audio_hop_length", 256))
        min_separation_seconds = float(getattr(self.config, "audio_peak_min_separation_seconds", 1.2))
        min_distance_frames = max(1, int(min_separation_seconds * sample_rate / hop_length))
        peaks = self._find_peaks(score, threshold=threshold, min_distance=min_distance_frames)
        raw_peaks = self._find_peaks(score, threshold=float("-inf"), min_distance=1)
        proposals: List[AudioCandidate] = []
        raw_rows: List[Dict[str, Any]] = []
        min_timestamp = float(getattr(self.config, "audio_ignore_before_seconds", 0.35))
        min_audio_score = float(getattr(self.config, "audio_min_score", 4.5))
        min_hf_ratio = float(getattr(self.config, "audio_min_hf_ratio", 0.115))
        early_peak_score = float(getattr(self.config, "audio_early_peak_score", 4.0))
        early_peak_max_seconds = float(getattr(self.config, "audio_early_peak_max_seconds", 0.8))
        early_peak_max_hf_ratio = float(getattr(self.config, "audio_early_peak_max_hf_ratio", 0.6))
        early_peak_max_centroid_hz = float(getattr(self.config, "audio_early_peak_max_centroid_hz", 2200.0))
        early_peak_max_flatness = float(getattr(self.config, "audio_early_peak_max_flatness", 0.45))
        min_pattern_score = float(getattr(self.config, "audio_pattern_min_score", 0.4))
        tail_persistence_weight = float(getattr(self.config, "pre_candidate_tail_persistence_weight", 0.0))
        cluster_support_weight = float(getattr(self.config, "pre_candidate_cluster_support_weight", 0.0))
        region_descriptor_enabled = bool(getattr(self.config, "frontend_region_descriptor_enabled", False))
        dive_trend_enabled = bool(getattr(self.config, "frontend_dive_trend_enabled", False))
        region_env = (
            self._frontend_region_descriptor_envelope(
                flux=features["flux"],
                rms=features["rms"],
                sample_rate=sample_rate,
                hop_length=hop_length,
            )
            if region_descriptor_enabled or dive_trend_enabled
            else None
        )
        peak_index_set = set(peaks)
        audio_model_min_probability = float(getattr(self.config, "audio_model_min_probability", 0.0))
        for peak_idx in raw_peaks:
            backtracked_idx = self._backtrack_onset(score, peak_idx)
            timestamp = backtracked_idx * hop_length / sample_rate
            peak_timestamp = peak_idx * hop_length / sample_rate
            peak_score = float(score[peak_idx])
            peak_hf_ratio = float(features["hf_ratio"][peak_idx])
            peak_centroid_hz = float(features["spectral_centroid_hz"][peak_idx])
            peak_flatness = float(features["spectral_flatness"][peak_idx])
            post_flux_ratio = self._forward_ratio(features["flux"], peak_idx, 10)
            post_rms_ratio = self._forward_ratio(features["rms"], peak_idx, 10)
            local_prominence = self._local_prominence(score, peak_idx, 30, 3)
            nearby_peaks_8s = self._count_nearby_peaks(peaks, peak_idx, int(8.0 * sample_rate / hop_length))
            frontend_persistence = self._frontend_persistence_integral_features(
                flux=features["flux"],
                onset_sum=onset_sum,
                peak_idx=peak_idx,
                sample_rate=sample_rate,
                hop_length=hop_length,
            )
            tail_persistence = (
                self._tail_persistence_features(
                    flux=features["flux"],
                    rms=features["rms"],
                    peak_idx=peak_idx,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )
                if tail_persistence_weight > 0.0
                else {"tail_persistence_windows": [], "tail_persistence_score": 0.0}
            )
            cluster_support = (
                self._cluster_support_features(
                    score=score,
                    raw_peaks=raw_peaks,
                    peak_idx=peak_idx,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )
                if cluster_support_weight > 0.0
                else {
                    "cluster_support_window_seconds": float(getattr(self.config, "pre_candidate_cluster_support_window_seconds", 1.5)),
                    "cluster_support_min_peak_ratio": float(getattr(self.config, "pre_candidate_cluster_support_min_peak_ratio", 0.55)),
                    "cluster_support_count": 0,
                    "cluster_support_mass": 0.0,
                    "cluster_support_mass_ratio": 0.0,
                    "cluster_support_score": 0.0,
                    "cluster_support_peaks": [],
                }
            )
            region_descriptor = (
                self._frontend_region_descriptor_features(
                    env=region_env,
                    peak_idx=peak_idx,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )
                if region_env is not None
                else {
                    "frontend_region_descriptor_raw_score": 0.0,
                    "frontend_region_descriptor_probability": 0.0,
                    "frontend_region_descriptor_bonus": 0.0,
                    "frontend_region_descriptor_pre_seconds": float(
                        getattr(self.config, "frontend_region_descriptor_pre_seconds", 0.2)
                    ),
                    "frontend_region_descriptor_post_seconds": float(
                        getattr(self.config, "frontend_region_descriptor_post_seconds", 0.8)
                    ),
                    "frontend_region_peak_amplitude": 0.0,
                    "frontend_region_time_to_peak": 0.0,
                    "frontend_region_decay_slope": 0.0,
                    "frontend_region_early_energy": 0.0,
                    "frontend_region_mid_energy": 0.0,
                    "frontend_region_late_energy": 0.0,
                    "frontend_region_late_over_early": 0.0,
                    "frontend_region_duration_above_1p10": 0.0,
                }
            )
            dive_trend = (
                self._frontend_dive_trend_features(
                    flux=features["flux"],
                    rms=features["rms"],
                    onset_sum=onset_sum,
                    hf_ratio=features["hf_ratio"],
                    spectral_centroid_hz=features["spectral_centroid_hz"],
                    spectral_flatness=features["spectral_flatness"],
                    raw_peaks=raw_peaks,
                    peak_idx=peak_idx,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )
                if dive_trend_enabled and region_env is not None
                else {
                    "frontend_dive_trend_flatness_slope": 0.0,
                    "frontend_dive_trend_centroid_slope": 0.0,
                    "frontend_dive_trend_hf_lf_slope": 0.0,
                    "frontend_dive_trend_time_to_peak": 0.0,
                    "frontend_dive_trend_cluster_density": 0.0,
                    "frontend_dive_trend_raw_score": 0.0,
                    "frontend_dive_trend_probability": 0.0,
                    "frontend_dive_trend_bonus": 0.0,
                }
            )
            proposal_evidence_boost = (
                tail_persistence_weight * float(tail_persistence["tail_persistence_score"])
                + cluster_support_weight * float(cluster_support["cluster_support_score"])
            )
            effective_peak_score = float(peak_score + float(frontend_persistence["frontend_persistence_integral_bonus"]))
            early_peak_allowed = (
                timestamp <= early_peak_max_seconds
                and effective_peak_score >= early_peak_score
                and peak_hf_ratio <= early_peak_max_hf_ratio
                and peak_centroid_hz <= early_peak_max_centroid_hz
                and peak_flatness <= early_peak_max_flatness
            )
            audio_pattern_score = self._audio_pattern_score(
                post_flux_ratio=post_flux_ratio,
                post_rms_ratio=post_rms_ratio,
                local_prominence=local_prominence,
                spectral_flatness=peak_flatness,
                spectral_centroid_hz=peak_centroid_hz,
                hf_ratio=peak_hf_ratio,
                nearby_peaks_8s=nearby_peaks_8s,
            )
            pattern_persistence_bonus = 0.0
            pattern_persistence_weight = float(getattr(self.config, "frontend_pattern_persistence_bonus_weight", 0.0))
            pattern_persistence_max = float(getattr(self.config, "frontend_pattern_persistence_bonus_max", 0.0))
            if pattern_persistence_weight > 0.0 and pattern_persistence_max > 0.0:
                pattern_persistence_min_bonus = float(
                    getattr(self.config, "frontend_pattern_persistence_bonus_min_bonus", 0.0)
                )
                pattern_persistence_min_post_flux = float(
                    getattr(self.config, "frontend_pattern_persistence_bonus_min_post_flux_ratio", 1.0)
                )
                pattern_persistence_min_post_rms = float(
                    getattr(self.config, "frontend_pattern_persistence_bonus_min_post_rms_ratio", 1.0)
                )
                pattern_persistence_min_prominence = float(
                    getattr(self.config, "frontend_pattern_persistence_bonus_min_prominence", 0.0)
                )
                if (
                    float(frontend_persistence["frontend_persistence_integral_bonus"]) >= pattern_persistence_min_bonus
                    and post_flux_ratio >= pattern_persistence_min_post_flux
                    and post_rms_ratio >= pattern_persistence_min_post_rms
                    and local_prominence >= pattern_persistence_min_prominence
                ):
                    tail_strength = max(
                        min(post_flux_ratio - pattern_persistence_min_post_flux, post_rms_ratio - pattern_persistence_min_post_rms),
                        0.0,
                    )
                    persistence_strength = max(
                        float(frontend_persistence["frontend_persistence_integral_bonus"]) - pattern_persistence_min_bonus,
                        0.0,
                    )
                    prominence_strength = max(local_prominence - pattern_persistence_min_prominence, 0.0)
                    pattern_persistence_bonus = min(
                        pattern_persistence_max,
                        pattern_persistence_weight
                        * (persistence_strength + 0.6 * tail_strength + 0.05 * prominence_strength),
                    )
                    audio_pattern_score += pattern_persistence_bonus
            if front_end_is_advanced(frontend_name):
                audio_pattern_score += 0.25 * max(float(onset_sum[peak_idx]) if onset_sum is not None else 0.0, 0.0)
                audio_pattern_score += 0.15 * max(float(onset_peak[peak_idx]) if onset_peak is not None else 0.0, 0.0)
            audio_pattern_score += proposal_evidence_boost
            threshold_passed = bool(
                peak_idx in peak_index_set
                or (
                    (proposal_evidence_boost > 0.0 or float(frontend_persistence["frontend_persistence_integral_bonus"]) > 0.0)
                    and float(effective_peak_score + proposal_evidence_boost) >= threshold
                )
            )
            timestamp_allowed = not (timestamp < min_timestamp and not early_peak_allowed)
            hf_allowed = peak_hf_ratio >= min_hf_ratio
            score_allowed = early_peak_allowed or (effective_peak_score + proposal_evidence_boost) >= min_audio_score
            region_pattern_tiebreak_bonus = 0.0
            region_pattern_tiebreak_band = float(
                getattr(self.config, "frontend_region_descriptor_pattern_tiebreak_band", 0.35)
            )
            region_pattern_tiebreak_applied = False
            if (
                float(region_descriptor["frontend_region_descriptor_bonus"]) > 0.0
                and threshold_passed
                and timestamp_allowed
                and hf_allowed
                and score_allowed
                and not early_peak_allowed
                and audio_pattern_score < min_pattern_score
                and audio_pattern_score >= (min_pattern_score - region_pattern_tiebreak_band)
            ):
                region_pattern_tiebreak_bonus = float(region_descriptor["frontend_region_descriptor_bonus"])
                audio_pattern_score += region_pattern_tiebreak_bonus
                region_pattern_tiebreak_applied = region_pattern_tiebreak_bonus > 0.0
            sustained_noise_reject = (
                post_flux_ratio >= 1.6
                and post_rms_ratio >= 1.8
                and effective_peak_score < 7.0
                and local_prominence < 6.5
                and peak_centroid_hz >= 1800.0
                and peak_flatness >= 0.39
                and (float(onset_peak[peak_idx]) if onset_peak is not None else 0.0) < 3.2
            )
            sustained_noise_exception = (
                bool(getattr(self.config, "frontend_sustained_noise_exception_enabled", False))
                and float(frontend_persistence["frontend_persistence_integral_bonus"])
                >= float(getattr(self.config, "frontend_sustained_noise_exception_min_bonus", 0.0))
                and float(frontend_persistence["frontend_persistence_flux_ratio"])
                >= float(getattr(self.config, "frontend_sustained_noise_exception_min_flux_ratio", 1.0))
                and post_flux_ratio
                >= float(getattr(self.config, "frontend_sustained_noise_exception_min_post_flux_ratio", 1.0))
                and post_rms_ratio
                >= float(getattr(self.config, "frontend_sustained_noise_exception_min_post_rms_ratio", 1.0))
                and local_prominence
                >= float(getattr(self.config, "frontend_sustained_noise_exception_min_prominence", 0.0))
                and float(frontend_persistence["frontend_persistence_pcen_ratio"])
                >= float(getattr(self.config, "frontend_sustained_noise_exception_min_pcen_ratio", 0.0))
            )
            sustained_noise_reject = sustained_noise_reject and not sustained_noise_exception
            strong_impulse_candidate = effective_peak_score >= 8.0 and local_prominence >= 7.5
            region_pattern_exception = (
                bool(getattr(self.config, "frontend_region_pattern_exception_enabled", False))
                and threshold_passed
                and timestamp_allowed
                and hf_allowed
                and score_allowed
                and not early_peak_allowed
                and not strong_impulse_candidate
                and effective_peak_score >= float(getattr(self.config, "frontend_region_pattern_exception_min_score", 0.0))
                and local_prominence >= float(getattr(self.config, "frontend_region_pattern_exception_min_prominence", 0.0))
                and post_flux_ratio >= float(getattr(self.config, "frontend_region_pattern_exception_min_post_flux_ratio", 1.0))
                and post_rms_ratio >= float(getattr(self.config, "frontend_region_pattern_exception_min_post_rms_ratio", 1.0))
                and float(region_descriptor["frontend_region_descriptor_bonus"])
                >= float(getattr(self.config, "frontend_region_pattern_exception_min_bonus", 0.0))
            )
            dense_pcen_pattern_exception = (
                bool(getattr(self.config, "frontend_dense_pcen_pattern_exception_enabled", False))
                and str(frontend_name) == "pcen_multiband"
                and threshold_passed
                and timestamp_allowed
                and hf_allowed
                and score_allowed
                and not early_peak_allowed
                and not strong_impulse_candidate
                and effective_peak_score >= float(getattr(self.config, "frontend_dense_pcen_pattern_exception_min_score", 0.0))
                and local_prominence >= float(getattr(self.config, "frontend_dense_pcen_pattern_exception_min_prominence", 0.0))
                and post_flux_ratio >= float(getattr(self.config, "frontend_dense_pcen_pattern_exception_min_post_flux_ratio", 1.0))
                and post_rms_ratio >= float(getattr(self.config, "frontend_dense_pcen_pattern_exception_min_post_rms_ratio", 1.0))
                and int(nearby_peaks_8s) >= int(getattr(self.config, "frontend_dense_pcen_pattern_exception_min_nearby_peaks", 0))
                and peak_flatness <= float(getattr(self.config, "frontend_dense_pcen_pattern_exception_max_flatness", 1.0))
            )
            region_tail_imbalance_exception = (
                bool(getattr(self.config, "frontend_region_tail_imbalance_exception_enabled", False))
                and str(frontend_name) == "pcen_multiband"
                and threshold_passed
                and timestamp_allowed
                and hf_allowed
                and score_allowed
                and not early_peak_allowed
                and not strong_impulse_candidate
                and effective_peak_score >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_score", 0.0))
                and local_prominence >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_prominence", 0.0))
                and post_flux_ratio >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_post_flux_ratio", 1.0))
                and post_rms_ratio >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_post_rms_ratio", 1.0))
                and post_rms_ratio <= float(getattr(self.config, "frontend_region_tail_imbalance_exception_max_post_rms_ratio", 999.0))
                and float(region_descriptor["frontend_region_descriptor_bonus"])
                >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_bonus", 0.0))
                and float(region_descriptor["frontend_region_late_over_early"])
                >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_late_over_early", 0.0))
                and float(region_descriptor["frontend_region_duration_above_1p10"])
                >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_duration_above_1p10", 0.0))
                and float(region_descriptor["frontend_region_time_to_peak"])
                >= float(getattr(self.config, "frontend_region_tail_imbalance_exception_min_time_to_peak", 0.0))
                and float(region_descriptor["frontend_region_time_to_peak"])
                <= float(getattr(self.config, "frontend_region_tail_imbalance_exception_max_time_to_peak", 999.0))
                and peak_flatness <= float(getattr(self.config, "frontend_region_tail_imbalance_exception_max_flatness", 1.0))
            )
            short_region_tail_exception = (
                bool(getattr(self.config, "frontend_short_region_tail_exception_enabled", False))
                and str(frontend_name) == "pcen_multiband"
                and threshold_passed
                and timestamp_allowed
                and hf_allowed
                and score_allowed
                and not early_peak_allowed
                and not strong_impulse_candidate
                and effective_peak_score >= float(getattr(self.config, "frontend_short_region_tail_exception_min_score", 0.0))
                and local_prominence >= float(getattr(self.config, "frontend_short_region_tail_exception_min_prominence", 0.0))
                and post_flux_ratio >= float(getattr(self.config, "frontend_short_region_tail_exception_min_post_flux_ratio", 1.0))
                and post_rms_ratio >= float(getattr(self.config, "frontend_short_region_tail_exception_min_post_rms_ratio", 1.0))
                and float(region_descriptor["frontend_region_descriptor_bonus"])
                >= float(getattr(self.config, "frontend_short_region_tail_exception_min_bonus", 0.0))
                and float(region_descriptor["frontend_region_descriptor_bonus"])
                <= float(getattr(self.config, "frontend_short_region_tail_exception_max_bonus", 999.0))
                and float(region_descriptor["frontend_region_late_over_early"])
                >= float(getattr(self.config, "frontend_short_region_tail_exception_min_late_over_early", 0.0))
                and float(region_descriptor["frontend_region_duration_above_1p10"])
                >= float(getattr(self.config, "frontend_short_region_tail_exception_min_duration_above_1p10", 0.0))
                and float(region_descriptor["frontend_region_duration_above_1p10"])
                <= float(getattr(self.config, "frontend_short_region_tail_exception_max_duration_above_1p10", 999.0))
                and int(nearby_peaks_8s) >= int(getattr(self.config, "frontend_short_region_tail_exception_min_nearby_peaks", 0))
                and int(nearby_peaks_8s) <= int(getattr(self.config, "frontend_short_region_tail_exception_max_nearby_peaks", 999))
                and float(region_descriptor["frontend_region_time_to_peak"])
                >= float(getattr(self.config, "frontend_short_region_tail_exception_min_time_to_peak", 0.0))
                and float(region_descriptor["frontend_region_time_to_peak"])
                <= float(getattr(self.config, "frontend_short_region_tail_exception_max_time_to_peak", 999.0))
                and peak_flatness <= float(getattr(self.config, "frontend_short_region_tail_exception_max_flatness", 1.0))
            )
            proposal = AudioCandidate(
                timestamp=timestamp,
                audio_score=effective_peak_score,
                spectral_flux=float(features["flux"][peak_idx]),
                rms=float(features["rms"][peak_idx]),
                hf_ratio=peak_hf_ratio,
                spectral_centroid_hz=peak_centroid_hz,
                spectral_flatness=peak_flatness,
                post_flux_ratio=post_flux_ratio,
                post_rms_ratio=post_rms_ratio,
                local_prominence=local_prominence,
                nearby_peaks_8s=nearby_peaks_8s,
            )
            setattr(proposal, "_pre_candidate_tail_persistence_score", float(tail_persistence["tail_persistence_score"]))
            setattr(proposal, "_pre_candidate_cluster_support_score", float(cluster_support["cluster_support_score"]))
            setattr(
                proposal,
                "_pre_candidate_region_descriptor_bonus",
                float(region_descriptor["frontend_region_descriptor_bonus"]),
            )
            setattr(proposal, "_pre_candidate_dive_trend_flatness_slope", float(dive_trend["frontend_dive_trend_flatness_slope"]))
            setattr(proposal, "_pre_candidate_dive_trend_centroid_slope", float(dive_trend["frontend_dive_trend_centroid_slope"]))
            setattr(proposal, "_pre_candidate_dive_trend_hf_lf_slope", float(dive_trend["frontend_dive_trend_hf_lf_slope"]))
            setattr(proposal, "_pre_candidate_dive_trend_time_to_peak", float(dive_trend["frontend_dive_trend_time_to_peak"]))
            setattr(proposal, "_pre_candidate_dive_trend_cluster_density", float(dive_trend["frontend_dive_trend_cluster_density"]))
            setattr(proposal, "_pre_candidate_dive_trend_raw_score", 0.0)
            setattr(proposal, "_pre_candidate_dive_trend_probability", 0.0)
            setattr(proposal, "_pre_candidate_dive_trend_bonus", 0.0)
            setattr(proposal, "_pre_candidate_evidence_boost", float(proposal_evidence_boost))
            details = self._proposal_details(proposal)
            details["proposal_frontend"] = frontend_name
            details["proposal_threshold"] = float(threshold)
            details["peak_score_minus_threshold"] = float(peak_score - threshold)
            details["frontend_effective_score"] = float(effective_peak_score)
            details["frontend_effective_score_minus_threshold"] = float(effective_peak_score - threshold)
            details.update(frontend_persistence)
            details["peak_frame_index"] = int(peak_idx)
            details["backtracked_frame_index"] = int(backtracked_idx)
            details["peak_timestamp_seconds"] = float(peak_timestamp)
            details["proposal_timestamp_seconds"] = float(timestamp)
            details["selected_by_peak_threshold"] = bool(peak_idx in peak_index_set)
            details["tail_persistence_windows"] = tail_persistence["tail_persistence_windows"]
            details["tail_persistence_score"] = float(tail_persistence["tail_persistence_score"])
            details["cluster_support_window_seconds"] = float(cluster_support["cluster_support_window_seconds"])
            details["cluster_support_min_peak_ratio"] = float(cluster_support["cluster_support_min_peak_ratio"])
            details["cluster_support_count"] = int(cluster_support["cluster_support_count"])
            details["cluster_support_mass"] = float(cluster_support["cluster_support_mass"])
            details["cluster_support_mass_ratio"] = float(cluster_support["cluster_support_mass_ratio"])
            details["cluster_support_score"] = float(cluster_support["cluster_support_score"])
            details["cluster_support_peaks"] = cluster_support["cluster_support_peaks"]
            details.update(region_descriptor)
            details.update(dive_trend)
            details["proposal_evidence_boost"] = float(proposal_evidence_boost)
            if onset_sum is not None:
                details["pcen_onset_mean"] = float(onset_sum[peak_idx])
            if onset_peak is not None:
                details["pcen_onset_peak"] = float(onset_peak[peak_idx])
            pattern_allowed = (
                early_peak_allowed
                or strong_impulse_candidate
                or region_pattern_exception
                or dense_pcen_pattern_exception
                or region_tail_imbalance_exception
                or short_region_tail_exception
                or audio_pattern_score >= min_pattern_score
            )
            audio_model_probability = None
            audio_model_allowed = True
            if self.audio_candidate_model is not None and not early_peak_allowed:
                audio_model_probability = self._audio_model_probability(proposal)
                audio_model_allowed = audio_model_probability >= audio_model_min_probability
            details["audio_model_probability"] = audio_model_probability if audio_model_probability is not None else details.get("audio_model_probability", 0.0)
            details["audio_pattern_score_before_region_tiebreak"] = float(audio_pattern_score - region_pattern_tiebreak_bonus)
            details["audio_pattern_score"] = float(audio_pattern_score)
            details["frontend_pattern_persistence_bonus"] = float(pattern_persistence_bonus)
            details["frontend_region_pattern_tiebreak_bonus"] = float(region_pattern_tiebreak_bonus)
            details["frontend_region_pattern_tiebreak_applied"] = bool(region_pattern_tiebreak_applied)
            details["frontend_region_pattern_tiebreak_band"] = float(region_pattern_tiebreak_band)
            details["frontend_region_pattern_exception"] = bool(region_pattern_exception)
            details["frontend_dense_pcen_pattern_exception"] = bool(dense_pcen_pattern_exception)
            details["frontend_region_tail_imbalance_exception"] = bool(region_tail_imbalance_exception)
            details["frontend_short_region_tail_exception"] = bool(short_region_tail_exception)
            details["early_peak_allowed"] = bool(early_peak_allowed)
            details["strong_impulse_candidate"] = bool(strong_impulse_candidate)
            details["threshold_passed"] = bool(threshold_passed)
            details["timestamp_allowed"] = bool(timestamp_allowed)
            details["hf_allowed"] = bool(hf_allowed)
            details["score_allowed"] = bool(score_allowed)
            details["pattern_allowed"] = bool(pattern_allowed)
            details["sustained_noise_reject"] = bool(sustained_noise_reject)
            details["sustained_noise_exception"] = bool(sustained_noise_exception)
            details["audio_model_allowed"] = bool(audio_model_allowed)
            details["evidence_threshold_promoted"] = bool(peak_idx not in peak_index_set and threshold_passed)
            ranking_components = self._proposal_ranking_components(proposal)
            details.update(ranking_components)
            rejection_stage = "accepted"
            if not threshold_passed:
                rejection_stage = "below_threshold"
            elif not timestamp_allowed:
                rejection_stage = "ignored_before_start"
            elif not hf_allowed:
                rejection_stage = "low_hf_ratio"
            elif not score_allowed:
                rejection_stage = "low_audio_score"
            elif sustained_noise_reject:
                rejection_stage = "sustained_noise_reject"
            elif not pattern_allowed:
                rejection_stage = "weak_pattern_score"
            elif not audio_model_allowed:
                rejection_stage = "audio_model_rejected"
            details["rejection_stage"] = rejection_stage
            proposal = self._attach_details(proposal, details)
            raw_rows.append(
                {
                    "proposal_frontend": frontend_name,
                    "timestamp": float(timestamp),
                    "peak_timestamp_seconds": float(peak_timestamp),
                    "peak_frame_index": int(peak_idx),
                    "backtracked_frame_index": int(backtracked_idx),
                    "raw_peak_score": float(peak_score),
                    "raw_proposal_score": float(effective_peak_score),
                    "proposal_threshold": float(threshold),
                    "peak_score_minus_threshold": float(peak_score - threshold),
                    "frontend_effective_score_minus_threshold": float(effective_peak_score - threshold),
                    "accepted_as_proposal": rejection_stage == "accepted",
                    "evidence_threshold_promoted": bool(peak_idx not in peak_index_set and threshold_passed),
                    "rejection_stage": rejection_stage,
                    **details,
                }
            )
            if rejection_stage == "accepted":
                proposals.append(proposal)
        duration_seconds = float(features["duration_seconds"])
        self._annotate_local_peak_consolidation(proposals)
        proposal_details_by_id = {
            self._proposal_identity(proposal): self._proposal_details(proposal)
            for proposal in proposals
        }
        for row in raw_rows:
            if row.get("rejection_stage") != "accepted":
                continue
            details = proposal_details_by_id.get(self._proposal_identity_from_row(row))
            if not details:
                continue
            row.update(
                {
                    "tail_persistence_score": float(details.get("tail_persistence_score", 0.0) or 0.0),
                    "cluster_support_score": float(details.get("cluster_support_score", 0.0) or 0.0),
                    "frontend_region_descriptor_bonus": float(
                        details.get("frontend_region_descriptor_bonus", 0.0) or 0.0
                    ),
                    "frontend_region_descriptor_probability": float(
                        details.get("frontend_region_descriptor_probability", 0.0) or 0.0
                    ),
                    "proposal_evidence_boost": float(details.get("proposal_evidence_boost", 0.0) or 0.0),
                    "consolidation_score": float(details.get("consolidation_score", 0.0) or 0.0),
                    "consolidation_bonus": float(details.get("consolidation_bonus", 0.0) or 0.0),
                    "consolidation_group_count": int(details.get("consolidation_group_count", 0) or 0),
                    "consolidation_compactness": float(details.get("consolidation_compactness", 0.0) or 0.0),
                    "consolidation_persistence": float(details.get("consolidation_persistence", 0.0) or 0.0),
                    "rank_bonus": float(details.get("rank_bonus", 0.0) or 0.0),
                    "rank_score": float(details.get("rank_score", row.get("raw_proposal_score", 0.0)) or 0.0),
                    "dive_likeness": float(details.get("dive_likeness", 0.0) or 0.0),
                }
            )
        filtered_proposals, noisy_filter_events = self._filter_noisy_audio_files_with_events(proposals, duration_seconds)
        kept_identities = {self._proposal_identity(proposal) for proposal in filtered_proposals}
        local_rescue_identities = {
            self._proposal_identity(proposal)
            for proposal in filtered_proposals
            if bool(getattr(proposal, "_pre_candidate_local_rescue", False))
        }
        protected_identities = {
            self._proposal_identity(proposal)
            for proposal in filtered_proposals
            if bool(getattr(proposal, "_pre_candidate_protected_survivor", False))
        }
        dropped_by_cap = {self._proposal_identity_from_event(event) for event in noisy_filter_events}
        for row in raw_rows:
            if row["rejection_stage"] != "accepted":
                row["frontend_candidate_survived"] = False
                row["pre_candidate_loss_stage"] = row["rejection_stage"]
                continue
            row_identity = self._proposal_identity_from_row(row)
            survived = row_identity in kept_identities
            local_rescue_survivor = row_identity in local_rescue_identities
            protected_survivor = row_identity in protected_identities
            row["frontend_candidate_survived"] = bool(survived)
            row["local_rescue_survivor"] = bool(local_rescue_survivor)
            row["protected_survivor"] = bool(protected_survivor)
            row["pre_candidate_loss_stage"] = (
                "local_rescue_survivor"
                if local_rescue_survivor
                else "protected_survivor"
                if protected_survivor
                else "frontend_candidate_survived"
                if survived
                else "dropped_by_noisy_session_cap"
                if row_identity in dropped_by_cap
                else "filtered_after_accept"
            )
        return filtered_proposals, raw_rows

    def _classify_audio_candidates(
        self,
        signal: np.ndarray,
        sample_rate: int,
        proposals: Sequence[AudioCandidate],
    ) -> tuple[List[AudioCandidate], List[AudioCandidate]]:
        accepted: List[AudioCandidate] = []
        ambiguous: List[AudioCandidate] = []
        for proposal in self._score_audio_candidates(signal, sample_rate, proposals):
            bucket = str(self._proposal_details(proposal).get("audio_clip_bucket", "rejected"))
            if bucket in {"accepted", "accepted_no_model"}:
                accepted.append(proposal)
            elif bucket == "ambiguous":
                ambiguous.append(proposal)
        return accepted, ambiguous

    def _score_audio_candidates(
        self,
        signal: np.ndarray,
        sample_rate: int,
        proposals: Sequence[AudioCandidate],
    ) -> List[AudioCandidate]:
        scored: List[AudioCandidate] = []
        low = float(getattr(self.config, "audio_clip_classifier_ambiguity_low", 0.35))
        high = float(getattr(self.config, "audio_clip_classifier_ambiguity_high", 0.65))
        min_probability = float(getattr(self.config, "audio_clip_model_min_probability", 0.5))
        window_seconds = float(getattr(self.config, "audio_clip_classifier_window_seconds", 3.0))
        frame_length = int(getattr(self.config, "audio_frame_length", 1024))
        hop_length = int(getattr(self.config, "audio_hop_length", 256))
        for proposal in proposals:
            features = extract_clip_feature_map(
                signal,
                sample_rate,
                proposal.timestamp,
                window_seconds=window_seconds,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            details = self._proposal_details(proposal)
            details.update(features)
            details["clip_feature_window_seconds"] = window_seconds
            details["audio_clip_classifier_ambiguity_low"] = low
            details["audio_clip_classifier_ambiguity_high"] = high
            details["audio_clip_model_min_probability"] = min_probability
            if self.audio_clip_model is None:
                details["audio_clip_probability"] = None
                details["audio_clip_bucket"] = "accepted_no_model"
                details["audio_clip_model_available"] = False
            else:
                probability = self.audio_clip_model.predict_probability(features)
                details["audio_clip_probability"] = probability
                details["audio_clip_model_available"] = True
                if probability >= max(min_probability, high):
                    details["audio_clip_bucket"] = "accepted"
                elif probability >= low:
                    details["audio_clip_bucket"] = "ambiguous"
                else:
                    details["audio_clip_bucket"] = "rejected"
            scored.append(self._attach_details(proposal, details))
        return scored

    def _filter_noisy_audio_files(self, proposals: Sequence[AudioCandidate], duration_seconds: float) -> List[AudioCandidate]:
        kept, _ = self._filter_noisy_audio_files_with_events(proposals, duration_seconds)
        return kept

    def _filter_noisy_audio_files_with_events(
        self,
        proposals: Sequence[AudioCandidate],
        duration_seconds: float,
    ) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        if len(proposals) <= 1:
            return list(proposals), []

        noisy_peak_count = int(getattr(self.config, "audio_noise_max_peak_count", 5))
        noisy_peak_ratio = float(getattr(self.config, "audio_noise_max_top_ratio", 1.8))
        long_session_seconds = float(getattr(self.config, "audio_long_session_seconds", 120.0))
        long_session_max_candidates = int(getattr(self.config, "audio_long_session_max_candidates", 120))
        soft_ratio = float(getattr(self.config, "pre_candidate_soft_ratio", 0.0))
        max_extra = int(getattr(self.config, "pre_candidate_max_extra_candidates", 0))
        local_window_seconds = float(getattr(self.config, "pre_candidate_local_window_seconds", 0.0))
        local_ratio = float(getattr(self.config, "pre_candidate_local_ratio", 0.0))
        tail_ratio = float(getattr(self.config, "pre_candidate_tail_ratio", 0.0))
        tail_boost = float(getattr(self.config, "pre_candidate_tail_boost", 0.0))
        protect_window_seconds = float(getattr(self.config, "pre_candidate_protect_survivor_window_seconds", 0.0))
        protect_max_per_bucket = max(0, int(getattr(self.config, "pre_candidate_protect_survivor_max_per_bucket", 1)))
        protect_min_score = float(getattr(self.config, "pre_candidate_protect_survivor_min_score", 0.0))
        protect_min_dive_likeness = float(getattr(self.config, "pre_candidate_protect_survivor_min_dive_likeness", 0.0))
        protect_min_prominence = float(getattr(self.config, "pre_candidate_protect_survivor_min_prominence", 0.0))
        protect_min_tail_ratio = float(getattr(self.config, "pre_candidate_protect_survivor_min_tail_ratio", 0.0))
        rescue_window_seconds = float(getattr(self.config, "pre_candidate_local_rescue_window_seconds", 0.0))
        rescue_max_per_bucket = max(0, int(getattr(self.config, "pre_candidate_local_rescue_max_per_bucket", 1)))
        rescue_max_per_session = max(0, int(getattr(self.config, "pre_candidate_local_rescue_max_per_session", 2)))
        rescue_anchor_min_rank_score = float(getattr(self.config, "pre_candidate_local_rescue_anchor_min_rank_score", 0.0))
        rescue_min_score = float(getattr(self.config, "pre_candidate_local_rescue_min_score", 0.0))
        rescue_min_dive_likeness = float(getattr(self.config, "pre_candidate_local_rescue_min_dive_likeness", 0.0))
        rescue_min_prominence = float(getattr(self.config, "pre_candidate_local_rescue_min_prominence", 0.0))
        rescue_min_tail_persistence_score = float(getattr(self.config, "pre_candidate_local_rescue_min_tail_persistence_score", 0.0))
        rescue_min_cluster_support_score = float(getattr(self.config, "pre_candidate_local_rescue_min_cluster_support_score", 0.0))
        self._prepare_dive_trend_rank_bonus(proposals)
        scored_items: list[dict[str, Any]] = []
        for proposal in proposals:
            self._apply_consolidation_centering(proposal)
            score_value = float(proposal.audio_score)
            tail_boosted = False
            if tail_ratio > 0.0 and proposal.post_flux_ratio >= tail_ratio:
                score_value += tail_boost
                tail_boosted = True
            ranking_components = self._proposal_ranking_components(proposal)
            rank_bonus = float(ranking_components["rank_bonus"])
            rank_score = score_value + rank_bonus
            setattr(proposal, "_pre_candidate_score_value", score_value)
            setattr(proposal, "_pre_candidate_tail_boosted", tail_boosted)
            setattr(proposal, "_pre_candidate_rank_bonus", rank_bonus)
            setattr(proposal, "_pre_candidate_rank_score", rank_score)
            setattr(proposal, "_pre_candidate_dive_likeness", float(ranking_components["dive_likeness"]))
            setattr(proposal, "_pre_candidate_promotion_eligible", bool(ranking_components["promotion_eligible"]))
            self._proposal_details(proposal).update(ranking_components)
            scored_items.append({"proposal": proposal, "score": rank_score})

        ranked = sorted(scored_items, key=lambda item: item["score"], reverse=True)
        events: List[Dict[str, Any]] = []

        if len(ranked) <= noisy_peak_count or ranked[1]["score"] <= 0:
            return sorted(proposals, key=lambda p: p.timestamp), events

        top_ratio = ranked[0]["score"] / ranked[1]["score"]
        if top_ratio >= noisy_peak_ratio:
            return sorted(proposals, key=lambda p: p.timestamp), events

        cap = long_session_max_candidates if duration_seconds >= long_session_seconds else max(noisy_peak_count * 3, 12)
        if cap <= 0:
            return [], events

        bucket_seconds = float(getattr(self.config, "audio_noise_diversity_bucket_seconds", 0.0) or 0.0)
        if bucket_seconds > 0.0:
            bucket_seconds = max(1.0, bucket_seconds)
            buckets: dict[int, list[AudioCandidate]] = {}
            for item in ranked:
                proposal = item["proposal"]
                bucket_idx = int(float(proposal.timestamp) // bucket_seconds)
                buckets.setdefault(bucket_idx, []).append(proposal)
            selected: list[AudioCandidate] = []
            bucket_order = sorted(buckets.keys())
            round_index = 0
            while len(selected) < cap:
                added_any = False
                for bucket_idx in bucket_order:
                    bucket_rows = buckets[bucket_idx]
                    if round_index >= len(bucket_rows):
                        continue
                    selected.append(bucket_rows[round_index])
                    added_any = True
                    if len(selected) >= cap:
                        break
                if not added_any:
                    break
                round_index += 1
            base_kept = sorted(selected, key=lambda p: p.timestamp) if selected else sorted([item["proposal"] for item in ranked[: max(1, cap)]], key=lambda p: p.timestamp)
        else:
            base_kept = sorted([item["proposal"] for item in ranked[: max(1, cap)]], key=lambda p: p.timestamp)

        return self._finalize_pre_candidate_selection(
            kept_candidates=base_kept,
            ranked_proposals=[item["proposal"] for item in ranked],
            cap=cap,
            top_score=float(ranked[0]["score"]),
            soft_ratio=soft_ratio,
            max_extra=max_extra,
            local_window_seconds=local_window_seconds,
            local_ratio=local_ratio,
            protect_window_seconds=protect_window_seconds,
            protect_max_per_bucket=protect_max_per_bucket,
            protect_min_score=protect_min_score,
            protect_min_dive_likeness=protect_min_dive_likeness,
            protect_min_prominence=protect_min_prominence,
            protect_min_tail_ratio=protect_min_tail_ratio,
            rescue_window_seconds=rescue_window_seconds,
            rescue_max_per_bucket=rescue_max_per_bucket,
            rescue_max_per_session=rescue_max_per_session,
            rescue_anchor_min_rank_score=rescue_anchor_min_rank_score,
            rescue_min_score=rescue_min_score,
            rescue_min_dive_likeness=rescue_min_dive_likeness,
            rescue_min_prominence=rescue_min_prominence,
            rescue_min_tail_persistence_score=rescue_min_tail_persistence_score,
            rescue_min_cluster_support_score=rescue_min_cluster_support_score,
            events=events,
        )

    def _finalize_pre_candidate_selection(
        self,
        *,
        kept_candidates: list[AudioCandidate],
        ranked_proposals: list[AudioCandidate],
        cap: int,
        top_score: float,
        soft_ratio: float,
        max_extra: int,
        local_window_seconds: float,
        local_ratio: float,
        protect_window_seconds: float,
        protect_max_per_bucket: int,
        protect_min_score: float,
        protect_min_dive_likeness: float,
        protect_min_prominence: float,
        protect_min_tail_ratio: float,
        rescue_window_seconds: float,
        rescue_max_per_bucket: int,
        rescue_max_per_session: int,
        rescue_anchor_min_rank_score: float,
        rescue_min_score: float,
        rescue_min_dive_likeness: float,
        rescue_min_prominence: float,
        rescue_min_tail_persistence_score: float,
        rescue_min_cluster_support_score: float,
        events: list[dict[str, Any]],
    ) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        kept_identities = {self._proposal_identity(candidate) for candidate in kept_candidates}
        bucket_window = local_window_seconds if local_window_seconds > 0 else 1.0
        extra_counts: dict[int, int] = defaultdict(int)
        for proposal in ranked_proposals:
            identity = self._proposal_identity(proposal)
            if identity in kept_identities:
                continue
            score_value = float(getattr(proposal, "_pre_candidate_score_value", proposal.audio_score))
            rank_score = float(getattr(proposal, "_pre_candidate_rank_score", score_value))
            ratio_condition = soft_ratio > 0.0 and top_score > 0.0 and score_value >= top_score * soft_ratio
            local_condition = False
            if local_window_seconds > 0.0 and local_ratio > 0.0:
                has_nearby = any(
                    abs(kept.timestamp - proposal.timestamp) <= local_window_seconds for kept in kept_candidates
                )
                if not has_nearby and score_value >= top_score * local_ratio:
                    local_condition = True
            if not (ratio_condition or local_condition):
                continue
            bucket_idx = int(proposal.timestamp // bucket_window) if bucket_window > 0.0 else 0
            if max_extra > 0 and extra_counts[bucket_idx] >= max_extra:
                continue
            extra_counts[bucket_idx] += 1
            kept_candidates.append(proposal)
            kept_identities.add(identity)
            selection_mode = "extra_soft_ratio" if ratio_condition else "extra_local_window"
            events.append(
                {
                    "event_type": "extra_candidate",
                    "selection_mode": selection_mode,
                    "cap": cap,
                    "proposal_frontend": getattr(proposal, "proposal_frontend", "unknown"),
                    "suppressed_timestamp": float(proposal.timestamp),
                    "rank_score": float(rank_score),
                    "rank_bonus": float(getattr(proposal, "_pre_candidate_rank_bonus", 0.0)),
                    "promotion_eligible": bool(getattr(proposal, "_pre_candidate_promotion_eligible", False)),
                }
            )
        if protect_window_seconds > 0.0 and protect_max_per_bucket > 0:
            protected_by_bucket: dict[int, AudioCandidate] = {}
            for proposal in ranked_proposals:
                identity = self._proposal_identity(proposal)
                if identity in kept_identities:
                    continue
                details = self._proposal_details(proposal)
                dive_likeness = float(details.get("dive_likeness", 0.0) or 0.0)
                tail_component = float(details.get("tail_component", 0.0) or 0.0)
                if (
                    float(proposal.audio_score) < protect_min_score
                    or float(proposal.local_prominence) < protect_min_prominence
                    or dive_likeness < protect_min_dive_likeness
                    or tail_component < protect_min_tail_ratio
                ):
                    continue
                bucket_idx = int(proposal.timestamp // protect_window_seconds)
                existing = protected_by_bucket.get(bucket_idx)
                if existing is None:
                    protected_by_bucket[bucket_idx] = proposal
                    continue
                existing_score = float(getattr(existing, "_pre_candidate_rank_score", existing.audio_score))
                proposal_score = float(getattr(proposal, "_pre_candidate_rank_score", proposal.audio_score))
                if proposal_score > existing_score:
                    protected_by_bucket[bucket_idx] = proposal
            for bucket_idx in sorted(protected_by_bucket.keys()):
                proposal = protected_by_bucket[bucket_idx]
                identity = self._proposal_identity(proposal)
                if identity in kept_identities:
                    continue
                same_bucket = [candidate for candidate in kept_candidates if int(candidate.timestamp // protect_window_seconds) == bucket_idx]
                if same_bucket:
                    victim = min(same_bucket, key=lambda candidate: float(getattr(candidate, "_pre_candidate_rank_score", candidate.audio_score)))
                elif kept_candidates:
                    victim = min(kept_candidates, key=lambda candidate: float(getattr(candidate, "_pre_candidate_rank_score", candidate.audio_score)))
                else:
                    victim = None
                if victim is None:
                    continue
                kept_candidates.remove(victim)
                kept_identities.remove(self._proposal_identity(victim))
                setattr(proposal, "_pre_candidate_protected_survivor", True)
                kept_candidates.append(proposal)
                kept_identities.add(identity)
                events.append(
                    {
                        "event_type": "protected_survivor",
                        "proposal_frontend": getattr(proposal, "proposal_frontend", "unknown"),
                        "suppressed_timestamp": float(victim.timestamp),
                        "survivor_timestamp": float(proposal.timestamp),
                        "survivor_score": float(getattr(proposal, "_pre_candidate_rank_score", proposal.audio_score)),
                        "victim_timestamp": float(victim.timestamp),
                        "victim_score": float(getattr(victim, "_pre_candidate_rank_score", victim.audio_score)),
                        "bucket_seconds": float(protect_window_seconds),
                        "bucket_idx": int(bucket_idx),
                        "rank_score": float(getattr(proposal, "_pre_candidate_rank_score", proposal.audio_score)),
                        "dive_likeness": float(self._proposal_details(proposal).get("dive_likeness", 0.0) or 0.0),
                        "tail_component": float(self._proposal_details(proposal).get("tail_component", 0.0) or 0.0),
                    }
                )
        kept_candidates, events = self._apply_local_rescue_survivors(
            kept_candidates=kept_candidates,
            ranked_proposals=ranked_proposals,
            rescue_window_seconds=rescue_window_seconds,
            rescue_max_per_bucket=rescue_max_per_bucket,
            rescue_max_per_session=rescue_max_per_session,
            rescue_anchor_min_rank_score=rescue_anchor_min_rank_score,
            rescue_min_score=rescue_min_score,
            rescue_min_dive_likeness=rescue_min_dive_likeness,
            rescue_min_prominence=rescue_min_prominence,
            rescue_min_tail_persistence_score=rescue_min_tail_persistence_score,
            rescue_min_cluster_support_score=rescue_min_cluster_support_score,
            events=events,
        )
        dropped_identities: set[tuple[str, int, int, int]] = set()
        for proposal in ranked_proposals:
            identity = self._proposal_identity(proposal)
            if identity in kept_identities or identity in dropped_identities:
                continue
            events.append(self._build_noisy_cap_event(proposal, cap=cap, selection_mode="score_rank"))
            dropped_identities.add(identity)
        final_kept = sorted(kept_candidates, key=lambda p: p.timestamp)
        return final_kept, events

    def _apply_local_rescue_survivors(
        self,
        *,
        kept_candidates: list[AudioCandidate],
        ranked_proposals: list[AudioCandidate],
        rescue_window_seconds: float,
        rescue_max_per_bucket: int,
        rescue_max_per_session: int,
        rescue_anchor_min_rank_score: float,
        rescue_min_score: float,
        rescue_min_dive_likeness: float,
        rescue_min_prominence: float,
        rescue_min_tail_persistence_score: float,
        rescue_min_cluster_support_score: float,
        events: list[dict[str, Any]],
    ) -> tuple[list[AudioCandidate], list[Dict[str, Any]]]:
        if rescue_window_seconds <= 0.0 or rescue_max_per_bucket <= 0 or rescue_max_per_session <= 0:
            return kept_candidates, events

        kept_identities = {self._proposal_identity(candidate) for candidate in kept_candidates}
        rescue_candidates: dict[int, dict[str, Any]] = {}
        rescue_count = 0
        for proposal in ranked_proposals:
            identity = self._proposal_identity(proposal)
            if identity in kept_identities:
                continue
            details = self._proposal_details(proposal)
            tail_persistence_score = float(details.get("tail_persistence_score", 0.0) or 0.0)
            cluster_support_score = float(details.get("cluster_support_score", 0.0) or 0.0)
            dive_likeness = float(details.get("dive_likeness", 0.0) or 0.0)
            if (
                float(proposal.audio_score) < rescue_min_score
                or float(proposal.local_prominence) < rescue_min_prominence
                or dive_likeness < rescue_min_dive_likeness
                or tail_persistence_score < rescue_min_tail_persistence_score
                or cluster_support_score < rescue_min_cluster_support_score
            ):
                continue
            bucket_idx = int(proposal.timestamp // rescue_window_seconds)
            anchor_candidates = [
                candidate
                for candidate in kept_candidates
                if int(candidate.timestamp // rescue_window_seconds) == bucket_idx
                and abs(float(candidate.timestamp) - float(proposal.timestamp)) <= rescue_window_seconds
                and float(getattr(candidate, "_pre_candidate_rank_score", candidate.audio_score)) >= rescue_anchor_min_rank_score
            ]
            if not anchor_candidates:
                continue
            rescue_priority = float(
                proposal.audio_score
                + 0.35 * tail_persistence_score
                + 0.35 * cluster_support_score
                + 0.2 * max(dive_likeness, 0.0)
                + 0.1 * max(float(proposal.local_prominence) - rescue_min_prominence, 0.0)
            )
            existing = rescue_candidates.get(bucket_idx)
            if existing is None or rescue_priority > float(existing["rescue_priority"]):
                rescue_candidates[bucket_idx] = {
                    "proposal": proposal,
                    "anchor": max(
                        anchor_candidates,
                        key=lambda candidate: float(getattr(candidate, "_pre_candidate_rank_score", candidate.audio_score)),
                    ),
                    "rescue_priority": rescue_priority,
                    "tail_persistence_score": tail_persistence_score,
                    "cluster_support_score": cluster_support_score,
                    "dive_likeness": dive_likeness,
                }

        for bucket_idx, info in sorted(rescue_candidates.items(), key=lambda item: float(item[1]["rescue_priority"]), reverse=True):
            if rescue_count >= rescue_max_per_session:
                break
            proposal = info["proposal"]
            identity = self._proposal_identity(proposal)
            if identity in kept_identities:
                continue
            same_bucket = [
                candidate
                for candidate in kept_candidates
                if int(candidate.timestamp // rescue_window_seconds) == bucket_idx
            ]
            if not same_bucket:
                continue
            if not any(
                float(getattr(candidate, "_pre_candidate_rank_score", candidate.audio_score)) >= rescue_anchor_min_rank_score
                for candidate in same_bucket
            ):
                continue
            victim = min(same_bucket, key=lambda candidate: float(getattr(candidate, "_pre_candidate_rank_score", candidate.audio_score)))
            victim_score = float(getattr(victim, "_pre_candidate_rank_score", victim.audio_score))
            rescue_priority = float(info["rescue_priority"])
            if rescue_priority <= victim_score:
                continue
            kept_candidates.remove(victim)
            kept_identities.remove(self._proposal_identity(victim))
            setattr(proposal, "_pre_candidate_local_rescue", True)
            setattr(proposal, "_pre_candidate_local_rescue_score", rescue_priority)
            kept_candidates.append(proposal)
            kept_identities.add(identity)
            rescue_count += 1
            events.append(
                {
                    "event_type": "local_rescue_survivor",
                    "proposal_frontend": getattr(proposal, "proposal_frontend", "unknown"),
                    "rescue_timestamp": float(proposal.timestamp),
                    "anchor_timestamp": float(info["anchor"].timestamp),
                    "victim_timestamp": float(victim.timestamp),
                    "rescue_window_seconds": float(rescue_window_seconds),
                    "rescue_priority": float(rescue_priority),
                    "anchor_rank_score": float(getattr(info["anchor"], "_pre_candidate_rank_score", info["anchor"].audio_score)),
                    "victim_rank_score": float(victim_score),
                    "tail_persistence_score": float(info["tail_persistence_score"]),
                    "cluster_support_score": float(info["cluster_support_score"]),
                    "dive_likeness": float(info["dive_likeness"]),
                }
            )
        return kept_candidates, events

    def _suppress_rebound_precursors(self, proposals: Sequence[AudioCandidate]) -> List[AudioCandidate]:
        kept, _ = self._suppress_rebound_precursors_with_events(proposals)
        return kept

    def _suppress_rebound_precursors_with_events(self, proposals: Sequence[AudioCandidate]) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        if len(proposals) <= 1:
            return list(proposals), []

        sorted_proposals = sorted(proposals, key=lambda p: p.timestamp)
        kept: List[AudioCandidate] = []
        events: List[Dict[str, Any]] = []
        lookahead_seconds = 3.0
        min_score_gain = 4.0
        min_hf_gain = 0.12
        min_centroid_gain_hz = 500.0
        min_post_flux_gain = 1.0
        precursor_max_hf_ratio = 0.28
        precursor_max_centroid_hz = 1700.0
        precursor_max_flatness = 0.38

        for index, proposal in enumerate(sorted_proposals):
            suppress = False
            for follower in sorted_proposals[index + 1:]:
                delta = float(follower.timestamp - proposal.timestamp)
                if delta > lookahead_seconds:
                    break
                if (
                    proposal.hf_ratio <= precursor_max_hf_ratio
                    and proposal.spectral_centroid_hz <= precursor_max_centroid_hz
                    and proposal.spectral_flatness <= precursor_max_flatness
                    and follower.audio_score >= proposal.audio_score + min_score_gain
                    and follower.hf_ratio >= proposal.hf_ratio + min_hf_gain
                    and follower.spectral_centroid_hz >= proposal.spectral_centroid_hz + min_centroid_gain_hz
                        and follower.post_flux_ratio >= proposal.post_flux_ratio + min_post_flux_gain
                ):
                    suppress = True
                    events.append(
                        {
                            "event_type": "suppressed_rebound_precursor",
                            "suppressed_timestamp": float(proposal.timestamp),
                            "suppressed_score": float(proposal.audio_score),
                            "survivor_timestamp": float(follower.timestamp),
                            "survivor_score": float(follower.audio_score),
                            "offset_seconds": float(delta),
                        }
                    )
                    break
            if not suppress:
                kept.append(proposal)
        return kept, events

    def _merge_audio_candidates(self, primary: Sequence[AudioCandidate], secondary: Sequence[AudioCandidate]) -> List[AudioCandidate]:
        merged, _ = self._merge_audio_candidates_with_events(primary, secondary)
        return merged

    def _merge_audio_candidates_with_events(
        self,
        primary: Sequence[AudioCandidate],
        secondary: Sequence[AudioCandidate],
    ) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        merged = sorted([*primary, *secondary], key=lambda item: item.timestamp)
        if not merged:
            return [], []
        merge_window = float(getattr(self.config, "audio_visual_merge_seconds", 2.0))
        deduped: List[AudioCandidate] = []
        events: List[Dict[str, Any]] = []
        for candidate in merged:
            if not deduped or candidate.timestamp - deduped[-1].timestamp > merge_window:
                deduped.append(candidate)
                continue
            if candidate.audio_score > deduped[-1].audio_score:
                events.append(
                    {
                        "event_type": "merged_replaced_by_stronger_neighbor",
                        "suppressed_timestamp": float(deduped[-1].timestamp),
                        "suppressed_score": float(deduped[-1].audio_score),
                        "survivor_timestamp": float(candidate.timestamp),
                        "survivor_score": float(candidate.audio_score),
                        "offset_seconds": float(candidate.timestamp - deduped[-1].timestamp),
                    }
                )
                deduped[-1] = candidate
            else:
                events.append(
                    {
                        "event_type": "merged_into_stronger_neighbor",
                        "suppressed_timestamp": float(candidate.timestamp),
                        "suppressed_score": float(candidate.audio_score),
                        "survivor_timestamp": float(deduped[-1].timestamp),
                        "survivor_score": float(deduped[-1].audio_score),
                        "offset_seconds": float(candidate.timestamp - deduped[-1].timestamp),
                    }
                )
        return deduped, events

    def _suppress_dominant_duplicate_followers(self, proposals: Sequence[AudioCandidate]) -> List[AudioCandidate]:
        kept, _ = self._suppress_dominant_duplicate_followers_with_events(proposals)
        return kept

    def _suppress_dominant_duplicate_followers_with_events(self, proposals: Sequence[AudioCandidate]) -> tuple[List[AudioCandidate], List[Dict[str, Any]]]:
        if len(proposals) <= 1:
            return list(proposals), []

        sorted_proposals = sorted(proposals, key=lambda p: p.timestamp)
        suppress_window = float(getattr(self.config, "audio_duplicate_suppress_window_seconds", 0.9))
        leader_min_score = float(getattr(self.config, "audio_duplicate_leader_min_score", 12.0))
        leader_min_prominence = float(getattr(self.config, "audio_duplicate_leader_min_prominence", 10.0))
        follower_max_ratio = float(getattr(self.config, "audio_duplicate_follower_max_score_ratio", 0.55))
        kept: List[AudioCandidate] = []
        events: List[Dict[str, Any]] = []
        cluster: List[AudioCandidate] = [sorted_proposals[0]]

        def flush_cluster(items: Sequence[AudioCandidate]) -> None:
            if not items:
                return
            if len(items) == 1:
                kept.extend(items)
                return
            leader = max(items, key=lambda item: item.audio_score)
            if leader.audio_score < leader_min_score or leader.local_prominence < leader_min_prominence:
                kept.extend(items)
                return
            threshold = leader.audio_score * follower_max_ratio
            for item in items:
                if item is leader or item.audio_score > threshold:
                    kept.append(item)
                else:
                    events.append(
                        {
                            "event_type": "suppressed_duplicate_follower",
                            "suppressed_timestamp": float(item.timestamp),
                            "suppressed_score": float(item.audio_score),
                            "survivor_timestamp": float(leader.timestamp),
                            "survivor_score": float(leader.audio_score),
                            "offset_seconds": float(item.timestamp - leader.timestamp),
                            "follower_score_threshold": float(threshold),
                        }
                    )

        for candidate in sorted_proposals[1:]:
            if candidate.timestamp - cluster[-1].timestamp <= suppress_window:
                cluster.append(candidate)
                continue
            flush_cluster(cluster)
            cluster = [candidate]
        flush_cluster(cluster)
        return kept, events

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
        verify_scale = max_verify_width / float(width) if max_verify_width > 0 and width > max_verify_width else 1.0
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
        selected_proposals = sorted(proposals, key=lambda p: p.audio_score, reverse=True)[: max(1, max_proposals)]
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
            splash_peak = float(np.max(local_splash_motion[max(0, event_local_idx - 4): min(len(local_splash_motion), event_local_idx + 6)]))
            pre_diver_peak = float(np.max(local_diver_motion[max(0, event_local_idx - int(1.5 * effective_fps)): max(1, event_local_idx)]))
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
            audio_rescue_passed = proposal.audio_score >= audio_rescue_score and video_score >= hard_video_floor and splash_ratio >= rescue_splash_ratio
            if not (video_gate_passed or audio_rescue_passed):
                continue
            takeoff_idx = self._estimate_takeoff_index(local_diver_motion, event_local_idx, effective_fps)
            clip_end_idx = self._estimate_end_index(local_splash_motion, event_local_idx, effective_fps)
            refined_start_time = start_frame / fps + (takeoff_idx / effective_fps)
            refined_end_time = start_frame / fps + (clip_end_idx / effective_fps)
            audio_weight = min(0.95, max(0.5, float(getattr(self.config, "audio_priority_weight", 0.85))))
            combined_score = audio_weight * proposal.audio_score + (1.0 - audio_weight) * video_score
            confidence = "high" if combined_score >= 7.5 else "medium" if combined_score >= 3.8 else "low"
            min_combined_score = float(getattr(self.config, "audio_visual_min_combined_score", 3.8))
            if combined_score < min_combined_score:
                continue
            details = self._proposal_details(proposal)
            details.update(
                {
                    "video_score": float(video_score),
                    "video_gate_passed": video_gate_passed,
                    "audio_rescue_passed": audio_rescue_passed,
                    "splash_ratio": float(splash_ratio),
                    "splash_peak": splash_peak,
                    "pre_diver_peak": pre_diver_peak,
                    "detector": str(getattr(self.config, "detector_id", "audio_v1_heuristic")),
                }
            )
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
                    details=details,
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
        return max(0, int(above[0]) + search_start - int(0.35 * fps))

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

    def _serialize_proposal_candidate(
        self,
        proposal: AudioCandidate,
        *,
        source_path: str,
        source_file: str,
        detector_id: str,
        pipeline_stage: str,
    ) -> Dict[str, Any]:
        details = self._proposal_details(proposal)
        classifier_bucket = str(details.get("audio_clip_bucket", "unclassified"))
        row = {
            "source_video_path": str(source_path),
            "source_file": source_file,
            "timestamp": float(proposal.timestamp),
            "proposal_frontend": str(details.get("proposal_frontend", "unknown")),
            "raw_proposal_score": float(proposal.audio_score),
            "audio_clip_probability": details.get("audio_clip_probability"),
            "audio_model_probability": details.get("audio_model_probability"),
            "classifier_bucket": classifier_bucket,
            "classifier_decision": "dive" if classifier_bucket in {"accepted", "accepted_no_model"} else "non-dive",
            "detector_id": detector_id,
            "pipeline_stage": pipeline_stage,
            "details": details,
        }
        for key in (
            "proposal_threshold",
            "peak_score_minus_threshold",
            "peak_frame_index",
            "backtracked_frame_index",
            "peak_timestamp_seconds",
            "proposal_timestamp_seconds",
            "selected_by_peak_threshold",
            "audio_pattern_score",
            "early_peak_allowed",
            "strong_impulse_candidate",
            "threshold_passed",
            "timestamp_allowed",
            "hf_allowed",
            "score_allowed",
            "pattern_allowed",
            "sustained_noise_reject",
            "audio_model_allowed",
            "rejection_stage",
            "pcen_onset_mean",
            "pcen_onset_peak",
            "rank_bonus",
            "rank_score",
            "frontend_dive_trend_flatness_slope",
            "frontend_dive_trend_centroid_slope",
            "frontend_dive_trend_hf_lf_slope",
            "frontend_dive_trend_time_to_peak",
            "frontend_dive_trend_cluster_density",
            "frontend_dive_trend_raw_score",
            "frontend_dive_trend_probability",
            "frontend_dive_trend_bonus",
            "tail_component",
            "asymmetry_component",
            "broadband_component",
            "decay_component",
            "dive_likeness",
            "promotion_eligible",
            "protected_survivor",
        ):
            row[key] = details.get(key)
        return row

    def _proposal_identity(self, proposal: AudioCandidate) -> tuple[str, int, int, int]:
        details = self._proposal_details(proposal)
        return (
            str(details.get("proposal_frontend", "unknown")),
            int(details.get("peak_frame_index", -1) or -1),
            int(details.get("backtracked_frame_index", -1) or -1),
            int(round(float(proposal.timestamp) * 1000.0)),
        )

    def _proposal_identity_from_row(self, row: Dict[str, Any]) -> tuple[str, int, int, int]:
        return (
            str(row.get("proposal_frontend", "unknown")),
            int(row.get("peak_frame_index", -1) or -1),
            int(row.get("backtracked_frame_index", -1) or -1),
            int(round(float(row.get("timestamp", row.get("proposal_timestamp_seconds", 0.0)) or 0.0) * 1000.0)),
        )

    def _proposal_identity_from_event(self, event: Dict[str, Any]) -> tuple[str, int, int, int]:
        return (
            str(event.get("proposal_frontend", "unknown")),
            int(event.get("peak_frame_index", -1) or -1),
            int(event.get("backtracked_frame_index", -1) or -1),
            int(round(float(event.get("suppressed_timestamp", 0.0) or 0.0) * 1000.0)),
        )

    def _build_noisy_cap_event(self, proposal: AudioCandidate, *, cap: int, selection_mode: str) -> Dict[str, Any]:
        details = self._proposal_details(proposal)
        return {
            "event_type": "dropped_by_noisy_session_cap",
            "suppressed_timestamp": float(proposal.timestamp),
            "suppressed_score": float(proposal.audio_score),
            "survivor_timestamp": None,
            "survivor_score": None,
            "offset_seconds": None,
            "selection_mode": str(selection_mode),
            "cap": int(cap),
            "proposal_frontend": str(details.get("proposal_frontend", "unknown")),
            "peak_frame_index": int(details.get("peak_frame_index", -1) or -1),
            "backtracked_frame_index": int(details.get("backtracked_frame_index", -1) or -1),
        }

    def _collect_signal_peaks(
        self,
        values: np.ndarray,
        sample_rate: int,
        *,
        frontend_name: str,
        signal_name: str,
    ) -> List[Dict[str, Any]]:
        if values.size == 0:
            return []
        hop_length = int(getattr(self.config, "audio_hop_length", 256))
        peak_indices = self._find_peaks(values, threshold=float("-inf"), min_distance=1)
        rows: List[Dict[str, Any]] = []
        for peak_idx in peak_indices:
            timestamp = peak_idx * hop_length / sample_rate
            rows.append(
                {
                    "proposal_frontend": frontend_name,
                    "transient_signal": signal_name,
                    "timestamp": float(timestamp),
                    "frame_index": int(peak_idx),
                    "value": float(values[peak_idx]),
                }
            )
        return rows

    def _build_frontend_stage_summary(
        self,
        *,
        frontend_name: str,
        transient_peaks: Sequence[Dict[str, Any]],
        frontend_score_peaks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rejection_counts: Dict[str, int] = {}
        pre_candidate_loss_counts: Dict[str, int] = {}
        for row in frontend_score_peaks:
            rejection_stage = str(row.get("rejection_stage") or "unknown")
            rejection_counts[rejection_stage] = rejection_counts.get(rejection_stage, 0) + 1
            loss_stage = str(row.get("pre_candidate_loss_stage") or "unknown")
            pre_candidate_loss_counts[loss_stage] = pre_candidate_loss_counts.get(loss_stage, 0) + 1
        return {
            "frontend_name": frontend_name,
            "transient_peak_count": len(transient_peaks),
            "frontend_score_peak_count": len(frontend_score_peaks),
            "selected_by_peak_threshold_count": sum(1 for row in frontend_score_peaks if row.get("selected_by_peak_threshold")),
            "accepted_pre_candidate_count": sum(1 for row in frontend_score_peaks if row.get("rejection_stage") == "accepted"),
            "frontend_candidate_survived_count": sum(1 for row in frontend_score_peaks if row.get("frontend_candidate_survived")),
            "frontend_score_rejection_counts": rejection_counts,
            "pre_candidate_loss_counts": pre_candidate_loss_counts,
        }

    def _build_false_negative_neighborhood(
        self,
        timestamp_seconds: float,
        traces: Sequence[Dict[str, Any]],
        *,
        source_path: str,
        window_seconds: float,
        trace_stride_frames: int,
    ) -> Dict[str, Any]:
        sample_rate = int(getattr(self.config, "audio_sample_rate", 16000))
        hop_length = int(getattr(self.config, "audio_hop_length", 256))
        stride = max(1, int(trace_stride_frames))
        frontend_rows: List[Dict[str, Any]] = []
        for trace in traces:
            score = np.asarray(trace["score"], dtype=np.float32)
            center_frame = int(round(float(timestamp_seconds) * sample_rate / hop_length))
            window_frames = max(1, int(round(float(window_seconds) * sample_rate / hop_length)))
            start_idx = max(0, center_frame - window_frames)
            end_idx = min(int(score.size), center_frame + window_frames + 1)
            trace_points = []
            for frame_idx in range(start_idx, end_idx, stride):
                frame_timestamp = frame_idx * hop_length / sample_rate
                trace_points.append(
                    {
                        "frame_index": int(frame_idx),
                        "timestamp_seconds": float(frame_timestamp),
                        "offset_seconds": float(frame_timestamp - timestamp_seconds),
                        "score": float(score[frame_idx]),
                    }
                )
            transient_peaks = [
                row
                for row in trace["transient_peaks"]
                if abs(float(row.get("timestamp", 0.0) or 0.0) - timestamp_seconds) <= window_seconds
            ]
            score_peaks = [
                row
                for row in trace["frontend_score_peaks"]
                if abs(float(row.get("timestamp", 0.0) or 0.0) - timestamp_seconds) <= window_seconds
            ]
            frontend_candidates = [
                self._serialize_proposal_candidate(
                    proposal,
                    source_path=source_path,
                    source_file=Path(source_path).name,
                    detector_id=str(getattr(self.config, "detector_id", "audio_v1_heuristic") or "audio_v1_heuristic"),
                    pipeline_stage="frontend_candidate",
                )
                for proposal in trace["frontend_candidates"]
                if abs(float(proposal.timestamp) - timestamp_seconds) <= window_seconds
            ]
            nearest_transient_peak = min(
                transient_peaks,
                key=lambda row: abs(float(row.get("timestamp", 0.0) or 0.0) - timestamp_seconds),
                default=None,
            )
            nearest_score_peak = min(
                score_peaks,
                key=lambda row: abs(float(row.get("timestamp", 0.0) or 0.0) - timestamp_seconds),
                default=None,
            )
            nearest_frontend_candidate = min(
                frontend_candidates,
                key=lambda row: abs(float(row.get("timestamp", 0.0) or 0.0) - timestamp_seconds),
                default=None,
            )
            local_loss_stage = None
            if nearest_frontend_candidate is not None:
                local_loss_stage = "frontend_candidate_survived"
            elif nearest_score_peak is None:
                local_loss_stage = "no_frontend_score_peak_nearby"
            else:
                local_loss_stage = str(nearest_score_peak.get("pre_candidate_loss_stage") or nearest_score_peak.get("rejection_stage") or "unknown")
            frontend_rows.append(
                {
                    "frontend_name": trace["frontend_name"],
                    "threshold": float(trace["threshold"]),
                    "local_trace_max": float(np.max(score[start_idx:end_idx])) if end_idx > start_idx else None,
                    "local_trace_argmax_offset_seconds": (
                        ((start_idx + int(np.argmax(score[start_idx:end_idx]))) * hop_length / sample_rate) - timestamp_seconds
                        if end_idx > start_idx
                        else None
                    ),
                    "trace_points": trace_points,
                    "transient_peaks": transient_peaks,
                    "frontend_score_peaks": score_peaks,
                    "frontend_candidates": frontend_candidates,
                    "nearest_transient_peak": nearest_transient_peak,
                    "nearest_frontend_score_peak": nearest_score_peak,
                    "nearest_frontend_candidate": nearest_frontend_candidate,
                    "local_loss_stage": local_loss_stage,
                }
            )
        return {
            "timestamp_seconds": float(timestamp_seconds),
            "window_seconds": float(window_seconds),
            "frontends": frontend_rows,
        }

    def _compute_audio_base_features(self, signal: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        frame_length = int(getattr(self.config, "audio_frame_length", 1024))
        hop_length = int(getattr(self.config, "audio_hop_length", 256))
        frames = frame_audio(signal, frame_length, hop_length)
        if frames.size == 0:
            return {
                "flux": np.empty(0, dtype=np.float32),
                "rms": np.empty(0, dtype=np.float32),
                "hf_ratio": np.empty(0, dtype=np.float32),
                "spectral_centroid_hz": np.empty(0, dtype=np.float32),
                "spectral_flatness": np.empty(0, dtype=np.float32),
                "duration_seconds": float(signal.size) / float(sample_rate),
            }
        window = np.hanning(frame_length).astype(np.float32)
        windowed = frames * window[None, :]
        spectrum = np.abs(np.fft.rfft(windowed, axis=1))
        flux = np.maximum(0.0, spectrum[1:] - spectrum[:-1]).sum(axis=1)
        flux = np.concatenate([[0.0], flux]).astype(np.float32)
        rms = np.sqrt(np.mean(windowed ** 2, axis=1)).astype(np.float32)
        freqs = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
        hf_mask = freqs >= float(getattr(self.config, "audio_high_freq_cutoff_hz", 1800.0))
        hf_energy = spectrum[:, hf_mask].sum(axis=1)
        total_energy = spectrum.sum(axis=1) + 1e-8
        hf_ratio = (hf_energy / total_energy).astype(np.float32)
        spectral_centroid_hz = ((spectrum * freqs[None, :]).sum(axis=1) / total_energy).astype(np.float32)
        spectral_flatness = (
            np.exp(np.mean(np.log(spectrum + 1e-8), axis=1)) / (np.mean(spectrum, axis=1) + 1e-8)
        ).astype(np.float32)
        return {
            "flux": flux,
            "rms": rms,
            "hf_ratio": hf_ratio,
            "spectral_centroid_hz": spectral_centroid_hz,
            "spectral_flatness": spectral_flatness,
            "duration_seconds": float(signal.size) / float(sample_rate),
        }

    def _find_peaks(self, values: np.ndarray, threshold: float, min_distance: int) -> List[int]:
        peaks: List[int] = []
        last_peak = -min_distance
        for idx in range(1, len(values) - 1):
            if values[idx] < threshold or values[idx] < values[idx - 1] or values[idx] < values[idx + 1]:
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

    def _tail_persistence_features(
        self,
        *,
        flux: np.ndarray,
        rms: np.ndarray,
        peak_idx: int,
        sample_rate: int,
        hop_length: int,
    ) -> Dict[str, Any]:
        windows_seconds = (
            float(getattr(self.config, "pre_candidate_tail_persistence_short_seconds", 0.5)),
            float(getattr(self.config, "pre_candidate_tail_persistence_medium_seconds", 1.0)),
            float(getattr(self.config, "pre_candidate_tail_persistence_long_seconds", 2.0)),
        )
        rows: list[dict[str, Any]] = []
        scores: list[float] = []
        for window_seconds in windows_seconds:
            window_frames = max(1, int(round(window_seconds * sample_rate / max(hop_length, 1))))
            flux_ratio = self._forward_ratio(flux, peak_idx, window_frames)
            rms_ratio = self._forward_ratio(rms, peak_idx, window_frames)
            flux_component = max(flux_ratio - 1.0, 0.0)
            rms_component = max(rms_ratio - 1.0, 0.0)
            score = 0.65 * flux_component + 0.35 * rms_component
            rows.append(
                {
                    "window_seconds": float(window_seconds),
                    "flux_ratio": float(flux_ratio),
                    "rms_ratio": float(rms_ratio),
                    "score": float(score),
                }
            )
            scores.append(float(score))
        return {
            "tail_persistence_windows": rows,
            "tail_persistence_score": float(np.mean(scores)) if scores else 0.0,
        }

    def _frontend_persistence_integral_features(
        self,
        *,
        flux: np.ndarray,
        onset_sum: np.ndarray | None,
        peak_idx: int,
        sample_rate: int,
        hop_length: int,
    ) -> Dict[str, Any]:
        weight = float(getattr(self.config, "frontend_persistence_integral_weight", 0.0))
        max_bonus = float(getattr(self.config, "frontend_persistence_integral_max_bonus", 0.0))
        start_seconds = float(getattr(self.config, "frontend_persistence_integral_start_seconds", 0.15))
        end_seconds = float(getattr(self.config, "frontend_persistence_integral_end_seconds", 0.8))
        pre_seconds = float(getattr(self.config, "frontend_persistence_integral_pre_seconds", 0.4))
        pcen_weight = float(getattr(self.config, "frontend_persistence_integral_pcen_weight", 0.6))
        if weight <= 0.0 or max_bonus <= 0.0 or end_seconds <= start_seconds:
            return {
                "frontend_persistence_integral_score": 0.0,
                "frontend_persistence_integral_bonus": 0.0,
                "frontend_persistence_flux_ratio": 0.0,
                "frontend_persistence_pcen_ratio": 0.0,
                "frontend_persistence_window_start_seconds": float(start_seconds),
                "frontend_persistence_window_end_seconds": float(end_seconds),
                "frontend_persistence_pre_seconds": float(pre_seconds),
            }
        start_frames = max(1, int(round(start_seconds * sample_rate / max(hop_length, 1))))
        end_frames = max(start_frames + 1, int(round(end_seconds * sample_rate / max(hop_length, 1))))
        pre_frames = max(1, int(round(pre_seconds * sample_rate / max(hop_length, 1))))

        post_flux = flux[min(len(flux), peak_idx + start_frames):min(len(flux), peak_idx + end_frames)]
        pre_flux = flux[max(0, peak_idx - pre_frames):peak_idx]
        flux_ratio = self._safe_ratio(
            float(np.mean(post_flux)) if post_flux.size else 0.0,
            float(np.mean(pre_flux)) if pre_flux.size else 0.0,
        )

        pcen_ratio = 0.0
        if onset_sum is not None:
            post_pcen = onset_sum[min(len(onset_sum), peak_idx + start_frames):min(len(onset_sum), peak_idx + end_frames)]
            pre_pcen = onset_sum[max(0, peak_idx - pre_frames):peak_idx]
            pcen_ratio = self._safe_ratio(
                float(np.mean(post_pcen)) if post_pcen.size else 0.0,
                float(np.mean(pre_pcen)) if pre_pcen.size else 0.0,
            )

        flux_component = max(flux_ratio - 1.0, 0.0)
        pcen_component = max(pcen_ratio - 1.0, 0.0)
        score = (1.0 - pcen_weight) * flux_component + pcen_weight * pcen_component
        bonus = min(max_bonus, max(0.0, weight * score))
        return {
            "frontend_persistence_integral_score": float(score),
            "frontend_persistence_integral_bonus": float(bonus),
            "frontend_persistence_flux_ratio": float(flux_ratio),
            "frontend_persistence_pcen_ratio": float(pcen_ratio),
            "frontend_persistence_window_start_seconds": float(start_seconds),
            "frontend_persistence_window_end_seconds": float(end_seconds),
            "frontend_persistence_pre_seconds": float(pre_seconds),
        }

    def _ema_series(
        self,
        values: np.ndarray,
        tau_seconds: float,
        sample_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        alpha = float(np.exp(-hop_length / max(tau_seconds * sample_rate, 1e-6)))
        out = np.empty_like(values, dtype=np.float32)
        acc = float(values[0])
        for index, value in enumerate(values):
            acc = alpha * acc + (1.0 - alpha) * float(value)
            out[index] = acc
        return out

    def _frontend_region_descriptor_envelope(
        self,
        *,
        flux: np.ndarray,
        rms: np.ndarray,
        sample_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        flux_base = self._ema_series(flux, 0.35, sample_rate, hop_length) + 1e-6
        rms_base = self._ema_series(rms, 0.35, sample_rate, hop_length) + 1e-6
        return (0.65 * (flux / flux_base) + 0.35 * (rms / rms_base)).astype(np.float32)

    def _duration_above_threshold(self, values: np.ndarray, threshold: float, sample_rate: int, hop_length: int) -> float:
        if values.size == 0:
            return 0.0
        above = values >= threshold
        longest = 0
        active_start = None
        for index, flag in enumerate(above):
            if flag and active_start is None:
                active_start = index
            elif not flag and active_start is not None:
                longest = max(longest, index - active_start)
                active_start = None
        if active_start is not None:
            longest = max(longest, len(values) - active_start)
        return float(longest * hop_length / sample_rate)

    def _frontend_region_descriptor_features(
        self,
        *,
        env: np.ndarray,
        peak_idx: int,
        sample_rate: int,
        hop_length: int,
    ) -> Dict[str, Any]:
        enabled = bool(getattr(self.config, "frontend_region_descriptor_enabled", False))
        weight = float(getattr(self.config, "frontend_region_descriptor_weight", 0.0))
        max_bonus = float(getattr(self.config, "frontend_region_descriptor_max_bonus", 0.0))
        pre_seconds = float(getattr(self.config, "frontend_region_descriptor_pre_seconds", 0.2))
        post_seconds = float(getattr(self.config, "frontend_region_descriptor_post_seconds", 0.8))
        base = {
            "frontend_region_descriptor_raw_score": 0.0,
            "frontend_region_descriptor_probability": 0.0,
            "frontend_region_descriptor_bonus": 0.0,
            "frontend_region_descriptor_pre_seconds": float(pre_seconds),
            "frontend_region_descriptor_post_seconds": float(post_seconds),
            "frontend_region_peak_amplitude": 0.0,
            "frontend_region_time_to_peak": 0.0,
            "frontend_region_decay_slope": 0.0,
            "frontend_region_early_energy": 0.0,
            "frontend_region_mid_energy": 0.0,
            "frontend_region_late_energy": 0.0,
            "frontend_region_late_over_early": 0.0,
            "frontend_region_duration_above_1p10": 0.0,
        }
        if not enabled or weight <= 0.0 or max_bonus <= 0.0 or env.size == 0:
            return base

        pre_frames = max(1, int(round(pre_seconds * sample_rate / max(hop_length, 1))))
        post_frames = max(1, int(round(post_seconds * sample_rate / max(hop_length, 1))))
        start_index = max(0, int(peak_idx) - pre_frames)
        end_index = min(len(env), int(peak_idx) + post_frames)
        region = env[start_index:end_index]
        if region.size == 0:
            return base

        time_axis = np.arange(region.size, dtype=np.float32) * hop_length / sample_rate

        def time_span(start: float, end: float) -> np.ndarray:
            begin = max(0, int(round(start * sample_rate / max(hop_length, 1))))
            finish = min(region.size, int(round(end * sample_rate / max(hop_length, 1))))
            if finish <= begin:
                return region[0:0]
            return region[begin:finish]

        early = time_span(0.0, 0.15)
        mid = time_span(0.15, 0.40)
        late = time_span(0.40, 0.80)
        peak_offset = int(np.argmax(region))
        peak_time = float(time_axis[peak_offset]) if time_axis.size else 0.0
        decay_end = min(region.size, peak_offset + max(1, int(round(0.6 * sample_rate / max(hop_length, 1)))))
        decay_segment = region[peak_offset:decay_end]
        decay_time = time_axis[peak_offset:decay_end]
        decay_slope = float(np.polyfit(decay_time, decay_segment, 1)[0]) if decay_segment.size >= 2 else 0.0

        feature_values = {
            "decay_slope": float(decay_slope),
            "early_energy": float(np.sum(early)),
            "mid_energy": float(np.sum(mid)),
            "late_energy": float(np.sum(late)),
            "late_over_early": float(np.sum(late) / (np.sum(early) + 1e-6)),
            "duration_above_1p10": float(self._duration_above_threshold(region, 1.10, sample_rate, hop_length)),
        }
        means = {
            "decay_slope": -2.2543070055384544,
            "early_energy": 10.685887813795613,
            "mid_energy": 18.133771475489812,
            "late_energy": 23.533556844441947,
            "late_over_early": 2.4132621387707935,
            "duration_above_1p10": 0.11700763358778633,
        }
        stds = {
            "decay_slope": 4.478045155805434,
            "early_energy": 3.2181968246999104,
            "mid_energy": 3.933853347519769,
            "late_energy": 4.868877563357533,
            "late_over_early": 0.9248458202450075,
            "duration_above_1p10": 0.07258782573286013,
        }
        weights = {
            "decay_slope": 1.0064918075119564,
            "early_energy": -0.05984636043726436,
            "mid_energy": -0.15566075480401506,
            "late_energy": 0.35519146274026536,
            "late_over_early": -0.26202704447630015,
            "duration_above_1p10": 0.6253123588220937,
        }
        raw_score = -2.721715148427047
        for key, weight_value in weights.items():
            raw_score += weight_value * ((feature_values[key] - means[key]) / max(stds[key], 1e-6))
        probability = float(1.0 / (1.0 + np.exp(-np.clip(raw_score, -40.0, 40.0))))
        bonus = min(max_bonus, max(0.0, weight * probability))
        return {
            "frontend_region_descriptor_raw_score": float(raw_score),
            "frontend_region_descriptor_probability": probability,
            "frontend_region_descriptor_bonus": float(bonus),
            "frontend_region_descriptor_pre_seconds": float(pre_seconds),
            "frontend_region_descriptor_post_seconds": float(post_seconds),
            "frontend_region_peak_amplitude": float(np.max(region)),
            "frontend_region_time_to_peak": float(peak_time),
            "frontend_region_decay_slope": float(feature_values["decay_slope"]),
            "frontend_region_early_energy": float(feature_values["early_energy"]),
            "frontend_region_mid_energy": float(feature_values["mid_energy"]),
            "frontend_region_late_energy": float(feature_values["late_energy"]),
            "frontend_region_late_over_early": float(feature_values["late_over_early"]),
            "frontend_region_duration_above_1p10": float(feature_values["duration_above_1p10"]),
        }

    def _frontend_dive_trend_features(
        self,
        *,
        flux: np.ndarray,
        rms: np.ndarray,
        onset_sum: np.ndarray | None,
        hf_ratio: np.ndarray,
        spectral_centroid_hz: np.ndarray,
        spectral_flatness: np.ndarray,
        raw_peaks: List[int],
        peak_idx: int,
        sample_rate: int,
        hop_length: int,
    ) -> Dict[str, Any]:
        start_seconds = -1.0
        end_seconds = 2.0
        start_frames = int(round(start_seconds * sample_rate / max(hop_length, 1)))
        end_frames = int(round(end_seconds * sample_rate / max(hop_length, 1)))
        window_start = max(0, peak_idx + start_frames)
        window_end = min(len(flux), peak_idx + end_frames)
        base = {
            "frontend_dive_trend_flatness_slope": 0.0,
            "frontend_dive_trend_centroid_slope": 0.0,
            "frontend_dive_trend_hf_lf_slope": 0.0,
            "frontend_dive_trend_time_to_peak": 0.0,
            "frontend_dive_trend_cluster_density": 0.0,
            "frontend_dive_trend_raw_score": 0.0,
            "frontend_dive_trend_probability": 0.0,
            "frontend_dive_trend_bonus": 0.0,
        }
        if window_end <= window_start:
            return base
        flux_window = flux[window_start:window_end]
        rms_window = rms[window_start:window_end]
        centroid_window = spectral_centroid_hz[window_start:window_end]
        flatness_window = spectral_flatness[window_start:window_end]
        hf_window = hf_ratio[window_start:window_end]
        onset_window = onset_sum[window_start:window_end] if onset_sum is not None else np.zeros_like(flux_window)
        if flux_window.size == 0:
            return base
        flux_base = self._ema_series(flux_window, 0.35, sample_rate, hop_length) + 1e-6
        rms_base = self._ema_series(rms_window, 0.35, sample_rate, hop_length) + 1e-6
        onset_base = self._ema_series(onset_window, 0.35, sample_rate, hop_length) + 1e-6
        combined_env = (
            0.4 * (flux_window / flux_base)
            + 0.3 * (rms_window / rms_base)
            + 0.3 * (onset_window / onset_base)
        ).astype(np.float32)
        time_axis = np.arange(combined_env.size, dtype=np.float32) * hop_length / sample_rate + float(start_seconds)
        peak_time = float(time_axis[int(np.argmax(combined_env))]) if combined_env.size else 0.0
        nearby_peak_count = sum(1 for raw_peak in raw_peaks if window_start <= int(raw_peak) <= window_end)
        cluster_density = float(nearby_peak_count / max(end_seconds - start_seconds, 1e-6))
        return {
            "frontend_dive_trend_flatness_slope": self._linear_fit_slope(time_axis, flatness_window),
            "frontend_dive_trend_centroid_slope": self._linear_fit_slope(time_axis, centroid_window),
            "frontend_dive_trend_hf_lf_slope": self._linear_fit_slope(time_axis, hf_window),
            "frontend_dive_trend_time_to_peak": float(peak_time),
            "frontend_dive_trend_cluster_density": float(cluster_density),
            "frontend_dive_trend_raw_score": 0.0,
            "frontend_dive_trend_probability": 0.0,
            "frontend_dive_trend_bonus": 0.0,
        }

    def _linear_fit_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return 0.0
        coeffs = np.polyfit(x.astype(np.float64), y.astype(np.float64), 1)
        return float(coeffs[0])

    def _frontend_persistence_integral_features(
        self,
        *,
        flux: np.ndarray,
        onset_sum: np.ndarray | None,
        peak_idx: int,
        sample_rate: int,
        hop_length: int,
    ) -> Dict[str, Any]:
        weight = float(getattr(self.config, "frontend_persistence_integral_weight", 0.0))
        max_bonus = float(getattr(self.config, "frontend_persistence_integral_max_bonus", 0.0))
        start_seconds = float(getattr(self.config, "frontend_persistence_integral_start_seconds", 0.15))
        end_seconds = float(getattr(self.config, "frontend_persistence_integral_end_seconds", 0.8))
        pre_seconds = float(getattr(self.config, "frontend_persistence_integral_pre_seconds", 0.4))
        pcen_weight = float(getattr(self.config, "frontend_persistence_integral_pcen_weight", 0.6))
        if weight <= 0.0 or max_bonus <= 0.0 or end_seconds <= start_seconds:
            return {
                "frontend_persistence_integral_score": 0.0,
                "frontend_persistence_integral_bonus": 0.0,
                "frontend_persistence_flux_ratio": 0.0,
                "frontend_persistence_pcen_ratio": 0.0,
                "frontend_persistence_window_start_seconds": float(start_seconds),
                "frontend_persistence_window_end_seconds": float(end_seconds),
                "frontend_persistence_pre_seconds": float(pre_seconds),
            }
        start_frames = max(1, int(round(start_seconds * sample_rate / max(hop_length, 1))))
        end_frames = max(start_frames + 1, int(round(end_seconds * sample_rate / max(hop_length, 1))))
        pre_frames = max(1, int(round(pre_seconds * sample_rate / max(hop_length, 1))))

        post_flux = flux[min(len(flux), peak_idx + start_frames):min(len(flux), peak_idx + end_frames)]
        pre_flux = flux[max(0, peak_idx - pre_frames):peak_idx]
        flux_ratio = self._safe_ratio(
            float(np.mean(post_flux)) if post_flux.size else 0.0,
            float(np.mean(pre_flux)) if pre_flux.size else 0.0,
        )

        pcen_ratio = 0.0
        if onset_sum is not None:
            post_pcen = onset_sum[min(len(onset_sum), peak_idx + start_frames):min(len(onset_sum), peak_idx + end_frames)]
            pre_pcen = onset_sum[max(0, peak_idx - pre_frames):peak_idx]
            pcen_ratio = self._safe_ratio(
                float(np.mean(post_pcen)) if post_pcen.size else 0.0,
                float(np.mean(pre_pcen)) if pre_pcen.size else 0.0,
            )

        flux_component = max(flux_ratio - 1.0, 0.0)
        pcen_component = max(pcen_ratio - 1.0, 0.0)
        score = (1.0 - pcen_weight) * flux_component + pcen_weight * pcen_component
        bonus = min(max_bonus, max(0.0, weight * score))
        return {
            "frontend_persistence_integral_score": float(score),
            "frontend_persistence_integral_bonus": float(bonus),
            "frontend_persistence_flux_ratio": float(flux_ratio),
            "frontend_persistence_pcen_ratio": float(pcen_ratio),
            "frontend_persistence_window_start_seconds": float(start_seconds),
            "frontend_persistence_window_end_seconds": float(end_seconds),
            "frontend_persistence_pre_seconds": float(pre_seconds),
        }

    def _cluster_support_features(
        self,
        *,
        score: np.ndarray,
        raw_peaks: Sequence[int],
        peak_idx: int,
        sample_rate: int,
        hop_length: int,
    ) -> Dict[str, Any]:
        window_seconds = float(getattr(self.config, "pre_candidate_cluster_support_window_seconds", 1.5))
        min_peak_ratio = float(getattr(self.config, "pre_candidate_cluster_support_min_peak_ratio", 0.55))
        window_frames = max(1, int(round(window_seconds * sample_rate / max(hop_length, 1))))
        peak_score = float(score[peak_idx]) if peak_idx < len(score) else 0.0
        nearby_rows: list[dict[str, Any]] = []
        support_scores: list[float] = []
        for other_idx in raw_peaks:
            if other_idx == peak_idx:
                continue
            if abs(int(other_idx) - int(peak_idx)) > window_frames:
                continue
            other_score = float(score[other_idx]) if 0 <= int(other_idx) < len(score) else 0.0
            if peak_score > 0.0 and other_score < peak_score * min_peak_ratio:
                continue
            nearby_rows.append(
                {
                    "frame_index": int(other_idx),
                    "timestamp_seconds": float(other_idx * hop_length / sample_rate),
                    "score": float(other_score),
                }
            )
            support_scores.append(float(other_score))
        cluster_mass = float(sum(support_scores))
        cluster_count = len(support_scores)
        cluster_mass_ratio = cluster_mass / max(peak_score, 1e-6)
        cluster_score = cluster_mass_ratio + 0.2 * min(cluster_count, 5)
        return {
            "cluster_support_window_seconds": float(window_seconds),
            "cluster_support_min_peak_ratio": float(min_peak_ratio),
            "cluster_support_count": int(cluster_count),
            "cluster_support_mass": float(cluster_mass),
            "cluster_support_mass_ratio": float(cluster_mass_ratio),
            "cluster_support_score": float(cluster_score),
            "cluster_support_peaks": nearby_rows,
        }

    def _local_prominence(self, values: np.ndarray, peak_idx: int, radius: int, guard: int) -> float:
        left = values[max(0, peak_idx - radius):max(0, peak_idx - guard)]
        right = values[min(len(values), peak_idx + guard + 1):min(len(values), peak_idx + radius + 1)]
        background = np.concatenate([left, right]) if left.size or right.size else values[max(0, peak_idx - radius):min(len(values), peak_idx + radius + 1)]
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
        return float(
            1.0 * max(post_flux_ratio - 1.0, 0.0)
            + 0.35 * max(post_rms_ratio - 1.0, 0.0)
            + 0.15 * max(local_prominence - 4.0, 0.0)
            - 1.2 * max(spectral_flatness - 0.35, 0.0)
            - 0.4 * max(spectral_centroid_hz - 1800.0, 0.0) / 1000.0
            - 0.6 * max(hf_ratio - 0.45, 0.0)
            - 0.35 * max(float(nearby_peaks_8s) - 1.0, 0.0)
        )

    def _annotate_local_peak_consolidation(self, proposals: Sequence[AudioCandidate]) -> None:
        weight = float(getattr(self.config, "pre_candidate_consolidation_weight", 0.0))
        window_seconds = float(getattr(self.config, "pre_candidate_consolidation_window_seconds", 0.0))
        top_peaks = max(1, int(getattr(self.config, "pre_candidate_consolidation_top_peaks", 3)))
        min_score = float(getattr(self.config, "pre_candidate_consolidation_min_score", 0.0))
        min_cluster_size = max(2, int(getattr(self.config, "pre_candidate_consolidation_min_cluster_size", 2)))
        merge_gap_seconds = float(getattr(self.config, "pre_candidate_consolidation_merge_gap_seconds", 0.12))
        max_bonus = float(getattr(self.config, "pre_candidate_consolidation_max_bonus", 0.0))
        group_by_peak_timestamps = bool(
            getattr(self.config, "pre_candidate_consolidation_group_by_peak_timestamps", False)
        )
        if weight <= 0.0 or window_seconds <= 0.0 or max_bonus <= 0.0 or len(proposals) < min_cluster_size:
            for proposal in proposals:
                setattr(proposal, "_pre_candidate_consolidation_score", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_bonus", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_group_count", 0)
                setattr(proposal, "_pre_candidate_consolidation_compactness", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_persistence", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_center_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_consolidation_center_shift", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_anchor_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_consolidation_peak_centroid_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_consolidation_proposal_centroid_timestamp", float(proposal.timestamp))
            return

        def _proposal_group_timestamp(proposal: AudioCandidate) -> float:
            if group_by_peak_timestamps:
                details = self._proposal_details(proposal)
                return float(details.get("peak_timestamp_seconds", proposal.timestamp) or proposal.timestamp)
            return float(proposal.timestamp)

        sorted_proposals = sorted(proposals, key=_proposal_group_timestamp)

        def _group_neighbors(center: AudioCandidate) -> list[dict[str, Any]]:
            center_group_timestamp = _proposal_group_timestamp(center)
            neighbors = [
                proposal
                for proposal in sorted_proposals
                if abs(_proposal_group_timestamp(proposal) - center_group_timestamp) <= window_seconds
                and float(proposal.audio_score) >= min_score
            ]
            if not neighbors:
                return []
            groups: list[dict[str, Any]] = []
            current: list[AudioCandidate] = []
            for proposal in neighbors:
                if not current or abs(_proposal_group_timestamp(proposal) - _proposal_group_timestamp(current[-1])) <= merge_gap_seconds:
                    current.append(proposal)
                    continue
                groups.append(self._summarize_consolidation_group(current))
                current = [proposal]
            if current:
                groups.append(self._summarize_consolidation_group(current))
            return groups

        for proposal in sorted_proposals:
            groups = _group_neighbors(proposal)
            if len(groups) < min_cluster_size:
                setattr(proposal, "_pre_candidate_consolidation_score", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_bonus", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_group_count", len(groups))
                setattr(proposal, "_pre_candidate_consolidation_compactness", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_persistence", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_center_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_consolidation_center_shift", 0.0)
                setattr(proposal, "_pre_candidate_consolidation_anchor_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_consolidation_peak_centroid_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_consolidation_proposal_centroid_timestamp", float(proposal.timestamp))
                continue

            groups = sorted(groups, key=lambda row: float(row["score"]), reverse=True)[:top_peaks]
            anchor = max(float(groups[0]["score"]), 1e-6)
            mass_ratio = max(sum(float(group["score"]) for group in groups) / anchor - 1.0, 0.0)
            proposal_group_timestamp = _proposal_group_timestamp(proposal)
            offsets = [abs(float(group["group_timestamp"]) - proposal_group_timestamp) for group in groups[1:]]
            mean_offset = float(np.mean(offsets)) if offsets else 0.0
            compactness = max(0.0, 1.0 - (mean_offset / max(window_seconds, 1e-6)))
            persistence = float(np.mean([float(group["persistence"]) for group in groups]))
            anchor_timestamp = float(groups[0]["proposal_timestamp"])
            peak_weights = np.asarray([float(group["score"]) for group in groups], dtype=np.float64)
            if float(np.sum(peak_weights)) <= 0.0:
                peak_weights = np.ones(len(groups), dtype=np.float64)
            proposal_centroid_timestamp = float(
                np.average(
                    np.asarray([float(group["proposal_timestamp"]) for group in groups], dtype=np.float64),
                    weights=peak_weights,
                )
            )
            peak_centroid_timestamp = float(
                np.average(
                    np.asarray([float(group["peak_timestamp"]) for group in groups], dtype=np.float64),
                    weights=peak_weights,
                )
            )
            consolidation_score = max(mass_ratio * compactness * (1.0 + 0.25 * persistence), 0.0)
            consolidation_bonus = min(max_bonus, weight * consolidation_score)
            setattr(proposal, "_pre_candidate_consolidation_score", float(consolidation_score))
            setattr(proposal, "_pre_candidate_consolidation_bonus", float(consolidation_bonus))
            setattr(proposal, "_pre_candidate_consolidation_group_count", len(groups))
            setattr(proposal, "_pre_candidate_consolidation_compactness", float(compactness))
            setattr(proposal, "_pre_candidate_consolidation_persistence", float(persistence))
            setattr(proposal, "_pre_candidate_consolidation_center_timestamp", float(proposal_centroid_timestamp))
            setattr(proposal, "_pre_candidate_consolidation_center_shift", float(proposal_centroid_timestamp - float(proposal.timestamp)))
            setattr(proposal, "_pre_candidate_consolidation_anchor_timestamp", float(anchor_timestamp))
            setattr(proposal, "_pre_candidate_consolidation_peak_centroid_timestamp", float(peak_centroid_timestamp))
            setattr(proposal, "_pre_candidate_consolidation_proposal_centroid_timestamp", float(proposal_centroid_timestamp))
            setattr(proposal, "_pre_candidate_consolidation_grouping_basis", "peak" if group_by_peak_timestamps else "proposal")

        self._annotate_overlap_agreement(sorted_proposals)

    def _summarize_consolidation_group(self, proposals: Sequence[AudioCandidate]) -> Dict[str, Any]:
        strongest = max(proposals, key=lambda proposal: float(proposal.audio_score))
        strongest_details = self._proposal_details(strongest)
        persistence = float(
            np.mean(
                [
                    max(min(float(proposal.post_flux_ratio), float(proposal.post_rms_ratio)) - 1.0, 0.0)
                    for proposal in proposals
                ]
            )
        )
        return {
            "timestamp": float(strongest.timestamp),
            "group_timestamp": float(strongest_details.get("peak_timestamp_seconds", strongest.timestamp) or strongest.timestamp),
            "proposal_timestamp": float(strongest.timestamp),
            "peak_timestamp": float(strongest_details.get("peak_timestamp_seconds", strongest.timestamp) or strongest.timestamp),
            "score": float(max(float(proposal.audio_score) for proposal in proposals)),
            "persistence": persistence,
        }

    def _annotate_overlap_agreement(self, proposals: Sequence[AudioCandidate]) -> None:
        weight = float(getattr(self.config, "pre_candidate_overlap_agreement_weight", 0.0))
        window_seconds = float(getattr(self.config, "pre_candidate_overlap_window_seconds", 0.12))
        min_pcen_persistence = float(getattr(self.config, "pre_candidate_overlap_min_pcen_persistence", 0.0))
        min_total_score = float(getattr(self.config, "pre_candidate_overlap_min_total_score", 0.0))
        pcen_center_weight = float(getattr(self.config, "pre_candidate_overlap_pcen_center_weight", 0.0))

        def _peak_timestamp(proposal: AudioCandidate) -> float:
            details = self._proposal_details(proposal)
            return float(details.get("peak_timestamp_seconds", proposal.timestamp) or proposal.timestamp)

        if weight <= 0.0 or window_seconds <= 0.0 or len(proposals) < 2:
            for proposal in proposals:
                setattr(proposal, "_pre_candidate_overlap_agreement_bonus", 0.0)
                setattr(proposal, "_pre_candidate_overlap_agreement_score", 0.0)
                setattr(proposal, "_pre_candidate_overlap_center_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_overlap_center_shift", 0.0)
                setattr(proposal, "_pre_candidate_overlap_member_count", 0)
                setattr(proposal, "_pre_candidate_overlap_pcen_score_mass", 0.0)
                setattr(proposal, "_pre_candidate_overlap_total_score_mass", 0.0)
            return

        for proposal in proposals:
            frontend = str(self._proposal_details(proposal).get("proposal_frontend", "unknown"))
            center_peak_timestamp = _peak_timestamp(proposal)
            neighbors = [
                neighbor
                for neighbor in proposals
                if neighbor is not proposal
                and abs(_peak_timestamp(neighbor) - center_peak_timestamp) <= window_seconds
            ]
            mixed = [proposal, *neighbors]
            frontends = {str(self._proposal_details(candidate).get("proposal_frontend", "unknown")) for candidate in mixed}
            if "heuristic" not in frontends or "pcen_multiband" not in frontends:
                setattr(proposal, "_pre_candidate_overlap_agreement_bonus", 0.0)
                setattr(proposal, "_pre_candidate_overlap_agreement_score", 0.0)
                setattr(proposal, "_pre_candidate_overlap_center_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_overlap_center_shift", 0.0)
                setattr(proposal, "_pre_candidate_overlap_member_count", len(mixed))
                setattr(proposal, "_pre_candidate_overlap_pcen_score_mass", 0.0)
                setattr(proposal, "_pre_candidate_overlap_total_score_mass", float(sum(float(candidate.audio_score) for candidate in mixed)))
                continue

            pcen_members = [
                candidate
                for candidate in mixed
                if str(self._proposal_details(candidate).get("proposal_frontend", "unknown")) == "pcen_multiband"
                and max(min(float(candidate.post_flux_ratio), float(candidate.post_rms_ratio)) - 1.0, 0.0) >= min_pcen_persistence
            ]
            if not pcen_members:
                setattr(proposal, "_pre_candidate_overlap_agreement_bonus", 0.0)
                setattr(proposal, "_pre_candidate_overlap_agreement_score", 0.0)
                setattr(proposal, "_pre_candidate_overlap_center_timestamp", float(proposal.timestamp))
                setattr(proposal, "_pre_candidate_overlap_center_shift", 0.0)
                setattr(proposal, "_pre_candidate_overlap_member_count", len(mixed))
                setattr(proposal, "_pre_candidate_overlap_pcen_score_mass", 0.0)
                setattr(proposal, "_pre_candidate_overlap_total_score_mass", float(sum(float(candidate.audio_score) for candidate in mixed)))
                continue

            total_score_mass = float(sum(float(candidate.audio_score) for candidate in mixed))
            pcen_score_mass = float(sum(float(candidate.audio_score) for candidate in pcen_members))
            if total_score_mass < min_total_score:
                overlap_score = 0.0
                overlap_bonus = 0.0
            else:
                offsets = [abs(_peak_timestamp(candidate) - center_peak_timestamp) for candidate in mixed if candidate is not proposal]
                mean_offset = float(np.mean(offsets)) if offsets else 0.0
                compactness = max(0.0, 1.0 - (mean_offset / max(window_seconds, 1e-6)))
                pcen_ratio = pcen_score_mass / max(total_score_mass, 1e-6)
                overlap_score = max(compactness * pcen_ratio, 0.0)
                overlap_bonus = max(weight * overlap_score, 0.0)

            weighted_members: list[tuple[float, float]] = []
            for candidate in mixed:
                candidate_frontend = str(self._proposal_details(candidate).get("proposal_frontend", "unknown"))
                candidate_weight = float(candidate.audio_score)
                if candidate_frontend == "pcen_multiband":
                    candidate_weight *= (1.0 + pcen_center_weight)
                weighted_members.append((float(candidate.timestamp), max(candidate_weight, 1e-6)))
            overlap_center_timestamp = float(
                np.average(
                    np.asarray([timestamp for timestamp, _ in weighted_members], dtype=np.float64),
                    weights=np.asarray([weight for _, weight in weighted_members], dtype=np.float64),
                )
            )
            setattr(proposal, "_pre_candidate_overlap_agreement_bonus", float(overlap_bonus))
            setattr(proposal, "_pre_candidate_overlap_agreement_score", float(overlap_score))
            setattr(proposal, "_pre_candidate_overlap_center_timestamp", float(overlap_center_timestamp))
            setattr(proposal, "_pre_candidate_overlap_center_shift", float(overlap_center_timestamp - float(proposal.timestamp)))
            setattr(proposal, "_pre_candidate_overlap_member_count", len(mixed))
            setattr(proposal, "_pre_candidate_overlap_pcen_score_mass", float(pcen_score_mass))
            setattr(proposal, "_pre_candidate_overlap_total_score_mass", float(total_score_mass))

    def _prepare_dive_trend_rank_bonus(self, proposals: Sequence[AudioCandidate]) -> None:
        enabled = bool(getattr(self.config, "frontend_dive_trend_enabled", False))
        weight = float(getattr(self.config, "frontend_dive_trend_weight", 0.0))
        max_bonus = float(getattr(self.config, "frontend_dive_trend_max_bonus", 0.0))
        feature_names = ("flatness_slope", "centroid_slope", "hf_lf_slope", "time_to_peak", "cluster_density")
        if not enabled or weight <= 0.0 or max_bonus <= 0.0 or not proposals:
            for proposal in proposals:
                setattr(proposal, "_pre_candidate_dive_trend_raw_score", 0.0)
                setattr(proposal, "_pre_candidate_dive_trend_probability", 0.0)
                setattr(proposal, "_pre_candidate_dive_trend_bonus", 0.0)
                setattr(proposal, "_pre_candidate_dive_trend_rank_bonus", 0.0)
            return
        feature_values: dict[str, list[float]] = {name: [] for name in feature_names}
        for proposal in proposals:
            feature_values["flatness_slope"].append(float(getattr(proposal, "_pre_candidate_dive_trend_flatness_slope", 0.0)))
            feature_values["centroid_slope"].append(float(getattr(proposal, "_pre_candidate_dive_trend_centroid_slope", 0.0)))
            feature_values["hf_lf_slope"].append(float(getattr(proposal, "_pre_candidate_dive_trend_hf_lf_slope", 0.0)))
            feature_values["time_to_peak"].append(float(getattr(proposal, "_pre_candidate_dive_trend_time_to_peak", 0.0)))
            feature_values["cluster_density"].append(float(getattr(proposal, "_pre_candidate_dive_trend_cluster_density", 0.0)))
        feature_stats = {
            name: (
                float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0,
                float(np.std(np.asarray(values, dtype=np.float64))) if values else 1.0,
            )
            for name, values in feature_values.items()
        }
        for proposal in proposals:
            flatness_slope = float(getattr(proposal, "_pre_candidate_dive_trend_flatness_slope", 0.0))
            centroid_slope = float(getattr(proposal, "_pre_candidate_dive_trend_centroid_slope", 0.0))
            hf_lf_slope = float(getattr(proposal, "_pre_candidate_dive_trend_hf_lf_slope", 0.0))
            time_to_peak = float(getattr(proposal, "_pre_candidate_dive_trend_time_to_peak", 0.0))
            cluster_density = float(getattr(proposal, "_pre_candidate_dive_trend_cluster_density", 0.0))
            raw_score = (
                0.30 * ((flatness_slope - feature_stats["flatness_slope"][0]) / max(feature_stats["flatness_slope"][1], 1e-6))
                + 0.30 * ((centroid_slope - feature_stats["centroid_slope"][0]) / max(feature_stats["centroid_slope"][1], 1e-6))
                + 0.20 * ((hf_lf_slope - feature_stats["hf_lf_slope"][0]) / max(feature_stats["hf_lf_slope"][1], 1e-6))
                - 0.20 * ((time_to_peak - feature_stats["time_to_peak"][0]) / max(feature_stats["time_to_peak"][1], 1e-6))
                - 0.15 * ((cluster_density - feature_stats["cluster_density"][0]) / max(feature_stats["cluster_density"][1], 1e-6))
            )
            probability = float(1.0 / (1.0 + np.exp(-np.clip(raw_score, -40.0, 40.0))))
            bonus = min(max_bonus, max(0.0, weight * max((probability - 0.5) * 2.0, 0.0)))
            setattr(proposal, "_pre_candidate_dive_trend_raw_score", float(raw_score))
            setattr(proposal, "_pre_candidate_dive_trend_probability", float(probability))
            setattr(proposal, "_pre_candidate_dive_trend_bonus", float(bonus))
            setattr(proposal, "_pre_candidate_dive_trend_rank_bonus", 0.0)

        cluster_window_seconds = 0.75
        cluster_bonuses: dict[int, float] = {}
        for i, proposal in enumerate(proposals):
            center_timestamp = float(proposal.timestamp)
            cluster = [
                peer
                for peer in proposals
                if abs(float(peer.timestamp) - center_timestamp) <= cluster_window_seconds
            ]
            if len(cluster) < 2:
                cluster_bonuses[id(proposal)] = 0.0
                continue
            cluster_raw_scores = np.asarray(
                [float(getattr(peer, "_pre_candidate_dive_trend_raw_score", 0.0)) for peer in cluster],
                dtype=np.float64,
            )
            cluster_mean = float(np.mean(cluster_raw_scores)) if cluster_raw_scores.size else 0.0
            cluster_std = float(np.std(cluster_raw_scores)) if cluster_raw_scores.size else 1.0
            cluster_std = max(cluster_std, 1e-6)
            proposal_raw = float(getattr(proposal, "_pre_candidate_dive_trend_raw_score", 0.0))
            normalized = (proposal_raw - cluster_mean) / cluster_std
            rank_bonus = float(np.clip(weight * normalized, -max_bonus, max_bonus))
            cluster_bonuses[id(proposal)] = rank_bonus
        for proposal in proposals:
            setattr(
                proposal,
                "_pre_candidate_dive_trend_rank_bonus",
                float(cluster_bonuses.get(id(proposal), 0.0)),
            )

    def _apply_consolidation_centering(self, proposal: AudioCandidate) -> None:
        centering_weight = float(getattr(self.config, "pre_candidate_consolidation_centering_weight", 0.0))
        center_timestamp = float(
            getattr(proposal, "_pre_candidate_consolidation_center_timestamp", float(proposal.timestamp))
        )
        overlap_center_timestamp = float(
            getattr(proposal, "_pre_candidate_overlap_center_timestamp", center_timestamp)
        )
        overlap_bonus = float(getattr(proposal, "_pre_candidate_overlap_agreement_bonus", 0.0))
        original_timestamp = float(getattr(proposal, "_original_timestamp", float(proposal.timestamp)))
        setattr(proposal, "_original_timestamp", original_timestamp)
        if centering_weight <= 0.0:
            setattr(proposal, "_pre_candidate_consolidation_applied_timestamp", float(proposal.timestamp))
            setattr(proposal, "_pre_candidate_consolidation_applied_shift", float(float(proposal.timestamp) - original_timestamp))
            return
        group_count = int(getattr(proposal, "_pre_candidate_consolidation_group_count", 0))
        if group_count < 2:
            setattr(proposal, "_pre_candidate_consolidation_applied_timestamp", float(proposal.timestamp))
            setattr(proposal, "_pre_candidate_consolidation_applied_shift", float(float(proposal.timestamp) - original_timestamp))
            return
        if overlap_bonus > 0.0:
            center_timestamp = overlap_center_timestamp
        centered_timestamp = (
            (1.0 - centering_weight) * float(proposal.timestamp)
            + centering_weight * center_timestamp
        )
        proposal.timestamp = float(centered_timestamp)
        setattr(proposal, "_pre_candidate_consolidation_applied_timestamp", float(centered_timestamp))
        setattr(proposal, "_pre_candidate_consolidation_applied_shift", float(centered_timestamp - original_timestamp))

    def _proposal_ranking_components(self, proposal: AudioCandidate) -> Dict[str, Any]:
        tail_component = max(float(proposal.post_flux_ratio) - 1.0, 0.0)
        asymmetry_component = abs(float(proposal.post_flux_ratio) - float(proposal.post_rms_ratio))
        broadband_component = max(float(proposal.spectral_flatness), 0.0)
        decay_component = max(float(proposal.local_prominence) - 4.0, 0.0)
        dive_likeness = float(
            tail_component
            + asymmetry_component
            + broadband_component
            + 0.1 * decay_component
        )
        rank_tail_weight = float(getattr(self.config, "pre_candidate_rank_tail_weight", 0.0))
        rank_asymmetry_weight = float(getattr(self.config, "pre_candidate_rank_asymmetry_weight", 0.0))
        rank_broadband_weight = float(getattr(self.config, "pre_candidate_rank_broadband_weight", 0.0))
        rank_decay_weight = float(getattr(self.config, "pre_candidate_rank_decay_weight", 0.0))
        tail_persistence_weight = float(getattr(self.config, "pre_candidate_tail_persistence_weight", 0.0))
        cluster_support_weight = float(getattr(self.config, "pre_candidate_cluster_support_weight", 0.0))
        tail_persistence_score = float(getattr(proposal, "_pre_candidate_tail_persistence_score", 0.0))
        cluster_support_score = float(getattr(proposal, "_pre_candidate_cluster_support_score", 0.0))
        region_descriptor_bonus = float(getattr(proposal, "_pre_candidate_region_descriptor_bonus", 0.0))
        consolidation_score = float(getattr(proposal, "_pre_candidate_consolidation_score", 0.0))
        consolidation_bonus = float(getattr(proposal, "_pre_candidate_consolidation_bonus", 0.0))
        consolidation_group_count = int(getattr(proposal, "_pre_candidate_consolidation_group_count", 0))
        consolidation_compactness = float(getattr(proposal, "_pre_candidate_consolidation_compactness", 0.0))
        consolidation_persistence = float(getattr(proposal, "_pre_candidate_consolidation_persistence", 0.0))
        consolidation_center_timestamp = float(
            getattr(proposal, "_pre_candidate_consolidation_center_timestamp", float(proposal.timestamp))
        )
        consolidation_center_shift = float(getattr(proposal, "_pre_candidate_consolidation_center_shift", 0.0))
        consolidation_anchor_timestamp = float(
            getattr(proposal, "_pre_candidate_consolidation_anchor_timestamp", float(proposal.timestamp))
        )
        consolidation_peak_centroid_timestamp = float(
            getattr(proposal, "_pre_candidate_consolidation_peak_centroid_timestamp", float(proposal.timestamp))
        )
        consolidation_proposal_centroid_timestamp = float(
            getattr(proposal, "_pre_candidate_consolidation_proposal_centroid_timestamp", float(proposal.timestamp))
        )
        consolidation_applied_timestamp = float(
            getattr(proposal, "_pre_candidate_consolidation_applied_timestamp", float(proposal.timestamp))
        )
        consolidation_applied_shift = float(
            getattr(proposal, "_pre_candidate_consolidation_applied_shift", 0.0)
        )
        consolidation_grouping_basis = str(
            getattr(proposal, "_pre_candidate_consolidation_grouping_basis", "proposal")
        )
        overlap_agreement_bonus = float(getattr(proposal, "_pre_candidate_overlap_agreement_bonus", 0.0))
        overlap_agreement_score = float(getattr(proposal, "_pre_candidate_overlap_agreement_score", 0.0))
        overlap_center_timestamp = float(getattr(proposal, "_pre_candidate_overlap_center_timestamp", float(proposal.timestamp)))
        overlap_center_shift = float(getattr(proposal, "_pre_candidate_overlap_center_shift", 0.0))
        overlap_member_count = int(getattr(proposal, "_pre_candidate_overlap_member_count", 0))
        overlap_pcen_score_mass = float(getattr(proposal, "_pre_candidate_overlap_pcen_score_mass", 0.0))
        overlap_total_score_mass = float(getattr(proposal, "_pre_candidate_overlap_total_score_mass", 0.0))
        dive_trend_flatness_slope = float(getattr(proposal, "_pre_candidate_dive_trend_flatness_slope", 0.0))
        dive_trend_centroid_slope = float(getattr(proposal, "_pre_candidate_dive_trend_centroid_slope", 0.0))
        dive_trend_hf_lf_slope = float(getattr(proposal, "_pre_candidate_dive_trend_hf_lf_slope", 0.0))
        dive_trend_time_to_peak = float(getattr(proposal, "_pre_candidate_dive_trend_time_to_peak", 0.0))
        dive_trend_cluster_density = float(getattr(proposal, "_pre_candidate_dive_trend_cluster_density", 0.0))
        dive_trend_raw_score = float(getattr(proposal, "_pre_candidate_dive_trend_raw_score", 0.0))
        dive_trend_probability = float(getattr(proposal, "_pre_candidate_dive_trend_probability", 0.0))
        dive_trend_bonus = float(getattr(proposal, "_pre_candidate_dive_trend_bonus", 0.0))
        dive_trend_rank_bonus = float(getattr(proposal, "_pre_candidate_dive_trend_rank_bonus", 0.0))
        proposal_evidence_boost = (
            tail_persistence_weight * tail_persistence_score
            + cluster_support_weight * cluster_support_score
            + consolidation_bonus
            + overlap_agreement_bonus
        )
        promotion_bonus = float(getattr(self.config, "pre_candidate_rank_promotion_bonus", 0.0))
        promotion_min_score = float(getattr(self.config, "pre_candidate_rank_promotion_min_score", 0.0))
        promotion_min_dive_likeness = float(getattr(self.config, "pre_candidate_rank_promotion_min_dive_likeness", 0.0))
        promotion_min_prominence = float(getattr(self.config, "pre_candidate_rank_promotion_min_prominence", 0.0))
        promotion_min_nearby_peaks = int(getattr(self.config, "pre_candidate_rank_promotion_min_nearby_peaks", 0))
        rank_bonus = (
            rank_tail_weight * tail_component
            + rank_asymmetry_weight * asymmetry_component
            + rank_broadband_weight * broadband_component
            + rank_decay_weight * decay_component
            + proposal_evidence_boost
            + dive_trend_rank_bonus
        )
        promotion_eligible = (
            promotion_bonus > 0.0
            and float(proposal.audio_score) >= promotion_min_score
            and float(proposal.local_prominence) >= promotion_min_prominence
            and int(proposal.nearby_peaks_8s) >= promotion_min_nearby_peaks
            and dive_likeness >= promotion_min_dive_likeness
        )
        if promotion_eligible:
            rank_bonus += promotion_bonus
        return {
            "tail_component": float(tail_component),
            "asymmetry_component": float(asymmetry_component),
            "broadband_component": float(broadband_component),
            "decay_component": float(decay_component),
            "tail_persistence_score": float(tail_persistence_score),
            "cluster_support_score": float(cluster_support_score),
            "frontend_region_descriptor_bonus": float(region_descriptor_bonus),
            "consolidation_score": float(consolidation_score),
            "consolidation_bonus": float(consolidation_bonus),
            "consolidation_group_count": int(consolidation_group_count),
            "consolidation_compactness": float(consolidation_compactness),
            "consolidation_persistence": float(consolidation_persistence),
            "consolidation_center_timestamp": float(consolidation_center_timestamp),
            "consolidation_center_shift": float(consolidation_center_shift),
            "consolidation_anchor_timestamp": float(consolidation_anchor_timestamp),
            "consolidation_peak_centroid_timestamp": float(consolidation_peak_centroid_timestamp),
            "consolidation_proposal_centroid_timestamp": float(consolidation_proposal_centroid_timestamp),
            "consolidation_applied_timestamp": float(consolidation_applied_timestamp),
            "consolidation_applied_shift": float(consolidation_applied_shift),
            "consolidation_grouping_basis": consolidation_grouping_basis,
            "overlap_agreement_score": float(overlap_agreement_score),
            "overlap_agreement_bonus": float(overlap_agreement_bonus),
            "overlap_center_timestamp": float(overlap_center_timestamp),
            "overlap_center_shift": float(overlap_center_shift),
            "overlap_member_count": int(overlap_member_count),
            "overlap_pcen_score_mass": float(overlap_pcen_score_mass),
            "overlap_total_score_mass": float(overlap_total_score_mass),
            "frontend_dive_trend_flatness_slope": float(dive_trend_flatness_slope),
            "frontend_dive_trend_centroid_slope": float(dive_trend_centroid_slope),
            "frontend_dive_trend_hf_lf_slope": float(dive_trend_hf_lf_slope),
            "frontend_dive_trend_time_to_peak": float(dive_trend_time_to_peak),
            "frontend_dive_trend_cluster_density": float(dive_trend_cluster_density),
            "frontend_dive_trend_raw_score": float(dive_trend_raw_score),
            "frontend_dive_trend_probability": float(dive_trend_probability),
            "frontend_dive_trend_bonus": float(dive_trend_bonus),
            "frontend_dive_trend_rank_bonus": float(dive_trend_rank_bonus),
            "proposal_evidence_boost": float(proposal_evidence_boost),
            "dive_likeness": float(dive_likeness),
            "rank_bonus": float(rank_bonus),
            "rank_score": float(proposal.audio_score + rank_bonus),
            "promotion_eligible": bool(promotion_eligible),
        }

    def _proposal_details(self, proposal: AudioCandidate) -> Dict[str, Any]:
        details = getattr(proposal, "details", None)
        base = dict(details) if isinstance(details, dict) else {
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
            "tail_persistence_score": float(getattr(proposal, "_pre_candidate_tail_persistence_score", 0.0)),
            "cluster_support_score": float(getattr(proposal, "_pre_candidate_cluster_support_score", 0.0)),
            "consolidation_score": float(getattr(proposal, "_pre_candidate_consolidation_score", 0.0)),
            "consolidation_bonus": float(getattr(proposal, "_pre_candidate_consolidation_bonus", 0.0)),
            "consolidation_group_count": int(getattr(proposal, "_pre_candidate_consolidation_group_count", 0)),
            "consolidation_compactness": float(getattr(proposal, "_pre_candidate_consolidation_compactness", 0.0)),
            "consolidation_persistence": float(getattr(proposal, "_pre_candidate_consolidation_persistence", 0.0)),
            "original_proposal_timestamp_seconds": float(getattr(proposal, "_original_timestamp", proposal.timestamp)),
            "consolidation_center_applied": bool(
                float(getattr(proposal, "_pre_candidate_consolidation_applied_shift", 0.0)) != 0.0
            ),
            "overlap_agreement_bonus": float(getattr(proposal, "_pre_candidate_overlap_agreement_bonus", 0.0)),
            "overlap_agreement_score": float(getattr(proposal, "_pre_candidate_overlap_agreement_score", 0.0)),
            "overlap_center_timestamp": float(getattr(proposal, "_pre_candidate_overlap_center_timestamp", proposal.timestamp)),
            "overlap_center_shift": float(getattr(proposal, "_pre_candidate_overlap_center_shift", 0.0)),
            "overlap_member_count": int(getattr(proposal, "_pre_candidate_overlap_member_count", 0)),
            "overlap_pcen_score_mass": float(getattr(proposal, "_pre_candidate_overlap_pcen_score_mass", 0.0)),
            "overlap_total_score_mass": float(getattr(proposal, "_pre_candidate_overlap_total_score_mass", 0.0)),
            "proposal_evidence_boost": float(getattr(proposal, "_pre_candidate_evidence_boost", 0.0)),
        }
        base.update(self._proposal_ranking_components(proposal))
        base["local_rescue_survivor"] = bool(getattr(proposal, "_pre_candidate_local_rescue", False))
        base["local_rescue_score"] = float(getattr(proposal, "_pre_candidate_local_rescue_score", 0.0))
        base["protected_survivor"] = bool(getattr(proposal, "_pre_candidate_protected_survivor", False))
        return base

    def _attach_details(self, proposal: AudioCandidate, details: Dict[str, Any]) -> AudioCandidate:
        setattr(proposal, "details", details)
        return proposal

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

    def _load_audio_clip_model(self) -> Optional[AudioClipModel]:
        model_path = str(getattr(self.config, "audio_clip_model_path", "") or "").strip()
        if not model_path:
            return None
        candidate = Path(model_path)
        if not candidate.exists():
            return None
        try:
            return AudioClipModel.load(candidate)
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


def front_end_is_advanced(frontend_name: str) -> bool:
    return frontend_name != "heuristic"
