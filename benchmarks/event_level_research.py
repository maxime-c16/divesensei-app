from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPS = 1e-6
HOP = 256
WIN = 512
WINDOW_PRE_SECONDS = 0.2
WINDOW_POST_SECONDS = 1.0
CLUSTER_GAP_SECONDS = 0.5
EVENT_SEARCH_RADIUS_SECONDS = 1.5


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def robust_auc(a: np.ndarray, b: np.ndarray) -> float:
    total = len(a) * len(b)
    if total == 0:
        return 0.5
    gt = 0
    eq = 0
    for value in a:
        diff = value - b
        gt += int(np.sum(diff > 0))
        eq += int(np.sum(diff == 0))
    auc = (gt + 0.5 * eq) / total
    return float(max(auc, 1.0 - auc))


def linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2 or len(values) < 2:
        return 0.0
    x = times.astype(np.float64)
    y = values.astype(np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= EPS:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= EPS:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def to_serializable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_serializable(val) for val in value]
    return value


@dataclass
class ReviewEvent:
    label: str
    subtype: str | None
    timestamp: float
    source_id: str


class EventLevelResearch:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.report = load_json(session_dir / "session_pipeline_report.json")
        self.raw_rows = load_jsonl(session_dir / "proposal_raw_peaks.jsonl")
        self.frontend_rows = load_jsonl(session_dir / "proposal_frontend_candidates.jsonl")
        self.diagnostic_rows = load_jsonl(session_dir / "proposal_diagnostics.jsonl")
        self.review = load_json(session_dir / "evaluation_review.json")
        self.false_negative_rows = load_jsonl(session_dir / "exports" / "evaluation-review" / "false_negatives.jsonl")
        self.summary = load_json(session_dir / "exports" / "evaluation-review" / "evaluation_export_summary.json")
        self.detections = self._load_detections(session_dir / "detections.csv")
        self.audio_path = Path(self.report["source_audio_path"])
        self._load_audio()
        self._prepare_rows()
        self._build_anchor_features()

    def _load_detections(self, path: Path) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        with path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                detection_id = f"det-{int(row['index']):04d}"
                rows[detection_id] = {
                    "timestamp": float(row["timestamp"]),
                    "audio_score": float(row["audio_score"]),
                    "combined_score": float(row["combined_score"]),
                }
        return rows

    def _load_audio(self) -> None:
        with wave.open(str(self.audio_path), "rb") as wav_handle:
            self.sr = wav_handle.getframerate()
            frame_count = wav_handle.getnframes()
            channels = wav_handle.getnchannels()
            sample_width = wav_handle.getsampwidth()
            data = wav_handle.readframes(frame_count)
        if sample_width != 2:
            raise RuntimeError(f"Unsupported sample width {sample_width}")
        signal = np.frombuffer(data, dtype="<i2").astype(np.float32)
        if channels > 1:
            signal = signal.reshape(-1, channels).mean(axis=1)
        signal /= 32768.0
        self.signal = signal
        self.num_frames = 1 + (len(signal) - WIN) // HOP
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(self.num_frames, WIN),
            strides=(signal.strides[0] * HOP, signal.strides[0]),
        )
        window = np.hanning(WIN).astype(np.float32)
        windowed = frames * window
        spectrum = np.abs(np.fft.rfft(windowed, axis=1)).astype(np.float32) + EPS
        self.rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
        self.flux = np.zeros(self.num_frames, dtype=np.float32)
        self.flux[1:] = np.maximum(spectrum[1:] - spectrum[:-1], 0.0).sum(axis=1)
        self.flatness = (np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)).astype(np.float32)
        freqs = np.fft.rfftfreq(WIN, d=1.0 / self.sr)
        self.centroid = (np.sum(spectrum * freqs[None, :], axis=1) / np.sum(spectrum, axis=1)).astype(np.float32)
        low_mask = freqs < 1000.0
        high_mask = freqs >= 3000.0
        band_low = np.sum(spectrum[:, low_mask], axis=1).astype(np.float32)
        band_high = np.sum(spectrum[:, high_mask], axis=1).astype(np.float32)
        self.hf_lf = (band_high / np.maximum(band_low, EPS)).astype(np.float32)
        flux_norm = self.flux / max(float(np.percentile(self.flux, 95)), EPS)
        rms_norm = self.rms / max(float(np.percentile(self.rms, 95)), EPS)
        self.env = (0.65 * flux_norm + 0.35 * rms_norm).astype(np.float32)

    def _prepare_rows(self) -> None:
        for index, row in enumerate(self.raw_rows):
            row["_row_index"] = index
            row["_anchor_timestamp"] = float(
                row.get("proposal_timestamp_seconds", row.get("timestamp", row.get("peak_timestamp_seconds", 0.0)))
            )
            row["_peak_timestamp"] = float(row.get("peak_timestamp_seconds", row["_anchor_timestamp"]))
            row["_raw_score"] = float(row.get("raw_proposal_score", row.get("audio_score", 0.0)))
            row["_threshold"] = float(row.get("proposal_threshold", 0.0))
        self.raw_rows.sort(key=lambda item: item["_anchor_timestamp"])
        self.raw_anchor_timestamps = [row["_anchor_timestamp"] for row in self.raw_rows]

    def _window_frame_bounds(self, anchor_timestamp: float) -> tuple[int, int, float]:
        start_seconds = anchor_timestamp - WINDOW_PRE_SECONDS
        end_seconds = anchor_timestamp + WINDOW_POST_SECONDS
        start_frame = max(0, int(math.floor(start_seconds * self.sr / HOP)))
        end_frame = min(len(self.env), int(math.ceil(end_seconds * self.sr / HOP)))
        return start_frame, end_frame, start_seconds

    def _extract_event_features(self, row: dict) -> dict:
        anchor = row["_anchor_timestamp"]
        start_frame, end_frame, start_seconds = self._window_frame_bounds(anchor)
        env = self.env[start_frame:end_frame]
        centroid = self.centroid[start_frame:end_frame]
        flatness = self.flatness[start_frame:end_frame]
        hf_lf = self.hf_lf[start_frame:end_frame]
        if len(env) == 0:
            return {}
        times = (np.arange(start_frame, end_frame) * HOP / self.sr) - start_seconds
        pre_mask = times < WINDOW_PRE_SECONDS
        baseline = float(np.mean(env[pre_mask])) if np.any(pre_mask) else float(np.mean(env[: max(1, min(len(env), 3))]))
        baseline = max(baseline, EPS)
        norm_env = env / baseline

        early_end = 0.35
        mid_end = 0.75
        early_mask = times < early_end
        mid_mask = (times >= early_end) & (times < mid_end)
        late_mask = times >= mid_end

        early_energy = float(np.mean(norm_env[early_mask])) if np.any(early_mask) else 0.0
        mid_energy = float(np.mean(norm_env[mid_mask])) if np.any(mid_mask) else 0.0
        late_energy = float(np.mean(norm_env[late_mask])) if np.any(late_mask) else 0.0
        global_peak_index = int(np.argmax(norm_env))
        time_to_peak = float(times[global_peak_index] - WINDOW_PRE_SECONDS)
        duration_above = float(np.sum(norm_env >= 1.10) * HOP / self.sr)
        sustain_minus_decay = late_energy - early_energy

        left = bisect.bisect_left(self.raw_anchor_timestamps, anchor - WINDOW_PRE_SECONDS)
        right = bisect.bisect_right(self.raw_anchor_timestamps, anchor + WINDOW_POST_SECONDS)
        local_rows = self.raw_rows[left:right]
        local_rows.sort(key=lambda item: item["_anchor_timestamp"])
        peak_scores = [max(candidate["_raw_score"], 0.0) for candidate in local_rows]
        peak_times = [candidate["_anchor_timestamp"] for candidate in local_rows]
        multi_peak_count = int(sum(score >= max(1.5, 0.4 * max(peak_scores or [0.0])) for score in peak_scores))
        spacing = np.diff(np.asarray(peak_times, dtype=np.float32)) if len(peak_times) >= 2 else np.asarray([], dtype=np.float32)
        peak_spacing_variance = float(np.var(spacing)) if len(spacing) else 0.0
        cluster_density = float(len(local_rows) / (WINDOW_PRE_SECONDS + WINDOW_POST_SECONDS))
        sorted_scores = sorted(peak_scores, reverse=True)
        second_peak_ratio = float(sorted_scores[1] / max(sorted_scores[0], EPS)) if len(sorted_scores) >= 2 else 0.0
        first_minus_second = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else float(sorted_scores[0] if sorted_scores else 0.0)

        return {
            "anchor_timestamp": anchor,
            "proposal_frontend": row.get("proposal_frontend"),
            "raw_proposal_score": row["_raw_score"],
            "proposal_threshold": row["_threshold"],
            "threshold_passed": bool(row.get("threshold_passed", row["_raw_score"] >= row["_threshold"])),
            "peak_timestamp_seconds": row["_peak_timestamp"],
            "event_window_start_seconds": anchor - WINDOW_PRE_SECONDS,
            "event_window_end_seconds": anchor + WINDOW_POST_SECONDS,
            "energy_early": early_energy,
            "energy_mid": mid_energy,
            "energy_late": late_energy,
            "late_over_early": float(late_energy / max(early_energy, EPS)),
            "time_to_peak_global": time_to_peak,
            "duration_above_1p10": duration_above,
            "multi_peak_count": multi_peak_count,
            "peak_spacing_variance": peak_spacing_variance,
            "centroid_slope": linear_slope(times, centroid),
            "flatness_slope": linear_slope(times, flatness),
            "hf_lf_slope": linear_slope(times, hf_lf),
            "cluster_density": cluster_density,
            "energy_decay_vs_sustain": sustain_minus_decay,
            "second_peak_strength_ratio": second_peak_ratio,
            "first_minus_second_peak": first_minus_second,
            "local_prominence": float(row.get("local_prominence", 0.0)),
            "post_flux_ratio": float(row.get("post_flux_ratio", 0.0)),
            "post_rms_ratio": float(row.get("post_rms_ratio", 0.0)),
            "spectral_flatness": float(row.get("spectral_flatness", 0.0)),
            "hf_ratio": float(row.get("hf_ratio", 0.0)),
            "spectral_centroid_hz": float(row.get("spectral_centroid_hz", 0.0)),
            "frontend_region_descriptor_bonus": float(row.get("frontend_region_descriptor_bonus", 0.0)),
            "frontend_region_late_over_early": float(row.get("frontend_region_late_over_early", 0.0)),
            "frontend_region_duration_above_1p10": float(row.get("frontend_region_duration_above_1p10", 0.0)),
            "frontend_region_time_to_peak": float(row.get("frontend_region_time_to_peak", 0.0)),
            "rejection_stage": row.get("rejection_stage"),
            "pre_candidate_loss_stage": row.get("pre_candidate_loss_stage"),
        }

    def _build_anchor_features(self) -> None:
        self.anchor_features: dict[int, dict] = {}
        for row in self.raw_rows:
            self.anchor_features[row["_row_index"]] = self._extract_event_features(row)
        feature_names = [
            "energy_early",
            "energy_mid",
            "energy_late",
            "late_over_early",
            "time_to_peak_global",
            "duration_above_1p10",
            "multi_peak_count",
            "peak_spacing_variance",
            "centroid_slope",
            "flatness_slope",
            "hf_lf_slope",
            "cluster_density",
            "energy_decay_vs_sustain",
            "second_peak_strength_ratio",
            "first_minus_second_peak",
            "frontend_region_descriptor_bonus",
            "frontend_region_late_over_early",
            "frontend_region_duration_above_1p10",
            "frontend_region_time_to_peak",
        ]
        zmap: dict[str, np.ndarray] = {}
        for name in feature_names:
            values = np.asarray([self.anchor_features[row["_row_index"]].get(name, 0.0) for row in self.raw_rows], dtype=np.float32)
            zmap[name] = zscore(values)
        for idx, row in enumerate(self.raw_rows):
            features = self.anchor_features[row["_row_index"]]
            dive_score = (
                1.4 * float(zmap["late_over_early"][idx])
                + 1.1 * float(zmap["time_to_peak_global"][idx])
                + 1.0 * float(zmap["duration_above_1p10"][idx])
                + 0.9 * float(zmap["energy_decay_vs_sustain"][idx])
                + 0.8 * float(zmap["second_peak_strength_ratio"][idx])
                + 0.6 * float(zmap["frontend_region_descriptor_bonus"][idx])
                + 0.6 * float(zmap["frontend_region_late_over_early"][idx])
                - 0.8 * float(zmap["flatness_slope"][idx])
            )
            rebound_score = (
                1.4 * float(zmap["energy_early"][idx])
                - 1.2 * float(zmap["energy_late"][idx])
                - 1.0 * float(zmap["time_to_peak_global"][idx])
                - 0.9 * float(zmap["duration_above_1p10"][idx])
                + 0.7 * float(zmap["first_minus_second_peak"][idx])
                + 0.6 * float(zmap["cluster_density"][idx])
                + 0.6 * float(zmap["peak_spacing_variance"][idx])
            )
            terminal_score = (
                1.4 * float(zmap["energy_late"][idx])
                + 1.1 * float(zmap["time_to_peak_global"][idx])
                + 0.8 * float(zmap["duration_above_1p10"][idx])
                - 0.8 * float(zmap["multi_peak_count"][idx])
                - 0.6 * float(zmap["cluster_density"][idx])
                - 0.6 * float(zmap["energy_early"][idx])
            )
            selection_score = max(dive_score, terminal_score) - rebound_score
            features["dive_event_score"] = float(dive_score)
            features["rebound_event_score"] = float(rebound_score)
            features["terminal_peak_score"] = float(terminal_score)
            features["event_selection_score"] = float(selection_score)

    def reviewed_events(self) -> tuple[list[ReviewEvent], list[ReviewEvent], list[ReviewEvent]]:
        dives: list[ReviewEvent] = []
        rebounds: list[ReviewEvent] = []
        false_negatives: list[ReviewEvent] = []
        for decision in self.review.get("decisions", []):
            detection_id = decision.get("detectionId")
            detection = self.detections.get(detection_id)
            if detection is None:
                continue
            label = decision.get("label")
            event = ReviewEvent(
                label=label,
                subtype=decision.get("subtype"),
                timestamp=float(detection["timestamp"]),
                source_id=detection_id,
            )
            if label == "dive":
                dives.append(event)
            elif label == "non_dive" and decision.get("subtype") == "board_rebound":
                rebounds.append(event)
        for row in self.false_negative_rows:
            false_negatives.append(
                ReviewEvent(
                    label="false_negative",
                    subtype=row.get("subtype"),
                    timestamp=float(row["timestamp_seconds"]),
                    source_id=str(row.get("entry_type", "false_negative")),
                )
            )
        return dives, rebounds, false_negatives

    def _cluster_rows_for_timestamp(self, timestamp: float) -> list[dict]:
        left = bisect.bisect_left(self.raw_anchor_timestamps, timestamp - EVENT_SEARCH_RADIUS_SECONDS)
        right = bisect.bisect_right(self.raw_anchor_timestamps, timestamp + EVENT_SEARCH_RADIUS_SECONDS)
        local = self.raw_rows[left:right]
        if not local:
            return []
        local.sort(key=lambda item: item["_anchor_timestamp"])
        clusters: list[list[dict]] = []
        current: list[dict] = []
        for row in local:
            if not current or (row["_anchor_timestamp"] - current[-1]["_anchor_timestamp"]) <= CLUSTER_GAP_SECONDS:
                current.append(row)
            else:
                clusters.append(current)
                current = [row]
        if current:
            clusters.append(current)
        clusters.sort(key=lambda cluster: min(abs(row["_anchor_timestamp"] - timestamp) for row in cluster))
        return clusters[0]

    def _anchor_candidates(self, cluster: list[dict]) -> list[dict]:
        if not cluster:
            return []
        cluster = sorted(cluster, key=lambda item: item["_anchor_timestamp"])
        max_score = max(row["_raw_score"] for row in cluster)
        earliest_strong = None
        for row in cluster:
            if row["_raw_score"] >= max(1.5, 0.6 * max_score):
                earliest_strong = row
                break
        candidates = list(cluster)
        if earliest_strong is not None and all(existing["_row_index"] != earliest_strong["_row_index"] for existing in candidates):
            candidates.append(earliest_strong)
        return candidates

    def analyze_cases(self) -> tuple[list[dict], list[dict], list[dict]]:
        dives, rebounds, false_negatives = self.reviewed_events()
        return (
            [self._analyze_case(case) for case in dives],
            [self._analyze_case(case) for case in rebounds],
            [self._analyze_case(case) for case in false_negatives],
        )

    def _analyze_case(self, event: ReviewEvent) -> dict:
        cluster = self._cluster_rows_for_timestamp(event.timestamp)
        anchors = self._anchor_candidates(cluster)
        if not anchors:
            return {
                "label": event.label,
                "subtype": event.subtype,
                "timestamp": event.timestamp,
                "source_id": event.source_id,
                "cluster_peak_count": 0,
                "baseline_top_offset_seconds": None,
                "event_top_offset_seconds": None,
                "correct_anchor_raw_rank": None,
                "correct_anchor_event_rank": None,
            }
        anchors_sorted_raw = sorted(anchors, key=lambda row: row["_raw_score"], reverse=True)
        anchors_sorted_event = sorted(
            anchors,
            key=lambda row: self.anchor_features[row["_row_index"]]["event_selection_score"],
            reverse=True,
        )
        correct_row = min(anchors, key=lambda row: abs(row["_anchor_timestamp"] - event.timestamp))
        raw_rank = 1 + next(index for index, row in enumerate(anchors_sorted_raw) if row["_row_index"] == correct_row["_row_index"])
        event_rank = 1 + next(index for index, row in enumerate(anchors_sorted_event) if row["_row_index"] == correct_row["_row_index"])
        baseline_top = anchors_sorted_raw[0]
        event_top = anchors_sorted_event[0]
        return {
            "label": event.label,
            "subtype": event.subtype,
            "timestamp": event.timestamp,
            "source_id": event.source_id,
            "cluster_peak_count": len(cluster),
            "candidate_anchor_count": len(anchors),
            "baseline_top_anchor_timestamp": baseline_top["_anchor_timestamp"],
            "baseline_top_offset_seconds": baseline_top["_anchor_timestamp"] - event.timestamp,
            "baseline_top_raw_score": baseline_top["_raw_score"],
            "baseline_top_event_selection_score": self.anchor_features[baseline_top["_row_index"]]["event_selection_score"],
            "event_top_anchor_timestamp": event_top["_anchor_timestamp"],
            "event_top_offset_seconds": event_top["_anchor_timestamp"] - event.timestamp,
            "event_top_raw_score": event_top["_raw_score"],
            "event_top_event_selection_score": self.anchor_features[event_top["_row_index"]]["event_selection_score"],
            "correct_anchor_timestamp": correct_row["_anchor_timestamp"],
            "correct_anchor_offset_seconds": correct_row["_anchor_timestamp"] - event.timestamp,
            "correct_anchor_raw_rank": raw_rank,
            "correct_anchor_event_rank": event_rank,
            "correct_anchor_raw_score": correct_row["_raw_score"],
            "correct_anchor_event_selection_score": self.anchor_features[correct_row["_row_index"]]["event_selection_score"],
            "correct_anchor_features": self.anchor_features[correct_row["_row_index"]],
            "baseline_top_features": self.anchor_features[baseline_top["_row_index"]],
            "event_top_features": self.anchor_features[event_top["_row_index"]],
        }

    def summarize(self) -> dict:
        dive_cases, rebound_cases, fn_cases = self.analyze_cases()
        valid_dive_cases = [case for case in dive_cases if case["cluster_peak_count"] > 0]
        valid_rebound_cases = [case for case in rebound_cases if case["cluster_peak_count"] > 0]
        valid_fn_cases = [case for case in fn_cases if case["cluster_peak_count"] > 0]

        dive_raw = np.asarray([case["baseline_top_raw_score"] for case in valid_dive_cases], dtype=np.float32)
        rebound_raw = np.asarray([case["baseline_top_raw_score"] for case in valid_rebound_cases], dtype=np.float32)
        dive_event = np.asarray([case["event_top_event_selection_score"] for case in valid_dive_cases], dtype=np.float32)
        rebound_event = np.asarray([case["event_top_event_selection_score"] for case in valid_rebound_cases], dtype=np.float32)
        fn_raw = np.asarray([case["baseline_top_raw_score"] for case in valid_fn_cases], dtype=np.float32)
        fn_event = np.asarray([case["event_top_event_selection_score"] for case in valid_fn_cases], dtype=np.float32)

        fn_rank_deltas = [
            case["correct_anchor_raw_rank"] - case["correct_anchor_event_rank"]
            for case in valid_fn_cases
            if case["correct_anchor_raw_rank"] is not None and case["correct_anchor_event_rank"] is not None
        ]
        top1_raw = sum(case["correct_anchor_raw_rank"] == 1 for case in valid_fn_cases)
        top1_event = sum(case["correct_anchor_event_rank"] == 1 for case in valid_fn_cases)
        practical_1_raw = sum(abs(float(case["baseline_top_offset_seconds"])) <= 1.0 for case in valid_fn_cases)
        practical_1_event = sum(abs(float(case["event_top_offset_seconds"])) <= 1.0 for case in valid_fn_cases)
        practical_15_raw = sum(abs(float(case["baseline_top_offset_seconds"])) <= 1.5 for case in valid_fn_cases)
        practical_15_event = sum(abs(float(case["event_top_offset_seconds"])) <= 1.5 for case in valid_fn_cases)

        dive_vs_rebound_auc_raw = robust_auc(dive_raw, rebound_raw)
        dive_vs_rebound_auc_event = robust_auc(dive_event, rebound_event)
        fn_vs_rebound_auc_raw = robust_auc(fn_raw, rebound_raw)
        fn_vs_rebound_auc_event = robust_auc(fn_event, rebound_event)

        improved_cases = [
            case
            for case in valid_fn_cases
            if (
                case["correct_anchor_event_rank"] is not None
                and case["correct_anchor_raw_rank"] is not None
                and case["correct_anchor_event_rank"] < case["correct_anchor_raw_rank"]
            )
        ]
        worsened_cases = [
            case
            for case in valid_fn_cases
            if (
                case["correct_anchor_event_rank"] is not None
                and case["correct_anchor_raw_rank"] is not None
                and case["correct_anchor_event_rank"] > case["correct_anchor_raw_rank"]
            )
        ]

        viable = (
            (fn_vs_rebound_auc_event - fn_vs_rebound_auc_raw) >= 0.05
            or (float(np.mean(fn_rank_deltas)) if fn_rank_deltas else 0.0) >= 0.5
            or practical_1_event > practical_1_raw
        ) and (dive_vs_rebound_auc_event + 1e-6 >= dive_vs_rebound_auc_raw)

        result = {
            "session_dir": str(self.session_dir),
            "source_audio_path": str(self.audio_path),
            "review_summary": {
                "reviewed_dive_count": len(valid_dive_cases),
                "reviewed_board_rebound_count": len(valid_rebound_cases),
                "reviewed_false_negative_count": len(valid_fn_cases),
                "springboard_non_dive_counts": self.summary.get("hard_negative_subtype_counts", {}),
            },
            "auc": {
                "dive_vs_rebound_raw_peak_auc": dive_vs_rebound_auc_raw,
                "dive_vs_rebound_event_score_auc": dive_vs_rebound_auc_event,
                "fn_vs_rebound_raw_peak_auc": fn_vs_rebound_auc_raw,
                "fn_vs_rebound_event_score_auc": fn_vs_rebound_auc_event,
            },
            "false_negative_ranking": {
                "mean_rank_delta_raw_minus_event": float(np.mean(fn_rank_deltas)) if fn_rank_deltas else 0.0,
                "median_rank_delta_raw_minus_event": float(np.median(fn_rank_deltas)) if fn_rank_deltas else 0.0,
                "top1_raw_count": top1_raw,
                "top1_event_count": top1_event,
                "practical_1p0_raw_count": practical_1_raw,
                "practical_1p0_event_count": practical_1_event,
                "practical_1p5_raw_count": practical_15_raw,
                "practical_1p5_event_count": practical_15_event,
                "improved_case_count": len(improved_cases),
                "worsened_case_count": len(worsened_cases),
            },
            "cluster_level_examples": {
                "improved_false_negative_cases": improved_cases[:6],
                "worsened_false_negative_cases": worsened_cases[:6],
                "sample_false_negative_cases": valid_fn_cases[:6],
            },
            "decision": {
                "viable": bool(viable),
                "reason": (
                    "Event-level structure looks viable offline."
                    if viable
                    else "Event-level structure does not improve enough beyond peak ranking on the validated springboard session."
                ),
            },
        }
        return result


def render_markdown(result: dict) -> str:
    auc = result["auc"]
    ranking = result["false_negative_ranking"]
    decision = result["decision"]
    review = result["review_summary"]
    lines = [
        "# Event-Level Research",
        "",
        f"Session: `{result['session_dir']}`",
        "",
        "## Review Cohorts",
        "",
        f"- reviewed dives: `{review['reviewed_dive_count']}`",
        f"- reviewed board rebounds: `{review['reviewed_board_rebound_count']}`",
        f"- reviewed false negatives: `{review['reviewed_false_negative_count']}`",
        "",
        "## AUC",
        "",
        f"- dive vs rebound, raw peak baseline: `{auc['dive_vs_rebound_raw_peak_auc']:.4f}`",
        f"- dive vs rebound, event score: `{auc['dive_vs_rebound_event_score_auc']:.4f}`",
        f"- false negative vs rebound, raw peak baseline: `{auc['fn_vs_rebound_raw_peak_auc']:.4f}`",
        f"- false negative vs rebound, event score: `{auc['fn_vs_rebound_event_score_auc']:.4f}`",
        "",
        "## False-Negative Cluster Ranking",
        "",
        f"- mean rank delta (raw rank - event rank): `{ranking['mean_rank_delta_raw_minus_event']:.4f}`",
        f"- median rank delta (raw rank - event rank): `{ranking['median_rank_delta_raw_minus_event']:.4f}`",
        f"- top-1 correct anchor count, raw: `{ranking['top1_raw_count']}`",
        f"- top-1 correct anchor count, event: `{ranking['top1_event_count']}`",
        f"- practical `±1.0s` top-1 count, raw: `{ranking['practical_1p0_raw_count']}`",
        f"- practical `±1.0s` top-1 count, event: `{ranking['practical_1p0_event_count']}`",
        f"- practical `±1.5s` top-1 count, raw: `{ranking['practical_1p5_raw_count']}`",
        f"- practical `±1.5s` top-1 count, event: `{ranking['practical_1p5_event_count']}`",
        f"- improved FN cluster cases: `{ranking['improved_case_count']}`",
        f"- worsened FN cluster cases: `{ranking['worsened_case_count']}`",
        "",
        "## Decision",
        "",
        f"- viable: `{decision['viable']}`",
        f"- conclusion: {decision['reason']}",
        "",
    ]
    examples = result["cluster_level_examples"]["improved_false_negative_cases"]
    if examples:
        lines.extend(
            [
                "## Improved False-Negative Examples",
                "",
            ]
        )
        for case in examples:
            lines.append(
                f"- ts `{case['timestamp']:.3f}`: raw rank `{case['correct_anchor_raw_rank']}` -> event rank `{case['correct_anchor_event_rank']}`, "
                f"raw top offset `{case['baseline_top_offset_seconds']:.3f}s`, event top offset `{case['event_top_offset_seconds']:.3f}s`"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline event-level research on validated springboard session artifacts.")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("outputs/evaluation_insep_15min_validated"),
        help="Validated session root containing raw peaks, replayed review, and export artifacts.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs/analysis_event_level.json"),
        help="Where to write the JSON analysis summary.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("outputs/analysis_event_level.md"),
        help="Where to write the Markdown analysis summary.",
    )
    args = parser.parse_args()

    research = EventLevelResearch(args.session_dir)
    result = research.summarize()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(to_serializable(result), indent=2))
    args.md_out.write_text(render_markdown(to_serializable(result)))
    print(json.dumps({"json_out": str(args.json_out), "md_out": str(args.md_out), "viable": result["decision"]["viable"]}, indent=2))


if __name__ == "__main__":
    main()
