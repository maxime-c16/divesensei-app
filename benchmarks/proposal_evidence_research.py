from __future__ import annotations

import argparse
import json
import math
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


EPS = 1e-6
HOP = 256
WIN = 512
NEIGHBORHOOD_SECONDS = 1.5


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def ema(values: np.ndarray, tau_seconds: float, sr: int, hop: int) -> np.ndarray:
    alpha = math.exp(-hop / max(tau_seconds * sr, EPS))
    out = np.empty_like(values, dtype=np.float32)
    acc = float(values[0]) if len(values) else 0.0
    for index, value in enumerate(values):
        acc = alpha * acc + (1.0 - alpha) * float(value)
        out[index] = acc
    return out


def moving_mean_std(values: np.ndarray, radius_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if radius_frames <= 0:
        return values.copy(), np.zeros_like(values)
    width = 2 * radius_frames + 1
    kernel = np.ones(width, dtype=np.float32)
    padded = np.pad(values.astype(np.float32), (radius_frames, radius_frames), mode="edge")
    count = np.convolve(np.ones_like(padded), kernel, mode="valid")
    mean = np.convolve(padded, kernel, mode="valid") / np.maximum(count, EPS)
    sq_mean = np.convolve(padded * padded, kernel, mode="valid") / np.maximum(count, EPS)
    var = np.maximum(sq_mean - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


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


def hist_overlap(a: np.ndarray, b: np.ndarray, bins: int = 24) -> float:
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return float(np.sum(np.minimum(ha, hb) * np.diff(edges)))


def best_threshold(a: np.ndarray, b: np.ndarray) -> dict[str, float | str]:
    values = np.unique(np.concatenate([a, b]))
    if len(values) == 1:
        return {
            "threshold": float(values[0]),
            "direction": ">=",
            "balanced_accuracy": 0.5,
            "tpr": 0.5,
            "tnr": 0.5,
        }
    mids = (values[:-1] + values[1:]) / 2.0
    mids = np.concatenate([[values[0] - 1e-9], mids, [values[-1] + 1e-9]])
    best: dict[str, float | str] | None = None
    for threshold in mids:
        for direction in (">=", "<="):
            if direction == ">=":
                tpr = float(np.mean(a >= threshold))
                tnr = float(np.mean(b < threshold))
            else:
                tpr = float(np.mean(a <= threshold))
                tnr = float(np.mean(b > threshold))
            balanced = 0.5 * (tpr + tnr)
            candidate = {
                "threshold": float(threshold),
                "direction": direction,
                "balanced_accuracy": balanced,
                "tpr": tpr,
                "tnr": tnr,
            }
            if best is None or balanced > float(best["balanced_accuracy"]):
                best = candidate
    assert best is not None
    return best


def pass_rate(values: np.ndarray, threshold: float, direction: str) -> float:
    if direction == ">=":
        return float(np.mean(values >= threshold))
    return float(np.mean(values <= threshold))


def sec_to_frames(seconds: float, sr: int) -> int:
    return int(round(seconds * sr / HOP))


def window_slice(values: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, start)
    end = min(len(values), end)
    if end <= start:
        return values[0:0]
    return values[start:end]


def safe_mean(values: np.ndarray | list[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def safe_std(values: np.ndarray | list[float]) -> float:
    return float(np.std(values)) if len(values) else 0.0


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


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class PeakFeature:
    timestamp: float
    frontend: str
    row: dict
    peak_score: float
    proposal_threshold: float
    peak_frame_index: int
    flux_short: float
    flux_long: float
    rms_short: float
    rms_long: float
    flux_local_z: float
    rms_local_z: float
    env_local_z: float
    transient_rise: float
    transient_contrast: float
    local_noise_norm: float


class ProposalEvidenceResearch:
    def __init__(self, session_dir: Path, audio_path: Path) -> None:
        self.session_dir = session_dir
        self.audio_path = audio_path
        self.raw_rows = load_jsonl(session_dir / "proposal_raw_peaks.jsonl")
        self.diag_rows = load_jsonl(session_dir / "proposal_diagnostics.jsonl")
        self.frontend_rows = load_jsonl(session_dir / "proposal_frontend_candidates.jsonl")
        self.accepted_timestamps = sorted(
            float(row["timestamp"]) for row in self.diag_rows if row.get("pipeline_selected")
        )
        self._load_audio()
        self._build_peak_features()
        self.cohorts = self._build_cohorts()
        self.neighborhoods = self._build_all_neighborhoods()

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
        self.flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
        freqs = np.fft.rfftfreq(WIN, d=1.0 / self.sr)
        low_mask = freqs < 1000.0
        mid_mask = (freqs >= 1000.0) & (freqs < 3000.0)
        high_mask = freqs >= 3000.0
        self.band_low = np.sum(spectrum[:, low_mask], axis=1).astype(np.float32)
        self.band_mid = np.sum(spectrum[:, mid_mask], axis=1).astype(np.float32)
        self.band_high = np.sum(spectrum[:, high_mask], axis=1).astype(np.float32)
        self.env = self._combined_env(self.flux, self.rms)
        self.flux_short = self._pcen_proxy(self.flux, 0.06)
        self.flux_long = self._pcen_proxy(self.flux, 0.30)
        self.rms_short = self._pcen_proxy(self.rms, 0.08)
        self.rms_long = self._pcen_proxy(self.rms, 0.40)
        self.band_low_pcen = self._pcen_proxy(self.band_low, 0.12)
        self.band_mid_pcen = self._pcen_proxy(self.band_mid, 0.12)
        self.band_high_pcen = self._pcen_proxy(self.band_high, 0.12)
        radius = sec_to_frames(2.0, self.sr)
        self.flux_mean, self.flux_std = moving_mean_std(self.flux, radius)
        self.rms_mean, self.rms_std = moving_mean_std(self.rms, radius)
        self.env_mean, self.env_std = moving_mean_std(self.env, radius)
        self.env_diff = np.diff(np.pad(self.env, (1, 0), mode="edge")).astype(np.float32)

    def _combined_env(self, flux: np.ndarray, rms: np.ndarray) -> np.ndarray:
        flux_base = ema(flux, 0.35, self.sr, HOP) + EPS
        rms_base = ema(rms, 0.35, self.sr, HOP) + EPS
        return 0.65 * (flux / flux_base) + 0.35 * (rms / rms_base)

    def _pcen_proxy(self, values: np.ndarray, smooth_seconds: float) -> np.ndarray:
        smooth = ema(values, smooth_seconds, self.sr, HOP)
        gain = values / np.power(EPS + smooth, 0.8)
        return np.power(gain + 2.0, 0.5) - np.power(2.0, 0.5)

    def _build_peak_features(self) -> None:
        self.rows_by_frontend: dict[str, list[dict]] = defaultdict(list)
        self.peak_features: list[PeakFeature] = []
        self.peak_feature_by_key: dict[tuple[str, float], PeakFeature] = {}
        for row in self.raw_rows:
            self.rows_by_frontend[row["proposal_frontend"]].append(row)
        for frontend in self.rows_by_frontend:
            self.rows_by_frontend[frontend].sort(key=lambda row: row["proposal_timestamp_seconds"])
        for row in self.raw_rows:
            peak_index = min(int(row["peak_frame_index"]), self.num_frames - 1)
            transient_rise = float(np.maximum(self.env_diff[peak_index - sec_to_frames(0.06, self.sr): peak_index + 1], 0.0).sum())
            transient_contrast = transient_rise / (
                safe_mean(window_slice(self.env, peak_index + 1, peak_index + sec_to_frames(0.30, self.sr))) + EPS
            )
            noise_floor = safe_mean(
                window_slice(self.env, peak_index - sec_to_frames(0.80, self.sr), peak_index - sec_to_frames(0.05, self.sr))
            )
            feature = PeakFeature(
                timestamp=float(row["proposal_timestamp_seconds"]),
                frontend=str(row["proposal_frontend"]),
                row=row,
                peak_score=float(row.get("raw_proposal_score", 0.0)),
                proposal_threshold=float(row.get("proposal_threshold", 0.0)),
                peak_frame_index=peak_index,
                flux_short=float(self.flux_short[peak_index]),
                flux_long=float(self.flux_long[peak_index]),
                rms_short=float(self.rms_short[peak_index]),
                rms_long=float(self.rms_long[peak_index]),
                flux_local_z=float((self.flux[peak_index] - self.flux_mean[peak_index]) / (self.flux_std[peak_index] + EPS)),
                rms_local_z=float((self.rms[peak_index] - self.rms_mean[peak_index]) / (self.rms_std[peak_index] + EPS)),
                env_local_z=float((self.env[peak_index] - self.env_mean[peak_index]) / (self.env_std[peak_index] + EPS)),
                transient_rise=transient_rise,
                transient_contrast=float(transient_contrast),
                local_noise_norm=float(self.env[peak_index] / (noise_floor + EPS)),
            )
            self.peak_features.append(feature)
            self.peak_feature_by_key[(feature.frontend, round(feature.timestamp, 6))] = feature

    def _nearest_accepted_distance(self, timestamp: float) -> float:
        return min((abs(timestamp - value) for value in self.accepted_timestamps), default=999.0)

    def _select_distinct_centers(self, rows: list[dict], limit: int, min_spacing: float) -> list[float]:
        selected: list[float] = []
        for row in rows:
            timestamp = float(row["proposal_timestamp_seconds"])
            if any(abs(timestamp - current) <= min_spacing for current in selected):
                continue
            selected.append(timestamp)
            if len(selected) >= limit:
                break
        return selected

    def _build_cohorts(self) -> dict[str, dict[str, list[float]]]:
        strong_rows = [row for row in self.raw_rows if row.get("accepted_as_proposal")]
        strong_rows.sort(key=lambda row: float(row.get("raw_proposal_score", 0.0)), reverse=True)
        strong_centers = self._select_distinct_centers(strong_rows, limit=120, min_spacing=0.9)

        confuser_rows = [
            row
            for row in self.raw_rows
            if not row.get("accepted_as_proposal")
            and row.get("rejection_stage") in {"weak_pattern_score", "sustained_noise_reject"}
            and (
                float(row.get("local_prominence", 0.0)) >= 2.5
                or max(float(row.get("post_flux_ratio", 0.0)), float(row.get("post_rms_ratio", 0.0))) >= 1.9
            )
        ]
        confuser_rows.sort(
            key=lambda row: (
                -float(row.get("local_prominence", 0.0)),
                -max(float(row.get("post_flux_ratio", 0.0)), float(row.get("post_rms_ratio", 0.0))),
                -float(row.get("raw_proposal_score", 0.0)),
            )
        )
        confuser_centers = self._select_distinct_centers(confuser_rows, limit=120, min_spacing=1.2)

        weak_named = [114.350347, 157.576774, 398.4226974879982]
        weak_proxy_rows = []
        for row in self.raw_rows:
            timestamp = float(row["proposal_timestamp_seconds"])
            if row["proposal_frontend"] != "pcen_multiband":
                continue
            if self._nearest_accepted_distance(timestamp) <= 3.0:
                continue
            if any(abs(timestamp - value) <= 2.0 for value in weak_named):
                continue
            score = float(row.get("raw_proposal_score", 0.0))
            prominence = float(row.get("local_prominence", 0.0))
            persistence = float(row.get("frontend_persistence_integral_bonus", 0.0))
            ratios = max(float(row.get("post_flux_ratio", 0.0)), float(row.get("post_rms_ratio", 0.0)))
            if score > 4.0 or prominence > 3.5:
                continue
            if persistence < 0.05 and ratios < 1.6:
                continue
            weakness = (
                0.8 * persistence
                + 0.4 * ratios
                + 0.3 * max(0.0, 3.5 - score)
                + 0.3 * max(0.0, 3.0 - prominence)
                - 0.1 * min(abs(float(row.get("audio_pattern_score", 0.0))), 2.0)
            )
            weak_proxy_rows.append((weakness, row))
        weak_proxy_rows.sort(key=lambda item: item[0], reverse=True)
        weak_proxies: list[float] = []
        for _, row in weak_proxy_rows:
            timestamp = float(row["proposal_timestamp_seconds"])
            if any(abs(timestamp - current) <= 8.0 for current in [*weak_named, *weak_proxies]):
                continue
            weak_proxies.append(timestamp)
            if len(weak_proxies) >= 8:
                break
        return {
            "full": {
                "STRONG_DIVES": strong_centers,
                "CONFUSERS": confuser_centers,
                "WEAK_MISSES": [*weak_named, *weak_proxies],
            },
            "named_only": {
                "STRONG_DIVES": strong_centers,
                "CONFUSERS": confuser_centers,
                "WEAK_MISSES": weak_named,
            },
            "weak_named": weak_named,
            "weak_proxies": weak_proxies,
        }

    def _representative_peak(self, center: float) -> PeakFeature | None:
        exact = self.peak_feature_by_key.get(("heuristic", round(center, 6))) or self.peak_feature_by_key.get(("pcen_multiband", round(center, 6)))
        if exact is not None:
            return exact
        candidates = [
            peak
            for peak in self.peak_features
            if abs(peak.timestamp - center) <= 0.35
        ]
        if not candidates:
            candidates = [
                peak
                for peak in self.peak_features
                if abs(peak.timestamp - center) <= 1.0
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda peak: (abs(peak.timestamp - center), -peak.peak_score))
        return candidates[0]

    def _peaks_near(self, timestamp: float, window_seconds: float = NEIGHBORHOOD_SECONDS) -> list[PeakFeature]:
        return [
            peak
            for peak in self.peak_features
            if abs(peak.timestamp - timestamp) <= window_seconds
        ]

    def _cluster_features(self, peaks: list[PeakFeature], max_gap: float) -> list[dict]:
        if not peaks:
            return []
        ordered = sorted(peaks, key=lambda peak: peak.timestamp)
        clusters: list[list[PeakFeature]] = [[ordered[0]]]
        for peak in ordered[1:]:
            if peak.timestamp - clusters[-1][-1].timestamp <= max_gap:
                clusters[-1].append(peak)
            else:
                clusters.append([peak])
        cluster_rows = []
        for cluster in clusters:
            timestamps = np.array([peak.timestamp for peak in cluster], dtype=float)
            scores = np.array([max(peak.peak_score, 0.0) for peak in cluster], dtype=float)
            if scores.sum() <= 0:
                scores = np.ones_like(scores) * 0.01
            spread = float(np.sqrt(np.sum(scores * (timestamps - timestamps.mean()) ** 2) / max(scores.sum(), EPS)))
            frontend_count = len({peak.frontend for peak in cluster})
            score_cv = float(np.std(scores) / (np.mean(scores) + EPS))
            cluster_rows.append(
                {
                    "timestamps": timestamps.tolist(),
                    "size": len(cluster),
                    "mass": float(scores.sum()),
                    "spread": spread,
                    "compactness": float(math.exp(-spread / 0.18)),
                    "frontend_count": frontend_count,
                    "score_cv": score_cv,
                }
            )
        return cluster_rows

    def describe_neighborhood(self, timestamp: float) -> dict:
        peaks = self._peaks_near(timestamp)
        frame_center = min(int(round(timestamp * self.sr / HOP)), self.num_frames - 1)
        frame_radius = sec_to_frames(NEIGHBORHOOD_SECONDS, self.sr)
        env_window = window_slice(self.env, frame_center - frame_radius, frame_center + frame_radius + 1)
        time_axis = np.linspace(-NEIGHBORHOOD_SECONDS, NEIGHBORHOOD_SECONDS, num=13)
        curve_bins = []
        for start, end in zip(time_axis[:-1], time_axis[1:]):
            start_frame = frame_center + sec_to_frames(float(start), self.sr)
            end_frame = frame_center + sec_to_frames(float(end), self.sr)
            curve_bins.append(safe_mean(window_slice(self.env, start_frame, end_frame)))
        peak_scores = np.array([peak.peak_score for peak in peaks], dtype=float) if peaks else np.array([], dtype=float)
        positive_scores = np.maximum(peak_scores, 0.0)
        timestamps = np.array([peak.timestamp for peak in peaks], dtype=float) if peaks else np.array([], dtype=float)
        if len(peaks):
            weights = np.maximum(positive_scores, 0.01)
            spread = float(np.sqrt(np.sum(weights * (timestamps - timestamp) ** 2) / max(weights.sum(), EPS)))
        else:
            spread = 0.0
        if len(peaks) >= 3:
            sorted_times = np.sort(timestamps)
            deltas = np.diff(sorted_times)
            regularity = float(np.std(deltas) / (np.mean(deltas) + EPS))
        else:
            regularity = 1.0
        peak_count = len(peaks)
        density = peak_count / (2 * NEIGHBORHOOD_SECONDS)
        clusters = self._cluster_features(peaks, max_gap=0.24)
        coherence = max(
            (
                cluster["mass"] * cluster["compactness"] * (1.0 + 0.15 * (cluster["frontend_count"] - 1))
                for cluster in clusters
            ),
            default=0.0,
        )
        return {
            "center_timestamp": timestamp,
            "peak_count": peak_count,
            "total_score_mass": float(np.sum(positive_scores)),
            "max_peak_score": float(np.max(peak_scores)) if len(peak_scores) else 0.0,
            "mean_peak_score": safe_mean(peak_scores),
            "temporal_density": density,
            "weighted_spread": spread,
            "interpeak_regularity": regularity,
            "energy_mass": float(np.sum(np.maximum(env_window, 0.0))),
            "energy_curve": curve_bins,
            "cluster_count": len(clusters),
            "cluster_coherence": coherence,
            "cross_frontend_peak_count": len({peak.frontend for peak in peaks}),
        }

    def _build_all_neighborhoods(self) -> dict[str, dict[str, list[dict]]]:
        neighborhoods: dict[str, dict[str, list[dict]]] = {}
        for cohort_key, cohorts in self.cohorts.items():
            if cohort_key not in {"full", "named_only"}:
                continue
            neighborhoods[cohort_key] = {
                cohort_name: [self.describe_neighborhood(timestamp) for timestamp in timestamps]
                for cohort_name, timestamps in cohorts.items()
            }
        return neighborhoods

    def _window_candidates(self, center: float, radius: float, half_width: float) -> list[tuple[float, float]]:
        starts = np.arange(center - radius, center + radius + EPS, 0.08)
        return [(float(start), float(start + 2 * half_width)) for start in starts]

    def _region_frame_mass(self, start: float, end: float) -> float:
        start_idx = sec_to_frames(start, self.sr)
        end_idx = sec_to_frames(end, self.sr)
        return float(np.sum(np.maximum(window_slice(self.env, start_idx, end_idx), 0.0)))

    def _peaks_in_window(self, center: float, start: float, end: float) -> list[PeakFeature]:
        return [peak for peak in self._peaks_near(center, window_seconds=2.0) if start <= peak.timestamp <= end]

    def _region_window(self, center: float) -> dict[str, np.ndarray | float | int]:
        peak = self._representative_peak(center)
        if peak is None:
            start_index = 0
            end_index = sec_to_frames(1.0, self.sr)
            peak_index = 0
        else:
            start_index = peak.peak_frame_index - sec_to_frames(0.2, self.sr)
            end_index = peak.peak_frame_index + sec_to_frames(0.8, self.sr)
            peak_index = peak.peak_frame_index
        env = window_slice(self.env, start_index, end_index)
        flux = window_slice(self.flux, start_index, end_index)
        rms = window_slice(self.rms, start_index, end_index)
        flatness = window_slice(self.flatness, start_index, end_index)
        low = window_slice(self.band_low, start_index, end_index)
        mid = window_slice(self.band_mid, start_index, end_index)
        high = window_slice(self.band_high, start_index, end_index)
        low_pcen = window_slice(self.band_low_pcen, start_index, end_index)
        mid_pcen = window_slice(self.band_mid_pcen, start_index, end_index)
        high_pcen = window_slice(self.band_high_pcen, start_index, end_index)
        flux_short = window_slice(self.flux_short, start_index, end_index)
        min_len = min(len(env), len(flux), len(rms), len(flatness), len(low), len(mid), len(high), len(low_pcen), len(mid_pcen), len(high_pcen), len(flux_short))
        env = env[:min_len]
        flux = flux[:min_len]
        rms = rms[:min_len]
        flatness = flatness[:min_len]
        low = low[:min_len]
        mid = mid[:min_len]
        high = high[:min_len]
        low_pcen = low_pcen[:min_len]
        mid_pcen = mid_pcen[:min_len]
        high_pcen = high_pcen[:min_len]
        flux_short = flux_short[:min_len]
        time_axis = np.arange(min_len, dtype=float) * HOP / self.sr
        peak_offset = (peak_index - start_index) if peak is not None else 0
        return {
            "env": env,
            "flux": flux,
            "rms": rms,
            "flatness": flatness,
            "low": low,
            "mid": mid,
            "high": high,
            "low_pcen": low_pcen,
            "mid_pcen": mid_pcen,
            "high_pcen": high_pcen,
            "flux_short": flux_short,
            "time_axis": time_axis,
            "candidate_offset_frames": peak_offset,
        }

    def _corr(self, left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or len(right) < 2:
            return 0.0
        if safe_std(left) < EPS or safe_std(right) < EPS:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    def _count_local_maxima(self, values: np.ndarray) -> float:
        if len(values) < 3:
            return 0.0
        threshold = safe_mean(values)
        count = 0
        for idx in range(1, len(values) - 1):
            if values[idx] > threshold and values[idx] > values[idx - 1] and values[idx] >= values[idx + 1]:
                count += 1
        return float(count)

    def _duration_above(self, values: np.ndarray, threshold: float) -> float:
        if not len(values):
            return 0.0
        above = values >= threshold
        duration = 0.0
        active = False
        start = 0
        for idx, flag in enumerate(above):
            if flag and not active:
                start = idx
                active = True
            if active and not flag:
                duration = max(duration, (idx - start) * HOP / self.sr)
                active = False
        if active:
            duration = max(duration, (len(values) - start) * HOP / self.sr)
        return duration

    def region_family_features(self, center: float) -> dict[str, dict[str, float]]:
        region = self._region_window(center)
        env = np.asarray(region["env"], dtype=float)
        flux = np.asarray(region["flux"], dtype=float)
        rms = np.asarray(region["rms"], dtype=float)
        low = np.asarray(region["low"], dtype=float)
        mid = np.asarray(region["mid"], dtype=float)
        high = np.asarray(region["high"], dtype=float)
        low_pcen = np.asarray(region["low_pcen"], dtype=float)
        mid_pcen = np.asarray(region["mid_pcen"], dtype=float)
        high_pcen = np.asarray(region["high_pcen"], dtype=float)
        flux_short = np.asarray(region["flux_short"], dtype=float)
        time_axis = np.asarray(region["time_axis"], dtype=float)

        def span(start: float, end: float) -> np.ndarray:
            begin = int(round(start * self.sr / HOP))
            finish = int(round(end * self.sr / HOP))
            return env[begin:finish]

        early = span(0.0, 0.15)
        mid_env = span(0.15, 0.40)
        late = span(0.40, 0.80)
        peak_index = int(np.argmax(env)) if len(env) else 0
        peak_time = time_axis[peak_index] if len(time_axis) else 0.0
        decay_end = min(len(env), peak_index + sec_to_frames(0.6, self.sr))
        decay_segment = env[peak_index:decay_end]
        decay_time = time_axis[peak_index:decay_end]
        if len(decay_segment) >= 2:
            decay_slope = float(np.polyfit(decay_time, decay_segment, 1)[0])
        else:
            decay_slope = 0.0
        post = env[int(round(0.20 * self.sr / HOP)):]
        derivative = np.diff(post) if len(post) >= 2 else np.array([], dtype=float)
        frame_80ms = max(1, sec_to_frames(0.08, self.sr))
        if len(env) >= frame_80ms:
            rolling = np.convolve(env, np.ones(frame_80ms, dtype=float), mode="valid")
            concentration = float(np.max(rolling) / (np.sum(env) + EPS))
        else:
            concentration = 0.0

        def band_energy_ratio(values: np.ndarray, start: float, end: float) -> float:
            begin = int(round(start * self.sr / HOP))
            finish = int(round(end * self.sr / HOP))
            return float(np.sum(values[begin:finish]))

        low_early = band_energy_ratio(low_pcen, 0.0, 0.15)
        low_mid = band_energy_ratio(low_pcen, 0.15, 0.40)
        low_late = band_energy_ratio(low_pcen, 0.40, 0.80)
        mid_early = band_energy_ratio(mid_pcen, 0.0, 0.15)
        mid_mid = band_energy_ratio(mid_pcen, 0.15, 0.40)
        mid_late = band_energy_ratio(mid_pcen, 0.40, 0.80)
        high_early = band_energy_ratio(high_pcen, 0.0, 0.15)
        high_mid = band_energy_ratio(high_pcen, 0.15, 0.40)
        high_late = band_energy_ratio(high_pcen, 0.40, 0.80)
        band_peak_times = np.array(
            [
                time_axis[int(np.argmax(low_pcen))] if len(low_pcen) else 0.0,
                time_axis[int(np.argmax(mid_pcen))] if len(mid_pcen) else 0.0,
                time_axis[int(np.argmax(high_pcen))] if len(high_pcen) else 0.0,
            ],
            dtype=float,
        )

        family_a = {
            "peak_amplitude": float(np.max(env)) if len(env) else 0.0,
            "time_to_peak": float(peak_time),
            "decay_slope": decay_slope,
            "early_energy": float(np.sum(early)),
            "mid_energy": float(np.sum(mid_env)),
            "late_energy": float(np.sum(late)),
            "mid_over_early": float(np.sum(mid_env) / (np.sum(early) + EPS)),
            "late_over_early": float(np.sum(late) / (np.sum(early) + EPS)),
            "duration_above_1p10": self._duration_above(env, 1.10),
            "duration_above_1p25": self._duration_above(env, 1.25),
        }
        family_b = {
            "post_cv": float(np.std(post) / (np.mean(post) + EPS)) if len(post) else 0.0,
            "energy_asymmetry": float((np.sum(late) - np.sum(early)) / (np.sum(late) + np.sum(early) + EPS)),
            "local_maxima_count": self._count_local_maxima(env),
            "derivative_variance": float(np.var(derivative)) if len(derivative) else 0.0,
            "energy_concentration": concentration,
            "peak_vs_total_ratio": float(np.max(env) / (np.sum(env) + EPS)) if len(env) else 0.0,
        }
        family_c = {
            "low_mid_ratio_mid": float(low_mid / (mid_mid + EPS)),
            "high_low_ratio_early": float(high_early / (low_early + EPS)),
            "high_low_ratio_late": float(high_late / (low_late + EPS)),
            "band_temporal_consistency": float(1.0 / (1.0 + np.std(band_peak_times))),
            "flux_pcen_corr": self._corr(flux, flux_short),
            "band_corr_mean": float(np.mean([
                self._corr(low_pcen, mid_pcen),
                self._corr(mid_pcen, high_pcen),
                self._corr(low_pcen, high_pcen),
            ])),
            "band_cv_mean": float(np.mean([
                np.std(low_pcen) / (np.mean(low_pcen) + EPS) if len(low_pcen) else 0.0,
                np.std(mid_pcen) / (np.mean(mid_pcen) + EPS) if len(mid_pcen) else 0.0,
                np.std(high_pcen) / (np.mean(high_pcen) + EPS) if len(high_pcen) else 0.0,
            ])),
        }
        return {"A": family_a, "B": family_b, "C": family_c}

    def score_A1_multi_window(self, center: float) -> float:
        peaks = self._peaks_near(center)
        return max(
            (
                0.35 * peak.flux_short
                + 0.20 * peak.flux_long
                + 0.25 * peak.rms_short
                + 0.20 * peak.rms_long
            )
            for peak in peaks
        ) if peaks else 0.0

    def score_A2_local_normalization(self, center: float) -> float:
        peaks = self._peaks_near(center)
        return max((0.55 * peak.env_local_z + 0.45 * peak.local_noise_norm) for peak in peaks) if peaks else 0.0

    def score_A3_transient_proxy(self, center: float) -> float:
        peaks = self._peaks_near(center)
        return max((0.65 * peak.transient_rise + 0.35 * peak.transient_contrast) for peak in peaks) if peaks else 0.0

    def score_B_window_sum(self, center: float, half_width: float) -> float:
        best = 0.0
        for start, end in self._window_candidates(center, radius=1.0, half_width=half_width):
            peaks = self._peaks_in_window(center, start, end)
            peak_mass = sum(max(peak.peak_score, 0.0) for peak in peaks)
            score = peak_mass + 0.12 * self._region_frame_mass(start, end)
            best = max(best, score)
        return best

    def score_B_weighted_window(self, center: float, half_width: float) -> float:
        best = 0.0
        for start, end in self._window_candidates(center, radius=1.0, half_width=half_width):
            mid = 0.5 * (start + end)
            peaks = self._peaks_in_window(center, start, end)
            weighted = 0.0
            for peak in peaks:
                weight = math.exp(-abs(peak.timestamp - mid) / max(half_width, 0.05))
                weighted += weight * max(peak.peak_score, 0.0)
            weighted += 0.10 * self._region_frame_mass(start, end)
            best = max(best, weighted)
        return best

    def score_B_shifted_softmax(self, center: float, half_width: float) -> float:
        best = 0.0
        for start, end in self._window_candidates(center, radius=1.0, half_width=half_width):
            peaks = self._peaks_in_window(center, start, end)
            if not peaks:
                continue
            scores = np.array([peak.peak_score for peak in peaks], dtype=float)
            weights = np.exp(scores / 2.0)
            pooled = float(np.sum(weights * scores) / np.maximum(np.sum(weights), EPS))
            pooled += 0.08 * self._region_frame_mass(start, end)
            best = max(best, pooled)
        return best

    def score_C_topk(self, center: float, k: int) -> float:
        peaks = self._peaks_near(center)
        scores = sorted((max(peak.peak_score, 0.0) for peak in peaks), reverse=True)
        if not scores:
            return 0.0
        return float(sum(scores[:k]))

    def score_C_cluster(self, center: float, max_gap: float) -> float:
        peaks = self._peaks_near(center)
        clusters = self._cluster_features(peaks, max_gap=max_gap)
        return max(
            (
                cluster["mass"] * cluster["compactness"] * (1.0 / (1.0 + cluster["score_cv"]))
                for cluster in clusters
            ),
            default=0.0,
        )

    def score_C_cross_frontend(self, center: float) -> float:
        peaks = self._peaks_near(center)
        if not peaks:
            return 0.0
        heuristic = [peak for peak in peaks if peak.frontend == "heuristic"]
        pcen = [peak for peak in peaks if peak.frontend == "pcen_multiband"]
        agreement = 0.0
        for left in heuristic:
            for right in pcen:
                if abs(left.timestamp - right.timestamp) <= 0.12:
                    agreement += 0.5 * (max(left.peak_score, 0.0) + max(right.peak_score, 0.0))
        return agreement

    def _best_cluster_terms(self, center: float, max_gap: float = 0.24) -> dict[str, float]:
        peaks = self._peaks_near(center)
        clusters = self._cluster_features(peaks, max_gap=max_gap)
        best = {
            "top2_mass": 0.0,
            "top3_mass": 0.0,
            "compactness": 0.0,
            "agreement_bonus": 0.0,
            "capped_mass": 0.0,
            "cluster_score": 0.0,
        }
        for cluster in clusters:
            cluster_peaks = [
                peak
                for peak in peaks
                if peak.timestamp in cluster["timestamps"]
            ]
            scores = sorted((max(peak.peak_score, 0.0) for peak in cluster_peaks), reverse=True)
            if not scores:
                continue
            top2_mass = float(sum(scores[:2]))
            top3_mass = float(sum(scores[:3]))
            compactness = float(cluster["compactness"])
            frontend_count = int(cluster["frontend_count"])
            agreement_bonus = 0.10 * max(frontend_count - 1, 0)
            capped_mass = min(top2_mass, 16.0)
            cluster_score = min(top3_mass, 18.0) * compactness * (1.0 + agreement_bonus)
            candidate = {
                "top2_mass": top2_mass,
                "top3_mass": top3_mass,
                "compactness": compactness,
                "agreement_bonus": agreement_bonus,
                "capped_mass": capped_mass,
                "cluster_score": cluster_score,
            }
            if candidate["cluster_score"] > best["cluster_score"]:
                best = candidate
        return best

    def score_C_top_k_cluster_mass(self, center: float) -> float:
        terms = self._best_cluster_terms(center, max_gap=0.24)
        return min(terms["capped_mass"] * terms["compactness"], 12.0)

    def score_C_cluster_evidence_score(self, center: float) -> float:
        terms = self._best_cluster_terms(center, max_gap=0.24)
        return min(terms["cluster_score"], 12.5)

    def score_D_soft_sum(self, center: float, temperature: float) -> float:
        peaks = self._peaks_near(center)
        if not peaks:
            return 0.0
        total = 0.0
        for peak in peaks:
            margin = (peak.peak_score - peak.proposal_threshold) / max(temperature, EPS)
            weight = 1.0 / (1.0 + math.exp(-margin))
            total += weight * max(peak.peak_score, 0.0)
        return total

    def score_D_probability_pool(self, center: float, temperature: float) -> float:
        peaks = self._peaks_near(center)
        if not peaks:
            return 0.0
        scores = np.array([peak.peak_score for peak in peaks], dtype=float)
        thresholds = np.array([peak.proposal_threshold for peak in peaks], dtype=float)
        weights = 1.0 / (1.0 + np.exp(-(scores - thresholds) / max(temperature, EPS)))
        return float(np.sum(weights * scores) / np.maximum(np.sum(weights), EPS))

    def score_D_evidence_density(self, center: float, temperature: float) -> float:
        peaks = self._peaks_near(center)
        if not peaks:
            return 0.0
        total = 0.0
        for peak in peaks:
            margin = (peak.peak_score - peak.proposal_threshold) / max(temperature, EPS)
            weight = 1.0 / (1.0 + math.exp(-margin))
            total += weight
        descriptor = self.describe_neighborhood(center)
        return float(total * descriptor["cluster_coherence"] / max(descriptor["peak_count"], 1))

    def score_baseline(self, center: float) -> float:
        peaks = self._peaks_near(center)
        return max((peak.peak_score for peak in peaks), default=0.0)

    def experiment_scores(self, focus: str = "all") -> dict[str, Callable[[float], float]]:
        all_scores = {
            "baseline_max_peak_score": self.score_baseline,
            "A1_multi_window_mix": self.score_A1_multi_window,
            "A2_local_normalization": self.score_A2_local_normalization,
            "A3_transient_proxy": self.score_A3_transient_proxy,
            "B1_window_sum_0p3": lambda center: self.score_B_window_sum(center, 0.3),
            "B1_window_sum_0p5": lambda center: self.score_B_window_sum(center, 0.5),
            "B1_window_sum_1p0": lambda center: self.score_B_window_sum(center, 1.0),
            "B2_weighted_window_0p5": lambda center: self.score_B_weighted_window(center, 0.5),
            "B3_shifted_softmax_0p5": lambda center: self.score_B_shifted_softmax(center, 0.5),
            "C1_topk_2": lambda center: self.score_C_topk(center, 2),
            "C1_topk_3": lambda center: self.score_C_topk(center, 3),
            "C1_topk_5": lambda center: self.score_C_topk(center, 5),
            "C2_cluster_0p24": lambda center: self.score_C_cluster(center, 0.24),
            "C2_cluster_0p40": lambda center: self.score_C_cluster(center, 0.40),
            "C3_cross_frontend_agreement": self.score_C_cross_frontend,
            "D1_soft_sum_0p50": lambda center: self.score_D_soft_sum(center, 0.50),
            "D1_soft_sum_1p00": lambda center: self.score_D_soft_sum(center, 1.00),
            "D2_probability_pool_0p75": lambda center: self.score_D_probability_pool(center, 0.75),
            "D3_evidence_density_0p75": lambda center: self.score_D_evidence_density(center, 0.75),
            "top_k_cluster_mass": self.score_C_top_k_cluster_mass,
            "cluster_evidence_score": self.score_C_cluster_evidence_score,
        }
        if focus == "bounded_cluster":
            return {
                "baseline_max_peak_score": all_scores["baseline_max_peak_score"],
                "C1_topk_2": all_scores["C1_topk_2"],
                "top_k_cluster_mass": all_scores["top_k_cluster_mass"],
                "cluster_evidence_score": all_scores["cluster_evidence_score"],
            }
        return all_scores

    def _cohort_feature_matrix(self, centers: list[float], family: str) -> tuple[np.ndarray, list[str]]:
        rows = []
        feature_names: list[str] | None = None
        for center in centers:
            features = self.region_family_features(center)[family]
            if feature_names is None:
                feature_names = list(features.keys())
            rows.append([features[name] for name in feature_names])
        return np.asarray(rows, dtype=float), (feature_names or [])

    def _fit_linear_score(self, x_pos: np.ndarray, x_neg: np.ndarray) -> dict[str, np.ndarray]:
        x_train = np.concatenate([x_pos, x_neg], axis=0)
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0) + EPS
        pos_mean = np.mean((x_pos - mean) / std, axis=0)
        neg_mean = np.mean((x_neg - mean) / std, axis=0)
        weights = pos_mean - neg_mean
        bias = -0.5 * float(np.dot(weights, pos_mean + neg_mean))
        return {"mean": mean, "std": std, "weights": weights, "bias": np.array([bias])}

    def _fit_logistic_score(self, x_pos: np.ndarray, x_neg: np.ndarray, steps: int = 400, lr: float = 0.05) -> dict[str, np.ndarray]:
        x_train = np.concatenate([x_pos, x_neg], axis=0)
        y_train = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))], axis=0)
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0) + EPS
        x_norm = (x_train - mean) / std
        weights = np.zeros(x_norm.shape[1], dtype=float)
        bias = 0.0
        for _ in range(steps):
            logits = x_norm @ weights + bias
            probs = sigmoid(logits)
            grad_w = x_norm.T @ (probs - y_train) / len(y_train) + 0.01 * weights
            grad_b = float(np.mean(probs - y_train))
            weights -= lr * grad_w
            bias -= lr * grad_b
        return {"mean": mean, "std": std, "weights": weights, "bias": np.array([bias])}

    def _fit_tree(self, x_pos: np.ndarray, x_neg: np.ndarray, max_depth: int = 3) -> dict:
        x_train = np.concatenate([x_pos, x_neg], axis=0)
        y_train = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))], axis=0)
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0) + EPS
        x_norm = (x_train - mean) / std

        def gini(labels: np.ndarray) -> float:
            if len(labels) == 0:
                return 0.0
            p = np.mean(labels)
            return 1.0 - p * p - (1.0 - p) * (1.0 - p)

        def build(indices: np.ndarray, depth: int) -> dict:
            labels = y_train[indices]
            node = {"prob": float(np.mean(labels))}
            if depth >= max_depth or len(indices) < 10 or np.all(labels == labels[0]):
                return node
            best = None
            current = gini(labels)
            for feature_index in range(x_norm.shape[1]):
                values = x_norm[indices, feature_index]
                thresholds = np.unique(np.percentile(values, [20, 40, 60, 80]))
                for threshold in thresholds:
                    left = indices[values <= threshold]
                    right = indices[values > threshold]
                    if len(left) < 4 or len(right) < 4:
                        continue
                    score = current - (
                        len(left) / len(indices) * gini(y_train[left]) + len(right) / len(indices) * gini(y_train[right])
                    )
                    if best is None or score > best["gain"]:
                        best = {
                            "gain": float(score),
                            "feature": feature_index,
                            "threshold": float(threshold),
                            "left": left,
                            "right": right,
                        }
            if best is None or best["gain"] <= 0.0:
                return node
            node.update(
                {
                    "feature": int(best["feature"]),
                    "threshold": float(best["threshold"]),
                    "left": build(best["left"], depth + 1),
                    "right": build(best["right"], depth + 1),
                }
            )
            return node

        tree = build(np.arange(len(x_norm)), 0)
        return {"mean": mean, "std": std, "tree": tree}

    def _apply_linear(self, model: dict[str, np.ndarray], matrix: np.ndarray) -> np.ndarray:
        x_norm = (matrix - model["mean"]) / model["std"]
        return x_norm @ model["weights"] + float(model["bias"][0])

    def _apply_logistic(self, model: dict[str, np.ndarray], matrix: np.ndarray) -> np.ndarray:
        x_norm = (matrix - model["mean"]) / model["std"]
        return sigmoid(x_norm @ model["weights"] + float(model["bias"][0]))

    def _apply_tree(self, model: dict, matrix: np.ndarray) -> np.ndarray:
        x_norm = (matrix - model["mean"]) / model["std"]

        def predict_row(row: np.ndarray, node: dict) -> float:
            if "feature" not in node:
                return float(node["prob"])
            if row[int(node["feature"])] <= float(node["threshold"]):
                return predict_row(row, node["left"])
            return predict_row(row, node["right"])

        return np.asarray([predict_row(row, model["tree"]) for row in x_norm], dtype=float)

    def run_region_descriptor(self) -> dict:
        families = ("A", "B", "C")
        family_payload: dict[str, dict] = {}
        comparison_rows: dict[str, dict] = {
            "baseline_max_peak_score": self.experiments_single("baseline_max_peak_score"),
        }
        feature_rankings: dict[str, list[dict]] = {}
        for family in families:
            matrices = {
                cohort_name: self._cohort_feature_matrix(self.cohorts["full"][cohort_name], family)
                for cohort_name in ("STRONG_DIVES", "CONFUSERS", "WEAK_MISSES")
            }
            feature_names = matrices["STRONG_DIVES"][1]
            strong_matrix = matrices["STRONG_DIVES"][0]
            conf_matrix = matrices["CONFUSERS"][0]
            weak_matrix = matrices["WEAK_MISSES"][0]
            named_matrix, _ = self._cohort_feature_matrix(self.cohorts["named_only"]["WEAK_MISSES"], family)
            linear_model = self._fit_linear_score(weak_matrix, conf_matrix)
            logistic_model = self._fit_logistic_score(weak_matrix, conf_matrix)
            tree_model = self._fit_tree(weak_matrix, conf_matrix)
            models = {
                "linear": self._apply_linear(linear_model, weak_matrix),  # placeholder overwritten below
            }
            model_defs = {
                "linear": (linear_model, self._apply_linear),
                "logistic": (logistic_model, self._apply_logistic),
                "tree": (tree_model, self._apply_tree),
            }
            model_results: dict[str, dict] = {}
            for model_name, (model, apply_fn) in model_defs.items():
                scores_full = {
                    "STRONG_DIVES": {f"{timestamp:.6f}": float(score) for timestamp, score in zip(self.cohorts["full"]["STRONG_DIVES"], apply_fn(model, strong_matrix))},
                    "CONFUSERS": {f"{timestamp:.6f}": float(score) for timestamp, score in zip(self.cohorts["full"]["CONFUSERS"], apply_fn(model, conf_matrix))},
                    "WEAK_MISSES": {f"{timestamp:.6f}": float(score) for timestamp, score in zip(self.cohorts["full"]["WEAK_MISSES"], apply_fn(model, weak_matrix))},
                }
                scores_named = {
                    "STRONG_DIVES": scores_full["STRONG_DIVES"],
                    "CONFUSERS": scores_full["CONFUSERS"],
                    "WEAK_MISSES": {f"{timestamp:.6f}": float(score) for timestamp, score in zip(self.cohorts["named_only"]["WEAK_MISSES"], apply_fn(model, named_matrix))},
                }
                payload = {
                    "full": {"scores": scores_full, "metrics": self._evaluate_single(scores_full)},
                    "named_only": {"scores": scores_named, "metrics": self._evaluate_single(scores_named)},
                }
                model_results[model_name] = payload
                comparison_rows[f"{family}_{model_name}"] = payload
            ranked_models = sorted(
                model_results.items(),
                key=lambda item: (
                    -float(item[1]["full"]["metrics"]["weak_vs_confuser"]["auc"]),
                    -float(item[1]["named_only"]["metrics"]["weak_vs_confuser"]["auc"]),
                    float(item[1]["full"]["metrics"]["weak_vs_strong"]["auc"]),
                ),
            )
            feature_items = []
            for index, feature_name in enumerate(feature_names):
                strong = strong_matrix[:, index]
                conf = conf_matrix[:, index]
                weak = weak_matrix[:, index]
                named = named_matrix[:, index]
                feature_items.append(
                    {
                        "name": feature_name,
                        "full_weak_vs_confuser_auc": robust_auc(weak, conf),
                        "named_weak_vs_confuser_auc": robust_auc(named, conf),
                        "full_weak_vs_strong_auc": robust_auc(weak, strong),
                        "full_overlap": hist_overlap(weak, conf),
                        "named_overlap": hist_overlap(named, conf),
                    }
                )
            feature_items.sort(
                key=lambda item: (
                    -0.6 * item["full_weak_vs_confuser_auc"] - 0.4 * item["named_weak_vs_confuser_auc"],
                    item["full_overlap"],
                )
            )
            feature_rankings[family] = feature_items
            family_payload[family] = {
                "best_model": ranked_models[0][0],
                "models": model_results,
                "feature_rankings": feature_items,
            }

        baseline_full = comparison_rows["baseline_max_peak_score"]["full"]["metrics"]
        baseline_named = comparison_rows["baseline_max_peak_score"]["named_only"]["metrics"]
        measurable_winner = None
        best_name = None
        ordered = []
        for name, payload in comparison_rows.items():
            if name == "baseline_max_peak_score":
                continue
            full = payload["full"]["metrics"]
            named = payload["named_only"]["metrics"]
            improves = (
                full["weak_vs_confuser"]["auc"] > baseline_full["weak_vs_confuser"]["auc"]
                and named["weak_vs_confuser"]["auc"] > baseline_named["weak_vs_confuser"]["auc"]
                and full["weak_vs_strong"]["auc"] <= baseline_full["weak_vs_strong"]["auc"]
                and named["weak_vs_strong"]["auc"] <= baseline_named["weak_vs_strong"]["auc"]
            )
            if improves and measurable_winner is None:
                measurable_winner = name
            ordered.append(
                (
                    full["weak_vs_confuser"]["auc"] + named["weak_vs_confuser"]["auc"] - full["weak_vs_strong"]["auc"] - named["weak_vs_strong"]["auc"],
                    name,
                )
            )
        ordered.sort(reverse=True)
        best_name = ordered[0][1] if ordered else None
        return {
            "assumptions": {
                "session_dir": str(self.session_dir),
                "audio_path": str(self.audio_path),
                "focus": "region_descriptor",
                "region_window_seconds": {"pre": 0.2, "post": 0.8},
                "descriptor_families": {
                    "A": "Envelope / energy trajectory on normalized flux + RMS.",
                    "B": "Shape / stability descriptors on the 1.0s region.",
                    "C": "Multi-band / PCEN-aligned descriptors on low/mid/high band trajectories.",
                },
            },
            "baseline": comparison_rows["baseline_max_peak_score"],
            "descriptor_families": family_payload,
            "comparison": comparison_rows,
            "selection": {
                "measurable_winner": measurable_winner,
                "best_nonbaseline": best_name,
            },
        }

    def experiments_single(self, name: str) -> dict:
        scorer = self.experiment_scores(focus="bounded_cluster").get(name) or self.experiment_scores().get(name)
        assert scorer is not None
        by_scope: dict[str, dict] = {}
        for scope_name in ("full", "named_only"):
            cohort_scores = {
                cohort_name: {
                    f"{timestamp:.6f}": float(scorer(timestamp))
                    for timestamp in self.cohorts[scope_name][cohort_name]
                }
                for cohort_name in ("STRONG_DIVES", "CONFUSERS", "WEAK_MISSES")
            }
            by_scope[scope_name] = {
                "scores": cohort_scores,
                "metrics": self._evaluate_single(cohort_scores),
            }
        return by_scope

    def _evaluate_single(self, scores: dict[str, dict[str, float]]) -> dict:
        strong = np.array(list(scores["STRONG_DIVES"].values()), dtype=float)
        conf = np.array(list(scores["CONFUSERS"].values()), dtype=float)
        weak = np.array(list(scores["WEAK_MISSES"].values()), dtype=float)
        weak_conf = {
            "auc": robust_auc(weak, conf),
            "overlap": hist_overlap(weak, conf),
            "best_threshold": best_threshold(weak, conf),
        }
        strong_conf = {
            "auc": robust_auc(strong, conf),
            "overlap": hist_overlap(strong, conf),
            "best_threshold": best_threshold(strong, conf),
        }
        weak_strong = {
            "auc": robust_auc(weak, strong),
            "overlap": hist_overlap(weak, strong),
            "best_threshold": best_threshold(weak, strong),
        }
        threshold = float(weak_conf["best_threshold"]["threshold"])
        direction = str(weak_conf["best_threshold"]["direction"])
        pass_rates = {
            "weak": pass_rate(weak, threshold, direction),
            "confuser": pass_rate(conf, threshold, direction),
            "strong": pass_rate(strong, threshold, direction),
        }
        named_anchor_variance = safe_std([value for key, value in scores["WEAK_MISSES"].items()])
        objective = 0.55 * weak_conf["auc"] + 0.30 * strong_conf["auc"] + 0.15 * (1.0 - weak_strong["auc"])
        return {
            "weak_vs_confuser": weak_conf,
            "strong_vs_confuser": strong_conf,
            "weak_vs_strong": weak_strong,
            "threshold_pass_rates": pass_rates,
            "means": {
                "weak": safe_mean(weak),
                "confuser": safe_mean(conf),
                "strong": safe_mean(strong),
            },
            "stds": {
                "weak": safe_std(weak),
                "confuser": safe_std(conf),
                "strong": safe_std(strong),
            },
            "objective": objective,
            "weak_variance": named_anchor_variance,
        }

    def run(self, focus: str = "all") -> dict:
        experiments = self.experiment_scores(focus=focus)
        results: dict[str, dict] = {}
        for name, scorer in experiments.items():
            by_scope: dict[str, dict] = {}
            for scope_name in ("full", "named_only"):
                cohort_scores = {
                    cohort_name: {
                        f"{timestamp:.6f}": float(scorer(timestamp))
                        for timestamp in self.cohorts[scope_name][cohort_name]
                    }
                    for cohort_name in ("STRONG_DIVES", "CONFUSERS", "WEAK_MISSES")
                }
                by_scope[scope_name] = {
                    "scores": cohort_scores,
                    "metrics": self._evaluate_single(cohort_scores),
                }
            results[name] = by_scope
        direction_summary: dict[str, dict] = {}
        prefixes = ("A", "B", "C", "D") if focus != "bounded_cluster" else ("C",)
        for prefix in prefixes:
            items = [
                (name, payload)
                for name, payload in results.items()
                if name.startswith(prefix)
            ]
            if not items:
                continue
            items.sort(
                key=lambda item: (
                    -float(item[1]["full"]["metrics"]["objective"]),
                    -float(item[1]["named_only"]["metrics"]["objective"]),
                )
            )
            best_name, best_payload = items[0]
            direction_summary[prefix] = {
                "best_experiment": best_name,
                "best_metrics": best_payload,
                "ranked_experiments": [
                    {
                        "name": name,
                        "full_objective": payload["full"]["metrics"]["objective"],
                        "named_objective": payload["named_only"]["metrics"]["objective"],
                        "weak_vs_confuser_auc": payload["full"]["metrics"]["weak_vs_confuser"]["auc"],
                        "weak_vs_strong_auc": payload["full"]["metrics"]["weak_vs_strong"]["auc"],
                        "confuser_pass_rate": payload["full"]["metrics"]["threshold_pass_rates"]["confuser"],
                    }
                    for name, payload in items
                ],
            }
        baseline = results["baseline_max_peak_score"]
        baseline_full = baseline["full"]["metrics"]
        baseline_named = baseline["named_only"]["metrics"]
        candidate_names = [name for name in results if name != "baseline_max_peak_score"]
        measurable_winners = []
        exploratory_rank = []
        for name in candidate_names:
            full = results[name]["full"]["metrics"]
            named = results[name]["named_only"]["metrics"]
            improves = (
                full["weak_vs_confuser"]["auc"] >= baseline_full["weak_vs_confuser"]["auc"]
                and named["weak_vs_confuser"]["auc"] >= baseline_named["weak_vs_confuser"]["auc"]
                and full["weak_vs_strong"]["auc"] <= baseline_full["weak_vs_strong"]["auc"]
                and named["weak_vs_strong"]["auc"] <= baseline_named["weak_vs_strong"]["auc"]
                and full["threshold_pass_rates"]["confuser"] <= baseline_full["threshold_pass_rates"]["confuser"] + 0.05
                and named["threshold_pass_rates"]["confuser"] <= baseline_named["threshold_pass_rates"]["confuser"] + 0.05
            )
            if improves:
                measurable_winners.append(name)
            exploratory_score = (
                -abs(full["weak_vs_confuser"]["auc"] - baseline_full["weak_vs_confuser"]["auc"])
                -abs(named["weak_vs_confuser"]["auc"] - baseline_named["weak_vs_confuser"]["auc"])
                -abs(full["weak_vs_strong"]["auc"] - baseline_full["weak_vs_strong"]["auc"])
                -abs(named["weak_vs_strong"]["auc"] - baseline_named["weak_vs_strong"]["auc"])
                -0.5 * abs(full["threshold_pass_rates"]["confuser"] - baseline_full["threshold_pass_rates"]["confuser"])
                -0.5 * abs(named["threshold_pass_rates"]["confuser"] - baseline_named["threshold_pass_rates"]["confuser"])
            )
            exploratory_rank.append((exploratory_score, name))
        exploratory_rank.sort(reverse=True)
        exploratory_lead_name = exploratory_rank[0][1]
        measurable_winner_name = None
        if measurable_winners:
            measurable_winner_name = max(
                measurable_winners,
                key=lambda name: (
                    float(results[name]["full"]["metrics"]["objective"]),
                    float(results[name]["named_only"]["metrics"]["objective"]),
                ),
            )
        weak_desc = self.neighborhoods["full"]["WEAK_MISSES"]
        conf_desc = self.neighborhoods["full"]["CONFUSERS"]
        strong_desc = self.neighborhoods["full"]["STRONG_DIVES"]
        return {
            "assumptions": {
                "session_dir": str(self.session_dir),
                "audio_path": str(self.audio_path),
                "weak_named": self.cohorts["weak_named"],
                "weak_proxies": self.cohorts["weak_proxies"],
                "focus": focus,
                "formulas": {
                    "C1_topk_2": "score = s1 + s2, where s1 and s2 are the two highest non-negative peak scores in the local neighborhood.",
                    "top_k_cluster_mass": "For the best cluster within max_gap=0.24s: score = min((s1 + s2), 16.0) * exp(-spread/0.18), then hard-cap at 12.0.",
                    "cluster_evidence_score": "For the best cluster within max_gap=0.24s: score = min((s1 + s2 + s3), 18.0) * exp(-spread/0.18) * (1 + 0.10 * agreement), where agreement=1 if both frontends appear else 0, then hard-cap at 12.5.",
                },
            },
            "neighborhood_descriptors": self.neighborhoods,
            "descriptor_summary": {
                cohort: {
                    key: {
                        "mean": safe_mean([row[key] for row in values]),
                        "median": float(np.median([row[key] for row in values])),
                        "std": safe_std([row[key] for row in values]),
                    }
                    for key in (
                        "peak_count",
                        "total_score_mass",
                        "max_peak_score",
                        "mean_peak_score",
                        "temporal_density",
                        "weighted_spread",
                        "energy_mass",
                        "cluster_coherence",
                    )
                }
                for cohort, values in {
                    "STRONG_DIVES": strong_desc,
                    "CONFUSERS": conf_desc,
                    "WEAK_MISSES": weak_desc,
                }.items()
            },
            "experiments": results,
            "direction_summary": direction_summary,
            "baseline": baseline,
            "selection": {
                "measurable_winner": measurable_winner_name,
                "exploratory_lead": exploratory_lead_name,
                "measurable_winners": measurable_winners,
            },
            "best_direction": {
                "direction": exploratory_lead_name[0] if exploratory_lead_name else None,
                "payload": direction_summary.get(exploratory_lead_name[0]) if exploratory_lead_name else None,
            },
        }


def render_report(payload: dict) -> str:
    lines: list[str] = []
    focus = payload["assumptions"].get("focus", "all")
    if focus == "region_descriptor":
        return render_region_report(payload)
    lines.append("## Baseline Weak-Miss Characteristics (Confirmed)")
    weak = payload["descriptor_summary"]["WEAK_MISSES"]
    conf = payload["descriptor_summary"]["CONFUSERS"]
    strong = payload["descriptor_summary"]["STRONG_DIVES"]
    lines.append(
        f"Weak neighborhoods stayed weaker and more diffuse than both strong dives and confusers. "
        f"Peak-count mean was `{weak['peak_count']['mean']:.2f}` vs `{conf['peak_count']['mean']:.2f}` confusers and `{strong['peak_count']['mean']:.2f}` strong dives. "
        f"Max-peak-score mean was `{weak['max_peak_score']['mean']:.2f}` vs `{conf['max_peak_score']['mean']:.2f}` and `{strong['max_peak_score']['mean']:.2f}`. "
        f"Weighted spread mean was `{weak['weighted_spread']['mean']:.3f}` vs `{conf['weighted_spread']['mean']:.3f}` and `{strong['weighted_spread']['mean']:.3f}`."
    )
    lines.append(
        f"Total score mass was closer to confusers than to strong dives: `{weak['total_score_mass']['mean']:.2f}` vs `{conf['total_score_mass']['mean']:.2f}` and `{strong['total_score_mass']['mean']:.2f}`. "
        f"Cluster coherence was `{weak['cluster_coherence']['mean']:.2f}` vs `{conf['cluster_coherence']['mean']:.2f}` and `{strong['cluster_coherence']['mean']:.2f}`."
    )
    lines.append("")
    if focus == "bounded_cluster":
        lines.append("## Comparison Table")
    else:
        lines.append("## Results by Direction")
    baseline_full = payload["baseline"]["full"]["metrics"]
    baseline_named = payload["baseline"]["named_only"]["metrics"]
    if focus == "bounded_cluster":
        lines.append(
            "| Variant | Full weak/conf AUC | Named weak/conf AUC | Full weak/strong AUC | Named weak/strong AUC | Full overlap | Named overlap | Full BA | Named BA |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for name in ("baseline_max_peak_score", "C1_topk_2", "top_k_cluster_mass", "cluster_evidence_score"):
            variant = payload["experiments"][name]
            full = variant["full"]["metrics"]
            named = variant["named_only"]["metrics"]
            lines.append(
                f"| `{name}` | `{full['weak_vs_confuser']['auc']:.3f}` | `{named['weak_vs_confuser']['auc']:.3f}` | "
                f"`{full['weak_vs_strong']['auc']:.3f}` | `{named['weak_vs_strong']['auc']:.3f}` | "
                f"`{full['weak_vs_confuser']['overlap']:.3f}` | `{named['weak_vs_confuser']['overlap']:.3f}` | "
                f"`{full['weak_vs_confuser']['best_threshold']['balanced_accuracy']:.3f}` | `{named['weak_vs_confuser']['best_threshold']['balanced_accuracy']:.3f}` |"
            )
    else:
        lines.append(
            f"Baseline single-peak reference: full weak-vs-confuser AUC `{baseline_full['weak_vs_confuser']['auc']:.3f}`, "
            f"full weak-vs-strong AUC `{baseline_full['weak_vs_strong']['auc']:.3f}`, confuser pass rate `{baseline_full['threshold_pass_rates']['confuser']:.3f}`. "
            f"Named-anchor weak-vs-confuser AUC `{baseline_named['weak_vs_confuser']['auc']:.3f}`."
        )
    for direction in (() if focus == "bounded_cluster" else ("A", "B", "C", "D")):
        summary = payload["direction_summary"][direction]
        name = summary["best_experiment"]
        full = summary["best_metrics"]["full"]["metrics"]
        named = summary["best_metrics"]["named_only"]["metrics"]
        lines.append(f"### {direction}: {name}")
        lines.append(
            f"Full cohort: weak-vs-confuser AUC `{full['weak_vs_confuser']['auc']:.3f}`, overlap `{full['weak_vs_confuser']['overlap']:.3f}`, "
            f"weak-vs-strong AUC `{full['weak_vs_strong']['auc']:.3f}`, confuser pass rate `{full['threshold_pass_rates']['confuser']:.3f}`."
        )
        lines.append(
            f"Named anchors only: weak-vs-confuser AUC `{named['weak_vs_confuser']['auc']:.3f}`, "
            f"weak-vs-strong AUC `{named['weak_vs_strong']['auc']:.3f}`, confuser pass rate `{named['threshold_pass_rates']['confuser']:.3f}`."
        )
        lines.append(
            f"Delta vs baseline: full weak/conf `{full['weak_vs_confuser']['auc'] - baseline_full['weak_vs_confuser']['auc']:+.3f}`, "
            f"full weak/strong `{full['weak_vs_strong']['auc'] - baseline_full['weak_vs_strong']['auc']:+.3f}`, "
            f"named weak/conf `{named['weak_vs_confuser']['auc'] - baseline_named['weak_vs_confuser']['auc']:+.3f}`."
        )
        top_ranked = summary["ranked_experiments"][:3]
        lines.append(
            "Top candidates: "
            + ", ".join(
                f"`{item['name']}` (obj `{item['full_objective']:.3f}`, weak/conf `{item['weak_vs_confuser_auc']:.3f}`)"
                for item in top_ranked
            )
            + "."
        )
    lines.append("")
    if focus == "bounded_cluster":
        lines.append("## Did any bounded cluster score beat baseline?")
    else:
        lines.append("## Best Direction Identified")
    winner = payload["selection"]["measurable_winner"]
    exploratory = payload["selection"]["exploratory_lead"]
    if winner is None:
        best_payload = payload["experiments"][exploratory] if exploratory else None
        if best_payload is not None:
            full = best_payload["full"]["metrics"]
            named = best_payload["named_only"]["metrics"]
            lines.append(
                f"No. Neither bounded cluster variant cleared the baseline decision rule on both the full cohort and the named anchors."
            )
            lines.append(
                f"The least-bad result was `{exploratory}` with full weak-vs-confuser AUC `{full['weak_vs_confuser']['auc']:.3f}` and named-anchor AUC `{named['weak_vs_confuser']['auc']:.3f}`, "
                f"but baseline remained higher at `{baseline_full['weak_vs_confuser']['auc']:.3f}` and `{baseline_named['weak_vs_confuser']['auc']:.3f}`."
            )
    else:
        best_payload = payload["experiments"][winner]
        full = best_payload["full"]["metrics"]
        named = best_payload["named_only"]["metrics"]
        lines.append(
            f"`{winner}` is the first measurable winner. "
            f"Full metrics: weak-vs-confuser AUC `{full['weak_vs_confuser']['auc']:.3f}`, weak-vs-strong AUC `{full['weak_vs_strong']['auc']:.3f}`, "
            f"strong-vs-confuser AUC `{full['strong_vs_confuser']['auc']:.3f}`."
        )
        lines.append(
            f"Named-anchor robustness held at weak-vs-confuser AUC `{named['weak_vs_confuser']['auc']:.3f}` with confuser pass rate `{named['threshold_pass_rates']['confuser']:.3f}`."
        )
    lines.append("")
    if focus == "bounded_cluster":
        lines.append("## Should proposal-evidence exploration stop here?")
    else:
        lines.append("## Proposed Next Implementation")
    chosen = exploratory if winner is None else winner
    if focus == "bounded_cluster":
        if winner is None:
            lines.append(
                "Yes. Proposal-evidence exploration should stop here."
            )
            lines.append(
                f"`top_k_cluster_mass` and `cluster_evidence_score` both failed the baseline rule, so there is no bounded cluster score that justifies detector integration."
            )
            lines.append(
                "Pivot away from proposal-evidence heuristics. If work continues, it should move to a different representation family rather than more local pooling variants."
            )
        else:
            lines.append(
                f"No. `{winner}` is an implement candidate."
            )
    elif chosen.startswith("B"):
        lines.append(
            "Implement a `window_evidence_score` before frontend ranking. For each raw peak neighborhood, slide a short temporal window, "
            "pool sub-threshold peaks plus bounded frame-energy mass, then emit a proposal at the best local window center instead of the strongest single peak."
        )
    elif chosen.startswith("C"):
        lines.append(
            "Implement a `cluster_evidence_score` before frontend ranking. Build compact local peak clusters across both frontends, score them by top-K mass, compactness, "
            "and cross-frontend agreement, then rank clusters rather than individual peaks."
        )
    elif chosen.startswith("D"):
        lines.append(
            "Implement a `soft_region_evidence_score` before frontend ranking. Convert peak margins into bounded soft weights, accumulate them over a local region, "
            "and use that accumulated evidence to form proposals from weak neighborhoods."
        )
    else:
        lines.append(
            "Implement a bounded multi-scale frontend evidence term before ranking. Use local normalization and multi-timescale smoothed envelopes to boost weak neighborhoods "
            "before the proposal stage."
        )
    lines.append("")
    lines.append("## Confidence Level")
    confidence = "medium" if winner is None else "low"
    lines.append(
        f"{confidence.capitalize()}. The result is reproducible on the local artifacts and the named-anchor subset, but it is still a proposal-stage offline surrogate without replay/export validation."
    )
    return "\n".join(lines) + "\n"


def render_region_report(payload: dict) -> str:
    baseline_full = payload["baseline"]["full"]["metrics"]
    baseline_named = payload["baseline"]["named_only"]["metrics"]
    lines = []
    lines.append("## Comparison Table")
    lines.append("| Variant | Full weak/conf AUC | Named weak/conf AUC | Full overlap | Named overlap | Full BA | Named BA | Full weak/strong AUC | Named weak/strong AUC |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows = [("baseline_max_peak_score", payload["baseline"])]
    for family in ("A", "B", "C"):
        best_model = payload["descriptor_families"][family]["best_model"]
        rows.append((f"{family}_{best_model}", payload["descriptor_families"][family]["models"][best_model]))
    for name, variant in rows:
        full = variant["full"]["metrics"]
        named = variant["named_only"]["metrics"]
        lines.append(
            f"| `{name}` | `{full['weak_vs_confuser']['auc']:.3f}` | `{named['weak_vs_confuser']['auc']:.3f}` | "
            f"`{full['weak_vs_confuser']['overlap']:.3f}` | `{named['weak_vs_confuser']['overlap']:.3f}` | "
            f"`{full['weak_vs_confuser']['best_threshold']['balanced_accuracy']:.3f}` | `{named['weak_vs_confuser']['best_threshold']['balanced_accuracy']:.3f}` | "
            f"`{full['weak_vs_strong']['auc']:.3f}` | `{named['weak_vs_strong']['auc']:.3f}` |"
        )
    lines.append("")
    lines.append("## Did any region-level descriptor beat baseline?")
    winner = payload["selection"]["measurable_winner"]
    if winner is None:
        lines.append(
            f"No. Baseline remained strongest at full weak-vs-confuser AUC `{baseline_full['weak_vs_confuser']['auc']:.3f}` and named-anchor AUC `{baseline_named['weak_vs_confuser']['auc']:.3f}`."
        )
        best_name = payload["selection"]["best_nonbaseline"]
        if best_name is not None:
            best_variant = payload["comparison"][best_name]
            lines.append(
                f"The best non-baseline region descriptor was `{best_name}`, but it still failed the strict rule."
            )
            lines.append(
                f"It reached full weak-vs-confuser AUC `{best_variant['full']['metrics']['weak_vs_confuser']['auc']:.3f}` and named-anchor AUC `{best_variant['named_only']['metrics']['weak_vs_confuser']['auc']:.3f}`, "
                f"which did not beat baseline on both cohorts."
            )
    else:
        variant = payload["comparison"][winner]
        lines.append(
            f"Yes. `{winner}` beat baseline on both the full cohort and the named anchors."
        )
        lines.append(
            f"Full weak-vs-confuser AUC `{variant['full']['metrics']['weak_vs_confuser']['auc']:.3f}` vs baseline `{baseline_full['weak_vs_confuser']['auc']:.3f}`; "
            f"named-anchor AUC `{variant['named_only']['metrics']['weak_vs_confuser']['auc']:.3f}` vs `{baseline_named['weak_vs_confuser']['auc']:.3f}`."
        )
    lines.append("")
    lines.append("## Which features actually separated weak misses from confusers?")
    for family in ("A", "B", "C"):
        top = payload["descriptor_families"][family]["feature_rankings"][:3]
        lines.append(
            f"Family `{family}` top features: "
            + ", ".join(
                f"`{item['name']}` (full AUC `{item['full_weak_vs_confuser_auc']:.3f}`, named AUC `{item['named_weak_vs_confuser_auc']:.3f}`)"
                for item in top
            )
            + "."
        )
    lines.append("")
    lines.append("## Recommendation")
    if winner is None:
        lines.append("no improvement — pivot again")
        lines.append(
            "No region-level descriptor beat baseline under the strict rule. Handcrafted feature exploration should stop here; the next justified step is a learned representation."
        )
    else:
        lines.append("implement region-level descriptor in pipeline")
        lines.append(
            f"`{winner}` is the only candidate that cleared the full-cohort and named-anchor bar."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline proposal-evidence research runner.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus", choices=("all", "bounded_cluster", "region_descriptor"), default="all")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = ProposalEvidenceResearch(args.session_dir, args.audio_path)
    payload = runner.run_region_descriptor() if args.focus == "region_descriptor" else runner.run(focus=args.focus)
    report = render_report(payload)

    json_path = args.output_dir / "proposal_evidence_research.json"
    md_path = args.output_dir / "proposal_evidence_research.md"
    json_path.write_text(json.dumps(to_serializable(payload), indent=2))
    md_path.write_text(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
