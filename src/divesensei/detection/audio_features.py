from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


AUDIO_CLIP_FEATURES = [
    "clip_rms_mean",
    "clip_rms_std",
    "clip_rms_peak",
    "clip_flux_mean",
    "clip_flux_peak",
    "clip_flux_std",
    "clip_hf_ratio_mean",
    "clip_hf_ratio_peak",
    "clip_centroid_mean",
    "clip_flatness_mean",
    "clip_pcen_onset_mean",
    "clip_pcen_onset_peak",
    "clip_pcen_onset_std",
    "clip_pcen_band_spread",
    "clip_pcen_peak_band",
    "clip_pcen_high_band_ratio",
    "clip_pcen_low_band_ratio",
    "clip_pre_post_flux_ratio",
    "clip_pre_post_rms_ratio",
    "clip_peak_to_tail_flux_ratio",
    "clip_peak_to_tail_rms_ratio",
    "clip_post_onset_decay_ratio",
    "clip_post_onset_autocorr_peak",
    "clip_post_band_spread_mean",
    "clip_post_peak_band_stability",
    "clip_temporal_asymmetry_flux",
    "clip_temporal_asymmetry_rms",
]


def frame_audio(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float32)
    windowed = np.lib.stride_tricks.sliding_window_view(signal, frame_length)
    return np.ascontiguousarray(windowed[::hop_length], dtype=np.float32)


def compute_fft_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> Dict[str, Any]:
    frames = frame_audio(signal, frame_length, hop_length)
    if frames.size == 0:
        return {
            "frames": frames,
            "windowed": frames,
            "spectrum": np.empty((0, frame_length // 2 + 1), dtype=np.float32),
            "flux": np.empty(0, dtype=np.float32),
            "rms": np.empty(0, dtype=np.float32),
            "hf_ratio": np.empty(0, dtype=np.float32),
            "spectral_centroid_hz": np.empty(0, dtype=np.float32),
            "spectral_flatness": np.empty(0, dtype=np.float32),
            "freqs": np.fft.rfftfreq(frame_length, d=1.0 / sample_rate),
        }

    window = np.hanning(frame_length).astype(np.float32)
    windowed = frames * window[None, :]
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)).astype(np.float32)
    flux = np.maximum(0.0, spectrum[1:] - spectrum[:-1]).sum(axis=1)
    flux = np.concatenate([[0.0], flux]).astype(np.float32)
    rms = np.sqrt(np.mean(windowed ** 2, axis=1)).astype(np.float32)

    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate).astype(np.float32)
    hf_mask = freqs >= 1800.0
    hf_energy = spectrum[:, hf_mask].sum(axis=1)
    total_energy = spectrum.sum(axis=1) + 1e-8
    hf_ratio = (hf_energy / total_energy).astype(np.float32)
    spectral_centroid_hz = ((spectrum * freqs[None, :]).sum(axis=1) / total_energy).astype(np.float32)
    spectral_flatness = (
        np.exp(np.mean(np.log(spectrum + 1e-8), axis=1)) / (np.mean(spectrum, axis=1) + 1e-8)
    ).astype(np.float32)
    return {
        "frames": frames,
        "windowed": windowed,
        "spectrum": spectrum,
        "flux": flux,
        "rms": rms,
        "hf_ratio": hf_ratio,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_flatness": spectral_flatness,
        "freqs": freqs,
    }


def build_multiband_energies(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    band_edges_hz: Tuple[float, ...] = (200.0, 450.0, 800.0, 1400.0, 2200.0, 3200.0, 4800.0, 7200.0),
) -> np.ndarray:
    if spectrum.size == 0:
        return np.empty((0, max(0, len(band_edges_hz) - 1)), dtype=np.float32)
    bands = []
    for start_hz, end_hz in zip(band_edges_hz[:-1], band_edges_hz[1:]):
        mask = (freqs >= start_hz) & (freqs < end_hz)
        if not np.any(mask):
            bands.append(np.zeros(spectrum.shape[0], dtype=np.float32))
            continue
        bands.append(np.sum(spectrum[:, mask], axis=1).astype(np.float32))
    return np.stack(bands, axis=1).astype(np.float32)


def compute_pcen(
    energies: np.ndarray,
    hop_seconds: float,
    *,
    smoothing_seconds: float = 0.06,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    if energies.size == 0:
        return energies.astype(np.float32)
    smoothing = min(1.0, max(hop_seconds / max(smoothing_seconds, hop_seconds), 1e-4))
    smoothed = np.empty_like(energies, dtype=np.float32)
    smoothed[0] = energies[0]
    for idx in range(1, len(energies)):
        smoothed[idx] = (1.0 - smoothing) * smoothed[idx - 1] + smoothing * energies[idx]
    pcen = (energies / np.power(smoothed + eps, gain) + bias) ** power - (bias ** power)
    return pcen.astype(np.float32)


def compute_multiband_pcen_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> Dict[str, Any]:
    fft_features = compute_fft_features(signal, sample_rate, frame_length, hop_length)
    band_energies = build_multiband_energies(fft_features["spectrum"], fft_features["freqs"])
    pcen = compute_pcen(band_energies, hop_length / float(sample_rate))
    onset = np.maximum(0.0, np.diff(pcen, axis=0, prepend=pcen[:1]))
    return {
        **fft_features,
        "band_energies": band_energies,
        "pcen": pcen,
        "pcen_onset": onset.astype(np.float32),
    }


def extract_clip_feature_map(
    signal: np.ndarray,
    sample_rate: int,
    center_time: float,
    *,
    window_seconds: float,
    frame_length: int,
    hop_length: int,
) -> Dict[str, float]:
    def safe_mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else 0.0

    def safe_std(values: np.ndarray) -> float:
        return float(np.std(values)) if values.size else 0.0

    def safe_peak(values: np.ndarray) -> float:
        return float(np.max(values)) if values.size else 0.0

    def safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / (denominator + 1e-6))

    def normalized_difference(left: float, right: float) -> float:
        return float((right - left) / (abs(left) + abs(right) + 1e-6))

    def frame_band_spread(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        peak = max(float(np.max(frame)), 1e-6)
        return float(np.mean(frame >= 0.65 * peak))

    def post_autocorr_peak(values: np.ndarray) -> float:
        if values.size < 4:
            return 0.0
        centered = values.astype(np.float32) - float(np.mean(values))
        denom = float(np.dot(centered, centered))
        if denom <= 1e-6:
            return 0.0
        best = 0.0
        max_lag = min(12, values.size - 1)
        for lag in range(2, max_lag + 1):
            corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
            best = max(best, corr)
        return best

    half_window = max(window_seconds / 2.0, hop_length / float(sample_rate))
    start = max(0, int(round((center_time - half_window) * sample_rate)))
    end = min(signal.size, int(round((center_time + half_window) * sample_rate)))
    clip = signal[start:end]
    features = compute_multiband_pcen_features(clip, sample_rate, frame_length, hop_length)
    flux = features["flux"]
    rms = features["rms"]
    hf_ratio = features["hf_ratio"]
    centroid = features["spectral_centroid_hz"]
    flatness = features["spectral_flatness"]
    onset = features["pcen_onset"]
    if onset.size == 0:
        onset_sum = np.empty(0, dtype=np.float32)
        onset_band_spread = 0.0
        peak_band = 0.0
        high_band_ratio = 0.0
        low_band_ratio = 0.0
        peak_idx = 0
    else:
        onset_sum = onset.mean(axis=1)
        peak_idx = int(np.argmax(onset_sum))
        peak_frame = onset[peak_idx]
        onset_band_spread = frame_band_spread(peak_frame)
        peak_band = float(np.argmax(peak_frame) / max(1, peak_frame.size - 1))
        high_band_ratio = float(np.sum(peak_frame[-max(1, peak_frame.size // 3):]) / (np.sum(peak_frame) + 1e-6))
        low_band_ratio = float(np.sum(peak_frame[: max(1, peak_frame.size // 3)]) / (np.sum(peak_frame) + 1e-6))

    if onset_sum.size == 0:
        peak_idx = int(np.argmax(flux)) if flux.size else 0
    pre_slice = slice(max(0, peak_idx - 6), peak_idx)
    early_post_slice = slice(min(peak_idx + 1, onset_sum.size), min(onset_sum.size, peak_idx + 6))
    tail_slice = slice(min(peak_idx + 6, onset_sum.size), min(onset_sum.size, peak_idx + 18))
    post_slice = slice(min(peak_idx + 1, onset_sum.size), min(onset_sum.size, peak_idx + 18))

    pre_flux = flux[pre_slice]
    post_flux = flux[post_slice]
    pre_rms = rms[pre_slice]
    post_rms = rms[post_slice]
    early_post_onset = onset_sum[early_post_slice]
    tail_onset = onset_sum[tail_slice]
    tail_flux = flux[tail_slice]
    tail_rms = rms[tail_slice]
    post_frames = onset[post_slice] if onset.size else np.empty((0, 0), dtype=np.float32)
    post_band_spread_mean = safe_mean(np.array([frame_band_spread(frame) for frame in post_frames], dtype=np.float32)) if post_frames.size else 0.0
    if post_frames.size:
        peak_bands = np.argmax(post_frames, axis=1)
        modal_peak_band = np.bincount(peak_bands).max() / max(1, peak_bands.size)
    else:
        modal_peak_band = 0.0

    return {
        "clip_rms_mean": safe_mean(rms),
        "clip_rms_std": safe_std(rms),
        "clip_rms_peak": safe_peak(rms),
        "clip_flux_mean": safe_mean(flux),
        "clip_flux_peak": safe_peak(flux),
        "clip_flux_std": safe_std(flux),
        "clip_hf_ratio_mean": safe_mean(hf_ratio),
        "clip_hf_ratio_peak": safe_peak(hf_ratio),
        "clip_centroid_mean": safe_mean(centroid),
        "clip_flatness_mean": safe_mean(flatness),
        "clip_pcen_onset_mean": safe_mean(onset_sum),
        "clip_pcen_onset_peak": safe_peak(onset_sum),
        "clip_pcen_onset_std": safe_std(onset_sum),
        "clip_pcen_band_spread": onset_band_spread,
        "clip_pcen_peak_band": peak_band,
        "clip_pcen_high_band_ratio": high_band_ratio,
        "clip_pcen_low_band_ratio": low_band_ratio,
        "clip_pre_post_flux_ratio": safe_ratio(safe_mean(post_flux), safe_mean(pre_flux)),
        "clip_pre_post_rms_ratio": safe_ratio(safe_mean(post_rms), safe_mean(pre_rms)),
        "clip_peak_to_tail_flux_ratio": safe_ratio(safe_peak(flux), safe_mean(tail_flux)),
        "clip_peak_to_tail_rms_ratio": safe_ratio(safe_peak(rms), safe_mean(tail_rms)),
        "clip_post_onset_decay_ratio": safe_ratio(safe_mean(early_post_onset), safe_mean(tail_onset)),
        "clip_post_onset_autocorr_peak": post_autocorr_peak(onset_sum[post_slice]),
        "clip_post_band_spread_mean": post_band_spread_mean,
        "clip_post_peak_band_stability": float(modal_peak_band),
        "clip_temporal_asymmetry_flux": normalized_difference(safe_mean(pre_flux), safe_mean(post_flux)),
        "clip_temporal_asymmetry_rms": normalized_difference(safe_mean(pre_rms), safe_mean(post_rms)),
    }


def load_wav_mono_float32(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        sample_width = int(handle.getsampwidth())
        channel_count = int(handle.getnchannels())
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported.")
    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channel_count > 1:
        signal = signal.reshape(-1, channel_count).mean(axis=1)
    return signal / 32768.0, sample_rate
