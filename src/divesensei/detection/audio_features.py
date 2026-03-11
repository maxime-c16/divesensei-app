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
    else:
        onset_sum = onset.mean(axis=1)
        peak_idx = int(np.argmax(onset_sum))
        peak_frame = onset[peak_idx]
        onset_band_spread = float(np.mean(peak_frame >= 0.65 * max(float(np.max(peak_frame)), 1e-6)))
        peak_band = float(np.argmax(peak_frame) / max(1, peak_frame.size - 1))
        high_band_ratio = float(np.sum(peak_frame[-max(1, peak_frame.size // 3):]) / (np.sum(peak_frame) + 1e-6))
        low_band_ratio = float(np.sum(peak_frame[: max(1, peak_frame.size // 3)]) / (np.sum(peak_frame) + 1e-6))

    return {
        "clip_rms_mean": float(np.mean(rms)) if rms.size else 0.0,
        "clip_rms_std": float(np.std(rms)) if rms.size else 0.0,
        "clip_rms_peak": float(np.max(rms)) if rms.size else 0.0,
        "clip_flux_mean": float(np.mean(flux)) if flux.size else 0.0,
        "clip_flux_peak": float(np.max(flux)) if flux.size else 0.0,
        "clip_flux_std": float(np.std(flux)) if flux.size else 0.0,
        "clip_hf_ratio_mean": float(np.mean(hf_ratio)) if hf_ratio.size else 0.0,
        "clip_hf_ratio_peak": float(np.max(hf_ratio)) if hf_ratio.size else 0.0,
        "clip_centroid_mean": float(np.mean(centroid)) if centroid.size else 0.0,
        "clip_flatness_mean": float(np.mean(flatness)) if flatness.size else 0.0,
        "clip_pcen_onset_mean": float(np.mean(onset_sum)) if onset_sum.size else 0.0,
        "clip_pcen_onset_peak": float(np.max(onset_sum)) if onset_sum.size else 0.0,
        "clip_pcen_onset_std": float(np.std(onset_sum)) if onset_sum.size else 0.0,
        "clip_pcen_band_spread": onset_band_spread,
        "clip_pcen_peak_band": peak_band,
        "clip_pcen_high_band_ratio": high_band_ratio,
        "clip_pcen_low_band_ratio": low_band_ratio,
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
