from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_PATH = Path("outputs/event_window_manifest_preview.jsonl")
LISTS_PATH = Path("outputs/phase5_regime_manifest_lists.json")
OUT_DATASET_NPZ = Path("outputs/platform_noise_es4_dataset.npz")
OUT_DATASET_ROWS_JSON = Path("outputs/platform_noise_es4_dataset_rows.json")
OUT_SUMMARY_JSON = Path("outputs/platform_noise_es4_dataset_summary.json")
OUT_SUMMARY_MD = Path("outputs/platform_noise_es4_dataset_summary.md")
SAMPLE_RATE = 16_000

PROBE1_FEATURE_NAMES = [
    "impact_peak_to_window_rms_ratio",
    "impact_peak_prominence_db",
    "transient_peak_count",
    "inter_peak_interval_cv",
    "post_impact_early_to_late_rms_ratio",
    "tail_half_life_ms",
    "spectral_flatness_post_mean",
    "tonal_peak_fraction_post_mean",
]

R2_FEATURE_NAMES = [
    "whistle_band_energy_fraction_post",
    "spectral_entropy_post_mean",
]

R4_FEATURE_NAMES = [
    "spectral_contrast_mean_post",
    "spectral_contrast_low_high_slope_post",
    "onset_tempogram_peak_ratio_post",
    "onset_density_0_300ms_post",
]

BASELINE_FEATURE_NAMES = [
    "audio_score",
    "audio_clip_probability",
    "event_anchor_timestamp_seconds",
    "is_false_negative_window",
]

ALL_FEATURE_NAMES = BASELINE_FEATURE_NAMES + PROBE1_FEATURE_NAMES + R2_FEATURE_NAMES + R4_FEATURE_NAMES


@dataclass(frozen=True)
class RowRef:
    row_key: str
    label: str
    row: dict[str, Any]


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except Exception:
        return float(default)


def decode_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def framing(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(signal) < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(signal)] = signal
        return padded.reshape(1, frame)
    n_frames = 1 + (len(signal) - frame) // hop
    return np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_frames, frame),
        strides=(signal.strides[0] * hop, signal.strides[0]),
        writeable=False,
    )


def local_peak_indices(values: np.ndarray, threshold: float) -> np.ndarray:
    if len(values) < 3:
        return np.asarray([], dtype=int)
    peaks: list[int] = []
    for idx in range(1, len(values) - 1):
        if values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1] and values[idx] >= threshold:
            peaks.append(idx)
    return np.asarray(peaks, dtype=int)


def octave_band_edges(fmin: float = 200.0, n_bands: int = 6, max_freq: float = 8000.0) -> list[tuple[float, float]]:
    edges = []
    low = 0.0
    first_high = min(fmin, max_freq)
    edges.append((low, first_high))
    low = first_high
    for _ in range(n_bands):
        high = min(low * 2.0, max_freq)
        if high <= low:
            break
        edges.append((low, high))
        low = high
        if high >= max_freq:
            break
    return edges


def spectral_contrast_features(power_post: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    eps = 1e-8
    bands = octave_band_edges(max_freq=float(freqs[-1]))
    if len(power_post) == 0:
        return 0.0, 0.0
    contrasts: list[np.ndarray] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if int(np.sum(mask)) < 3:
            continue
        band = power_post[:, mask]
        top = np.quantile(band, 0.98, axis=1)
        bottom = np.quantile(band, 0.02, axis=1)
        contrast = 10.0 * np.log10(np.maximum(top, eps) / np.maximum(bottom, eps))
        contrasts.append(contrast)
    if not contrasts:
        return 0.0, 0.0
    stacked = np.stack(contrasts, axis=1)
    mean_contrast = float(np.mean(stacked))
    low_band_mean = float(np.mean(stacked[:, 0]))
    high_band_mean = float(np.mean(stacked[:, -1]))
    return mean_contrast, high_band_mean - low_band_mean


def onset_flux_envelope(windowed: np.ndarray) -> np.ndarray:
    if len(windowed) < 2:
        return np.asarray([0.0], dtype=np.float64)
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    diff = np.maximum(power[1:] - power[:-1], 0.0)
    return np.sum(diff, axis=1).astype(np.float64)


def tempogram_peak_ratio(onset_env: np.ndarray, min_lag: int, max_lag: int) -> float:
    eps = 1e-8
    if len(onset_env) < max_lag + 2:
        return 0.0
    onset = onset_env - float(np.mean(onset_env))
    ac = np.correlate(onset, onset, mode="full")
    ac = ac[len(onset) - 1 :]
    zero = float(max(ac[0], eps))
    segment = ac[min_lag : min(max_lag + 1, len(ac))]
    if len(segment) == 0:
        return 0.0
    return float(np.max(segment) / zero)


def onset_density(onset_env: np.ndarray, threshold_scale: float = 0.5) -> float:
    if len(onset_env) < 3:
        return 0.0
    threshold = float(np.mean(onset_env) + threshold_scale * np.std(onset_env))
    peaks = local_peak_indices(onset_env, threshold=threshold)
    return float(len(peaks))


def extract_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    frame = 512
    hop = 128
    eps = 1e-8
    frames = framing(signal, frame=frame, hop=hop)
    if frames.size == 0:
        return {name: 0.0 for name in PROBE1_FEATURE_NAMES + R2_FEATURE_NAMES + R4_FEATURE_NAMES}

    window = np.hanning(frame).astype(np.float32)
    windowed = frames * window
    rms = np.sqrt(np.mean(np.square(windowed), axis=1) + eps)
    peak_idx = int(np.argmax(rms))
    peak_val = float(rms[peak_idx])
    mean_val = float(np.mean(rms))
    med_val = float(np.median(rms))
    std_val = float(np.std(rms))

    peak_threshold = mean_val + 0.5 * std_val
    peak_ids = local_peak_indices(rms, threshold=peak_threshold)
    transient_peak_count = float(len(peak_ids))
    if len(peak_ids) >= 3:
        intervals = np.diff(peak_ids).astype(np.float64)
        interval_cv = float(np.std(intervals) / max(np.mean(intervals), eps))
    else:
        interval_cv = 0.0

    frames_120 = max(1, int(round(0.120 * sample_rate / hop)))
    frames_300 = max(frames_120 + 1, int(round(0.300 * sample_rate / hop)))
    frames_600 = max(frames_300 + 1, int(round(0.600 * sample_rate / hop)))
    early = rms[peak_idx + 1 : peak_idx + 1 + frames_120]
    late = rms[peak_idx + 1 + frames_120 : peak_idx + 1 + frames_600]
    early_mean = float(np.mean(early)) if len(early) else 0.0
    late_mean = float(np.mean(late)) if len(late) else 0.0
    post_impact_early_to_late_rms_ratio = early_mean / max(late_mean, eps)

    half_level = 0.5 * peak_val
    tail = rms[peak_idx + 1 :]
    half_idx = next((idx for idx, value in enumerate(tail, start=1) if value <= half_level), len(tail))
    tail_half_life_ms = float(half_idx * hop * 1000.0 / sample_rate)

    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    post_frames = power[peak_idx : peak_idx + max(1, int(round(0.400 * sample_rate / hop)))]
    if len(post_frames) == 0:
        post_frames = power[max(0, peak_idx - 1) : peak_idx + 1]
    p = np.maximum(post_frames, eps)

    spectral_flatness = np.exp(np.mean(np.log(p), axis=1)) / np.maximum(np.mean(p, axis=1), eps)
    tonal_peak_fraction = np.max(p, axis=1) / np.maximum(np.sum(p, axis=1), eps)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    whistle_band = (freqs >= 1000.0) & (freqs <= 4000.0)
    whistle_band_energy_fraction_post = float(np.mean(np.sum(p[:, whistle_band], axis=1) / np.maximum(np.sum(p, axis=1), eps)))
    pnorm = p / np.maximum(np.sum(p, axis=1, keepdims=True), eps)
    spectral_entropy = -np.sum(pnorm * np.log(np.maximum(pnorm, eps)), axis=1) / math.log(p.shape[1])

    spectral_contrast_mean_post, spectral_contrast_low_high_slope_post = spectral_contrast_features(p, freqs=freqs)
    post_for_onset = windowed[peak_idx : peak_idx + max(2, frames_600)]
    onset_env = onset_flux_envelope(post_for_onset)
    onset_tempogram_peak_ratio_post = tempogram_peak_ratio(onset_env, min_lag=3, max_lag=25)
    onset_300_env = onset_env[: max(1, frames_300 - 1)]
    onset_density_0_300ms_post = onset_density(onset_300_env)

    return {
        "impact_peak_to_window_rms_ratio": peak_val / max(mean_val, eps),
        "impact_peak_prominence_db": 20.0 * math.log10(max(peak_val, eps) / max(med_val, eps)),
        "transient_peak_count": transient_peak_count,
        "inter_peak_interval_cv": interval_cv,
        "post_impact_early_to_late_rms_ratio": post_impact_early_to_late_rms_ratio,
        "tail_half_life_ms": tail_half_life_ms,
        "spectral_flatness_post_mean": float(np.mean(spectral_flatness)),
        "tonal_peak_fraction_post_mean": float(np.mean(tonal_peak_fraction)),
        "whistle_band_energy_fraction_post": whistle_band_energy_fraction_post,
        "spectral_entropy_post_mean": float(np.mean(spectral_entropy)),
        "spectral_contrast_mean_post": spectral_contrast_mean_post,
        "spectral_contrast_low_high_slope_post": spectral_contrast_low_high_slope_post,
        "onset_tempogram_peak_ratio_post": onset_tempogram_peak_ratio_post,
        "onset_density_0_300ms_post": onset_density_0_300ms_post,
    }


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        session_id = str(row["source_session_id"])
        counts[session_id] = counts.get(session_id, 0) + 1
        cid = row.get("legacy_candidate_id")
        rid = str(cid) if cid else f"row-{counts[session_id]:04d}"
        out[f"{session_id}::{rid}"] = row
    return out


def label_counts(items: list[RowRef]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item.label] = out.get(item.label, 0) + 1
    return dict(sorted(out.items()))


def vector_for(item: RowRef, fmap: dict[str, dict[str, float]]) -> list[float]:
    row = item.row
    return [
        to_float(row.get("audio_score")),
        to_float(row.get("audio_clip_probability")),
        to_float(row.get("event_anchor_timestamp_seconds")),
        1.0 if row.get("is_false_negative_window") else 0.0,
        *[to_float(fmap[item.row_key][name]) for name in PROBE1_FEATURE_NAMES],
        *[to_float(fmap[item.row_key][name]) for name in R2_FEATURE_NAMES],
        *[to_float(fmap[item.row_key][name]) for name in R4_FEATURE_NAMES],
    ]


def write_summary_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Platform/Noise ES4 Dataset Summary",
        "",
        "## Frozen dataset materialization",
        "",
        f"- train rows: `{summary['row_counts']['train_rows']}`",
        f"- scored validation rows: `{summary['row_counts']['scored_validation_rows']}`",
        f"- reporting-only rows excluded from scored evaluation: `{summary['row_counts']['reporting_only_rows_excluded']}`",
        f"- total materialized rows in model dataset: `{summary['row_counts']['materialized_rows_total']}`",
        "",
        "## Label counts",
        "",
        f"- train: `{json.dumps(summary['label_counts']['train'], sort_keys=True)}`",
        f"- scored validation: `{json.dumps(summary['label_counts']['scored_validation'], sort_keys=True)}`",
        f"- reporting-only excluded: `{json.dumps(summary['label_counts']['reporting_only_excluded'], sort_keys=True)}`",
        "",
        "## Frozen policy / leak prevention",
        "",
        f"- policy: `{summary['frozen_policy']['platform_noise_policy']}`",
        f"- holdout logic: `{summary['frozen_policy']['holdout_logic']}`",
        f"- scored slice source: `{summary['frozen_policy']['scored_slice_source']}`",
        f"- leak prevention: `{summary['frozen_policy']['leak_prevention_policy']}`",
        f"- overlap train vs scored holdout row_keys: `{summary['frozen_policy']['train_holdout_overlap_row_keys']}`",
        "",
        "## Features",
        "",
        f"- accepted platform/noise feature family included as-is: `{summary['feature_schema']['accepted_feature_family_included_as_is']}`",
        f"- feature columns count: `{len(summary['feature_schema']['feature_columns'])}`",
        "",
        "```json",
        json.dumps(summary["feature_schema"]["feature_columns"], indent=2),
        "```",
        "",
        "## Dataset artifact",
        "",
        f"- matrix artifact: `{summary['dataset_artifacts']['npz_path']}`",
        f"- row metadata: `{summary['dataset_artifacts']['rows_json_path']}`",
    ]
    OUT_SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    by_key = row_key_map(rows)
    lists = json.loads(LISTS_PATH.read_text())

    platform = lists["platform_noise_track"]
    train = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in platform["train_rows"]]
    holdout = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in platform["holdout_rows"]]

    mixed = lists.get("mixed_session_validation_track", {})
    sub_slices = mixed.get("sub_slices", {})
    reporting_names = mixed.get("reporting_only_slices", [])
    reporting_rows: list[dict[str, Any]] = []
    for name in reporting_names:
        reporting_rows.extend(sub_slices.get(name, {}).get("rows", []))
    reporting_refs = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in reporting_rows if item["row_key"] in by_key]

    audio_cache: dict[str, np.ndarray] = {}
    fmap: dict[str, dict[str, float]] = {}
    for item in train + holdout:
        sid = str(item.row["source_session_id"])
        if sid not in audio_cache:
            audio_cache[sid] = decode_audio_mono(Path(str(item.row["source_session_root"])) / "web" / "session_source_review.mp4", SAMPLE_RATE)
        signal = audio_cache[sid]
        start = max(0.0, to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, to_float(item.row.get("event_window_end_seconds")))
        s0 = int(round(start * SAMPLE_RATE))
        s1 = int(round(end * SAMPLE_RATE))
        fmap[item.row_key] = extract_features(signal[s0:s1], sample_rate=SAMPLE_RATE)

    x_train = np.asarray([vector_for(item, fmap) for item in train], dtype=np.float64)
    y_train = np.asarray([1 if item.label == "platform_dive" else 0 for item in train], dtype=np.int64)
    x_holdout = np.asarray([vector_for(item, fmap) for item in holdout], dtype=np.float64)
    y_holdout = np.asarray([1 if item.label == "platform_dive" else 0 for item in holdout], dtype=np.int64)

    OUT_DATASET_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DATASET_NPZ,
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        train_row_keys=np.asarray([item.row_key for item in train]),
        holdout_row_keys=np.asarray([item.row_key for item in holdout]),
        feature_names=np.asarray(ALL_FEATURE_NAMES),
    )

    rows_json = {
        "train_rows": [{"row_key": item.row_key, "label": item.label} for item in train],
        "holdout_rows": [{"row_key": item.row_key, "label": item.label} for item in holdout],
        "reporting_only_excluded_rows": [{"row_key": item.row_key, "label": item.label} for item in reporting_refs],
    }
    OUT_DATASET_ROWS_JSON.write_text(json.dumps(rows_json, indent=2))

    train_keys = {item.row_key for item in train}
    holdout_keys = {item.row_key for item in holdout}
    summary = {
        "scope": {
            "platform_noise_only": True,
            "detector_changed": False,
            "taxonomy_changed": False,
            "labels_changed": False,
            "springboard_changed": False,
        },
        "row_counts": {
            "train_rows": len(train),
            "scored_validation_rows": len(holdout),
            "reporting_only_rows_excluded": len(reporting_refs),
            "materialized_rows_total": len(train) + len(holdout),
        },
        "label_counts": {
            "train": label_counts(train),
            "scored_validation": label_counts(holdout),
            "reporting_only_excluded": label_counts(reporting_refs),
        },
        "frozen_policy": {
            "platform_noise_policy": platform.get("policy"),
            "holdout_logic": platform.get("holdout_logic"),
            "scored_slice_source": platform.get("scored_slice_source"),
            "leak_prevention_policy": platform.get("leak_prevention_policy"),
            "train_holdout_overlap_row_keys": len(train_keys & holdout_keys),
        },
        "feature_schema": {
            "accepted_feature_family_included_as_is": True,
            "feature_columns": ALL_FEATURE_NAMES,
        },
        "dataset_artifacts": {
            "npz_path": str(OUT_DATASET_NPZ),
            "rows_json_path": str(OUT_DATASET_ROWS_JSON),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    write_summary_md(summary)


if __name__ == "__main__":
    main()

