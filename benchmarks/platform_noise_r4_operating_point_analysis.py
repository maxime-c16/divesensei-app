from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MANIFEST_PATH = Path("outputs/event_window_manifest_preview.jsonl")
LISTS_PATH = Path("outputs/phase5_regime_manifest_lists.json")
R4_PATH = Path("outputs/platform_noise_feature_probe_r4.json")
OUT_JSON = Path("outputs/platform_noise_r4_operating_point_analysis.json")
OUT_MD = Path("outputs/platform_noise_r4_operating_point_analysis.md")
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


@dataclass(frozen=True)
class RowRef:
    row_key: str
    label: str
    row: dict


def to_float(value, default: float = 0.0) -> float:
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
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1] and values[i] >= threshold:
            peaks.append(i)
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
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if int(np.sum(mask)) < 3:
            continue
        band = power_post[:, mask]
        top = np.quantile(band, 0.98, axis=1)
        bottom = np.quantile(band, 0.02, axis=1)
        c = np.log(np.maximum(top, eps)) - np.log(np.maximum(bottom, eps))
        contrasts.append(c)
    if not contrasts:
        return 0.0, 0.0
    contrast_matrix = np.stack(contrasts, axis=0)
    band_means = np.mean(contrast_matrix, axis=1)
    contrast_mean_post = float(np.mean(contrast_matrix))
    lo_mean = float(np.mean(band_means[: max(1, len(band_means) // 2)]))
    hi_mean = float(np.mean(band_means[max(1, len(band_means) // 2) :]))
    return contrast_mean_post, hi_mean - lo_mean


def onset_flux_envelope(windowed: np.ndarray) -> np.ndarray:
    eps = 1e-8
    spec = np.abs(np.fft.rfft(windowed, axis=1))
    if len(spec) < 2:
        return np.zeros(1, dtype=np.float64)
    diff = spec[1:] - spec[:-1]
    flux = np.sum(np.maximum(diff, 0.0), axis=1)
    return (flux / max(float(np.max(flux)), eps)).astype(np.float64)


def tempogram_peak_ratio(onset_env: np.ndarray, min_lag: int, max_lag: int) -> float:
    eps = 1e-8
    if len(onset_env) < 4:
        return 0.0
    max_lag = min(max_lag, len(onset_env) - 1)
    if max_lag <= min_lag:
        return 0.0
    ac = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(0, max_lag + 1):
        a = onset_env[: len(onset_env) - lag]
        b = onset_env[lag:]
        ac[lag] = float(np.dot(a, b))
    return float(np.max(ac[min_lag : max_lag + 1]) / max(float(ac[0]), eps))


def onset_density(onset_env: np.ndarray) -> float:
    if len(onset_env) < 3:
        return 0.0
    threshold = float(np.mean(onset_env) + 0.5 * np.std(onset_env))
    return float(len(local_peak_indices(onset_env, threshold)))


def extract_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    frame = 512
    hop = 128
    eps = 1e-8
    frames = framing(signal, frame=frame, hop=hop)
    if frames.size == 0:
        names = PROBE1_FEATURE_NAMES + R2_FEATURE_NAMES + R4_FEATURE_NAMES
        return {name: 0.0 for name in names}

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
    early_late_ratio = early_mean / max(late_mean, eps)

    half_level = 0.5 * peak_val
    tail = rms[peak_idx + 1 :]
    half_idx = next((i for i, value in enumerate(tail, start=1) if value <= half_level), len(tail))
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
    whistle_frac = float(np.mean(np.sum(p[:, whistle_band], axis=1) / np.maximum(np.sum(p, axis=1), eps)))
    pnorm = p / np.maximum(np.sum(p, axis=1, keepdims=True), eps)
    spectral_entropy = -np.sum(pnorm * np.log(np.maximum(pnorm, eps)), axis=1) / math.log(p.shape[1])

    contrast_mean, contrast_slope = spectral_contrast_features(p, freqs)
    post_for_onset = windowed[peak_idx : peak_idx + max(2, frames_600)]
    onset_env = onset_flux_envelope(post_for_onset)
    onset_temp_ratio = tempogram_peak_ratio(onset_env, min_lag=3, max_lag=25)
    onset_density_300 = onset_density(onset_env[: max(1, frames_300 - 1)])

    return {
        "impact_peak_to_window_rms_ratio": peak_val / max(mean_val, eps),
        "impact_peak_prominence_db": 20.0 * math.log10(max(peak_val, eps) / max(med_val, eps)),
        "transient_peak_count": transient_peak_count,
        "inter_peak_interval_cv": interval_cv,
        "post_impact_early_to_late_rms_ratio": early_late_ratio,
        "tail_half_life_ms": tail_half_life_ms,
        "spectral_flatness_post_mean": float(np.mean(spectral_flatness)),
        "tonal_peak_fraction_post_mean": float(np.mean(tonal_peak_fraction)),
        "whistle_band_energy_fraction_post": whistle_frac,
        "spectral_entropy_post_mean": float(np.mean(spectral_entropy)),
        "spectral_contrast_mean_post": contrast_mean,
        "spectral_contrast_low_high_slope_post": contrast_slope,
        "onset_tempogram_peak_ratio_post": onset_temp_ratio,
        "onset_density_0_300ms_post": onset_density_300,
    }


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def train_binary_logreg(x: np.ndarray, y: np.ndarray, epochs: int = 1000, lr: float = 0.1, l2: float = 0.01) -> dict:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        error = probs - y
        grad_w = (x.T @ error) / len(x) + l2 * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return {"weights": weights, "bias": bias}


def predict_binary_logreg(model: dict, x: np.ndarray) -> np.ndarray:
    logits = x @ model["weights"] + model["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    total = len(pos) * len(neg)
    if total == 0:
        return 0.5
    gt = 0
    eq = 0
    for value in pos:
        diff = value - neg
        gt += int(np.sum(diff > 0))
        eq += int(np.sum(diff == 0))
    auc = (gt + 0.5 * eq) / total
    return float(max(auc, 1.0 - auc))


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.float64)
    tp = int(np.sum((y_true == 1.0) & (pred == 1.0)))
    fn = int(np.sum((y_true == 1.0) & (pred == 0.0)))
    fp = int(np.sum((y_true == 0.0) & (pred == 1.0)))
    tn = int(np.sum((y_true == 0.0) & (pred == 0.0)))
    acc = float((tp + tn) / len(y_true)) if len(y_true) else 0.0

    def f1_for(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    _, rec_platform, f1_platform = f1_for(tp, fp, fn)
    _, rec_noise, f1_noise = f1_for(tn, fn, fp)
    macro_f1 = float((f1_platform + f1_noise) / 2.0)
    return {
        "threshold": round(float(threshold), 6),
        "macro_f1": macro_f1,
        "accuracy": acc,
        "platform_dive_recall": rec_platform,
        "noise_or_other_recall": rec_noise,
        "confusion_matrix": [[tp, fn], [fp, tn]],
        "noise_to_platform_fp": fp,
        "platform_to_noise_fn": fn,
    }


def row_key_map(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        cid = row.get("legacy_candidate_id")
        rid = str(cid) if cid else f"row-{counts[sid]:04d}"
        out[f"{sid}::{rid}"] = row
    return out


def write_md(report: dict) -> None:
    lines = [
        "# Platform/Noise r4 Operating-Point Analysis",
        "",
        f"- decision: `{report['decision']}`",
        "",
        "## Part A — Score distribution",
        "",
        f"- true platform score range: `{report['part_a_score_distribution']['platform_score_range']}`",
        f"- true noise score range: `{report['part_a_score_distribution']['noise_score_range']}`",
        f"- overlap region: `{report['part_a_score_distribution']['overlap_region']}`",
        f"- better-than-default threshold plausible: `{report['part_a_score_distribution']['better_threshold_plausible']}`",
        "",
        "## Part B — Threshold sweep",
        "",
        f"- AUC (ranking, threshold-invariant): `{report['part_b_threshold_sweep']['auc']:.4f}`",
        "",
        "| threshold | macro F1 | accuracy | platform recall | noise recall | confusion matrix |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["part_b_threshold_sweep"]["threshold_results"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['platform_dive_recall']:.4f} | {row['noise_or_other_recall']:.4f} | `{row['confusion_matrix']}` |"
        )
    lines.extend(
        [
            "",
            "## Part C — Guardrail search",
            "",
            f"- constraint platform recall >= 0.75: `{report['part_c_guardrail_search']['constraint_platform_recall_floor']}`",
            f"- constraint macro F1 >= 0.50: `{report['part_c_guardrail_search']['constraint_macro_f1_floor']}`",
            f"- constraint noise FP improved vs r4 baseline (3): `{report['part_c_guardrail_search']['constraint_noise_fp_improved_vs_r4_baseline']}`",
            f"- feasible threshold exists: `{report['part_c_guardrail_search']['feasible_threshold_exists']}`",
            f"- best threshold candidate: `{report['part_c_guardrail_search']['best_threshold_candidate']}`",
            "",
            "## Part D — Final decision",
            "",
            f"- `{report['decision']}`",
            f"- rationale: `{report['decision_rationale']}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    r4 = json.loads(R4_PATH.read_text())
    r4_baseline_fp = r4["five_way_comparison"]["probe_r4_final_bundle"]["confusion_matrix"][1][0]

    rows = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    by_key = row_key_map(rows)
    lists = json.loads(LISTS_PATH.read_text())

    train = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in lists["platform_noise_track"]["train_rows"]]
    holdout = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in lists["platform_noise_track"]["holdout_rows"]]

    audio_cache: dict[str, np.ndarray] = {}
    fmap: dict[str, dict[str, float]] = {}
    for item in train + holdout:
        sid = item.row["source_session_id"]
        if sid not in audio_cache:
            audio_cache[sid] = decode_audio_mono(
                Path(str(item.row["source_session_root"])) / "web" / "session_source_review.mp4",
                SAMPLE_RATE,
            )
        signal = audio_cache[sid]
        start = max(0.0, to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, to_float(item.row.get("event_window_end_seconds")))
        s0 = int(round(start * SAMPLE_RATE))
        s1 = int(round(end * SAMPLE_RATE))
        fmap[item.row_key] = extract_features(signal[s0:s1], sample_rate=SAMPLE_RATE)

    def vec(item: RowRef) -> list[float]:
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

    x_train = np.asarray([vec(item) for item in train], dtype=np.float64)
    y_train = np.asarray([1.0 if item.label == "platform_dive" else 0.0 for item in train], dtype=np.float64)
    x_hold = np.asarray([vec(item) for item in holdout], dtype=np.float64)
    y_hold = np.asarray([1.0 if item.label == "platform_dive" else 0.0 for item in holdout], dtype=np.float64)

    mean, std = standardize_fit(x_train)
    x_train = standardize_apply(x_train, mean, std)
    x_hold = standardize_apply(x_hold, mean, std)
    model = train_binary_logreg(x_train, y_train)
    scores = predict_binary_logreg(model, x_hold)
    auc = binary_auc(y_hold, scores)

    platform_scores = scores[y_hold == 1.0]
    noise_scores = scores[y_hold == 0.0]
    pmin, pmax = float(np.min(platform_scores)), float(np.max(platform_scores))
    nmin, nmax = float(np.min(noise_scores)), float(np.max(noise_scores))
    overlap_lo = max(pmin, nmin)
    overlap_hi = min(pmax, nmax)
    overlap = [overlap_lo, overlap_hi] if overlap_lo <= overlap_hi else None

    thresholds = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]
    threshold_results = [metrics_at_threshold(y_hold, scores, t) for t in thresholds]

    feasible = [
        row
        for row in threshold_results
        if row["platform_dive_recall"] >= 0.75
        and row["macro_f1"] >= 0.50
        and row["noise_to_platform_fp"] < r4_baseline_fp
    ]

    best_candidate = None
    if feasible:
        feasible_sorted = sorted(
            feasible,
            key=lambda row: (-row["macro_f1"], -row["accuracy"], row["noise_to_platform_fp"], row["platform_to_noise_fn"]),
        )
        best_candidate = feasible_sorted[0]
    else:
        # Closest non-feasible point: maximize platform recall, then minimize FP, then maximize macro F1
        non_sorted = sorted(
            threshold_results,
            key=lambda row: (
                -row["platform_dive_recall"],
                row["noise_to_platform_fp"],
                -row["macro_f1"],
                -row["accuracy"],
            ),
        )
        best_candidate = non_sorted[0]

    decision = (
        "PLATFORM_NOISE_R4_OPERATING_POINT_READY_FOR_PHASE5_R7"
        if bool(feasible)
        else "STOP_AND_DOCUMENT_NEAR_PASS"
    )
    rationale = (
        "A threshold exists that preserves platform recall floor and macro F1 while reducing noise->platform FPs below the default r4 operating point."
        if feasible
        else "No threshold satisfies platform recall >= 0.75, macro F1 >= 0.50, and strict FP improvement vs default r4 simultaneously."
    )

    report = {
        "scope": {
            "platform_noise_only": True,
            "frozen_train_rows": len(train),
            "frozen_holdout_rows": len(holdout),
            "classifier_family_unchanged": True,
            "feature_bundle_fixed_to_accepted_r4": True,
            "phase5_rerun_performed": False,
            "springboard_touched": False,
        },
        "part_a_score_distribution": {
            "platform_score_range": [pmin, pmax],
            "noise_score_range": [nmin, nmax],
            "overlap_region": overlap,
            "better_threshold_plausible": overlap is not None and overlap_hi > 0.5 and overlap_lo < 0.5,
        },
        "part_b_threshold_sweep": {
            "auc": auc,
            "r4_default_threshold": 0.5,
            "threshold_results": threshold_results,
        },
        "part_c_guardrail_search": {
            "constraint_platform_recall_floor": 0.75,
            "constraint_macro_f1_floor": 0.50,
            "constraint_noise_fp_improved_vs_r4_baseline": {"baseline_fp_at_0_5": r4_baseline_fp, "required": "< baseline"},
            "feasible_threshold_exists": bool(feasible),
            "feasible_thresholds": feasible,
            "best_threshold_candidate": best_candidate,
        },
        "decision": decision,
        "decision_rationale": rationale,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_md(report)


if __name__ == "__main__":
    main()
