from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MANIFEST_PATH = Path("outputs/event_window_manifest_preview.jsonl")
LISTS_PATH = Path("outputs/phase5_regime_manifest_lists.json")
PROBE1_PATH = Path("outputs/platform_noise_feature_probe.json")
OUT_JSON = Path("outputs/platform_noise_feature_probe_r2.json")
OUT_MD = Path("outputs/platform_noise_feature_probe_r2.md")
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

R2_REFINEMENT_FEATURES = [
    "whistle_band_energy_fraction_post",
    "spectral_entropy_post_mean",
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
    cmd = [
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
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def framing(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(signal) < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(signal)] = signal
        return padded.reshape(1, frame)
    n_frames = 1 + (len(signal) - frame) // hop
    shape = (n_frames, frame)
    strides = (signal.strides[0] * hop, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides, writeable=False)


def local_peak_indices(values: np.ndarray, threshold: float) -> np.ndarray:
    if len(values) < 3:
        return np.asarray([], dtype=int)
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1] and values[i] >= threshold:
            peaks.append(i)
    return np.asarray(peaks, dtype=int)


def extract_probe_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    frame = 512
    hop = 128
    eps = 1e-8
    frames = framing(signal, frame=frame, hop=hop)
    if frames.size == 0:
        return {name: 0.0 for name in PROBE1_FEATURE_NAMES + R2_REFINEMENT_FEATURES}

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
    frames_600 = max(frames_120 + 1, int(round(0.600 * sample_rate / hop)))
    early = rms[peak_idx + 1 : peak_idx + 1 + frames_120]
    late = rms[peak_idx + 1 + frames_120 : peak_idx + 1 + frames_600]
    early_mean = float(np.mean(early)) if len(early) else 0.0
    late_mean = float(np.mean(late)) if len(late) else 0.0
    post_impact_early_to_late_rms_ratio = early_mean / max(late_mean, eps)

    half_level = 0.5 * peak_val
    tail = rms[peak_idx + 1 :]
    half_idx = next((i for i, value in enumerate(tail, start=1) if value <= half_level), len(tail))
    tail_half_life_ms = float(half_idx * hop * 1000.0 / sample_rate)

    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    post_frames = spectrum[peak_idx : peak_idx + max(1, int(round(0.400 * sample_rate / hop)))]
    if len(post_frames) == 0:
        post_frames = spectrum[max(0, peak_idx - 1) : peak_idx + 1]
    power = np.maximum(post_frames, eps)
    spectral_flatness = np.exp(np.mean(np.log(power), axis=1)) / np.maximum(np.mean(power, axis=1), eps)
    tonal_peak_fraction = np.max(power, axis=1) / np.maximum(np.sum(power, axis=1), eps)

    # r2 refinement: stronger tonal-vs-broadband separation for whistle-like residual FPs.
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    whistle_band = (freqs >= 1000.0) & (freqs <= 4000.0)
    whistle_band_energy_fraction_post = float(
        np.mean(np.sum(power[:, whistle_band], axis=1) / np.maximum(np.sum(power, axis=1), eps))
    )
    pnorm = power / np.maximum(np.sum(power, axis=1, keepdims=True), eps)
    spectral_entropy = -np.sum(pnorm * np.log(np.maximum(pnorm, eps)), axis=1) / math.log(power.shape[1])
    spectral_entropy_post_mean = float(np.mean(spectral_entropy))

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
        "spectral_entropy_post_mean": spectral_entropy_post_mean,
    }


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def train_binary_logreg(x: np.ndarray, y: np.ndarray, epochs: int = 1000, lr: float = 0.1, l2: float = 0.01) -> dict:
    n_samples, n_features = x.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        error = probs - y
        grad_w = (x.T @ error) / n_samples + l2 * weights
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


def evaluate(train: list[RowRef], holdout: list[RowRef], feature_map: dict[str, dict[str, float]]) -> dict:
    def vec(item: RowRef) -> list[float]:
        row = item.row
        return [
            to_float(row.get("audio_score")),
            to_float(row.get("audio_clip_probability")),
            to_float(row.get("event_anchor_timestamp_seconds")),
            1.0 if row.get("is_false_negative_window") else 0.0,
            *[to_float(feature_map[item.row_key][name]) for name in PROBE1_FEATURE_NAMES],
            *[to_float(feature_map[item.row_key][name]) for name in R2_REFINEMENT_FEATURES],
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
    pred = np.asarray([1.0 if score >= 0.5 else 0.0 for score in scores], dtype=np.float64)

    tp = int(np.sum((y_hold == 1.0) & (pred == 1.0)))
    fn = int(np.sum((y_hold == 1.0) & (pred == 0.0)))
    fp = int(np.sum((y_hold == 0.0) & (pred == 1.0)))
    tn = int(np.sum((y_hold == 0.0) & (pred == 0.0)))
    acc = float((tp + tn) / len(y_hold)) if len(y_hold) else 0.0
    auc = binary_auc(y_hold, scores)

    def f1_for(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    _, rec_platform, f1_platform = f1_for(tp, fp, fn)
    _, rec_noise, f1_noise = f1_for(tn, fn, fp)
    macro_f1 = float((f1_platform + f1_noise) / 2.0)

    fp_rows = [item.row_key for item, p in zip(holdout, pred.tolist()) if item.label == "noise_or_other" and p == 1.0]
    fn_rows = [item.row_key for item, p in zip(holdout, pred.tolist()) if item.label == "platform_dive" and p == 0.0]
    return {
        "auc": auc,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "platform_dive_recall": rec_platform,
        "noise_or_other_recall": rec_noise,
        "confusion_matrix": [[tp, fn], [fp, tn]],
        "false_positive_noise_to_platform_rows": fp_rows,
        "false_negative_platform_to_noise_rows": fn_rows,
    }


def row_key_map(rows: list[dict]) -> dict[str, dict]:
    by_key: dict[str, dict] = {}
    counters: dict[str, int] = {}
    for row in rows:
        session_id = str(row["source_session_id"])
        counters[session_id] = counters.get(session_id, 0) + 1
        candidate = row.get("legacy_candidate_id")
        row_id = str(candidate) if candidate else f"row-{counters[session_id]:04d}"
        by_key[f"{session_id}::{row_id}"] = row
    return by_key


def write_md(path: Path, report: dict) -> None:
    states = report["three_way_comparison"]
    base = states["baseline_current_feature_set"]
    p1 = states["probe_r1_feature_family"]
    p2 = states["probe_r2_feature_iteration"]
    lines = [
        "# Platform/Noise Feature Probe (r2)",
        "",
        f"- decision: `{report['decision']}`",
        "",
        "## Part A — Review of first feature probe",
        "",
        f"- probe1 added features: `{[f['name'] for f in report['part_a_review_probe1']['probe1_added_features']]}`",
        f"- probe1 improvements: `{report['part_a_review_probe1']['improvements_from_baseline']}`",
        f"- probe1 remaining failures: `{report['part_a_review_probe1']['remaining_failures']}`",
        f"- residual FP pattern: `{report['part_a_review_probe1']['residual_false_positive_pattern']}`",
        "",
        "## Part B/C — Three-way comparison",
        "",
        "| state | AUC | macro F1 | accuracy | platform recall | noise recall | confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| baseline current feature set | {base['auc']:.4f} | {base['macro_f1']:.4f} | {base['accuracy']:.4f} | {base['platform_dive_recall']:.4f} | {base['noise_or_other_recall']:.4f} | `{base['confusion_matrix']}` |",
        f"| first probe feature family | {p1['auc']:.4f} | {p1['macro_f1']:.4f} | {p1['accuracy']:.4f} | {p1['platform_dive_recall']:.4f} | {p1['noise_or_other_recall']:.4f} | `{p1['confusion_matrix']}` |",
        f"| second feature iteration (r2) | {p2['auc']:.4f} | {p2['macro_f1']:.4f} | {p2['accuracy']:.4f} | {p2['platform_dive_recall']:.4f} | {p2['noise_or_other_recall']:.4f} | `{p2['confusion_matrix']}` |",
        "",
        "## Part D — Error progression",
        "",
        f"- noise_or_other -> platform_dive: `{report['error_comparison']['noise_to_platform_fp_counts']}`",
        f"- platform_dive -> noise_or_other: `{report['error_comparison']['platform_to_noise_fn_counts']}`",
        f"- new harmful pattern: `{report['error_comparison']['new_harmful_pattern']}`",
        "",
        "## Part E — Decision",
        "",
        f"- `{report['decision']}`",
        f"- rationale: `{report['decision_rationale']}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    probe1 = json.loads(PROBE1_PATH.read_text())
    rows = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    by_key = row_key_map(rows)
    lists = json.loads(LISTS_PATH.read_text())
    track = lists["platform_noise_track"]
    train = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in track["train_rows"]]
    holdout = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in track["holdout_rows"]]

    # Extract per-row probe features from frozen windows/audio.
    audio_cache: dict[str, np.ndarray] = {}
    feature_map: dict[str, dict[str, float]] = {}
    for item in train + holdout:
        sid = item.row["source_session_id"]
        if sid not in audio_cache:
            source_root = Path(str(item.row["source_session_root"]))
            audio_path = source_root / "web" / "session_source_review.mp4"
            audio_cache[sid] = decode_audio_mono(audio_path, sample_rate=SAMPLE_RATE)
        signal = audio_cache[sid]
        start = max(0.0, to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, to_float(item.row.get("event_window_end_seconds")))
        start_idx = int(round(start * SAMPLE_RATE))
        end_idx = int(round(end * SAMPLE_RATE))
        segment = signal[start_idx:end_idx] if end_idx > start_idx else np.asarray([], dtype=np.float32)
        feature_map[item.row_key] = extract_probe_features(segment, sample_rate=SAMPLE_RATE)

    baseline = probe1["same_model_comparison"]["baseline_current_feature_set_r4_aligned_proxy"]
    probe1_state = probe1["same_model_comparison"]["baseline_plus_platform_noise_feature_family"]
    probe2_state = evaluate(train, holdout, feature_map=feature_map)

    fp_counts = [
        baseline["confusion_matrix"][1][0],
        probe1_state["confusion_matrix"][1][0],
        probe2_state["confusion_matrix"][1][0],
    ]
    fn_counts = [
        baseline["confusion_matrix"][0][1],
        probe1_state["confusion_matrix"][0][1],
        probe2_state["confusion_matrix"][0][1],
    ]

    harmful = "none_detected"
    if probe2_state["platform_dive_recall"] < probe1_state["platform_dive_recall"]:
        harmful = "platform_recall_dropped_vs_probe1"
    elif probe2_state["confusion_matrix"][1][0] > probe1_state["confusion_matrix"][1][0]:
        harmful = "noise_false_positives_increased_vs_probe1"
    elif set(probe2_state["false_negative_platform_to_noise_rows"]) != set(
        probe1_state["false_negative_platform_to_noise_rows"]
    ):
        harmful = "false_negative_identity_shift_without_count_increase"

    decision = (
        "PLATFORM_NOISE_FEATURE_R2_READY_FOR_PHASE5_INTEGRATION"
        if (
            probe2_state["platform_dive_recall"] >= probe1_state["platform_dive_recall"]
            and probe2_state["confusion_matrix"][1][0] <= probe1_state["confusion_matrix"][1][0]
            and probe2_state["auc"] >= probe1_state["auc"]
        )
        else "PLATFORM_NOISE_FEATURE_R2_NEEDS_MORE_REFINEMENT"
    )
    rationale = (
        "r2 keeps platform recall, keeps FP count from regressing, and improves rank-order separation (AUC)."
        if decision == "PLATFORM_NOISE_FEATURE_R2_READY_FOR_PHASE5_INTEGRATION"
        else "r2 does not sufficiently improve residual false positives without tradeoff."
    )

    report = {
        "scope": {
            "platform_noise_only": True,
            "frozen_train_rows": len(train),
            "frozen_holdout_rows": len(holdout),
            "classifier_family_unchanged": True,
            "phase5_rerun_performed": False,
        },
        "part_a_review_probe1": {
            "probe1_added_features": probe1["feature_extraction"]["new_features"],
            "improvements_from_baseline": {
                "auc": [baseline["auc"], probe1_state["auc"]],
                "macro_f1": [baseline["macro_f1"], probe1_state["macro_f1"]],
                "noise_recall": [baseline["noise_or_other_recall"], probe1_state["noise_or_other_recall"]],
                "noise_to_platform_fp": [baseline["confusion_matrix"][1][0], probe1_state["confusion_matrix"][1][0]],
            },
            "remaining_failures": {
                "residual_noise_to_platform_fp_count": probe1_state["confusion_matrix"][1][0],
                "residual_fp_rows": probe1_state["false_positive_noise_to_platform_rows"],
            },
            "residual_false_positive_pattern": "still concentrated in handling_noise/voice_whistle-like rows",
        },
        "part_b_r2_refinement": {
            "refinement_focus": "tonal_vs_broadband_discrimination",
            "new_features": [
                {
                    "name": "whistle_band_energy_fraction_post",
                    "definition": "post-impact energy fraction in 1-4 kHz whistle-like band.",
                },
                {
                    "name": "spectral_entropy_post_mean",
                    "definition": "normalized spectral entropy over post-impact frames.",
                },
            ],
            "bounded_change_only": True,
        },
        "three_way_comparison": {
            "baseline_current_feature_set": baseline,
            "probe_r1_feature_family": probe1_state,
            "probe_r2_feature_iteration": probe2_state,
        },
        "error_comparison": {
            "noise_to_platform_fp_counts": fp_counts,
            "platform_to_noise_fn_counts": fn_counts,
            "new_harmful_pattern": harmful,
        },
        "decision": decision,
        "decision_rationale": rationale,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_md(OUT_MD, report)


if __name__ == "__main__":
    main()
