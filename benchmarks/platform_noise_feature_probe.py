from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_MANIFEST = Path("outputs/event_window_manifest_preview.jsonl")
DEFAULT_LISTS = Path("outputs/phase5_regime_manifest_lists.json")
DEFAULT_JSON = Path("outputs/platform_noise_feature_probe.json")
DEFAULT_MD = Path("outputs/platform_noise_feature_probe.md")
DEFAULT_SAMPLE_RATE = 16_000


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class RowRef:
    row_key: str
    label: str
    row: dict


def row_key_map(rows: list[dict]) -> dict[str, dict]:
    by_key: dict[str, dict] = {}
    counters: dict[str, int] = {}
    for row in rows:
        session_id = str(row["source_session_id"])
        counters[session_id] = counters.get(session_id, 0) + 1
        candidate = row.get("legacy_candidate_id")
        row_id = str(candidate) if candidate else f"row-{counters[session_id]:04d}"
        key = f"{session_id}::{row_id}"
        by_key[key] = row
    return by_key


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


def extract_platform_noise_audio_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    frame = 512
    hop = 128
    eps = 1e-8
    frames = framing(signal, frame=frame, hop=hop)
    if frames.size == 0:
        return {name: 0.0 for name in AUDIO_FEATURE_NAMES}
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

    # 120 ms and 600 ms windows after peak for decay contrast.
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
    spectral_flatness_post_mean = float(np.mean(spectral_flatness))
    tonal_peak_fraction_post_mean = float(np.mean(np.max(power, axis=1) / np.maximum(np.sum(power, axis=1), eps)))

    return {
        "impact_peak_to_window_rms_ratio": peak_val / max(mean_val, eps),
        "impact_peak_prominence_db": 20.0 * math.log10(max(peak_val, eps) / max(med_val, eps)),
        "transient_peak_count": transient_peak_count,
        "inter_peak_interval_cv": interval_cv,
        "post_impact_early_to_late_rms_ratio": post_impact_early_to_late_rms_ratio,
        "tail_half_life_ms": tail_half_life_ms,
        "spectral_flatness_post_mean": spectral_flatness_post_mean,
        "tonal_peak_fraction_post_mean": tonal_peak_fraction_post_mean,
    }


AUDIO_FEATURE_NAMES = [
    "impact_peak_to_window_rms_ratio",
    "impact_peak_prominence_db",
    "transient_peak_count",
    "inter_peak_interval_cv",
    "post_impact_early_to_late_rms_ratio",
    "tail_half_life_ms",
    "spectral_flatness_post_mean",
    "tonal_peak_fraction_post_mean",
]


def baseline_feature_vector(row: dict) -> list[float]:
    return [
        to_float(row.get("audio_score")),
        to_float(row.get("audio_clip_probability")),
        to_float(row.get("event_anchor_timestamp_seconds")),
        1.0 if row.get("is_false_negative_window") else 0.0,
    ]


def augmented_feature_vector(row: dict, extra: dict[str, float]) -> list[float]:
    return baseline_feature_vector(row) + [to_float(extra.get(name)) for name in AUDIO_FEATURE_NAMES]


def fit_and_eval(
    train: list[RowRef],
    holdout: list[RowRef],
    vector_fn,
) -> dict:
    x_train = np.asarray([vector_fn(item.row) for item in train], dtype=np.float64)
    y_train = np.asarray([1.0 if item.label == "platform_dive" else 0.0 for item in train], dtype=np.float64)
    x_hold = np.asarray([vector_fn(item.row) for item in holdout], dtype=np.float64)
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

    def prf_pos(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    _, rec_platform, f1_platform = prf_pos(tp, fp, fn)
    _, rec_noise, f1_noise = prf_pos(tn, fn, fp)
    macro_f1 = float((f1_platform + f1_noise) / 2.0)

    fp_rows = [item.row_key for item, p in zip(holdout, pred.tolist()) if item.label == "noise_or_other" and p == 1.0]
    fn_rows = [item.row_key for item, p in zip(holdout, pred.tolist()) if item.label == "platform_dive" and p == 0.0]

    return {
        "auc": auc,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "confusion_matrix": [[tp, fn], [fp, tn]],
        "platform_dive_recall": rec_platform,
        "noise_or_other_recall": rec_noise,
        "false_positive_noise_to_platform_rows": fp_rows,
        "false_negative_platform_to_noise_rows": fn_rows,
    }


def write_markdown(path: Path, report: dict) -> None:
    base = report["same_model_comparison"]["baseline_current_feature_set_r4_aligned_proxy"]
    aug = report["same_model_comparison"]["baseline_plus_platform_noise_feature_family"]
    lines = [
        "# Platform/Noise Feature Probe",
        "",
        f"- decision: `{report['decision']}`",
        "",
        "## Part A — Feature extraction report",
        "",
        f"- new feature count: `{len(report['feature_extraction']['new_features'])}`",
        f"- extraction source: `{report['feature_extraction']['source_type']}`",
        "",
        "### Implemented features",
        "",
    ]
    for item in report["feature_extraction"]["new_features"]:
        lines.append(f"- `{item['name']}` ({item['group']}) — {item['definition']}")
    lines.extend(
        [
            "",
            "## Part B — Same-model comparison",
            "",
            "| setup | AUC | macro F1 | accuracy | confusion matrix | platform recall | noise recall |",
            "|---|---:|---:|---:|---|---:|---:|",
            f"| baseline (current r4-aligned proxy) | {base['auc']:.4f} | {base['macro_f1']:.4f} | {base['accuracy']:.4f} | `{base['confusion_matrix']}` | {base['platform_dive_recall']:.4f} | {base['noise_or_other_recall']:.4f} |",
            f"| baseline + new platform/noise features | {aug['auc']:.4f} | {aug['macro_f1']:.4f} | {aug['accuracy']:.4f} | `{aug['confusion_matrix']}` | {aug['platform_dive_recall']:.4f} | {aug['noise_or_other_recall']:.4f} |",
            "",
            "## Part C — Error reduction",
            "",
            f"- noise_or_other -> platform_dive: `{base['confusion_matrix'][1][0]} -> {aug['confusion_matrix'][1][0]}`",
            f"- platform_dive -> noise_or_other: `{base['confusion_matrix'][0][1]} -> {aug['confusion_matrix'][0][1]}`",
            f"- new failure pattern: `{report['error_reduction']['new_failure_pattern']}`",
            "",
            "## Part D — Interpretation",
            "",
            f"- verdict: `{report['decision']}`",
            f"- recommendation: `{report['interpretation']['next_step']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Platform/noise-only feature probe on frozen r4 rows.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lists", type=Path, default=DEFAULT_LISTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    by_key = row_key_map(rows)
    lists = json.loads(args.lists.read_text())
    track = lists["platform_noise_track"]
    train_rows = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in track["train_rows"]]
    holdout_rows = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in track["holdout_rows"]]

    # Decode session audio once and derive per-row platform/noise audio features.
    session_audio_cache: dict[str, np.ndarray] = {}
    extra_by_row: dict[str, dict[str, float]] = {}
    all_rows = train_rows + holdout_rows
    for item in all_rows:
        session_id = str(item.row["source_session_id"])
        if session_id not in session_audio_cache:
            source_root = Path(str(item.row["source_session_root"]))
            audio_video_path = source_root / "web" / "session_source_review.mp4"
            session_audio_cache[session_id] = decode_audio_mono(audio_video_path, sample_rate=args.sample_rate)
        signal = session_audio_cache[session_id]
        start = max(0.0, to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, to_float(item.row.get("event_window_end_seconds")))
        start_idx = int(round(start * args.sample_rate))
        end_idx = int(round(end * args.sample_rate))
        segment = signal[start_idx:end_idx] if end_idx > start_idx else np.asarray([], dtype=np.float32)
        extra_by_row[item.row_key] = extract_platform_noise_audio_features(segment, sample_rate=args.sample_rate)

    baseline = fit_and_eval(train_rows, holdout_rows, vector_fn=baseline_feature_vector)
    def fit_and_eval_augmented(train: list[RowRef], holdout: list[RowRef]) -> dict:
        x_train = np.asarray([augmented_feature_vector(item.row, extra_by_row[item.row_key]) for item in train], dtype=np.float64)
        y_train = np.asarray([1.0 if item.label == "platform_dive" else 0.0 for item in train], dtype=np.float64)
        x_hold = np.asarray([augmented_feature_vector(item.row, extra_by_row[item.row_key]) for item in holdout], dtype=np.float64)
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

        def prf_pos(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
            precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
            recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            return precision, recall, f1

        _, rec_platform, f1_platform = prf_pos(tp, fp, fn)
        _, rec_noise, f1_noise = prf_pos(tn, fn, fp)
        macro_f1 = float((f1_platform + f1_noise) / 2.0)

        fp_rows = [item.row_key for item, p in zip(holdout, pred.tolist()) if item.label == "noise_or_other" and p == 1.0]
        fn_rows = [item.row_key for item, p in zip(holdout, pred.tolist()) if item.label == "platform_dive" and p == 0.0]
        return {
            "auc": auc,
            "macro_f1": macro_f1,
            "accuracy": acc,
            "confusion_matrix": [[tp, fn], [fp, tn]],
            "platform_dive_recall": rec_platform,
            "noise_or_other_recall": rec_noise,
            "false_positive_noise_to_platform_rows": fp_rows,
            "false_negative_platform_to_noise_rows": fn_rows,
        }

    augmented = fit_and_eval_augmented(train_rows, holdout_rows)

    fp_before = baseline["confusion_matrix"][1][0]
    fp_after = augmented["confusion_matrix"][1][0]
    fn_before = baseline["confusion_matrix"][0][1]
    fn_after = augmented["confusion_matrix"][0][1]
    decision = (
        "PLATFORM_NOISE_FEATURE_FAMILY_PROMISING"
        if (fp_after < fp_before and augmented["macro_f1"] >= baseline["macro_f1"])
        else "PLATFORM_NOISE_FEATURE_FAMILY_NOT_PROMISING"
    )

    new_error_pattern = "none_detected"
    if fn_after > fn_before and fp_after <= fp_before:
        new_error_pattern = "platform_false_negatives_increased"
    if fp_after > fp_before:
        new_error_pattern = "noise_false_positives_increased"

    report = {
        "scope": {
            "frozen_train_rows": len(train_rows),
            "frozen_holdout_rows": len(holdout_rows),
            "classifier_family": "numpy_logistic_regression_binary",
            "phase5_rerun_performed": False,
        },
        "feature_extraction": {
            "source_type": "additional_offline_audio_window_extraction_from_session_source_review_mp4",
            "baseline_features_r4_aligned_proxy": [
                "audio_score",
                "audio_clip_probability",
                "event_anchor_timestamp_seconds",
                "is_false_negative_window",
            ],
            "new_features": [
                {
                    "name": "impact_peak_to_window_rms_ratio",
                    "group": "impact_compactness",
                    "definition": "max frame RMS within event window divided by mean frame RMS."
                },
                {
                    "name": "impact_peak_prominence_db",
                    "group": "impact_compactness",
                    "definition": "20*log10(peak RMS / median RMS) over the event window."
                },
                {
                    "name": "transient_peak_count",
                    "group": "isolation_vs_diffuse_clutter",
                    "definition": "count of local RMS peaks above mean + 0.5*std."
                },
                {
                    "name": "inter_peak_interval_cv",
                    "group": "isolation_vs_diffuse_clutter",
                    "definition": "coefficient of variation of spacing between transient peaks."
                },
                {
                    "name": "post_impact_early_to_late_rms_ratio",
                    "group": "broadband_splash_decay",
                    "definition": "mean RMS in 0-120 ms after impact peak divided by mean RMS in 120-600 ms."
                },
                {
                    "name": "tail_half_life_ms",
                    "group": "broadband_splash_decay",
                    "definition": "time after impact peak for RMS envelope to drop to 50% of peak."
                },
                {
                    "name": "spectral_flatness_post_mean",
                    "group": "tonal_vs_broadband_texture",
                    "definition": "mean spectral flatness in post-impact frames (~400 ms)."
                },
                {
                    "name": "tonal_peak_fraction_post_mean",
                    "group": "tonal_vs_broadband_texture",
                    "definition": "mean(max-bin power / total power) in post-impact frames."
                },
            ],
        },
        "same_model_comparison": {
            "baseline_current_feature_set_r4_aligned_proxy": baseline,
            "baseline_plus_platform_noise_feature_family": augmented,
        },
        "error_reduction": {
            "noise_or_other_to_platform_before_after": [fp_before, fp_after],
            "platform_dive_to_noise_before_after": [fn_before, fn_after],
            "new_failure_pattern": new_error_pattern,
        },
        "interpretation": {
            "meaningful_signal": decision == "PLATFORM_NOISE_FEATURE_FAMILY_PROMISING",
            "next_step": (
                "carry_feature_family_into_next_phase5_revision"
                if decision == "PLATFORM_NOISE_FEATURE_FAMILY_PROMISING"
                else "one_more_platform_noise_feature_iteration_first"
            ),
        },
        "decision": decision,
    }

    args.output_json.write_text(json.dumps(report, indent=2))
    write_markdown(args.output_md, report)


if __name__ == "__main__":
    main()
