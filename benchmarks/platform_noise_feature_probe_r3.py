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
PROBE2_PATH = Path("outputs/platform_noise_feature_probe_r2.json")
OUT_JSON = Path("outputs/platform_noise_feature_probe_r3.json")
OUT_MD = Path("outputs/platform_noise_feature_probe_r3.md")
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

R3_ONLY_FEATURE = "tonal_noise_penalty"


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


def extract_probe_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    frame = 512
    hop = 128
    eps = 1e-8
    frames = framing(signal, frame=frame, hop=hop)
    if frames.size == 0:
        names = PROBE1_FEATURE_NAMES + R2_FEATURE_NAMES + [R3_ONLY_FEATURE]
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

    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    whistle_band = (freqs >= 1000.0) & (freqs <= 4000.0)
    whistle_band_energy_fraction_post = float(
        np.mean(np.sum(power[:, whistle_band], axis=1) / np.maximum(np.sum(power, axis=1), eps))
    )
    pnorm = power / np.maximum(np.sum(power, axis=1, keepdims=True), eps)
    spectral_entropy = -np.sum(pnorm * np.log(np.maximum(pnorm, eps)), axis=1) / math.log(power.shape[1])
    spectral_entropy_post_mean = float(np.mean(spectral_entropy))

    # Only new r3 feature.
    tonal_noise_penalty = whistle_band_energy_fraction_post * float(np.mean(tonal_peak_fraction))

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
        "tonal_noise_penalty": tonal_noise_penalty,
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


def evaluate(train: list[RowRef], holdout: list[RowRef], fmap: dict[str, dict[str, float]]) -> dict:
    def vec(item: RowRef) -> list[float]:
        row = item.row
        return [
            to_float(row.get("audio_score")),
            to_float(row.get("audio_clip_probability")),
            to_float(row.get("event_anchor_timestamp_seconds")),
            1.0 if row.get("is_false_negative_window") else 0.0,
            *[to_float(fmap[item.row_key][name]) for name in PROBE1_FEATURE_NAMES],
            *[to_float(fmap[item.row_key][name]) for name in R2_FEATURE_NAMES],
            to_float(fmap[item.row_key][R3_ONLY_FEATURE]),
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
    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        cid = row.get("legacy_candidate_id")
        rid = str(cid) if cid else f"row-{counts[sid]:04d}"
        out[f"{sid}::{rid}"] = row
    return out


def write_md(path: Path, report: dict) -> None:
    comp = report["four_way_comparison"]
    b = comp["baseline_current_feature_set"]
    r1 = comp["probe_r1"]
    r2 = comp["probe_r2"]
    r3 = comp["probe_r3_with_tonal_noise_penalty"]
    lines = [
        "# Platform/Noise Feature Probe (r3)",
        "",
        f"- decision: `{report['decision']}`",
        "",
        "## Part A — Feature addition",
        "",
        "- added feature: `tonal_noise_penalty = whistle_band_energy_fraction_post * tonal_peak_fraction_post_mean`",
        "- no other new feature added in r3: `True`",
        "",
        "## Part B — Four-way comparison",
        "",
        "| state | AUC | macro F1 | accuracy | platform recall | noise recall | confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| baseline current feature set | {b['auc']:.4f} | {b['macro_f1']:.4f} | {b['accuracy']:.4f} | {b['platform_dive_recall']:.4f} | {b['noise_or_other_recall']:.4f} | `{b['confusion_matrix']}` |",
        f"| probe r1 | {r1['auc']:.4f} | {r1['macro_f1']:.4f} | {r1['accuracy']:.4f} | {r1['platform_dive_recall']:.4f} | {r1['noise_or_other_recall']:.4f} | `{r1['confusion_matrix']}` |",
        f"| probe r2 | {r2['auc']:.4f} | {r2['macro_f1']:.4f} | {r2['accuracy']:.4f} | {r2['platform_dive_recall']:.4f} | {r2['noise_or_other_recall']:.4f} | `{r2['confusion_matrix']}` |",
        f"| probe r3 (tonal_noise_penalty) | {r3['auc']:.4f} | {r3['macro_f1']:.4f} | {r3['accuracy']:.4f} | {r3['platform_dive_recall']:.4f} | {r3['noise_or_other_recall']:.4f} | `{r3['confusion_matrix']}` |",
        "",
        "## Part C — Error comparison",
        "",
        f"- noise_or_other -> platform_dive counts: `{report['error_comparison']['noise_to_platform_fp_counts']}`",
        f"- platform_dive -> noise_or_other counts: `{report['error_comparison']['platform_to_noise_fn_counts']}`",
        f"- harmful new pattern: `{report['error_comparison']['harmful_new_pattern']}`",
        f"- ranking-gap plausibility note: `{report['error_comparison']['ranking_gap_assessment']}`",
        "",
        "## Part D — Decision",
        "",
        f"- `{report['decision']}`",
        f"- rationale: `{report['decision_rationale']}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    probe1 = json.loads(PROBE1_PATH.read_text())
    probe2 = json.loads(PROBE2_PATH.read_text())
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
        fmap[item.row_key] = extract_probe_features(signal[s0:s1], sample_rate=SAMPLE_RATE)

    r3 = evaluate(train, holdout, fmap)
    baseline = probe1["same_model_comparison"]["baseline_current_feature_set_r4_aligned_proxy"]
    r1 = probe1["same_model_comparison"]["baseline_plus_platform_noise_feature_family"]
    r2 = probe2["three_way_comparison"]["probe_r2_feature_iteration"]

    fp_counts = [
        baseline["confusion_matrix"][1][0],
        r1["confusion_matrix"][1][0],
        r2["confusion_matrix"][1][0],
        r3["confusion_matrix"][1][0],
    ]
    fn_counts = [
        baseline["confusion_matrix"][0][1],
        r1["confusion_matrix"][0][1],
        r2["confusion_matrix"][0][1],
        r3["confusion_matrix"][0][1],
    ]

    harmful = "none_detected"
    if r3["platform_dive_recall"] < r2["platform_dive_recall"]:
        harmful = "platform_recall_dropped_vs_r2"
    elif r3["confusion_matrix"][1][0] > r2["confusion_matrix"][1][0]:
        harmful = "noise_to_platform_fp_increased_vs_r2"
    elif set(r3["false_negative_platform_to_noise_rows"]) != set(r2["false_negative_platform_to_noise_rows"]):
        harmful = "false_negative_identity_shift_without_count_increase"

    ready = (
        r3["auc"] >= 0.66
        and r3["platform_dive_recall"] >= r2["platform_dive_recall"]
        and r3["confusion_matrix"][1][0] <= r2["confusion_matrix"][1][0]
        and harmful == "none_detected"
    )
    decision = "PLATFORM_NOISE_R3_READY_FOR_PHASE5_R7" if ready else "PLATFORM_NOISE_R3_NOT_READY_FOR_PHASE5_R7"
    rationale = (
        "AUC now clears the remaining gap with no recall harm; proceed to full Phase 5 r7 with springboard unchanged and platform/noise on accepted r3 set."
        if ready
        else "AUC/ranking improvement is still insufficient to clear the remaining gap for confident r7 integration."
    )

    report = {
        "scope": {
            "platform_noise_only": True,
            "frozen_train_rows": len(train),
            "frozen_holdout_rows": len(holdout),
            "classifier_family_unchanged": True,
            "phase5_rerun_performed": False,
        },
        "part_a_feature_addition": {
            "added_feature_name": R3_ONLY_FEATURE,
            "added_feature_definition": "tonal_noise_penalty = whistle_band_energy_fraction_post * tonal_peak_fraction_post_mean",
            "only_one_new_feature_added": True,
            "all_r1_r2_features_kept_unchanged": True,
        },
        "four_way_comparison": {
            "baseline_current_feature_set": baseline,
            "probe_r1": r1,
            "probe_r2": r2,
            "probe_r3_with_tonal_noise_penalty": r3,
        },
        "error_comparison": {
            "noise_to_platform_fp_counts": fp_counts,
            "platform_to_noise_fn_counts": fn_counts,
            "ranking_gap_assessment": (
                "improved_enough_for_r7" if r3["auc"] >= 0.66 else "still_below_remaining_auc_gap_target"
            ),
            "harmful_new_pattern": harmful,
        },
        "decision": decision,
        "decision_rationale": rationale,
        "next_step_if_ready": (
            "Run full Phase 5 r7 with springboard unchanged (probe_r1_only) and platform/noise using accepted r3 feature set."
            if ready
            else None
        ),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_md(OUT_MD, report)


if __name__ == "__main__":
    main()
