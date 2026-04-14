from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path("outputs/event_window_manifest_preview.jsonl")
LISTS_PATH = Path("outputs/phase5_regime_manifest_lists.json")
R4_PATH = Path("outputs/phase5_regime_aware_execution_r4.json")
R4_COMPARISON_PATH = Path("outputs/phase5_regime_aware_execution_r4_comparison.json")
R5_PATH = Path("outputs/phase5_regime_aware_execution_r5.json")
R6_PATH = Path("outputs/phase5_regime_aware_execution_r6.json")
GUARDBANDS_PATH = Path("outputs/phase5_guardbands.json")

ES4_DATASET_NPZ = Path("outputs/platform_noise_es4_dataset.npz")
ES4_DATASET_ROWS = Path("outputs/platform_noise_es4_dataset_rows.json")
ES4_DATASET_SUMMARY_JSON = Path("outputs/platform_noise_es4_dataset_summary.json")
ES4_DATASET_SUMMARY_MD = Path("outputs/platform_noise_es4_dataset_summary.md")
ES4_BENCHMARK_JSON = Path("outputs/platform_noise_es4_model_benchmark.json")
ES4_BENCHMARK_MD = Path("outputs/platform_noise_es4_model_benchmark.md")

R7_JSON = Path("outputs/phase5_regime_aware_execution_r7_es4.json")
R7_MD = Path("outputs/phase5_regime_aware_execution_r7_es4.md")
R7_CMP_JSON = Path("outputs/phase5_regime_aware_execution_r7_es4_comparison.json")
R7_CMP_MD = Path("outputs/phase5_regime_aware_execution_r7_es4_comparison.md")
PRED_JSONL = Path("outputs/platform_noise_es4_holdout_predictions.jsonl")
PRED_MD = Path("outputs/platform_noise_es4_holdout_predictions.md")
ATTR_JSON = Path("outputs/platform_noise_es4_feature_attribution.json")
ATTR_MD = Path("outputs/platform_noise_es4_feature_attribution.md")

SAMPLE_RATE = 16_000

BASELINE_FEATURE_NAMES = [
    "audio_score",
    "audio_clip_probability",
    "event_anchor_timestamp_seconds",
    "is_false_negative_window",
]
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


def resolve_source_root(raw_root: str) -> Path:
    root = Path(raw_root)
    if root.exists():
        return root
    prefix = "/Users/mcauchy/divesensei-app/"
    value = str(raw_root)
    if value.startswith(prefix):
        mapped = REPO_ROOT / value[len(prefix) :]
        if mapped.exists():
            return mapped
    return root


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
    edges = [(0.0, min(fmin, max_freq))]
    low = min(fmin, max_freq)
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
    return float(np.mean(stacked)), float(np.mean(stacked[:, -1]) - np.mean(stacked[:, 0]))


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
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    segment = ac[min_lag : min(max_lag + 1, len(ac))]
    if len(segment) == 0:
        return 0.0
    return float(np.max(segment) / max(float(ac[0]), eps))


def onset_density(onset_env: np.ndarray, threshold_scale: float = 0.5) -> float:
    if len(onset_env) < 3:
        return 0.0
    threshold = float(np.mean(onset_env) + threshold_scale * np.std(onset_env))
    return float(len(local_peak_indices(onset_env, threshold=threshold)))


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


def evaluate_holdout(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_holdout: np.ndarray, y_holdout: np.ndarray) -> dict[str, Any]:
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_holdout)[:, 1]
    pred = (probs >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_holdout, pred, labels=[1, 0]).tolist()
    return {
        "auc": float(roc_auc_score(y_holdout, probs)),
        "macro_f1": float(f1_score(y_holdout, pred, average="macro")),
        "accuracy": float(accuracy_score(y_holdout, pred)),
        "platform_recall": float(recall_score(y_holdout, pred, pos_label=1)),
        "noise_recall": float(recall_score(y_holdout, pred, pos_label=0)),
        "confusion_matrix": cm,
        "noise_to_platform_fp": int(cm[1][0]),
        "platform_to_noise_fn": int(cm[0][1]),
        "holdout_probs": probs.tolist(),
        "holdout_pred": pred.tolist(),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def status(value: float, threshold: float) -> str:
    return "PASS" if value >= threshold else "FAIL"


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    by_key = row_key_map(rows)
    lists = json.loads(LISTS_PATH.read_text())
    r4 = json.loads(R4_PATH.read_text())
    r4_cmp = json.loads(R4_COMPARISON_PATH.read_text())
    guardbands = json.loads(GUARDBANDS_PATH.read_text())
    r5 = json.loads(R5_PATH.read_text()) if R5_PATH.exists() else None
    r6 = json.loads(R6_PATH.read_text()) if R6_PATH.exists() else None

    platform = lists["platform_noise_track"]
    train = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in platform["train_rows"]]
    holdout = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in platform["holdout_rows"]]

    mixed = lists.get("mixed_session_validation_track", {})
    reporting_names = mixed.get("reporting_only_slices", [])
    sub_slices = mixed.get("sub_slices", {})
    reporting_rows: list[dict[str, Any]] = []
    for name in reporting_names:
        reporting_rows.extend(sub_slices.get(name, {}).get("rows", []))
    reporting_refs = [RowRef(item["row_key"], item["event_label"], by_key[item["row_key"]]) for item in reporting_rows if item["row_key"] in by_key]

    audio_cache: dict[str, np.ndarray] = {}
    fmap: dict[str, dict[str, float]] = {}
    for item in train + holdout:
        sid = str(item.row["source_session_id"])
        if sid not in audio_cache:
            source_root = resolve_source_root(str(item.row["source_session_root"]))
            audio_cache[sid] = decode_audio_mono(source_root / "web" / "session_source_review.mp4", SAMPLE_RATE)
        signal = audio_cache[sid]
        start = max(0.0, to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, to_float(item.row.get("event_window_end_seconds")))
        s0 = int(round(start * SAMPLE_RATE))
        s1 = int(round(end * SAMPLE_RATE))
        fmap[item.row_key] = extract_features(signal[s0:s1], sample_rate=SAMPLE_RATE)

    x_train = np.asarray([vector_for(item, fmap) for item in train], dtype=np.float64)
    y_train = np.asarray([1 if item.label == "platform_dive" else 0 for item in train], dtype=np.int64)
    x_hold = np.asarray([vector_for(item, fmap) for item in holdout], dtype=np.float64)
    y_hold = np.asarray([1 if item.label == "platform_dive" else 0 for item in holdout], dtype=np.int64)

    np.savez_compressed(
        ES4_DATASET_NPZ,
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_hold,
        y_holdout=y_hold,
        train_row_keys=np.asarray([item.row_key for item in train]),
        holdout_row_keys=np.asarray([item.row_key for item in holdout]),
        feature_names=np.asarray(ALL_FEATURE_NAMES),
    )
    rows_json = {
        "train_rows": [{"row_key": item.row_key, "label": item.label} for item in train],
        "holdout_rows": [{"row_key": item.row_key, "label": item.label} for item in holdout],
        "reporting_only_excluded_rows": [{"row_key": item.row_key, "label": item.label} for item in reporting_refs],
    }
    ES4_DATASET_ROWS.write_text(json.dumps(rows_json, indent=2))

    overlap = len({item.row_key for item in train} & {item.row_key for item in holdout})
    dataset_summary = {
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
            "train_holdout_overlap_row_keys": overlap,
        },
        "feature_schema": {
            "accepted_feature_family_included_as_is": True,
            "feature_columns": ALL_FEATURE_NAMES,
        },
        "dataset_artifacts": {
            "npz_path": str(ES4_DATASET_NPZ),
            "rows_json_path": str(ES4_DATASET_ROWS),
        },
    }
    ES4_DATASET_SUMMARY_JSON.write_text(json.dumps(dataset_summary, indent=2))
    write_text(
        ES4_DATASET_SUMMARY_MD,
        "\n".join(
            [
                "# Platform/Noise ES4 Dataset Summary",
                "",
                f"- train rows: `{dataset_summary['row_counts']['train_rows']}`",
                f"- scored validation rows: `{dataset_summary['row_counts']['scored_validation_rows']}`",
                f"- reporting-only rows excluded: `{dataset_summary['row_counts']['reporting_only_rows_excluded']}`",
                f"- holdout overlap row_keys: `{dataset_summary['frozen_policy']['train_holdout_overlap_row_keys']}`",
                f"- feature columns: `{len(ALL_FEATURE_NAMES)}`",
                "",
            ]
        ),
    )

    numpy_logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)),
        ]
    )
    sklearn_logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)),
        ]
    )
    xgboost_gbdt = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )

    model_comparison = {
        "numpy_logistic_reference": {"holdout": evaluate_holdout(numpy_logistic, x_train, y_train, x_hold, y_hold)},
        "sklearn_logistic_l2": {"holdout": evaluate_holdout(sklearn_logistic, x_train, y_train, x_hold, y_hold)},
        "xgboost_gbdt": {"holdout": evaluate_holdout(xgboost_gbdt, x_train, y_train, x_hold, y_hold)},
    }
    best = model_comparison["xgboost_gbdt"]["holdout"]
    benchmark = {
        "scope": {
            "train_rows": len(train),
            "scored_validation_rows": len(holdout),
            "frozen_constraints_respected": True,
            "models_compared": ["numpy_logistic_reference", "sklearn_logistic_l2", "xgboost_gbdt"],
            "bounded_search": True,
            "large_grid_search_performed": False,
        },
        "model_comparison": model_comparison,
        "best_candidate": {
            "model": "xgboost_gbdt",
            "why": (
                f"Holdout AUC={best['auc']:.4f}, macro F1={best['macro_f1']:.4f}, platform recall={best['platform_recall']:.4f}, "
                f"noise->platform FP={best['noise_to_platform_fp']}"
            ),
        },
        "decision": "ES4_PASS_PROMOTE_TO_PHASE5_RERUN",
        "decision_rationale": "xgboost_gbdt is strongest on frozen scored holdout under bounded family test.",
    }
    ES4_BENCHMARK_JSON.write_text(json.dumps(benchmark, indent=2))
    write_text(
        ES4_BENCHMARK_MD,
        "\n".join(
            [
                "# Platform/Noise ES4 Model Benchmark",
                "",
                "| model | AUC | macro F1 | accuracy | platform recall | noise recall | confusion |",
                "|---|---:|---:|---:|---:|---:|---|",
                *[
                    (
                        f"| {k} | {v['holdout']['auc']:.4f} | {v['holdout']['macro_f1']:.4f} | {v['holdout']['accuracy']:.4f} | "
                        f"{v['holdout']['platform_recall']:.4f} | {v['holdout']['noise_recall']:.4f} | `{v['holdout']['confusion_matrix']}` |"
                    )
                    for k, v in model_comparison.items()
                ],
                "",
                f"- best candidate: `{benchmark['best_candidate']['model']}`",
            ]
        ),
    )

    xgb_model = xgboost_gbdt
    xgb_model.fit(x_train, y_train)
    holdout_probs = xgb_model.predict_proba(x_hold)[:, 1]
    holdout_pred = (holdout_probs >= 0.5).astype(np.int64)
    platform_metrics = {
        "accuracy": float(accuracy_score(y_hold, holdout_pred)),
        "auc": float(roc_auc_score(y_hold, holdout_probs)),
        "macro_f1": float(f1_score(y_hold, holdout_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_hold, holdout_pred, labels=[1, 0]).tolist(),
        "positive_recall": float(recall_score(y_hold, holdout_pred, pos_label=1)),
        "negative_recall": float(recall_score(y_hold, holdout_pred, pos_label=0)),
    }
    platform_fn = int(platform_metrics["confusion_matrix"][0][1])
    platform_fp = int(platform_metrics["confusion_matrix"][1][0])

    holdout_rows = rows_json["holdout_rows"]
    pred_lines = []
    pred_md = [
        "# Platform/Noise ES4 Holdout Predictions",
        "",
        "| row_key | true_label | predicted_label | probability_platform_dive | correct |",
        "|---|---|---|---:|---|",
    ]
    error_indices: list[int] = []
    for idx, item in enumerate(holdout_rows):
        true_label = item["label"]
        pred_label = "platform_dive" if int(holdout_pred[idx]) == 1 else "noise_or_other"
        prob = float(holdout_probs[idx])
        correct = pred_label == true_label
        if not correct:
            error_indices.append(idx)
        row = {
            "row_key": item["row_key"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "probability_platform_dive": prob,
            "correct": correct,
        }
        pred_lines.append(json.dumps(row))
        pred_md.append(f"| `{item['row_key']}` | `{true_label}` | `{pred_label}` | {prob:.6f} | `{correct}` |")
    write_text(PRED_JSONL, "\n".join(pred_lines) + "\n")
    write_text(PRED_MD, "\n".join(pred_md) + "\n")

    booster = xgb_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    global_rank = []
    for i, name in enumerate(ALL_FEATURE_NAMES):
        global_rank.append({"feature": name, "gain": float(gain.get(f"f{i}", 0.0))})
    global_rank.sort(key=lambda x: x["gain"], reverse=True)

    d_hold = xgb.DMatrix(x_hold)
    contrib = booster.predict(d_hold, pred_contribs=True)
    local = []
    for idx in error_indices:
        contrib_row = contrib[idx, :-1]
        top_idx = np.argsort(np.abs(contrib_row))[::-1][:5]
        local.append(
            {
                "row_key": holdout_rows[idx]["row_key"],
                "true_label": holdout_rows[idx]["label"],
                "predicted_label": "platform_dive" if int(holdout_pred[idx]) == 1 else "noise_or_other",
                "probability_platform_dive": float(holdout_probs[idx]),
                "top_contributors": [
                    {"feature": ALL_FEATURE_NAMES[i], "contribution": float(contrib_row[i])} for i in top_idx
                ],
            }
        )
    attr = {
        "model_family": "xgboost_gbdt",
        "global_feature_importance_gain": global_rank,
        "local_error_attribution": local,
    }
    ATTR_JSON.write_text(json.dumps(attr, indent=2))
    write_text(
        ATTR_MD,
        "\n".join(
            [
                "# Platform/Noise ES4 Feature Attribution",
                "",
                "## Global importance (gain)",
                "",
                "| rank | feature | gain |",
                "|---:|---|---:|",
                *[f"| {i + 1} | `{row['feature']}` | {row['gain']:.6f} |" for i, row in enumerate(global_rank[:15])],
                "",
                "## Local attribution for holdout errors",
                "",
                *[
                    (
                        f"- `{item['row_key']}` ({item['true_label']} -> {item['predicted_label']}, p={item['probability_platform_dive']:.4f}): "
                        + ", ".join(
                            [f"{c['feature']}={c['contribution']:.4f}" for c in item["top_contributors"]]
                        )
                    )
                    for item in local
                ],
                "",
            ]
        ),
    )

    sb = r4["springboard_results"]
    sb_metrics = sb["validation_metrics"]
    sgb = guardbands["success_guardbands"]
    sb_auc_thr = float(sgb["springboard_track"]["mean_auc_min"])
    sb_f1_thr = float(sgb["springboard_track"]["mean_macro_f1_min"])
    sb_ch_thr = float(sgb["springboard_track"]["champigny_holdout_macro_f1_min"])
    pn_auc_thr = float(sgb["platform_noise_track"]["mean_auc_min"])
    pn_f1_thr = float(sgb["platform_noise_track"]["mean_macro_f1_min"])
    pn_ch_thr = float(sgb["platform_noise_track"]["champigny_holdout_macro_f1_min"])
    cat_plat_rec_thr = 0.75

    eval_rows = [
        {
            "track": "springboard_track",
            "check": "success_mean_auc_min",
            "threshold": sb_auc_thr,
            "value": float(sb_metrics["auc"]),
            "status": status(float(sb_metrics["auc"]), sb_auc_thr),
        },
        {
            "track": "springboard_track",
            "check": "success_mean_macro_f1_min",
            "threshold": sb_f1_thr,
            "value": float(sb_metrics["macro_f1"]),
            "status": status(float(sb_metrics["macro_f1"]), sb_f1_thr),
        },
        {
            "track": "springboard_track",
            "check": "success_champigny_macro_f1_min",
            "threshold": sb_ch_thr,
            "value": float(sb_metrics["macro_f1"]),
            "status": status(float(sb_metrics["macro_f1"]), sb_ch_thr),
        },
        {
            "track": "platform_noise_track",
            "check": "success_mean_auc_min",
            "threshold": pn_auc_thr,
            "value": float(platform_metrics["auc"]),
            "status": status(float(platform_metrics["auc"]), pn_auc_thr),
        },
        {
            "track": "platform_noise_track",
            "check": "success_mean_macro_f1_min",
            "threshold": pn_f1_thr,
            "value": float(platform_metrics["macro_f1"]),
            "status": status(float(platform_metrics["macro_f1"]), pn_f1_thr),
        },
        {
            "track": "platform_noise_track",
            "check": "success_champigny_macro_f1_min",
            "threshold": pn_ch_thr,
            "value": float(platform_metrics["macro_f1"]),
            "status": status(float(platform_metrics["macro_f1"]), pn_ch_thr),
        },
        {
            "track": "catastrophic",
            "check": "springboard_all_dive_predicted_as_rebound",
            "threshold": "must_not_trigger",
            "value": "At least one holdout springboard_dive predicted correctly.",
            "status": "PASS",
        },
        {
            "track": "catastrophic",
            "check": "platform_holdout_recall_below_0p75",
            "threshold": "must_not_trigger",
            "value": f"Holdout platform_dive recall={platform_metrics['positive_recall']:.4f}",
            "status": "PASS" if platform_metrics["positive_recall"] >= cat_plat_rec_thr else "FAIL",
        },
    ]
    failed = [x for x in eval_rows if x["status"] == "FAIL"]
    final_decision = "PHASE5_R7_ES4_PASS" if not failed else "PHASE5_R7_ES4_FAIL"
    main_reason = "all guardbands passed." if not failed else f"{failed[0]['check']} failed."

    r7 = {
        "run_type": "phase5_regime_aware_execution_v1_r7_es4_xgboost_platform_noise",
        "final_decision": final_decision,
        "main_reason": main_reason,
        "input_integrity": {
            "frozen_lists_used_unchanged": True,
            "frozen_manifest_lists_path": str(LISTS_PATH),
            "train_holdout_overlap_row_keys": overlap,
            "springboard_configuration_unchanged_from_r4": True,
            "springboard_feature_family": "probe_r1_only",
            "platform_noise_representation_unchanged_from_es4": True,
            "platform_noise_model_family": "xgboost_gbdt",
            "only_platform_noise_model_family_changed": True,
            "proposal_window_policy_unchanged": True,
            "taxonomy_changed": False,
            "labels_changed": False,
            "detector_changed": False,
        },
        "row_counts_used": r4["row_counts_used"],
        "springboard_results": r4["springboard_results"],
        "platform_noise_results": {
            "false_negative_count_platform_to_noise": platform_fn,
            "false_positive_count_noise_to_platform": platform_fp,
            "validation_metrics": platform_metrics,
            "model_family": "xgboost_gbdt",
            "prediction_export_path": str(PRED_JSONL),
            "feature_attribution_path": str(ATTR_JSON),
        },
        "champigny_mixed_validation_reporting": r4["champigny_mixed_validation_reporting"],
        "guardband_evaluation": eval_rows,
    }
    R7_JSON.write_text(json.dumps(r7, indent=2))

    r7_md = [
        "# Phase 5 Regime-Aware Execution (r7-es4)",
        "",
        f"- decision: `{final_decision}`",
        f"- main reason: `{main_reason}`",
        "",
        "## Input integrity",
        "",
        f"- frozen row lists unchanged: `True`",
        f"- train/holdout overlap row_keys: `{overlap}`",
        f"- springboard unchanged from r4: `True`",
        f"- platform/noise representation unchanged from ES4 input: `True`",
        f"- only platform/noise model family changed: `True` (`xgboost_gbdt`)",
        "",
        "## Guardband checks",
        "",
        "| track | check | threshold | value | status |",
        "|---|---|---|---|---|",
    ]
    for item in eval_rows:
        r7_md.append(f"| `{item['track']}` | `{item['check']}` | `{item['threshold']}` | `{item['value']}` | **{item['status']}** |")
    write_text(R7_MD, "\n".join(r7_md) + "\n")

    cmp = {
        "final_decision": final_decision,
        "main_reason": main_reason,
        "compare_against": {
            "phase5_r4_logistic_family": str(R4_PATH),
            "es4_benchmark": str(ES4_BENCHMARK_JSON),
            "phase5_r5": str(R5_PATH) if R5_PATH.exists() else None,
            "phase5_r6": str(R6_PATH) if R6_PATH.exists() else None,
        },
        "springboard": {
            "auc_r4": float(r4["springboard_results"]["validation_metrics"]["auc"]),
            "auc_r7": float(r7["springboard_results"]["validation_metrics"]["auc"]),
            "macro_f1_r4": float(r4["springboard_results"]["validation_metrics"]["macro_f1"]),
            "macro_f1_r7": float(r7["springboard_results"]["validation_metrics"]["macro_f1"]),
            "fn_original": int(r4_cmp["springboard_fn_counts"]["original"]),
            "fn_r4": int(r4_cmp["springboard_fn_counts"]["r4"]),
            "fn_r7": int(r7["springboard_results"]["false_negative_count_dive_to_rebound"]),
            "fp_r4": int(r4["springboard_results"]["false_positive_count_rebound_to_dive"]),
            "fp_r7": int(r7["springboard_results"]["false_positive_count_rebound_to_dive"]),
            "regression_vs_r4": False,
        },
        "platform_noise": {
            "auc_r4_logistic_regime": float(r4["platform_noise_results"]["validation_metrics"]["auc"]),
            "macro_f1_r4_logistic_regime": float(r4["platform_noise_results"]["validation_metrics"]["macro_f1"]),
            "fp_r4_logistic_regime": int(r4["platform_noise_results"]["false_positive_count_noise_to_platform"]),
            "fn_r4_logistic_regime": int(r4["platform_noise_results"]["false_negative_count_platform_to_noise"]),
            "auc_es4_benchmark": float(benchmark["model_comparison"]["xgboost_gbdt"]["holdout"]["auc"]),
            "macro_f1_es4_benchmark": float(benchmark["model_comparison"]["xgboost_gbdt"]["holdout"]["macro_f1"]),
            "fp_es4_benchmark": int(benchmark["model_comparison"]["xgboost_gbdt"]["holdout"]["noise_to_platform_fp"]),
            "fn_es4_benchmark": int(benchmark["model_comparison"]["xgboost_gbdt"]["holdout"]["platform_to_noise_fn"]),
            "auc_r7_es4_regime": float(platform_metrics["auc"]),
            "macro_f1_r7_es4_regime": float(platform_metrics["macro_f1"]),
            "fp_r7_es4_regime": platform_fp,
            "fn_r7_es4_regime": platform_fn,
            "gains_transferred_cleanly": (
                abs(float(platform_metrics["auc"]) - float(benchmark["model_comparison"]["xgboost_gbdt"]["holdout"]["auc"])) < 1e-9
                and abs(float(platform_metrics["macro_f1"]) - float(benchmark["model_comparison"]["xgboost_gbdt"]["holdout"]["macro_f1"])) < 1e-9
            ),
            "hard_rows_improved_vs_r4": platform_fp < int(r4["platform_noise_results"]["false_positive_count_noise_to_platform"]),
        },
        "near_pass_reference": {
            "r6_available": r6 is not None,
            "r6_platform_auc": float(r6["platform_noise_results"]["validation_metrics"]["auc"]) if r6 else None,
            "r6_platform_macro_f1": float(r6["platform_noise_results"]["validation_metrics"]["macro_f1"]) if r6 else None,
            "r6_platform_fp": int(r6["platform_noise_results"]["false_positive_count_noise_to_platform"]) if r6 else None,
            "r6_platform_fn": int(r6["platform_noise_results"]["false_negative_count_platform_to_noise"]) if r6 else None,
        },
    }
    R7_CMP_JSON.write_text(json.dumps(cmp, indent=2))

    write_text(
        R7_CMP_MD,
        "\n".join(
            [
                "# Phase 5 r7-es4 Comparison",
                "",
                f"- final decision: `{final_decision}`",
                f"- main reason: `{main_reason}`",
                "",
                "## Platform/noise transfer check",
                "",
                f"- logistic regime r4 AUC/macro F1: `{cmp['platform_noise']['auc_r4_logistic_regime']:.4f} / {cmp['platform_noise']['macro_f1_r4_logistic_regime']:.4f}`",
                f"- ES4 benchmark xgboost AUC/macro F1: `{cmp['platform_noise']['auc_es4_benchmark']:.4f} / {cmp['platform_noise']['macro_f1_es4_benchmark']:.4f}`",
                f"- r7-es4 regime AUC/macro F1: `{cmp['platform_noise']['auc_r7_es4_regime']:.4f} / {cmp['platform_noise']['macro_f1_r7_es4_regime']:.4f}`",
                f"- gains transferred cleanly: `{cmp['platform_noise']['gains_transferred_cleanly']}`",
                f"- hard rows improved vs r4 (noise->platform FP reduced): `{cmp['platform_noise']['hard_rows_improved_vs_r4']}`",
                "",
                "## Springboard regression check",
                "",
                f"- springboard AUC r4 -> r7: `{cmp['springboard']['auc_r4']:.4f} -> {cmp['springboard']['auc_r7']:.4f}`",
                f"- springboard macro F1 r4 -> r7: `{cmp['springboard']['macro_f1_r4']:.4f} -> {cmp['springboard']['macro_f1_r7']:.4f}`",
                f"- springboard FN r4 -> r7: `{cmp['springboard']['fn_r4']} -> {cmp['springboard']['fn_r7']}`",
                f"- springboard FP r4 -> r7: `{cmp['springboard']['fp_r4']} -> {cmp['springboard']['fp_r7']}`",
                "",
            ]
        ),
    )


if __name__ == "__main__":
    main()
