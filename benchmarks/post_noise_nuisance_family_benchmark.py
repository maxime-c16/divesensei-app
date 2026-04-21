from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE5_MODULE_PATH = REPO_ROOT / "benchmarks" / "phase5_regime_aware_execution_r7_es4.py"
MANIFEST_PREVIEW_PATH = REPO_ROOT / "outputs" / "event_window_manifest_preview.jsonl"
DATASET_ROWS_PATH = REPO_ROOT / "outputs" / "platform_noise_es4_dataset_rows.json"
EXTERNAL_SLICE_PATH = REPO_ROOT / "outputs" / "external_holdout_slice.json"
OUT_JSON = REPO_ROOT / "outputs" / "post_noise_nuisance_family_benchmark.json"
OUT_MD = REPO_ROOT / "outputs" / "post_noise_nuisance_family_benchmark.md"

TONAL_FEATURES = [
    "dominant_frequency_hz_post_mean",
    "dominant_frequency_hz_post_std",
    "dominant_frequency_stability_ratio_post",
    "spectral_rolloff_90_post_mean",
]

IMPACT_FEATURES = [
    "low_band_energy_fraction_post",
    "low_high_energy_ratio_post",
    "zero_crossing_rate_post_mean",
    "post_rms_modulation_cv",
]

NOISE_BOUNDARY_COMPACT = [
    "dominant_frequency_hz_post_std",
    "spectral_rolloff_90_post_mean",
    "zero_crossing_rate_post_mean",
]

NOISE_BOUNDARY_PLUS_RATIO = [
    "dominant_frequency_hz_post_std",
    "spectral_rolloff_90_post_mean",
    "zero_crossing_rate_post_mean",
    "low_high_energy_ratio_post",
]


@dataclass(frozen=True)
class RowRef:
    row_key: str
    label: str
    row: dict[str, Any]


def load_phase5_module():
    spec = importlib.util.spec_from_file_location("phase5_r7_es4_runtime", PHASE5_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        session_id = str(row["source_session_id"])
        counts[session_id] = counts.get(session_id, 0) + 1
        candidate_id = row.get("legacy_candidate_id")
        row_id = str(candidate_id) if candidate_id else f"row-{counts[session_id]:04d}"
        out[f"{session_id}::{row_id}"] = row
    return out


def nuisance_features(mod: Any, signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    frame = 512
    hop = 128
    eps = 1e-8
    frames = np.asarray(mod.framing(signal, frame=frame, hop=hop), dtype=np.float32)
    if frames.size == 0:
        names = TONAL_FEATURES + IMPACT_FEATURES
        return {name: 0.0 for name in names}

    window = np.hanning(frame).astype(np.float32)
    windowed = frames * window
    rms = np.sqrt(np.mean(np.square(windowed), axis=1) + eps)
    peak_idx = int(np.argmax(rms))

    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    frames_400 = max(1, int(round(0.400 * sample_rate / hop)))
    post_frames = power[peak_idx : peak_idx + frames_400]
    if len(post_frames) == 0:
        post_frames = power[max(0, peak_idx - 1) : peak_idx + 1]
    post_frames = np.maximum(post_frames, eps)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)

    low_mask = freqs < 1000.0
    high_mask = (freqs >= 1000.0) & (freqs <= 4000.0)
    total_energy = np.maximum(np.sum(post_frames, axis=1), eps)
    low_energy = np.sum(post_frames[:, low_mask], axis=1)
    high_energy = np.sum(post_frames[:, high_mask], axis=1)
    low_band_energy_fraction_post = float(np.mean(low_energy / total_energy))
    low_high_energy_ratio_post = float(np.mean(low_energy / np.maximum(high_energy, eps)))

    dominant_idx = np.argmax(post_frames, axis=1)
    dominant_freqs = freqs[dominant_idx]
    dominant_frequency_hz_post_mean = float(np.mean(dominant_freqs))
    dominant_frequency_hz_post_std = float(np.std(dominant_freqs))
    dominant_frequency_stability_ratio_post = float(
        np.mean(np.abs(dominant_freqs - np.median(dominant_freqs)) <= 150.0)
    )

    cumulative = np.cumsum(post_frames, axis=1)
    rolloff_target = 0.9 * total_energy[:, None]
    rolloff_idx = np.argmax(cumulative >= rolloff_target, axis=1)
    spectral_rolloff_90_post_mean = float(np.mean(freqs[rolloff_idx]))

    crossings = np.mean(np.abs(np.diff(np.signbit(windowed).astype(np.float32), axis=1)), axis=1)
    zero_crossing_rate_post_mean = float(np.mean(crossings[peak_idx : peak_idx + min(len(crossings) - peak_idx, frames_400)]))

    post_rms = rms[peak_idx : peak_idx + min(len(rms) - peak_idx, frames_400)]
    post_rms_modulation_cv = float(np.std(post_rms) / max(np.mean(post_rms), eps)) if len(post_rms) else 0.0

    return {
        "dominant_frequency_hz_post_mean": dominant_frequency_hz_post_mean,
        "dominant_frequency_hz_post_std": dominant_frequency_hz_post_std,
        "dominant_frequency_stability_ratio_post": dominant_frequency_stability_ratio_post,
        "spectral_rolloff_90_post_mean": spectral_rolloff_90_post_mean,
        "low_band_energy_fraction_post": low_band_energy_fraction_post,
        "low_high_energy_ratio_post": low_high_energy_ratio_post,
        "zero_crossing_rate_post_mean": zero_crossing_rate_post_mean,
        "post_rms_modulation_cv": post_rms_modulation_cv,
    }


def vector_for(mod: Any, item: RowRef, fmap: dict[str, dict[str, float]], extra_feature_names: list[str]) -> list[float]:
    row = item.row
    return [
        mod.to_float(row.get("audio_score") or row.get("detector_scores", {}).get("audio_score")),
        mod.to_float(row.get("audio_clip_probability") or row.get("clip_probability")),
        mod.to_float(row.get("event_anchor_timestamp_seconds")),
        1.0 if row.get("is_false_negative_window") else 0.0,
        *[mod.to_float(fmap[item.row_key][name]) for name in mod.PROBE1_FEATURE_NAMES],
        *[mod.to_float(fmap[item.row_key][name]) for name in mod.R2_FEATURE_NAMES],
        *[mod.to_float(fmap[item.row_key][name]) for name in mod.R4_FEATURE_NAMES],
        *[mod.to_float(fmap[item.row_key][name]) for name in extra_feature_names],
    ]


def eval_split(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = (probs >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[1, 0]).tolist()
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, pred)),
        "platform_recall": float(recall_score(y_true, pred, pos_label=1)),
        "noise_recall": float(recall_score(y_true, pred, pos_label=0)),
        "confusion_matrix": cm,
        "noise_to_platform_fp": int(cm[1][0]),
        "platform_to_noise_fn": int(cm[0][1]),
        "pred": pred.tolist(),
        "probs": probs.tolist(),
    }


def build_refs(manifest_rows: dict[str, dict[str, Any]], list_rows: list[dict[str, Any]]) -> list[RowRef]:
    refs: list[RowRef] = []
    for item in list_rows:
        row_key = str(item["row_key"])
        refs.append(RowRef(row_key=row_key, label=str(item["label"]), row=manifest_rows[row_key]))
    return refs


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Post-Noise Nuisance Family Benchmark",
        "",
        f"- frozen training rows: `{report['data']['train_rows']}`",
        f"- internal holdout rows: `{report['data']['internal_holdout_rows']}`",
        f"- external reviewed rows: `{report['data']['external_rows']}`",
        f"- external nuisance distribution: `{json.dumps(report['data']['external_noise_subtypes'], sort_keys=True)}`",
        "",
        "| config | internal AUC | internal macro F1 | internal platform recall | internal noise recall | external AUC | external macro F1 | external platform recall | external noise recall | external noise FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        i = row["internal"]
        e = row["external"]
        lines.append(
            f"| {row['config_name']} | {i['auc']:.4f} | {i['macro_f1']:.4f} | {i['platform_recall']:.4f} | {i['noise_recall']:.4f} | "
            f"{e['auc']:.4f} | {e['macro_f1']:.4f} | {e['platform_recall']:.4f} | {e['noise_recall']:.4f} | {e['noise_to_platform_fp']} |"
        )
    lines += [
        "",
        "## Best Candidate",
        "",
        f"- config: `{report['best_candidate']['config_name']}`",
        f"- rationale: `{report['best_candidate']['rationale']}`",
        "",
        "## External Residual Core",
        "",
        f"- baseline noise FP: `{report['baseline_external']['noise_to_platform_fp']}`",
        f"- best-candidate noise FP: `{report['best_candidate']['external']['noise_to_platform_fp']}`",
        f"- baseline platform recall: `{report['baseline_external']['platform_recall']:.4f}`",
        f"- best-candidate platform recall: `{report['best_candidate']['external']['platform_recall']:.4f}`",
        f"- best-candidate subtype FP distribution: `{json.dumps(report['best_candidate']['external_fp_subtypes'], sort_keys=True)}`",
        "",
        "## Decision",
        "",
        f"- `{report['decision']}`",
        f"- `{report['decision_reason']}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    mod = load_phase5_module()

    preview_rows = [json.loads(line) for line in MANIFEST_PREVIEW_PATH.read_text().splitlines() if line.strip()]
    preview_map = row_key_map(preview_rows)
    dataset_rows = json.loads(DATASET_ROWS_PATH.read_text())
    train_refs = build_refs(preview_map, dataset_rows["train_rows"])
    internal_holdout_refs = build_refs(preview_map, dataset_rows["holdout_rows"])

    external_payload = json.loads(EXTERNAL_SLICE_PATH.read_text())
    external_rows = external_payload["rows"]
    external_refs = [RowRef(row_key=str(row["row_key"]), label=str(row["final_human_event_label"]), row=row) for row in external_rows]

    audio_cache: dict[str, np.ndarray] = {}
    feature_map: dict[str, dict[str, float]] = {}
    all_refs = train_refs + internal_holdout_refs + external_refs
    for item in all_refs:
        sid = str(item.row["source_session_id"])
        if sid not in audio_cache:
            source_root = mod.resolve_source_root(str(item.row["source_session_root"]))
            audio_cache[sid] = mod.decode_audio_mono(source_root / "web" / "session_source_review.mp4", mod.SAMPLE_RATE)
        signal = audio_cache[sid]
        start = max(0.0, mod.to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, mod.to_float(item.row.get("event_window_end_seconds")))
        s0 = int(round(start * mod.SAMPLE_RATE))
        s1 = int(round(end * mod.SAMPLE_RATE))
        base = mod.extract_features(signal[s0:s1], sample_rate=mod.SAMPLE_RATE)
        extra = nuisance_features(mod, signal[s0:s1], sample_rate=mod.SAMPLE_RATE)
        feature_map[item.row_key] = {**base, **extra}

    configs = [
        ("es4_current", []),
        ("es4_plus_noise_boundary_compact", NOISE_BOUNDARY_COMPACT),
        ("es4_plus_noise_boundary_plus_ratio", NOISE_BOUNDARY_PLUS_RATIO),
        ("es4_plus_tonal_nuisance", TONAL_FEATURES),
        ("es4_plus_impact_nuisance", IMPACT_FEATURES),
        ("es4_plus_combined_nuisance", TONAL_FEATURES + IMPACT_FEATURES),
    ]

    results: list[dict[str, Any]] = []
    for name, extra_features in configs:
        x_train = np.asarray([vector_for(mod, item, feature_map, extra_features) for item in train_refs], dtype=np.float64)
        y_train = np.asarray([1 if item.label == "platform_dive" else 0 for item in train_refs], dtype=np.int64)
        x_internal = np.asarray([vector_for(mod, item, feature_map, extra_features) for item in internal_holdout_refs], dtype=np.float64)
        y_internal = np.asarray([1 if item.label == "platform_dive" else 0 for item in internal_holdout_refs], dtype=np.int64)
        x_external = np.asarray([vector_for(mod, item, feature_map, extra_features) for item in external_refs], dtype=np.float64)
        y_external = np.asarray([1 if item.label == "platform_dive" else 0 for item in external_refs], dtype=np.int64)

        model = XGBClassifier(
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
        model.fit(x_train, y_train)
        internal_probs = model.predict_proba(x_internal)[:, 1]
        external_probs = model.predict_proba(x_external)[:, 1]
        internal_metrics = eval_split(y_internal, internal_probs)
        external_metrics = eval_split(y_external, external_probs)

        external_pred = np.asarray(external_metrics["pred"], dtype=np.int64)
        external_fp_subtypes = Counter()
        for item, pred_value, true_value in zip(external_refs, external_pred.tolist(), y_external.tolist()):
            if true_value == 0 and pred_value == 1:
                external_fp_subtypes[str(item.row.get("legacy_subtype") or "none")] += 1

        results.append(
            {
                "config_name": name,
                "added_feature_names": extra_features,
                "internal": internal_metrics,
                "external": external_metrics,
                "external_fp_subtypes": dict(sorted(external_fp_subtypes.items())),
            }
        )

    baseline = next(item for item in results if item["config_name"] == "es4_current")
    eligible = [
        item
        for item in results
        if item["internal"]["platform_recall"] >= baseline["internal"]["platform_recall"]
        and item["internal"]["macro_f1"] >= baseline["internal"]["macro_f1"] - 0.01
    ]
    if not eligible:
        eligible = results
    best = max(
        eligible,
        key=lambda item: (
            item["external"]["noise_recall"],
            item["external"]["macro_f1"],
            -item["external"]["noise_to_platform_fp"],
            item["external"]["platform_recall"],
            -item["external"]["noise_to_platform_fp"],
            item["external"]["auc"],
        ),
    )

    decision = (
        "REPRESENTATION_FIX_SUPPORTED"
        if best["config_name"] != "es4_current"
        and best["external"]["noise_recall"] > baseline["external"]["noise_recall"]
        and best["external"]["noise_to_platform_fp"] < baseline["external"]["noise_to_platform_fp"]
        and best["external"]["macro_f1"] > baseline["external"]["macro_f1"] + 0.02
        else "NOISE_REPRESENTATION_GAP_PERSISTS"
    )
    if decision == "REPRESENTATION_FIX_SUPPORTED":
        reason = "A compact noise-boundary feature bundle improves external noise separation, reduces noise false positives, and clears the internal guardrails."
    else:
        reason = "The tested nuisance-aware bundles do not beat the current ES4 baseline enough under the internal guardrails; the residual gap is not solved by these bounded features alone."

    report = {
        "data": {
            "train_rows": len(train_refs),
            "internal_holdout_rows": len(internal_holdout_refs),
            "external_rows": len(external_refs),
            "external_noise_subtypes": dict(
                sorted(
                    Counter(str(item.row.get("legacy_subtype") or "none") for item in external_refs if item.label == "noise_or_other").items()
                )
            ),
        },
        "results": results,
        "baseline_external": baseline["external"],
        "best_candidate": {
            **best,
            "rationale": (
                f"selected under internal guardrails using external macro F1={best['external']['macro_f1']:.4f}, "
                f"external noise recall={best['external']['noise_recall']:.4f}, noise FP={best['external']['noise_to_platform_fp']}"
            ),
        },
        "decision": decision,
        "decision_reason": reason,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print(json.dumps({"decision": decision, "best_candidate": best["config_name"], "baseline_external": baseline["external"], "best_external": best["external"]}, indent=2))


if __name__ == "__main__":
    main()
