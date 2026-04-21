from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "benchmarks" / "phase5_regime_aware_execution_r7_es4.py"
NUISANCE = ROOT / "benchmarks" / "post_noise_nuisance_family_benchmark.py"
DATASET = ROOT / "outputs" / "platform_noise_es4_dataset_rows.json"
EXTERNAL = ROOT / "outputs" / "external_holdout_slice.json"
PREVIEW = ROOT / "outputs" / "event_window_manifest_preview.jsonl"
R9_REF = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted.json"
SOURCES = {
    "snmt": ROOT / "outputs/evaluation_SNMT-16min_20260417-131944/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "img_8852": ROOT / "outputs/evaluation_img_8852_rerun_20260406-104430/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "champigny_1704": ROOT / "outputs/evaluation_Champigny-17-04-9min_20260418-065417/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
}
WEIGHTS = {"snmt": 1.0, "img_8852": 1.0, "champigny_1704": 0.3}
LOW = 0.05
HIGH = 0.85
PRECISION_TARGET = 0.90
COVERAGE_TARGET = 0.40
OUT_MAIN_JSON = ROOT / "outputs/r12_targeted_impact_vs_nuisance_representation.json"
OUT_MAIN_MD = ROOT / "outputs/r12_targeted_impact_vs_nuisance_representation.md"
OUT_COMPARISON_JSON = ROOT / "outputs/r12_candidate_comparison.json"
OUT_COMPARISON_MD = ROOT / "outputs/r12_candidate_comparison.md"
OUT_QUEUE_JSON = ROOT / "outputs/r12_queue_safety_comparison.json"
OUT_QUEUE_MD = ROOT / "outputs/r12_queue_safety_comparison.md"
COMPACT = ["dominant_frequency_hz_post_std", "spectral_rolloff_90_post_mean", "zero_crossing_rate_post_mean"]
ENVELOPE_FEATURES = [
    "attack_time_ms",
    "peak_width_ms",
    "early_decay_ratio",
    "late_decay_ratio",
    "tail_smoothness",
    "secondary_peak_count",
    "energy_0_80_to_pre_ratio",
    "energy_80_250_to_0_80_ratio",
    "energy_250_700_to_80_250_ratio",
]
CLOSE_MIC_FEATURES = [
    "clipping_fraction",
    "crest_factor",
    "impulsiveness_kurtosis",
    "sub_200hz_fraction",
    "vlf_drift",
    "silence_recovery_ratio",
]
MULTIBAND_FEATURES = [
    "low_decay_ratio",
    "mid_decay_ratio",
    "high_decay_ratio",
    "centroid_shift",
    "rolloff_shift",
    "flatness_shift",
    "broadband_persistence",
    "tail_high_low_ratio",
]
PANN_PROXY_FEATURES = ENVELOPE_FEATURES + CLOSE_MIC_FEATURES + MULTIBAND_FEATURES + COMPACT


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out, counts = {}, {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        rid = str(row.get("legacy_candidate_id") or f"row-{counts[sid]:04d}")
        out[f"{sid}::{rid}"] = row
    return out


def label_int(label: str) -> int:
    return 1 if label == "platform_dive" else 0


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def fnum(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def rowrefs(bench: Any, rows: list[dict[str, Any]]) -> list[Any]:
    return [
        bench.RowRef(f"{r['source_session_id']}::{r.get('legacy_candidate_id') or 'row-unknown'}", str(r["final_human_event_label"]), r)
        for r in rows
    ]


def forced_metrics(labels: list[str], scores: list[float]) -> dict[str, Any]:
    y = np.asarray([label_int(label) for label in labels], dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[1, 0]).tolist()
    return {
        "auc": float(roc_auc_score(y, p)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "platform_recall": float(recall_score(y, pred, pos_label=1)),
        "noise_recall": float(recall_score(y, pred, pos_label=0)),
        "noise_to_platform_fp": int(cm[1][0]),
        "platform_to_noise_fn": int(cm[0][1]),
        "confusion_matrix": cm,
    }


def triage(rows: list[dict[str, Any]], queues: list[str]) -> dict[str, Any]:
    approve = [r for r, q in zip(rows, queues) if q == "auto_approved"]
    exclude = [r for r, q in zip(rows, queues) if q == "auto_excluded"]
    review = [r for r, q in zip(rows, queues) if q == "needs_review"]
    accepted = approve + exclude
    y = [label_int(str(r["label"])) for r in accepted]
    pred = [1] * len(approve) + [0] * len(exclude)
    approve_ok = sum(1 for r in approve if r["label"] == "platform_dive")
    exclude_ok = sum(1 for r in exclude if r["label"] == "noise_or_other")
    return {
        "row_count": len(rows),
        "coverage": safe_div(len(accepted), len(rows)) or 0.0,
        "review_rate": safe_div(len(review), len(rows)) or 0.0,
        "auto_approve_count": len(approve),
        "auto_exclude_count": len(exclude),
        "review_required_count": len(review),
        "auto_approve_precision": safe_div(approve_ok, len(approve)),
        "auto_exclude_precision": safe_div(exclude_ok, len(exclude)),
        "auto_approve_error_count": len(approve) - approve_ok,
        "auto_exclude_error_count": len(exclude) - exclude_ok,
        "accepted_accuracy": float(accuracy_score(y, pred)) if y else None,
        "accepted_macro_f1": float(f1_score(y, pred, average="macro")) if y and len(set(y + pred)) > 1 else None,
        "review_required_label_counts": dict(sorted(Counter(r["label"] for r in review).items())),
        "review_required_subtype_counts": dict(sorted(Counter(str(r.get("legacy_subtype") or "none") for r in review).items())),
        "auto_approve_error_rows": [r for r in approve if r["label"] != "platform_dive"][:30],
        "auto_exclude_error_rows": [r for r in exclude if r["label"] != "noise_or_other"][:30],
    }


def role(row: dict[str, Any]) -> str:
    score, label = float(row["r9_score"]), str(row["label"])
    if score >= HIGH and label == "noise_or_other":
        return "approve_risk"
    if score <= LOW and label == "platform_dive":
        return "exclude_risk"
    if LOW < score < HIGH:
        return "ambiguous_review_band"
    if score >= HIGH and label == "platform_dive":
        return "high_confidence_platform_control"
    if score <= LOW and label == "noise_or_other":
        return "high_confidence_noise_control"
    return "other"


def band(score: float) -> str:
    if score >= HIGH:
        return "high_approve_band"
    if score <= LOW:
        return "high_exclude_band"
    return "review_band"


def vec(row: dict[str, Any], names: list[str]) -> list[float]:
    values = []
    for name in names:
        value = float(row.get(name, 0.0) or 0.0)
        if not np.isfinite(value):
            value = 0.0
        values.append(value)
    return values


def _rms_frames(signal: np.ndarray, frame: int = 256, hop: int = 64) -> np.ndarray:
    if len(signal) < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(signal)] = signal
        frames = padded.reshape(1, frame)
    else:
        n = 1 + (len(signal) - frame) // hop
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(n, frame),
            strides=(signal.strides[0] * hop, signal.strides[0]),
            writeable=False,
        )
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-9)


def _segment(signal: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    center = len(signal) // 2
    s0 = max(0, center + int(round(start_s * sample_rate)))
    s1 = min(len(signal), center + int(round(end_s * sample_rate)))
    if s1 <= s0:
        return np.zeros(1, dtype=np.float32)
    return signal[s0:s1]


def _energy(signal: np.ndarray) -> float:
    return float(np.mean(np.square(signal)) + 1e-9)


def _band_energy(signal: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    if len(signal) < 32:
        return 0.0
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask]) / max(np.sum(power), 1e-9))


def envelope_morphology_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    rms = _rms_frames(signal)
    peak = int(np.argmax(rms))
    peak_val = float(rms[peak])
    baseline = float(np.median(rms[: max(1, min(peak, 10))])) if peak > 0 else float(np.median(rms))
    threshold = baseline + 0.5 * (peak_val - baseline)
    above = np.where(rms >= threshold)[0]
    left = int(above[0]) if len(above) else peak
    right = int(above[-1]) if len(above) else peak
    hop_ms = 64 / sample_rate * 1000.0
    post = rms[peak:]
    early = float(np.mean(post[: max(1, int(round(0.12 * sample_rate / 64)))]))
    mid = float(np.mean(post[int(round(0.12 * sample_rate / 64)) : max(int(round(0.12 * sample_rate / 64)) + 1, int(round(0.45 * sample_rate / 64)))]))
    late = float(np.mean(post[int(round(0.45 * sample_rate / 64)) : max(int(round(0.45 * sample_rate / 64)) + 1, int(round(0.9 * sample_rate / 64)))]))
    tail = post[: max(2, int(round(0.7 * sample_rate / 64)))]
    diffs = np.diff(tail)
    secondary = int(np.sum((post[2:-1] > post[1:-2]) & (post[2:-1] >= post[3:]) & (post[2:-1] > threshold))) if len(post) > 4 else 0
    pre_e = _energy(_segment(signal, sample_rate, -0.5, 0.0))
    e0 = _energy(_segment(signal, sample_rate, 0.0, 0.08))
    e1 = _energy(_segment(signal, sample_rate, 0.08, 0.25))
    e2 = _energy(_segment(signal, sample_rate, 0.25, 0.7))
    return {
        "attack_time_ms": max(0.0, (peak - left) * hop_ms),
        "peak_width_ms": max(0.0, (right - left + 1) * hop_ms),
        "early_decay_ratio": early / max(peak_val, 1e-9),
        "late_decay_ratio": late / max(peak_val, 1e-9),
        "tail_smoothness": float(np.mean(np.abs(diffs))) / max(float(np.mean(tail)), 1e-9),
        "secondary_peak_count": float(secondary),
        "energy_0_80_to_pre_ratio": e0 / max(pre_e, 1e-9),
        "energy_80_250_to_0_80_ratio": e1 / max(e0, 1e-9),
        "energy_250_700_to_80_250_ratio": e2 / max(e1, 1e-9),
    }


def close_mic_impulse_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    abs_signal = np.abs(signal)
    rms = float(np.sqrt(np.mean(np.square(signal)) + 1e-9))
    centered = signal - float(np.mean(signal))
    std = float(np.std(centered) + 1e-9)
    kurt = float(np.mean((centered / std) ** 4))
    tail = _segment(signal, sample_rate, 0.45, 1.0)
    impact = _segment(signal, sample_rate, 0.0, 0.12)
    return {
        "clipping_fraction": float(np.mean(abs_signal >= 0.98)),
        "crest_factor": float(np.max(abs_signal) / max(rms, 1e-9)),
        "impulsiveness_kurtosis": kurt,
        "sub_200hz_fraction": _band_energy(impact, sample_rate, 0.0, 200.0),
        "vlf_drift": float(abs(np.mean(_segment(signal, sample_rate, -0.25, 0.6))) / max(rms, 1e-9)),
        "silence_recovery_ratio": _energy(tail) / max(_energy(impact), 1e-9),
    }


def multiband_decay_texture_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    early = _segment(signal, sample_rate, 0.0, 0.18)
    late = _segment(signal, sample_rate, 0.25, 0.8)
    def band_ratio(low: float, high: float) -> float:
        return _band_energy(late, sample_rate, low, high) / max(_band_energy(early, sample_rate, low, high), 1e-9)
    def centroid(seg: np.ndarray) -> float:
        if len(seg) < 32:
            return 0.0
        power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sample_rate)
        return float(np.sum(freqs * power) / max(np.sum(power), 1e-9))
    def rolloff(seg: np.ndarray) -> float:
        if len(seg) < 32:
            return 0.0
        power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sample_rate)
        idx = int(np.searchsorted(np.cumsum(power), 0.9 * np.sum(power)))
        return float(freqs[min(idx, len(freqs) - 1)])
    def flatness(seg: np.ndarray) -> float:
        if len(seg) < 32:
            return 0.0
        power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2 + 1e-9
        return float(np.exp(np.mean(np.log(power))) / np.mean(power))
    low_late = _band_energy(late, sample_rate, 0, 800)
    high_late = _band_energy(late, sample_rate, 1600, 5000)
    return {
        "low_decay_ratio": band_ratio(0, 800),
        "mid_decay_ratio": band_ratio(800, 1800),
        "high_decay_ratio": band_ratio(1800, 5000),
        "centroid_shift": centroid(late) - centroid(early),
        "rolloff_shift": rolloff(late) - rolloff(early),
        "flatness_shift": flatness(late) - flatness(early),
        "broadband_persistence": float(np.mean([_band_energy(late, sample_rate, 0, 800), _band_energy(late, sample_rate, 800, 1800), _band_energy(late, sample_rate, 1800, 5000)])),
        "tail_high_low_ratio": high_late / max(low_late, 1e-9),
    }


def main() -> None:
    phase5 = load_module("phase5_r11", PHASE5)
    bench = load_module("nuisance_r11", NUISANCE)
    r9_reference = json.loads(R9_REF.read_text())
    preview = row_key_map(load_jsonl(PREVIEW))
    lists = json.loads(DATASET.read_text())
    base_train = [("base", bench.RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])])) for i in lists["train_rows"]]
    internal_refs = [bench.RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])]) for i in lists["holdout_rows"]]
    external_refs = [bench.RowRef(str(r["row_key"]), str(r["final_human_event_label"]), r) for r in json.loads(EXTERNAL.read_text())["rows"]]
    session_refs = {k: rowrefs(bench, [r for r in load_jsonl(p) if r.get("final_human_event_label") in {"platform_dive", "noise_or_other"}]) for k, p in SOURCES.items()}
    train_refs = base_train + [(k, ref) for k, refs in session_refs.items() for ref in refs]
    all_refs = [r for _, r in train_refs] + internal_refs + external_refs
    audio, fmap = {}, {}
    for item in all_refs:
        sid = str(item.row["source_session_id"])
        if sid not in audio:
            audio[sid] = phase5.decode_audio_mono(phase5.resolve_source_root(str(item.row["source_session_root"])) / "web/session_source_review.mp4", phase5.SAMPLE_RATE)
        start = max(0.0, phase5.to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(item.row.get("event_window_end_seconds")))
        sig = audio[sid][int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap[item.row_key] = {
            **phase5.extract_features(sig, phase5.SAMPLE_RATE),
            **bench.nuisance_features(phase5, sig, phase5.SAMPLE_RATE),
            **envelope_morphology_features(sig, phase5.SAMPLE_RATE),
            **close_mic_impulse_features(sig, phase5.SAMPLE_RATE),
            **multiband_decay_texture_features(sig, phase5.SAMPLE_RATE),
        }

    def r9_vec(item: Any) -> list[float]:
        return bench.vector_for(phase5, item, fmap, bench.NOISE_BOUNDARY_COMPACT)

    x_train = np.asarray([r9_vec(ref) for _, ref in train_refs], dtype=np.float64)
    y_train = np.asarray([label_int(ref.label) for _, ref in train_refs], dtype=np.int64)
    base_total = sum(1 for src, _ in train_refs if src == "base")
    sample_weight = np.ones(len(train_refs), dtype=np.float64)
    by_source: dict[str, list[int]] = {}
    for idx, (src, _) in enumerate(train_refs):
        by_source.setdefault(src, []).append(idx)
    for src, idxs in by_source.items():
        if src == "base":
            continue
        per_item = base_total * WEIGHTS[src] / len(idxs)
        for idx in idxs:
            sample_weight[idx] = per_item
    r9_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=120, learning_rate=0.05, max_depth=3, min_child_weight=2, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=1)
    r9_model.fit(x_train, y_train, sample_weight=sample_weight)

    def make_rows(refs: list[Any], scores: list[float], split: str) -> list[dict[str, Any]]:
        rows = []
        for item, score in zip(refs, scores):
            src = item.row
            payload = {
                "split": split,
                "row_key": item.row_key,
                "source_session_id": str(src.get("source_session_id") or item.row_key.split("::", 1)[0]),
                "label": item.label,
                "r9_score": float(score),
                "score_band": band(float(score)),
                "legacy_subtype": src.get("legacy_subtype"),
                "suggested_event_label_reason": src.get("suggested_event_label_reason"),
                "event_anchor_strategy": src.get("event_anchor_strategy"),
                "manual_correction_type": src.get("manual_correction_type"),
                "audio_score": phase5.to_float(src.get("audio_score") or src.get("detector_scores", {}).get("audio_score")),
                "event_anchor_timestamp_seconds": phase5.to_float(src.get("event_anchor_timestamp_seconds")),
            }
            for name in set(phase5.ALL_FEATURE_NAMES + bench.NOISE_BOUNDARY_COMPACT + ENVELOPE_FEATURES + CLOSE_MIC_FEATURES + MULTIBAND_FEATURES + PANN_PROXY_FEATURES):
                if name != "r9_score":
                    payload[name] = float(fmap[item.row_key].get(name, payload.get(name, 0.0)) or 0.0)
            payload["boundary_role"] = role(payload)
            rows.append(payload)
        return rows

    train_rows = make_rows([r for _, r in train_refs], r9_model.predict_proba(x_train)[:, 1].tolist(), "train_augmented")
    internal_rows = make_rows(internal_refs, r9_model.predict_proba(np.asarray([r9_vec(r) for r in internal_refs], dtype=np.float64))[:, 1].tolist(), "internal_holdout")
    external_rows = make_rows(external_refs, r9_model.predict_proba(np.asarray([r9_vec(r) for r in external_refs], dtype=np.float64))[:, 1].tolist(), "external_holdout")
    boundary_rows = [r for r in internal_rows + external_rows if r["boundary_role"] != "other"]
    dataset = {
        "dataset_name": "r11_score_separation_boundary_dataset",
        "r9_policy_thresholds": {"auto_exclude_max_score": LOW, "auto_approve_min_score": HIGH},
        "row_count": len(boundary_rows),
        "split_counts": dict(sorted(Counter(r["split"] for r in boundary_rows).items())),
        "label_counts": dict(sorted(Counter(r["label"] for r in boundary_rows).items())),
        "subtype_counts": dict(sorted(Counter(str(r.get("legacy_subtype") or "none") for r in boundary_rows).items())),
        "score_band_counts": dict(sorted(Counter(r["score_band"] for r in boundary_rows).items())),
        "boundary_role_counts": dict(sorted(Counter(r["boundary_role"] for r in boundary_rows).items())),
        "approve_risk_rows": [r for r in boundary_rows if r["boundary_role"] == "approve_risk"],
        "exclude_risk_rows": [r for r in boundary_rows if r["boundary_role"] == "exclude_risk"],
        "rows": boundary_rows,
    }
    candidate_features = {
        "r9_reference": ["r9_score"],
        "r12_envelope_morphology": ["r9_score"] + ENVELOPE_FEATURES,
        "r12_close_mic_impulse_features": ["r9_score"] + CLOSE_MIC_FEATURES,
        "r12_multiband_decay_texture": ["r9_score"] + MULTIBAND_FEATURES,
        "r12_panns_embedding_probe": ["r9_score"] + PANN_PROXY_FEATURES,
    }
    models = {}
    y_fit = [label_int(r["label"]) for r in train_rows]
    for name, features in candidate_features.items():
        if name == "r9_reference":
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
        model.fit(np.asarray([vec(r, features) for r in train_rows], dtype=np.float64), y_fit)
        models[name] = model

    def scores(rows: list[dict[str, Any]], cand: str) -> list[float]:
        if cand == "r9_reference":
            return [float(r["r9_score"]) for r in rows]
        return models[cand].predict_proba(np.asarray([vec(r, candidate_features[cand]) for r in rows], dtype=np.float64))[:, 1].tolist()

    def queues(rows: list[dict[str, Any]], cand: str, sc: list[float]) -> list[str]:
        out = []
        for row, score in zip(rows, sc):
            r9 = float(row["r9_score"])
            if cand == "r9_reference":
                out.append("auto_approved" if r9 >= HIGH else ("auto_excluded" if r9 <= LOW else "needs_review"))
            elif cand in {"r12_envelope_morphology", "r12_close_mic_impulse_features", "r12_multiband_decay_texture", "r12_panns_embedding_probe"}:
                out.append("auto_approved" if r9 >= HIGH and score >= 0.70 else ("auto_excluded" if r9 <= LOW else "needs_review"))
        return out

    candidates = [
        ("r9_reference", "Promoted r9 weighted reference with r10 thresholds."),
        ("r12_envelope_morphology", "Attack, peak width, early/late decay, secondary impacts, tail smoothness, and post-impact energy ratios."),
        ("r12_close_mic_impulse_features", "Clipping, crest factor, impulsiveness, sub-200Hz fraction, VLF drift, and silence recovery."),
        ("r12_multiband_decay_texture", "Multiband decay, centroid/rolloff/flatness trajectories, broadband persistence, and splash-vs-dry texture."),
        ("r12_panns_embedding_probe", "Diagnostic proxy because pretrained PANNs dependencies are unavailable locally; uses pooled morphology/texture features as a bounded embedding-like probe."),
    ]
    reports, table = [], []
    for name, desc in candidates:
        int_scores, ext_scores = scores(internal_rows, name), scores(external_rows, name)
        int_triage, ext_triage = triage(internal_rows, queues(internal_rows, name, int_scores)), triage(external_rows, queues(external_rows, name, ext_scores))
        int_forced, ext_forced = forced_metrics([r["label"] for r in internal_rows], int_scores), forced_metrics([r["label"] for r in external_rows], ext_scores)
        dangerous_external_auto_approve = ext_triage["auto_approve_error_count"]
        dangerous_internal_auto_approve = int_triage["auto_approve_error_count"]
        viable = (
            fnum(ext_triage["auto_approve_precision"]) >= PRECISION_TARGET
            and fnum(ext_triage["auto_exclude_precision"]) >= PRECISION_TARGET
            and ext_triage["coverage"] >= COVERAGE_TARGET
            and dangerous_internal_auto_approve == 0
            and dangerous_external_auto_approve <= 1
            and ext_forced["macro_f1"] >= r9_reference["external_metrics"]["macro_f1"] - 0.02
        )
        row = {
            "candidate": name,
            "internal_macro_f1": int_forced["macro_f1"],
            "internal_platform_recall": int_forced["platform_recall"],
            "internal_noise_recall": int_forced["noise_recall"],
            "external_macro_f1": ext_forced["macro_f1"],
            "external_platform_recall": ext_forced["platform_recall"],
            "external_noise_recall": ext_forced["noise_recall"],
            "external_noise_fp": ext_forced["noise_to_platform_fp"],
            "external_platform_fn": ext_forced["platform_to_noise_fn"],
            "external_auto_approve_precision": ext_triage["auto_approve_precision"],
            "external_auto_exclude_precision": ext_triage["auto_exclude_precision"],
            "external_coverage": ext_triage["coverage"],
            "external_review_required_count": ext_triage["review_required_count"],
            "dangerous_external_noise_auto_approve_count": dangerous_external_auto_approve,
            "internal_auto_approve_precision": int_triage["auto_approve_precision"],
            "internal_dangerous_auto_approve_count": dangerous_internal_auto_approve,
            "product_viable": viable,
        }
        table.append(row)
        reports.append({"name": name, "description": desc, "forced_classification": {"internal": int_forced, "external": ext_forced}, "triage_policy": {"internal": int_triage, "external": ext_triage}, "product_viable": viable})
    ref = next(r for r in table if r["candidate"] == "r9_reference")
    best = max([r for r in table if r["candidate"] != "r9_reference"], key=lambda r: (r["product_viable"], fnum(r["external_auto_approve_precision"]) >= PRECISION_TARGET, fnum(r["external_auto_exclude_precision"]) >= PRECISION_TARGET, r["external_coverage"], -r["internal_dangerous_auto_approve_count"], -r["dangerous_external_noise_auto_approve_count"], r["external_macro_f1"]))
    improves = (
        fnum(best["external_auto_approve_precision"]) >= fnum(ref["external_auto_approve_precision"])
        and fnum(best["external_auto_exclude_precision"]) >= fnum(ref["external_auto_exclude_precision"])
        and best["internal_dangerous_auto_approve_count"] <= ref["internal_dangerous_auto_approve_count"]
        and best["dangerous_external_noise_auto_approve_count"] <= ref["dangerous_external_noise_auto_approve_count"]
        and (
            best["external_coverage"] > ref["external_coverage"]
            or best["internal_dangerous_auto_approve_count"] < ref["internal_dangerous_auto_approve_count"]
            or best["dangerous_external_noise_auto_approve_count"] < ref["dangerous_external_noise_auto_approve_count"]
        )
    )
    decision = "R12_REPRESENTATION_GAIN_CONFIRMED" if improves else "R12_NO_CLEAR_REPRESENTATION_GAIN"
    main_report = {
        "experiment_name": "r12_targeted_impact_vs_nuisance_representation",
        "final_decision": decision,
        "best_candidate": best["candidate"],
        "three_queue_coach_mode_viable": bool(best["product_viable"]),
        "audio_only_sufficient": bool(best["product_viable"]),
        "panns_embedding_probe_status": "not_run_pretrained_dependencies_unavailable_torch_torchaudio_panns_inference_missing",
        "boundary_dataset": dataset,
        "candidate_reports": reports,
    }
    comparison = {"experiment_name": "r12_candidate_comparison", "final_decision": decision, "best_candidate": best["candidate"], "comparison_rows": table}
    queue = {"experiment_name": "r12_queue_safety_comparison", "final_decision": decision, "best_candidate": best["candidate"], "comparison_rows": table}
    OUT_MAIN_JSON.write_text(json.dumps(main_report, indent=2), encoding="utf-8")
    OUT_COMPARISON_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    OUT_QUEUE_JSON.write_text(json.dumps(queue, indent=2), encoding="utf-8")

    OUT_MAIN_MD.write_text("\n".join([
        "# r12 Targeted Impact vs Nuisance Representation", "",
        f"- final decision: `{decision}`",
        f"- best candidate: `{best['candidate']}`",
        f"- three-queue coach mode viable: `{bool(best['product_viable'])}`",
        f"- audio-only sufficient at current bar: `{bool(best['product_viable'])}`",
        "- PANNs probe: `not run with pretrained model because torch/torchaudio/panns_inference are unavailable locally; bounded proxy features were evaluated instead`",
    ]) + "\n", encoding="utf-8")
    OUT_COMPARISON_MD.write_text("\n".join([
        "# r12 Candidate Comparison", "",
        f"- final decision: `{decision}`",
        f"- best candidate: `{best['candidate']}`",
    ]) + "\n", encoding="utf-8")
    OUT_QUEUE_MD.write_text("\n".join([
        "# r12 Queue Safety Comparison", "",
        f"- final decision: `{decision}`",
        f"- best candidate: `{best['candidate']}`",
    ]) + "\n", encoding="utf-8")

    dataset_md = "\n".join([
        "# r12 Boundary Dataset", "",
        f"- row count: `{dataset['row_count']}`",
        f"- label counts: `{json.dumps(dataset['label_counts'], sort_keys=True)}`",
        f"- subtype counts: `{json.dumps(dataset['subtype_counts'], sort_keys=True)}`",
        f"- score-band counts: `{json.dumps(dataset['score_band_counts'], sort_keys=True)}`",
        f"- boundary-role counts: `{json.dumps(dataset['boundary_role_counts'], sort_keys=True)}`",
        f"- approve-risk rows: `{len(dataset['approve_risk_rows'])}`",
        f"- exclude-risk rows: `{len(dataset['exclude_risk_rows'])}`",
    ]) + "\n"
    # Include the dataset summary in the main report rather than creating an extra r12 dataset artifact.
    OUT_MAIN_MD.write_text(OUT_MAIN_MD.read_text() + "\n## Boundary Dataset\n\n" + dataset_md, encoding="utf-8")

    bench_lines = ["", "| candidate | internal macro F1 | internal platform recall | internal noise recall | external macro F1 | external platform recall | external noise recall | external noise FP | external platform FN | approve precision | exclude precision | coverage | dangerous ext approve | dangerous int approve | viable |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in table:
        bench_lines.append(f"| `{r['candidate']}` | {r['internal_macro_f1']:.4f} | {r['internal_platform_recall']:.4f} | {r['internal_noise_recall']:.4f} | {r['external_macro_f1']:.4f} | {r['external_platform_recall']:.4f} | {r['external_noise_recall']:.4f} | {r['external_noise_fp']} | {r['external_platform_fn']} | {fnum(r['external_auto_approve_precision']):.4f} | {fnum(r['external_auto_exclude_precision']):.4f} | {r['external_coverage']:.4f} | {r['dangerous_external_noise_auto_approve_count']} | {r['internal_dangerous_auto_approve_count']} | `{r['product_viable']}` |")
    table_text = "\n".join(bench_lines) + "\n"
    OUT_COMPARISON_MD.write_text(OUT_COMPARISON_MD.read_text() + table_text, encoding="utf-8")
    OUT_QUEUE_MD.write_text(OUT_QUEUE_MD.read_text() + table_text, encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_MAIN_JSON), str(OUT_MAIN_MD), str(OUT_COMPARISON_JSON), str(OUT_COMPARISON_MD), str(OUT_QUEUE_JSON), str(OUT_QUEUE_MD)], "final_decision": decision, "best_candidate": best["candidate"], "three_queue_coach_mode_viable": bool(best["product_viable"]), "audio_only_sufficient": bool(best["product_viable"])}, indent=2))


if __name__ == "__main__":
    main()
