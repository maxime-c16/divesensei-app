from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
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
EMBED_CACHE = ROOT / "outputs" / "r13_ast_embedding_cache.npz"

SOURCES = {
    "snmt": ROOT
    / "outputs/evaluation_SNMT-16min_20260417-131944/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "img_8852": ROOT
    / "outputs/evaluation_img_8852_rerun_20260406-104430/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "champigny_1704": ROOT
    / "outputs/evaluation_Champigny-17-04-9min_20260418-065417/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
}
WEIGHTS = {"snmt": 1.0, "img_8852": 1.0, "champigny_1704": 0.3}

LOW = 0.05
HIGH = 0.85
PRECISION_TARGET = 0.90
COVERAGE_TARGET = 0.40

R13_JSON = ROOT / "outputs/r13_learned_audio_probe.json"
R13_MD = ROOT / "outputs/r13_learned_audio_probe.md"
R13_QUEUE_JSON = ROOT / "outputs/r13_learned_audio_probe_queue_safety.json"
R13_QUEUE_MD = ROOT / "outputs/r13_learned_audio_probe_queue_safety.md"
R14_JSON = ROOT / "outputs/r14_audio_video_verifier_benchmark.json"
R14_MD = ROOT / "outputs/r14_audio_video_verifier_benchmark.md"
R14_QUEUE_JSON = ROOT / "outputs/r14_audio_video_queue_safety.json"
R14_QUEUE_MD = ROOT / "outputs/r14_audio_video_queue_safety.md"

VIDEO_FEATURES_ENTRY = [
    "video_motion_mean",
    "video_motion_max",
    "video_upper_motion_mean",
    "video_lower_motion_mean",
    "video_center_motion_mean",
    "video_lower_upper_motion_ratio",
]
VIDEO_FEATURES_SPLASH = [
    "video_lower_motion_mean",
    "video_lower_motion_max",
    "video_lower_motion_persistence",
    "video_late_lower_motion_ratio",
    "video_motion_temporal_cv",
    "video_brightness_delta",
]
VIDEO_FEATURES_ALL = sorted(set(VIDEO_FEATURES_ENTRY + VIDEO_FEATURES_SPLASH))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class RowRef:
    row_key: str
    label: str
    row: dict[str, Any]


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
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        rid = str(row.get("legacy_candidate_id") or f"row-{counts[sid]:04d}")
        out[f"{sid}::{rid}"] = row
    return out


def rowrefs(rows: list[dict[str, Any]]) -> list[RowRef]:
    refs = []
    for row in rows:
        sid = str(row["source_session_id"])
        rid = str(row.get("legacy_candidate_id") or "row-unknown")
        refs.append(RowRef(f"{sid}::{rid}", str(row["final_human_event_label"]), row))
    return refs


def label_int(label: str) -> int:
    return 1 if label == "platform_dive" else 0


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def fnum(value: float | None) -> float:
    return -1.0 if value is None else float(value)


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
        "auto_approve_error_rows": [compact_row(r) for r in approve if r["label"] != "platform_dive"][:20],
        "auto_exclude_error_rows": [compact_row(r) for r in exclude if r["label"] != "noise_or_other"][:20],
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "label": row["label"],
        "r9_score": row.get("r9_score"),
        "score": row.get("score"),
        "legacy_subtype": row.get("legacy_subtype"),
        "suggested_event_label_reason": row.get("suggested_event_label_reason"),
    }


def product_viable(int_triage: dict[str, Any], ext_triage: dict[str, Any], ext_forced: dict[str, Any], r9_ext_macro_f1: float) -> bool:
    return (
        fnum(ext_triage["auto_approve_precision"]) >= PRECISION_TARGET
        and fnum(ext_triage["auto_exclude_precision"]) >= PRECISION_TARGET
        and ext_triage["coverage"] >= COVERAGE_TARGET
        and ext_triage["auto_approve_error_count"] <= 1
        and int_triage["auto_approve_error_count"] == 0
        and ext_forced["macro_f1"] >= r9_ext_macro_f1 - 0.02
    )


def vec(row: dict[str, Any], names: list[str]) -> list[float]:
    values = []
    for name in names:
        value = float(row.get(name, 0.0) or 0.0)
        values.append(value if math.isfinite(value) else 0.0)
    return values


def score_band(score: float) -> str:
    if score >= HIGH:
        return "high_approve_band"
    if score <= LOW:
        return "high_exclude_band"
    return "review_band"


def decode_video_frames(path: Path, start: float, duration: float, fps: float = 4.0, width: int = 160, height: int = 90) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-t",
        f"{max(0.1, duration):.3f}",
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = width * height
    if data.size < frame_size:
        return np.zeros((0, height, width), dtype=np.float32)
    n = data.size // frame_size
    return data[: n * frame_size].reshape(n, height, width).astype(np.float32) / 255.0


def video_features(frames: np.ndarray) -> dict[str, float]:
    if len(frames) < 2:
        return {name: 0.0 for name in VIDEO_FEATURES_ALL}
    diffs = np.abs(np.diff(frames, axis=0))
    h, w = frames.shape[1], frames.shape[2]
    upper = diffs[:, : h // 2, :]
    lower = diffs[:, h // 2 :, :]
    center = diffs[:, h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
    lower_series = np.mean(lower, axis=(1, 2))
    motion_series = np.mean(diffs, axis=(1, 2))
    late_start = max(0, int(round(len(lower_series) * 0.55)))
    early_end = max(1, int(round(len(lower_series) * 0.45)))
    return {
        "video_motion_mean": float(np.mean(diffs)),
        "video_motion_max": float(np.max(motion_series)),
        "video_upper_motion_mean": float(np.mean(upper)),
        "video_lower_motion_mean": float(np.mean(lower)),
        "video_center_motion_mean": float(np.mean(center)),
        "video_lower_upper_motion_ratio": float(np.mean(lower) / max(np.mean(upper), 1e-8)),
        "video_lower_motion_max": float(np.max(lower_series)),
        "video_lower_motion_persistence": float(np.mean(lower_series > (np.median(lower_series) + np.std(lower_series)))),
        "video_late_lower_motion_ratio": float(np.mean(lower_series[late_start:]) / max(np.mean(lower_series[:early_end]), 1e-8)),
        "video_motion_temporal_cv": float(np.std(motion_series) / max(np.mean(motion_series), 1e-8)),
        "video_brightness_delta": float(abs(np.mean(frames[-1]) - np.mean(frames[0]))),
    }


def queues_from_score(rows: list[dict[str, Any]], scores: list[float]) -> list[str]:
    return ["auto_approved" if s >= HIGH else ("auto_excluded" if s <= LOW else "needs_review") for s in scores]


def queues_guarded_approve(rows: list[dict[str, Any]], scores: list[float], approve_min: float = 0.70, exclude_max: float = 0.30) -> list[str]:
    out = []
    for row, score in zip(rows, scores):
        r9 = float(row["r9_score"])
        if r9 >= HIGH and score >= approve_min:
            out.append("auto_approved")
        elif r9 <= LOW and score <= exclude_max:
            out.append("auto_excluded")
        else:
            out.append("needs_review")
    return out


def write_table_md(path: Path, title: str, decision: str, rows: list[dict[str, Any]], extra: list[str] | None = None) -> None:
    lines = [f"# {title}", "", f"- final decision: `{decision}`"]
    if extra:
        lines.extend(["", *extra])
    lines.extend(
        [
            "",
            "| candidate | internal macro F1 | internal platform recall | internal noise recall | external macro F1 | external platform recall | external noise recall | external noise FP | external platform FN | approve precision | exclude precision | coverage | dangerous ext approve | dangerous int approve | viable |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in rows:
        lines.append(
            f"| `{r['candidate']}` | {r['internal_macro_f1']:.4f} | {r['internal_platform_recall']:.4f} | "
            f"{r['internal_noise_recall']:.4f} | {r['external_macro_f1']:.4f} | {r['external_platform_recall']:.4f} | "
            f"{r['external_noise_recall']:.4f} | {r['external_noise_fp']} | {r['external_platform_fn']} | "
            f"{fnum(r['external_auto_approve_precision']):.4f} | {fnum(r['external_auto_exclude_precision']):.4f} | "
            f"{r['external_coverage']:.4f} | {r['dangerous_external_noise_auto_approve_count']} | "
            f"{r['internal_dangerous_auto_approve_count']} | `{r['product_viable']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phase5 = load_module("phase5_r13_runtime", PHASE5)
    bench = load_module("nuisance_r13_runtime", NUISANCE)
    r9_reference = json.loads(R9_REF.read_text())
    r9_ext_macro_f1 = float(r9_reference["external_metrics"]["macro_f1"])

    preview = row_key_map(load_jsonl(PREVIEW))
    lists = json.loads(DATASET.read_text())
    base_train = [("base", RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])])) for i in lists["train_rows"]]
    internal_refs = [RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])]) for i in lists["holdout_rows"]]
    external_refs = [RowRef(str(r["row_key"]), str(r["final_human_event_label"]), r) for r in json.loads(EXTERNAL.read_text())["rows"]]
    session_refs = {
        k: rowrefs([r for r in load_jsonl(p) if r.get("final_human_event_label") in {"platform_dive", "noise_or_other"}])
        for k, p in SOURCES.items()
    }
    train_refs = base_train + [(k, ref) for k, refs in session_refs.items() for ref in refs]
    all_refs = [r for _, r in train_refs] + internal_refs + external_refs

    audio: dict[str, np.ndarray] = {}
    fmap: dict[str, dict[str, float]] = {}
    segments: dict[str, np.ndarray] = {}
    video_paths: dict[str, Path] = {}
    for item in all_refs:
        sid = str(item.row["source_session_id"])
        if sid not in audio:
            source_root = phase5.resolve_source_root(str(item.row["source_session_root"]))
            video_paths[sid] = source_root / "web/session_source_review.mp4"
            audio[sid] = phase5.decode_audio_mono(video_paths[sid], phase5.SAMPLE_RATE)
        start = max(0.0, phase5.to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(item.row.get("event_window_end_seconds")))
        sig = audio[sid][int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        segments[item.row_key] = sig.astype(np.float32)
        fmap[item.row_key] = {
            **phase5.extract_features(sig, phase5.SAMPLE_RATE),
            **bench.nuisance_features(phase5, sig, phase5.SAMPLE_RATE),
        }

    def r9_vec(item: RowRef) -> list[float]:
        return bench.vector_for(phase5, item, fmap, bench.NOISE_BOUNDARY_COMPACT)

    x_train = np.asarray([r9_vec(ref) for _, ref in train_refs], dtype=np.float64)
    y_train = np.asarray([label_int(ref.label) for _, ref in train_refs], dtype=np.int64)
    sample_weight = np.ones(len(train_refs), dtype=np.float64)
    base_total = sum(1 for source, _ in train_refs if source == "base")
    by_source: dict[str, list[int]] = {}
    for idx, (source, _) in enumerate(train_refs):
        by_source.setdefault(source, []).append(idx)
    for source, idxs in by_source.items():
        if source == "base":
            continue
        per_item = base_total * WEIGHTS[source] / len(idxs)
        for idx in idxs:
            sample_weight[idx] = per_item

    r9_model = XGBClassifier(
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
    r9_model.fit(x_train, y_train, sample_weight=sample_weight)

    def make_rows(refs: list[RowRef], scores: list[float], split: str) -> list[dict[str, Any]]:
        rows = []
        for item, score in zip(refs, scores):
            src = item.row
            rows.append(
                {
                    "split": split,
                    "row_key": item.row_key,
                    "source_session_id": str(src.get("source_session_id") or item.row_key.split("::", 1)[0]),
                    "label": item.label,
                    "r9_score": float(score),
                    "score_band": score_band(float(score)),
                    "legacy_subtype": src.get("legacy_subtype"),
                    "suggested_event_label_reason": src.get("suggested_event_label_reason"),
                    "event_window_start_seconds": phase5.to_float(src.get("event_window_start_seconds")),
                    "event_window_end_seconds": phase5.to_float(src.get("event_window_end_seconds")),
                    "source_session_root": str(src.get("source_session_root")),
                }
            )
        return rows

    train_items = [r for _, r in train_refs]
    internal_scores_r9 = r9_model.predict_proba(np.asarray([r9_vec(r) for r in internal_refs], dtype=np.float64))[:, 1].tolist()
    external_scores_r9 = r9_model.predict_proba(np.asarray([r9_vec(r) for r in external_refs], dtype=np.float64))[:, 1].tolist()
    train_rows = make_rows(train_items, r9_model.predict_proba(x_train)[:, 1].tolist(), "train_augmented")
    internal_rows = make_rows(internal_refs, internal_scores_r9, "internal_holdout")
    external_rows = make_rows(external_refs, external_scores_r9, "external_holdout")

    r9_int_triage = triage(internal_rows, queues_from_score(internal_rows, [r["r9_score"] for r in internal_rows]))
    r9_ext_triage = triage(external_rows, queues_from_score(external_rows, [r["r9_score"] for r in external_rows]))
    r9_int_forced = forced_metrics([r.label for r in internal_refs], internal_scores_r9)
    r9_ext_forced = forced_metrics([r.label for r in external_refs], external_scores_r9)

    feasibility = {
        "r13_panns_embedding_probe": {
            "status": "not_feasible_local_stack",
            "reason": "panns-inference requires librosa, which pulled llvmlite source build and failed because LLVMConfig.cmake is unavailable",
        },
        "r13_ast_embedding_probe": {"status": "attempted"},
        "r13_ssast_embedding_probe": {
            "status": "not_feasible_local_stack",
            "reason": "no directly supported SSAST extractor/model path is available in the installed bounded Transformers stack",
        },
    }

    candidate_reports = []
    comparison_rows = []

    def add_candidate(name: str, description: str, int_scores: list[float], ext_scores: list[float], int_queues: list[str], ext_queues: list[str]) -> None:
        for row, score in zip(internal_rows, int_scores):
            row["score"] = float(score)
        for row, score in zip(external_rows, ext_scores):
            row["score"] = float(score)
        int_tri = triage(internal_rows, int_queues)
        ext_tri = triage(external_rows, ext_queues)
        int_forced = forced_metrics([r["label"] for r in internal_rows], int_scores)
        ext_forced = forced_metrics([r["label"] for r in external_rows], ext_scores)
        viable = product_viable(int_tri, ext_tri, ext_forced, r9_ext_macro_f1)
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
            "external_auto_approve_precision": ext_tri["auto_approve_precision"],
            "external_auto_exclude_precision": ext_tri["auto_exclude_precision"],
            "external_coverage": ext_tri["coverage"],
            "external_review_required_count": ext_tri["review_required_count"],
            "dangerous_external_noise_auto_approve_count": ext_tri["auto_approve_error_count"],
            "internal_auto_approve_precision": int_tri["auto_approve_precision"],
            "internal_dangerous_auto_approve_count": int_tri["auto_approve_error_count"],
            "product_viable": viable,
        }
        comparison_rows.append(row)
        candidate_reports.append(
            {
                "name": name,
                "description": description,
                "forced_classification": {"internal": int_forced, "external": ext_forced},
                "triage_policy": {"internal": int_tri, "external": ext_tri},
                "product_viable": viable,
            }
        )

    add_candidate(
        "r9_weighted_audio_reference",
        "Current promoted weighted r9 audio reference.",
        internal_scores_r9,
        external_scores_r9,
        queues_from_score(internal_rows, internal_scores_r9),
        queues_from_score(external_rows, external_scores_r9),
    )

    ast_status = "not_run"
    ast_error = None
    try:
        from transformers import ASTFeatureExtractor, ASTModel
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        keys = [r.row_key for r in all_refs]
        if EMBED_CACHE.exists():
            cache = np.load(EMBED_CACHE, allow_pickle=True)
            cached_keys = [str(x) for x in cache["keys"].tolist()]
            if cached_keys == keys:
                embeddings = cache["embeddings"]
            else:
                EMBED_CACHE.unlink()
                embeddings = None
        else:
            embeddings = None
        if embeddings is None:
            extractor = ASTFeatureExtractor.from_pretrained(model_name)
            model = ASTModel.from_pretrained(model_name)
            model.eval()
            batches = []
            with torch.no_grad():
                for start in range(0, len(all_refs), 4):
                    refs = all_refs[start : start + 4]
                    waves = [segments[r.row_key] for r in refs]
                    inputs = extractor(waves, sampling_rate=phase5.SAMPLE_RATE, return_tensors="pt", padding=True)
                    outputs = model(**inputs)
                    pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    batches.append(pooled.astype(np.float32))
            embeddings = np.vstack(batches)
            np.savez_compressed(EMBED_CACHE, keys=np.asarray(keys, dtype=object), embeddings=embeddings)
        index = {key: idx for idx, key in enumerate(keys)}

        def emb(refs: list[RowRef]) -> np.ndarray:
            return np.asarray([embeddings[index[r.row_key]] for r in refs], dtype=np.float64)

        ast_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.25, random_state=42))
        ast_model.fit(emb(train_items), y_train, logisticregression__sample_weight=sample_weight)
        ast_int = ast_model.predict_proba(emb(internal_refs))[:, 1].tolist()
        ast_ext = ast_model.predict_proba(emb(external_refs))[:, 1].tolist()
        add_candidate(
            "r13_ast_embedding_probe",
            "True pretrained AST AudioSet embedding with a tiny weighted logistic probe.",
            ast_int,
            ast_ext,
            queues_from_score(internal_rows, ast_int),
            queues_from_score(external_rows, ast_ext),
        )

        fusion_train = np.hstack([emb(train_items), np.asarray([[r["r9_score"]] for r in train_rows], dtype=np.float64)])
        fusion_int_x = np.hstack([emb(internal_refs), np.asarray([[r["r9_score"]] for r in internal_rows], dtype=np.float64)])
        fusion_ext_x = np.hstack([emb(external_refs), np.asarray([[r["r9_score"]] for r in external_rows], dtype=np.float64)])
        fusion_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.25, random_state=42))
        fusion_model.fit(fusion_train, y_train, logisticregression__sample_weight=sample_weight)
        fusion_int = fusion_model.predict_proba(fusion_int_x)[:, 1].tolist()
        fusion_ext = fusion_model.predict_proba(fusion_ext_x)[:, 1].tolist()
        add_candidate(
            "r13_ast_embedding_plus_r9_guarded_probe",
            "AST embedding plus r9 score, with the same conservative guarded queue policy used for bounded verifier tests.",
            fusion_int,
            fusion_ext,
            queues_guarded_approve(internal_rows, fusion_int),
            queues_guarded_approve(external_rows, fusion_ext),
        )
        ast_status = "completed"
        feasibility["r13_ast_embedding_probe"]["status"] = "completed"
        feasibility["r13_ast_embedding_probe"]["model"] = model_name
    except Exception as exc:  # pragma: no cover - records environment/model download failures.
        ast_status = "failed"
        ast_error = f"{type(exc).__name__}: {exc}"
        feasibility["r13_ast_embedding_probe"]["status"] = "failed"
        feasibility["r13_ast_embedding_probe"]["reason"] = ast_error

    learned_candidates = [r for r in comparison_rows if r["candidate"] != "r9_weighted_audio_reference"]
    best_r13 = max(
        learned_candidates,
        key=lambda r: (
            r["product_viable"],
            fnum(r["external_auto_approve_precision"]),
            fnum(r["external_auto_exclude_precision"]),
            -r["dangerous_external_noise_auto_approve_count"],
            -r["internal_dangerous_auto_approve_count"],
            r["external_coverage"],
            r["external_macro_f1"],
        ),
    ) if learned_candidates else None
    r13_decision = "R13_AUDIO_ONLY_STILL_PLAUSIBLE" if best_r13 and best_r13["product_viable"] else "R13_AUDIO_ONLY_UPPER_BOUND_NOT_ENOUGH"
    r13_report = {
        "experiment_name": "r13_learned_audio_probe",
        "final_decision": r13_decision,
        "branch_a_success": r13_decision == "R13_AUDIO_ONLY_STILL_PLAUSIBLE",
        "best_candidate": best_r13["candidate"] if best_r13 else None,
        "feasibility": feasibility,
        "ast_status": ast_status,
        "ast_error": ast_error,
        "candidate_reports": candidate_reports,
        "comparison_rows": comparison_rows,
    }
    R13_JSON.write_text(json.dumps(r13_report, indent=2), encoding="utf-8")
    R13_QUEUE_JSON.write_text(json.dumps({"experiment_name": "r13_learned_audio_probe_queue_safety", "final_decision": r13_decision, "comparison_rows": comparison_rows}, indent=2), encoding="utf-8")
    write_table_md(
        R13_MD,
        "r13 Learned-Audio Probe",
        r13_decision,
        comparison_rows,
        [
            f"- best candidate: `{best_r13['candidate'] if best_r13 else None}`",
            f"- AST status: `{ast_status}`",
            f"- PANNs status: `{feasibility['r13_panns_embedding_probe']['status']}`",
            f"- SSAST status: `{feasibility['r13_ssast_embedding_probe']['status']}`",
        ],
    )
    write_table_md(R13_QUEUE_MD, "r13 Learned-Audio Probe Queue Safety", r13_decision, comparison_rows)

    if r13_decision == "R13_AUDIO_ONLY_STILL_PLAUSIBLE":
        print(json.dumps({"r13_decision": r13_decision, "branch_b_triggered": False, "wrote": [str(R13_JSON), str(R13_MD), str(R13_QUEUE_JSON), str(R13_QUEUE_MD)]}, indent=2))
        return

    video_cache: dict[str, np.ndarray] = {}
    def add_video(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            sid = row["source_session_id"]
            path = video_paths.get(sid)
            if path is None:
                path = phase5.resolve_source_root(row["source_session_root"]) / "web/session_source_review.mp4"
                video_paths[sid] = path
            key = f"{row['row_key']}::{row['event_window_start_seconds']:.3f}::{row['event_window_end_seconds']:.3f}"
            if key not in video_cache:
                start = max(0.0, float(row["event_window_start_seconds"]))
                duration = max(0.2, float(row["event_window_end_seconds"]) - start)
                video_cache[key] = decode_video_frames(path, start, duration)
            row.update(video_features(video_cache[key]))

    add_video(train_rows)
    add_video(internal_rows)
    add_video(external_rows)

    video_candidates = {
        "r14_video_entry_presence_verifier": VIDEO_FEATURES_ENTRY,
        "r14_video_splash_morphology_verifier": VIDEO_FEATURES_SPLASH,
        "r14_audio_video_boundary_fusion": ["r9_score"] + VIDEO_FEATURES_ALL,
    }
    r14_reports = []
    r14_rows = [
        next(r for r in comparison_rows if r["candidate"] == "r9_weighted_audio_reference")
    ]
    y_fit = [label_int(r["label"]) for r in train_rows]
    for name, features in video_candidates.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42))
        model.fit(np.asarray([vec(r, features) for r in train_rows], dtype=np.float64), y_fit, logisticregression__sample_weight=sample_weight)
        int_scores = model.predict_proba(np.asarray([vec(r, features) for r in internal_rows], dtype=np.float64))[:, 1].tolist()
        ext_scores = model.predict_proba(np.asarray([vec(r, features) for r in external_rows], dtype=np.float64))[:, 1].tolist()
        int_q = queues_guarded_approve(internal_rows, int_scores)
        ext_q = queues_guarded_approve(external_rows, ext_scores)
        for row, score in zip(internal_rows, int_scores):
            row["score"] = float(score)
        for row, score in zip(external_rows, ext_scores):
            row["score"] = float(score)
        int_tri = triage(internal_rows, int_q)
        ext_tri = triage(external_rows, ext_q)
        int_forced = forced_metrics([r["label"] for r in internal_rows], int_scores)
        ext_forced = forced_metrics([r["label"] for r in external_rows], ext_scores)
        viable = product_viable(int_tri, ext_tri, ext_forced, r9_ext_macro_f1)
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
            "external_auto_approve_precision": ext_tri["auto_approve_precision"],
            "external_auto_exclude_precision": ext_tri["auto_exclude_precision"],
            "external_coverage": ext_tri["coverage"],
            "external_review_required_count": ext_tri["review_required_count"],
            "dangerous_external_noise_auto_approve_count": ext_tri["auto_approve_error_count"],
            "internal_auto_approve_precision": int_tri["auto_approve_precision"],
            "internal_dangerous_auto_approve_count": int_tri["auto_approve_error_count"],
            "product_viable": viable,
        }
        r14_rows.append(row)
        r14_reports.append(
            {
                "name": name,
                "features": features,
                "forced_classification": {"internal": int_forced, "external": ext_forced},
                "triage_policy": {"internal": int_tri, "external": ext_tri},
                "product_viable": viable,
            }
        )

    best_r14 = max(
        [r for r in r14_rows if r["candidate"] != "r9_weighted_audio_reference"],
        key=lambda r: (
            r["product_viable"],
            fnum(r["external_auto_approve_precision"]),
            fnum(r["external_auto_exclude_precision"]),
            -r["dangerous_external_noise_auto_approve_count"],
            -r["internal_dangerous_auto_approve_count"],
            r["external_coverage"],
            r["external_macro_f1"],
        ),
    )
    r14_decision = "R14_BOUNDED_MULTIMODAL_GAIN_CONFIRMED" if best_r14["product_viable"] else "R14_BOUNDED_MULTIMODAL_NO_CLEAR_GAIN"
    r14_report = {
        "experiment_name": "r14_audio_video_verifier_benchmark",
        "final_decision": r14_decision,
        "branch_b_triggered": True,
        "best_candidate": best_r14["candidate"],
        "candidate_reports": r14_reports,
        "comparison_rows": r14_rows,
        "video_feature_method": "bounded ffmpeg-decoded grayscale 4fps 160x90 event-window motion features; no detector/taxonomy/model-family changes",
    }
    R14_JSON.write_text(json.dumps(r14_report, indent=2), encoding="utf-8")
    R14_QUEUE_JSON.write_text(json.dumps({"experiment_name": "r14_audio_video_queue_safety", "final_decision": r14_decision, "best_candidate": best_r14["candidate"], "comparison_rows": r14_rows}, indent=2), encoding="utf-8")
    write_table_md(R14_MD, "r14 Audio-Video Verifier Benchmark", r14_decision, r14_rows, [f"- best candidate: `{best_r14['candidate']}`"])
    write_table_md(R14_QUEUE_MD, "r14 Audio-Video Queue Safety", r14_decision, r14_rows, [f"- best candidate: `{best_r14['candidate']}`"])
    print(
        json.dumps(
            {
                "r13_decision": r13_decision,
                "branch_b_triggered": True,
                "r14_decision": r14_decision,
                "best_r14": best_r14["candidate"],
                "wrote": [str(R13_JSON), str(R13_MD), str(R13_QUEUE_JSON), str(R13_QUEUE_MD), str(R14_JSON), str(R14_MD), str(R14_QUEUE_JSON), str(R14_QUEUE_MD)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
