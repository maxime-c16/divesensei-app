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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
DATASET = ROOT / "outputs/platform_noise_es4_dataset_rows.json"
EXTERNAL = ROOT / "outputs/external_holdout_slice.json"
PREVIEW = ROOT / "outputs/event_window_manifest_preview.jsonl"
R9_REF = ROOT / "outputs/r9_compact_nuisance_generalization_weighted.json"
CLIP_CACHE = ROOT / "outputs/r15_clip_frame_embedding_cache.npz"

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

OUT_DATASET_JSON = ROOT / "outputs/r15_visual_verifier_dataset.json"
OUT_DATASET_MD = ROOT / "outputs/r15_visual_verifier_dataset.md"
OUT_BENCH_JSON = ROOT / "outputs/r15_visual_verifier_benchmark.json"
OUT_BENCH_MD = ROOT / "outputs/r15_visual_verifier_benchmark.md"
OUT_QUEUE_JSON = ROOT / "outputs/r15_audio_video_queue_safety.json"
OUT_QUEUE_MD = ROOT / "outputs/r15_audio_video_queue_safety.md"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


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
    out, counts = {}, {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        rid = str(row.get("legacy_candidate_id") or f"row-{counts[sid]:04d}")
        out[f"{sid}::{rid}"] = row
    return out


def manifest_refs(rows: list[dict[str, Any]]) -> list[RowRef]:
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


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "split": row["split"],
        "label": row["label"],
        "r9_score": row["r9_score"],
        "verifier_score": row.get("verifier_score"),
        "role": row["role"],
        "legacy_subtype": row.get("legacy_subtype"),
        "source_session_id": row["source_session_id"],
    }


def role(score: float, label: str) -> str:
    if score >= HIGH and label == "noise_or_other":
        return "approve_risk"
    if score >= HIGH and label == "platform_dive":
        return "high_confidence_platform_control"
    if score <= LOW and label == "platform_dive":
        return "exclude_risk"
    if score <= LOW and label == "noise_or_other":
        return "high_confidence_nuisance_control"
    if LOW < score < HIGH:
        return "ambiguous_review_band"
    return "other"


def triage(rows: list[dict[str, Any]], queues: list[str]) -> dict[str, Any]:
    approve = [r for r, q in zip(rows, queues) if q == "auto_approved"]
    exclude = [r for r, q in zip(rows, queues) if q == "auto_excluded"]
    review = [r for r, q in zip(rows, queues) if q == "needs_review"]
    accepted = approve + exclude
    y = [label_int(r["label"]) for r in accepted]
    pred = [1] * len(approve) + [0] * len(exclude)
    approve_ok = sum(1 for r in approve if r["label"] == "platform_dive")
    exclude_ok = sum(1 for r in exclude if r["label"] == "noise_or_other")
    return {
        "row_count": len(rows),
        "coverage": safe_div(len(accepted), len(rows)) or 0.0,
        "auto_approve_count": len(approve),
        "auto_exclude_count": len(exclude),
        "review_required_count": len(review),
        "review_band_size": len(review),
        "auto_approve_precision": safe_div(approve_ok, len(approve)),
        "auto_exclude_precision": safe_div(exclude_ok, len(exclude)),
        "dangerous_auto_approve_count": len(approve) - approve_ok,
        "dangerous_auto_exclude_count": len(exclude) - exclude_ok,
        "accepted_accuracy": float(accuracy_score(y, pred)) if y else None,
        "accepted_macro_f1": float(f1_score(y, pred, average="macro")) if y and len(set(y + pred)) > 1 else None,
        "review_required_label_counts": dict(sorted(Counter(r["label"] for r in review).items())),
        "review_required_role_counts": dict(sorted(Counter(r["role"] for r in review).items())),
        "auto_approve_error_rows": [compact_row(r) for r in approve if r["label"] != "platform_dive"][:20],
        "auto_exclude_error_rows": [compact_row(r) for r in exclude if r["label"] != "noise_or_other"][:20],
    }


def r9_queues(rows: list[dict[str, Any]]) -> list[str]:
    return ["auto_approved" if r["r9_score"] >= HIGH else ("auto_excluded" if r["r9_score"] <= LOW else "needs_review") for r in rows]


def verifier_queues(rows: list[dict[str, Any]], scores: list[float], policy: dict[str, float]) -> list[str]:
    out = []
    for row, score in zip(rows, scores):
        r9 = row["r9_score"]
        if r9 >= HIGH:
            out.append("auto_approved" if score >= policy["high_approve_min"] else "needs_review")
        elif r9 <= LOW:
            out.append("auto_excluded" if score <= policy["low_exclude_max"] else "needs_review")
        elif score >= policy["review_approve_min"]:
            out.append("auto_approved")
        elif score <= policy["review_exclude_max"]:
            out.append("auto_excluded")
        else:
            out.append("needs_review")
    return out


def product_viable(internal: dict[str, Any], external: dict[str, Any]) -> bool:
    return (
        external["dangerous_auto_approve_count"] <= 1
        and internal["dangerous_auto_approve_count"] == 0
        and fnum(external["auto_approve_precision"]) >= PRECISION_TARGET
        and fnum(external["auto_exclude_precision"]) >= PRECISION_TARGET
        and external["coverage"] >= COVERAGE_TARGET
    )


def decode_gray_frames(path: Path, start: float, duration: float, fps: float = 5.0, width: int = 224, height: int = 224) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-t",
        f"{max(0.2, duration):.3f}",
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


def decode_rgb_images(path: Path, start: float, duration: float, fps: float = 2.0, width: int = 224, height: int = 224) -> list[Image.Image]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-t",
        f"{max(0.2, duration):.3f}",
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={width}:{height},format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = width * height * 3
    if data.size < frame_size:
        return []
    n = data.size // frame_size
    frames = data[: n * frame_size].reshape(n, height, width, 3)
    return [Image.fromarray(frame) for frame in frames]


def morphology_v2_features(frames: np.ndarray) -> dict[str, float]:
    names = [
        "motion_mean",
        "motion_max",
        "lower_motion_mean",
        "lower_motion_max",
        "upper_motion_mean",
        "lower_upper_ratio",
        "center_motion_mean",
        "motion_persistence",
        "motion_peakiness",
        "late_early_lower_ratio",
        "spatial_spread_mean",
        "spatial_spread_max",
        "temporal_cv",
        "brightness_delta",
        "lower_brightness_delta",
    ]
    if len(frames) < 2:
        return {name: 0.0 for name in names}
    diffs = np.abs(np.diff(frames, axis=0))
    h, w = frames.shape[1], frames.shape[2]
    lower = diffs[:, h // 2 :, :]
    upper = diffs[:, : h // 2, :]
    center = diffs[:, h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
    lower_series = np.mean(lower, axis=(1, 2))
    motion_series = np.mean(diffs, axis=(1, 2))
    late_start = max(0, int(round(len(lower_series) * 0.55)))
    early_end = max(1, int(round(len(lower_series) * 0.45)))
    spread = []
    for diff in diffs:
        thresh = np.percentile(diff, 90)
        mask = diff >= thresh
        if not np.any(mask):
            spread.append(0.0)
            continue
        ys, xs = np.where(mask)
        spread.append(float((np.std(xs) / max(w, 1)) + (np.std(ys) / max(h, 1))))
    return {
        "motion_mean": float(np.mean(diffs)),
        "motion_max": float(np.max(motion_series)),
        "lower_motion_mean": float(np.mean(lower)),
        "lower_motion_max": float(np.max(lower_series)),
        "upper_motion_mean": float(np.mean(upper)),
        "lower_upper_ratio": float(np.mean(lower) / max(np.mean(upper), 1e-8)),
        "center_motion_mean": float(np.mean(center)),
        "motion_persistence": float(np.mean(motion_series > np.median(motion_series) + np.std(motion_series))),
        "motion_peakiness": float(np.max(motion_series) / max(np.mean(motion_series), 1e-8)),
        "late_early_lower_ratio": float(np.mean(lower_series[late_start:]) / max(np.mean(lower_series[:early_end]), 1e-8)),
        "spatial_spread_mean": float(np.mean(spread)),
        "spatial_spread_max": float(np.max(spread)),
        "temporal_cv": float(np.std(motion_series) / max(np.mean(motion_series), 1e-8)),
        "brightness_delta": float(abs(np.mean(frames[-1]) - np.mean(frames[0]))),
        "lower_brightness_delta": float(abs(np.mean(frames[-1, h // 2 :, :]) - np.mean(frames[0, h // 2 :, :]))),
    }


def vec(row: dict[str, Any], names: list[str]) -> list[float]:
    values = []
    for name in names:
        value = float(row.get(name, 0.0) or 0.0)
        values.append(value if math.isfinite(value) else 0.0)
    return values


def choose_policy(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, float]:
    candidates = []
    for high_approve in [0.55, 0.65, 0.75, 0.85]:
        for review_approve in [0.80, 0.90, 0.95]:
            for low_exclude in [0.20, 0.30, 0.40]:
                for review_exclude in [0.05, 0.10, 0.20]:
                    policy = {
                        "high_approve_min": high_approve,
                        "review_approve_min": review_approve,
                        "low_exclude_max": low_exclude,
                        "review_exclude_max": review_exclude,
                    }
                    queues = verifier_queues(rows, scores, policy)
                    tri = triage(rows, queues)
                    safe = (
                        tri["dangerous_auto_approve_count"] == 0
                        and fnum(tri["auto_approve_precision"]) >= PRECISION_TARGET
                        and (tri["auto_exclude_count"] == 0 or fnum(tri["auto_exclude_precision"]) >= PRECISION_TARGET)
                    )
                    rank = (
                        int(safe),
                        float(tri["coverage"]),
                        fnum(tri["auto_approve_precision"]),
                        fnum(tri["auto_exclude_precision"]),
                        -int(tri["dangerous_auto_approve_count"]),
                    )
                    candidates.append((rank, policy))
    return max(candidates, key=lambda item: item[0])[1]


def write_dataset_md(dataset: dict[str, Any]) -> None:
    lines = [
        "# r15 Visual Verifier Dataset",
        "",
        f"- row count: `{dataset['row_count']}`",
        f"- label counts: `{json.dumps(dataset['label_counts'], sort_keys=True)}`",
        f"- role counts: `{json.dumps(dataset['role_counts'], sort_keys=True)}`",
        f"- subtype counts: `{json.dumps(dataset['subtype_counts'], sort_keys=True)}`",
        f"- source-session counts: `{json.dumps(dataset['source_session_counts'], sort_keys=True)}`",
        f"- approve-risk rows: `{len(dataset['approve_risk_rows'])}`",
        f"- exclude-risk rows: `{len(dataset['exclude_risk_rows'])}`",
        "",
        "The benchmark materializes clip specifications and cached short-clip visual features/embeddings rather than storing duplicate video clips.",
    ]
    OUT_DATASET_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- final decision: `{payload['final_decision']}`",
        f"- best candidate: `{payload['best_candidate']}`",
        "",
        "| candidate | ext approve precision | ext exclude precision | ext coverage | ext dangerous approve | int dangerous approve | ext accepted accuracy | ext accepted macro F1 | review rows | viable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            f"| `{row['candidate']}` | {fnum(row['external_auto_approve_precision']):.4f} | "
            f"{fnum(row['external_auto_exclude_precision']):.4f} | {row['external_coverage']:.4f} | "
            f"{row['dangerous_external_noise_auto_approve_count']} | {row['dangerous_internal_auto_approve_count']} | "
            f"{fnum(row['external_accepted_accuracy']):.4f} | {fnum(row['external_accepted_macro_f1']):.4f} | "
            f"{row['external_review_required_count']} | `{row['product_viable']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phase5 = load_module("phase5_r15", PHASE5)
    bench = load_module("nuisance_r15", NUISANCE)

    preview = row_key_map(load_jsonl(PREVIEW))
    lists = json.loads(DATASET.read_text())
    base_train = [("base", RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])])) for i in lists["train_rows"]]
    internal_refs = [RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])]) for i in lists["holdout_rows"]]
    external_refs = [RowRef(str(r["row_key"]), str(r["final_human_event_label"]), r) for r in json.loads(EXTERNAL.read_text())["rows"]]
    session_refs = {
        k: manifest_refs([r for r in load_jsonl(p) if r.get("final_human_event_label") in {"platform_dive", "noise_or_other"}])
        for k, p in SOURCES.items()
    }
    train_refs = base_train + [(k, ref) for k, refs in session_refs.items() for ref in refs]
    train_items = [r for _, r in train_refs]
    all_refs = train_items + internal_refs + external_refs

    audio: dict[str, np.ndarray] = {}
    video_paths: dict[str, Path] = {}
    fmap: dict[str, dict[str, float]] = {}
    for item in all_refs:
        sid = str(item.row["source_session_id"])
        if sid not in audio:
            root = phase5.resolve_source_root(str(item.row["source_session_root"]))
            video_paths[sid] = root / "web/session_source_review.mp4"
            audio[sid] = phase5.decode_audio_mono(video_paths[sid], phase5.SAMPLE_RATE)
        start = max(0.0, phase5.to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(item.row.get("event_window_end_seconds")))
        sig = audio[sid][int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap[item.row_key] = {**phase5.extract_features(sig, phase5.SAMPLE_RATE), **bench.nuisance_features(phase5, sig, phase5.SAMPLE_RATE)}

    def r9_vec(item: RowRef) -> list[float]:
        return bench.vector_for(phase5, item, fmap, bench.NOISE_BOUNDARY_COMPACT)

    x_train = np.asarray([r9_vec(ref) for ref in train_items], dtype=np.float64)
    y_train = np.asarray([label_int(ref.label) for ref in train_items], dtype=np.int64)
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
    model.fit(x_train, y_train, sample_weight=sample_weight)

    def make_rows(refs: list[RowRef], scores: list[float], split: str) -> list[dict[str, Any]]:
        rows = []
        for ref, score in zip(refs, scores):
            row = ref.row
            start = max(0.0, phase5.to_float(row.get("event_window_start_seconds")))
            end = max(start + 0.05, phase5.to_float(row.get("event_window_end_seconds")))
            payload = {
                "split": split,
                "row_key": ref.row_key,
                "label": ref.label,
                "source_session_id": str(row.get("source_session_id") or ref.row_key.split("::", 1)[0]),
                "source_session_root": str(row.get("source_session_root")),
                "legacy_subtype": row.get("legacy_subtype"),
                "suggested_event_label_reason": row.get("suggested_event_label_reason"),
                "r9_score": float(score),
                "event_window_start_seconds": start,
                "event_window_end_seconds": end,
                "clip_spec": {"start_seconds": start, "end_seconds": end, "path": str(video_paths[str(row["source_session_id"])])},
            }
            payload["role"] = role(float(score), ref.label)
            rows.append(payload)
        return rows

    train_rows = make_rows(train_items, model.predict_proba(x_train)[:, 1].tolist(), "train_augmented")
    internal_rows = make_rows(internal_refs, model.predict_proba(np.asarray([r9_vec(r) for r in internal_refs], dtype=np.float64))[:, 1].tolist(), "internal_holdout")
    external_rows = make_rows(external_refs, model.predict_proba(np.asarray([r9_vec(r) for r in external_refs], dtype=np.float64))[:, 1].tolist(), "external_holdout")

    benchmark_rows = [r for r in internal_rows + external_rows if r["role"] != "other"]
    dataset = {
        "dataset_name": "r15_visual_verifier_dataset",
        "stage_1_reference": "r9_compact_nuisance_generalization_weighted",
        "row_count": len(benchmark_rows),
        "label_counts": dict(sorted(Counter(r["label"] for r in benchmark_rows).items())),
        "role_counts": dict(sorted(Counter(r["role"] for r in benchmark_rows).items())),
        "subtype_counts": dict(sorted(Counter(str(r.get("legacy_subtype") or "none") for r in benchmark_rows).items())),
        "source_session_counts": dict(sorted(Counter(r["source_session_id"] for r in benchmark_rows).items())),
        "approve_risk_rows": [compact_row(r) for r in benchmark_rows if r["role"] == "approve_risk"],
        "exclude_risk_rows": [compact_row(r) for r in benchmark_rows if r["role"] == "exclude_risk"],
        "rows": [compact_row(r) | {"clip_spec": r["clip_spec"]} for r in benchmark_rows],
    }
    OUT_DATASET_JSON.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    write_dataset_md(dataset)

    visual_rows = train_rows + internal_rows + external_rows
    morph_features = None
    clip_embeddings = None
    keys = [r["row_key"] for r in visual_rows]
    if CLIP_CACHE.exists():
        cache = np.load(CLIP_CACHE, allow_pickle=True)
        if [str(x) for x in cache["keys"].tolist()] == keys:
            clip_embeddings = cache["clip_embeddings"]
            morph_features = cache["morph_features"]
    if clip_embeddings is None or morph_features is None:
        from transformers import CLIPModel, CLIPProcessor
        import torch

        torch.set_num_threads(1)
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name)
        clip_model.eval()
        emb_batches = []
        morph_batches = []
        with torch.no_grad():
            for row in visual_rows:
                path = Path(row["clip_spec"]["path"])
                start = row["clip_spec"]["start_seconds"]
                duration = row["clip_spec"]["end_seconds"] - start
                frames = decode_rgb_images(path, start, duration)
                gray = decode_gray_frames(path, start, duration)
                morph = morphology_v2_features(gray)
                morph_batches.append([morph[k] for k in sorted(morph)])
                if not frames:
                    emb_batches.append(np.zeros(1536, dtype=np.float32))
                    continue
                inputs = processor(images=frames, return_tensors="pt", padding=True)
                features = clip_model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                arr = features.cpu().numpy().astype(np.float32)
                emb_batches.append(np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[-1] - arr[0]]).astype(np.float32))
        clip_embeddings = np.vstack(emb_batches)
        morph_features = np.asarray(morph_batches, dtype=np.float32)
        np.savez_compressed(CLIP_CACHE, keys=np.asarray(keys, dtype=object), clip_embeddings=clip_embeddings, morph_features=morph_features)

    idx = {key: i for i, key in enumerate(keys)}
    morph_names = sorted(morphology_v2_features(np.zeros((0, 224, 224), dtype=np.float32)))
    for row in visual_rows:
        mi = morph_features[idx[row["row_key"]]]
        for name, value in zip(morph_names, mi):
            row[name] = float(value)

    def rows_x(rows: list[dict[str, Any]], kind: str) -> np.ndarray:
        if kind == "clip":
            return np.asarray([clip_embeddings[idx[r["row_key"]]] for r in rows], dtype=np.float64)
        if kind == "entry":
            emb = np.asarray([clip_embeddings[idx[r["row_key"]]] for r in rows], dtype=np.float64)
            morph = np.asarray([vec(r, ["center_motion_mean", "lower_upper_ratio", "spatial_spread_mean", "motion_peakiness"]) for r in rows], dtype=np.float64)
            return np.hstack([emb, morph])
        if kind == "morph":
            return np.asarray([vec(r, morph_names) for r in rows], dtype=np.float64)
        if kind == "fusion":
            emb = np.asarray([clip_embeddings[idx[r["row_key"]]] for r in rows], dtype=np.float64)
            morph = np.asarray([vec(r, morph_names + ["r9_score"]) for r in rows], dtype=np.float64)
            return np.hstack([emb, morph])
        raise ValueError(kind)

    candidates = [
        ("r9_weighted_audio_reference", None, "Fixed stage-1 promoted audio reference; detector/taxonomy/forced path unchanged."),
        ("r15_clip_embedding_verifier", "clip", "Short-clip CLIP frame embeddings pooled across the event window with a tiny logistic probe."),
        ("r15_diver_entry_presence_verifier", "entry", "CLIP event-window embeddings plus bounded center/lower motion cues for diver-entry presence."),
        ("r15_splash_morphology_verifier_v2", "morph", "Stronger splash morphology from grayscale short-clip temporal/spatial motion features."),
        ("r15_audio_video_late_fusion", "fusion", "Late fusion of promoted r9 audio score with CLIP and stronger morphology features on risky rows."),
    ]
    train_labels = [label_int(r["label"]) for r in train_rows]
    reports = []
    table = []

    for name, kind, description in candidates:
        if kind is None:
            int_queues = r9_queues(internal_rows)
            ext_queues = r9_queues(external_rows)
            int_tri = triage(internal_rows, int_queues)
            ext_tri = triage(external_rows, ext_queues)
            policy = {"stage_2": "none"}
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42))
            clf.fit(rows_x(train_rows, kind), train_labels, logisticregression__sample_weight=sample_weight)
            train_scores = clf.predict_proba(rows_x(train_rows, kind))[:, 1].tolist()
            int_scores = clf.predict_proba(rows_x(internal_rows, kind))[:, 1].tolist()
            ext_scores = clf.predict_proba(rows_x(external_rows, kind))[:, 1].tolist()
            policy = choose_policy(train_rows, train_scores)
            for row, score in zip(internal_rows, int_scores):
                row["verifier_score"] = float(score)
            for row, score in zip(external_rows, ext_scores):
                row["verifier_score"] = float(score)
            int_tri = triage(internal_rows, verifier_queues(internal_rows, int_scores, policy))
            ext_tri = triage(external_rows, verifier_queues(external_rows, ext_scores, policy))
        viable = product_viable(int_tri, ext_tri)
        row = {
            "candidate": name,
            "description": description,
            "base_forced_classification_unchanged": True,
            "stage_2_applies_only_to_risky_rows": kind is not None,
            "policy": policy,
            "external_auto_approve_precision": ext_tri["auto_approve_precision"],
            "external_auto_exclude_precision": ext_tri["auto_exclude_precision"],
            "external_coverage": ext_tri["coverage"],
            "external_review_required_count": ext_tri["review_required_count"],
            "dangerous_external_noise_auto_approve_count": ext_tri["dangerous_auto_approve_count"],
            "dangerous_internal_auto_approve_count": int_tri["dangerous_auto_approve_count"],
            "external_accepted_accuracy": ext_tri["accepted_accuracy"],
            "external_accepted_macro_f1": ext_tri["accepted_macro_f1"],
            "product_viable": viable,
        }
        table.append(row)
        reports.append({"name": name, "description": description, "policy": policy, "triage": {"internal": int_tri, "external": ext_tri}, "product_viable": viable})

    best = max(
        [r for r in table if r["candidate"] != "r9_weighted_audio_reference"],
        key=lambda r: (
            r["product_viable"],
            -r["dangerous_external_noise_auto_approve_count"],
            -r["dangerous_internal_auto_approve_count"],
            fnum(r["external_auto_approve_precision"]),
            fnum(r["external_auto_exclude_precision"]),
            r["external_coverage"],
        ),
    )
    decision = "R15_VISUAL_VERIFIER_GAIN_CONFIRMED" if best["product_viable"] else "R15_VISUAL_VERIFIER_NO_CLEAR_GAIN"
    benchmark = {
        "experiment_name": "r15_stronger_visual_verifier_benchmark",
        "final_decision": decision,
        "best_candidate": best["candidate"],
        "stage_1_reference": "r9_compact_nuisance_generalization_weighted",
        "forced_classification_path_unchanged": True,
        "visual_embedding_cache": str(CLIP_CACHE),
        "candidate_reports": reports,
        "comparison_rows": table,
    }
    queue = {
        "experiment_name": "r15_audio_video_queue_safety",
        "final_decision": decision,
        "best_candidate": best["candidate"],
        "comparison_rows": table,
        "success_criteria": {
            "dangerous_external_noise_auto_approves_max": 1,
            "dangerous_internal_auto_approves": 0,
            "external_auto_approve_precision_min": PRECISION_TARGET,
            "external_auto_exclude_precision_min": PRECISION_TARGET,
            "combined_external_auto_coverage_min": COVERAGE_TARGET,
        },
    }
    OUT_BENCH_JSON.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    OUT_QUEUE_JSON.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    write_table(OUT_BENCH_MD, "r15 Visual Verifier Benchmark", benchmark)
    write_table(OUT_QUEUE_MD, "r15 Audio-Video Queue Safety", queue)
    print(json.dumps({"wrote": [str(OUT_DATASET_JSON), str(OUT_DATASET_MD), str(OUT_BENCH_JSON), str(OUT_BENCH_MD), str(OUT_QUEUE_JSON), str(OUT_QUEUE_MD)], "final_decision": decision, "best_candidate": best["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
