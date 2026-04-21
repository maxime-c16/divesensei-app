from __future__ import annotations

import importlib.util
import json
import sys
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
R15 = ROOT / "benchmarks/r15_stronger_visual_verifier_benchmark.py"
R20 = ROOT / "benchmarks/r20_source_aware_nuisance_hardening_for_approve_expansion.py"
PHASE5 = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
DATASET = ROOT / "outputs/platform_noise_es4_dataset_rows.json"
PREVIEW = ROOT / "outputs/event_window_manifest_preview.jsonl"
CLIP_CACHE = ROOT / "outputs/r15_clip_frame_embedding_cache.npz"
DEFAULT_SESSION_DIR = ROOT / "outputs/evaluation_CAO-1st-15min_20260421-072906"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "source"


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Backfill r23 visual shadow scores for a reviewed evaluation session.")
    parser.add_argument("--session-dir", default=str(DEFAULT_SESSION_DIR), help="Evaluation session directory with an event-reviewed manifest.")
    parser.add_argument("--output-prefix", default="", help="Optional output prefix under outputs/. Defaults to r23_<session-slug>_shadow_score_backfill.")
    return parser


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


def manifest_refs(r15: Any, rows: list[dict[str, Any]]) -> list[Any]:
    refs = []
    for row in rows:
        if row.get("final_human_event_label") not in {"platform_dive", "noise_or_other"}:
            continue
        sid = str(row["source_session_id"])
        rid = str(row.get("legacy_candidate_id") or "row-unknown")
        refs.append(r15.RowRef(f"{sid}::{rid}", str(row["final_human_event_label"]), row))
    return refs


def label_int(label: str) -> int:
    return 1 if label == "platform_dive" else 0


def extract_visual_features(r15: Any, rows: list[dict[str, Any]], cache_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        keys = [str(key) for key in cache["keys"].tolist()]
        if keys == [row["row_key"] for row in rows]:
            return (
                {key: cache["clip_embeddings"][idx] for idx, key in enumerate(keys)},
                {key: cache["morph_features"][idx] for idx, key in enumerate(keys)},
            )

    from transformers import CLIPModel, CLIPProcessor
    import torch

    torch.set_num_threads(1)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    embeddings = []
    morph_rows = []
    keys = []
    with torch.no_grad():
        for row in rows:
            keys.append(row["row_key"])
            path = Path(row["clip_spec"]["path"])
            start = float(row["clip_spec"]["start_seconds"])
            duration = float(row["clip_spec"]["end_seconds"]) - start
            frames = r15.decode_rgb_images(path, start, duration)
            gray = r15.decode_gray_frames(path, start, duration)
            morph = r15.morphology_v2_features(gray)
            morph_rows.append([morph[key] for key in sorted(morph)])
            if not frames:
                embeddings.append(np.zeros(1536, dtype=np.float32))
                continue
            inputs = processor(images=frames, return_tensors="pt", padding=True)
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            arr = features.cpu().numpy().astype(np.float32)
            embeddings.append(np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[-1] - arr[0]]).astype(np.float32))
    clip_embeddings = np.vstack(embeddings)
    morph_features = np.asarray(morph_rows, dtype=np.float32)
    np.savez_compressed(cache_path, keys=np.asarray(keys, dtype=object), clip_embeddings=clip_embeddings, morph_features=morph_features)
    return (
        {key: clip_embeddings[idx] for idx, key in enumerate(keys)},
        {key: morph_features[idx] for idx, key in enumerate(keys)},
    )


def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    manifest_path = session_dir / "exports/event-reviewed-manifest/event_reviewed_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Reviewed manifest not found: {manifest_path}")
    output_prefix = args.output_prefix or f"r23_{slugify(session_dir.name)}_shadow_score_backfill"
    out_json = ROOT / "outputs" / f"{output_prefix}.json"
    out_jsonl = ROOT / "outputs" / f"{output_prefix}_rows.jsonl"
    out_md = ROOT / "outputs" / f"{output_prefix}.md"
    out_cache = ROOT / "outputs" / f"{output_prefix}_clip_frame_embedding_cache.npz"

    r15 = load_module("r15_runtime_for_r23_backfill", R15)
    r20 = load_module("r20_runtime_for_r23_backfill", R20)
    phase5 = load_module("phase5_r23_backfill", PHASE5)
    bench = load_module("nuisance_r23_backfill", NUISANCE)

    preview = row_key_map(load_jsonl(PREVIEW))
    lists = json.loads(DATASET.read_text())
    base_refs = [("base", r15.RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])])) for item in lists["train_rows"]]
    session_refs = {source: manifest_refs(r15, load_jsonl(path)) for source, path in r20.SOURCES.items()}
    train_refs = base_refs + [(source, ref) for source, refs in session_refs.items() for ref in refs]
    train_items = [ref for _, ref in train_refs]
    target_refs = manifest_refs(r15, load_jsonl(manifest_path))
    all_refs = train_items + target_refs

    audio: dict[str, np.ndarray] = {}
    video_paths: dict[str, Path] = {}
    fmap: dict[str, dict[str, float]] = {}
    for ref in all_refs:
        sid = str(ref.row["source_session_id"])
        if sid not in audio:
            source_root = phase5.resolve_source_root(str(ref.row["source_session_root"]))
            video_paths[sid] = source_root / "web/session_source_review.mp4"
            audio[sid] = phase5.decode_audio_mono(video_paths[sid], phase5.SAMPLE_RATE)
        start = max(0.0, phase5.to_float(ref.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(ref.row.get("event_window_end_seconds")))
        sig = audio[sid][int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap[ref.row_key] = {**phase5.extract_features(sig, phase5.SAMPLE_RATE), **bench.nuisance_features(phase5, sig, phase5.SAMPLE_RATE)}

    def r9_vec(ref: Any) -> list[float]:
        return bench.vector_for(phase5, ref, fmap, bench.NOISE_BOUNDARY_COMPACT)

    x_train = np.asarray([r9_vec(ref) for ref in train_items], dtype=np.float64)
    y_train = np.asarray([label_int(ref.label) for ref in train_items], dtype=np.int64)
    weights = r20.train_weights(train_refs)
    model = r20.xgb_model()
    model.fit(x_train, y_train, sample_weight=weights)

    def make_rows(refs: list[Any], scores: list[float], split: str) -> list[dict[str, Any]]:
        rows = []
        for ref, score in zip(refs, scores):
            start = max(0.0, phase5.to_float(ref.row.get("event_window_start_seconds")))
            end = max(start + 0.05, phase5.to_float(ref.row.get("event_window_end_seconds")))
            rows.append(
                {
                    "row_key": ref.row_key,
                    "split": split,
                    "bank_split": split,
                    "source": split,
                    "source_session_id": str(ref.row["source_session_id"]),
                    "label": ref.label,
                    "legacy_subtype": ref.row.get("legacy_subtype"),
                    "suggested_event_label_reason": ref.row.get("suggested_event_label_reason"),
                    "r9_score": float(score),
                    "clip_spec": {
                        "path": str(video_paths[str(ref.row["source_session_id"])]),
                        "start_seconds": start,
                        "end_seconds": end,
                    },
                }
            )
        return rows

    train_rows = make_rows(train_items, model.predict_proba(x_train)[:, 1].tolist(), "train_augmented")
    target_rows = make_rows(target_refs, model.predict_proba(np.asarray([r9_vec(ref) for ref in target_refs], dtype=np.float64))[:, 1].tolist(), "fresh_shadow_backfill")

    cache = np.load(CLIP_CACHE, allow_pickle=True)
    clip_index = {str(key): idx for idx, key in enumerate(cache["keys"].tolist())}
    clip_embeddings = cache["clip_embeddings"]
    morph_features = cache["morph_features"]
    morph_names = sorted(r15.morphology_v2_features(np.zeros((0, 224, 224), dtype=np.float32)))

    target_clip, target_morph = extract_visual_features(r15, target_rows, out_cache)

    def clip_vec(row: dict[str, Any]) -> np.ndarray:
        key = row["row_key"]
        if key in clip_index:
            return clip_embeddings[clip_index[key]]
        return target_clip[key]

    def morph_vec(row: dict[str, Any]) -> np.ndarray:
        key = row["row_key"]
        if key in clip_index:
            return morph_features[clip_index[key]]
        return target_morph[key]

    for row in train_rows + target_rows:
        for name, value in zip(morph_names, morph_vec(row)):
            row[name] = float(value)

    def visual_x(rows: list[dict[str, Any]]) -> np.ndarray:
        emb = np.asarray([clip_vec(row) for row in rows], dtype=np.float64)
        morph = np.asarray([r20.vec_from_features(row, morph_names + ["r9_score"]) for row in rows], dtype=np.float64)
        return np.hstack([emb, morph])

    visual = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42))
    visual.fit(visual_x(train_rows), y_train, logisticregression__sample_weight=weights)
    visual_scores = visual.predict_proba(visual_x(target_rows))[:, 1].tolist()
    for row, score in zip(target_rows, visual_scores):
        row["visual_score"] = float(score)
        row["visual_late_fusion_logreg_c0.5"] = float(score)
        row.pop("clip_spec", None)

    out_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in target_rows) + "\n", encoding="utf-8")
    summary = {
        "experiment_name": "r23_shadow_score_backfill",
        "source_session_id": session_dir.name,
        "session_dir": str(session_dir),
        "row_count": len(target_rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in target_rows).items())),
        "subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in target_rows).items())),
        "v1_approved_count": sum(1 for row in target_rows if row["r9_score"] >= 0.92158),
        "v2_shadow_approved_count": sum(1 for row in target_rows if row["r9_score"] >= 0.92158 or (row["r9_score"] >= 0.84 and row["visual_score"] >= 0.55)),
        "v2_added_approve_count": sum(1 for row in target_rows if row["r9_score"] < 0.92158 and row["r9_score"] >= 0.84 and row["visual_score"] >= 0.55),
        "v2_added_label_counts": dict(sorted(Counter(row["label"] for row in target_rows if row["r9_score"] < 0.92158 and row["r9_score"] >= 0.84 and row["visual_score"] >= 0.55).items())),
        "rows_path": str(out_jsonl),
        "cache_path": str(out_cache),
    }
    out_json.write_text(json.dumps({"summary": summary, "rows": target_rows}, indent=2), encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "# r23 Shadow Score Backfill",
                "",
                f"- source: `{summary['source_session_id']}`",
                f"- rows scored: `{summary['row_count']}`",
                f"- label counts: `{json.dumps(summary['label_counts'], sort_keys=True)}`",
                f"- subtype counts: `{json.dumps(summary['subtype_counts'], sort_keys=True)}`",
                f"- v1 approved: `{summary['v1_approved_count']}`",
                f"- v2 shadow approved: `{summary['v2_shadow_approved_count']}`",
                f"- v2-only added approvals: `{summary['v2_added_approve_count']}`",
                f"- v2-only added label counts: `{json.dumps(summary['v2_added_label_counts'], sort_keys=True)}`",
                "",
                "The detector, taxonomy, and policy thresholds are unchanged. This only backfills the existing r20 visual late-fusion score for the reviewed CAO platform/noise rows.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": [str(out_json), str(out_jsonl), str(out_md), str(out_cache)], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
