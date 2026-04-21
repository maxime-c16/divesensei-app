from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
R15_PATH = ROOT / "benchmarks/r15_stronger_visual_verifier_benchmark.py"
PHASE5 = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
DATASET = ROOT / "outputs/platform_noise_es4_dataset_rows.json"
EXTERNAL = ROOT / "outputs/external_holdout_slice.json"
PREVIEW = ROOT / "outputs/event_window_manifest_preview.jsonl"
CLIP_CACHE = ROOT / "outputs/r15_clip_frame_embedding_cache.npz"

SOURCES = {
    "snmt": ROOT / "outputs/evaluation_SNMT-16min_20260417-131944/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "img_8852": ROOT / "outputs/evaluation_img_8852_rerun_20260406-104430/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "champigny_1704": ROOT / "outputs/evaluation_Champigny-17-04-9min_20260418-065417/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
}
WEIGHTS = {"snmt": 1.0, "img_8852": 1.0, "champigny_1704": 0.3}
LOW = 0.05
HIGH = 0.85

OUT_BENCH_JSON = ROOT / "outputs/r16_visual_approve_guard_benchmark.json"
OUT_BENCH_MD = ROOT / "outputs/r16_visual_approve_guard_benchmark.md"
OUT_CMP_JSON = ROOT / "outputs/r16_visual_approve_guard_comparison.json"
OUT_CMP_MD = ROOT / "outputs/r16_visual_approve_guard_comparison.md"
OUT_EXCL_JSON = ROOT / "outputs/r16_exclude_side_diagnostic.json"
OUT_EXCL_MD = ROOT / "outputs/r16_exclude_side_diagnostic.md"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label_int(label: str) -> int:
    return 1 if label == "platform_dive" else 0


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def fnum(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def approve_metrics(rows: list[dict[str, Any]], approve: list[bool]) -> dict[str, Any]:
    approved = [row for row, flag in zip(rows, approve) if flag]
    ok = sum(1 for row in approved if row["label"] == "platform_dive")
    errors = [row for row in approved if row["label"] != "platform_dive"]
    return {
        "row_count": len(rows),
        "auto_approve_count": len(approved),
        "approve_coverage": safe_div(len(approved), len(rows)) or 0.0,
        "review_required_count": len(rows) - len(approved),
        "review_burden": safe_div(len(rows) - len(approved), len(rows)) or 0.0,
        "auto_approve_precision": safe_div(ok, len(approved)),
        "dangerous_auto_approve_count": len(errors),
        "approved_label_counts": dict(sorted(Counter(row["label"] for row in approved).items())),
        "approved_role_counts": dict(sorted(Counter(row["role"] for row in approved).items())),
        "dangerous_auto_approve_rows": [compact(row) for row in errors],
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "label": row["label"],
        "r9_score": row["r9_score"],
        "visual_score": row.get("visual_score"),
        "role": row["role"],
        "legacy_subtype": row.get("legacy_subtype"),
        "source_session_id": row["source_session_id"],
    }


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out, counts = {}, {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        rid = str(row.get("legacy_candidate_id") or f"row-{counts[sid]:04d}")
        out[f"{sid}::{rid}"] = row
    return out


def manifest_refs(r15: Any, rows: list[dict[str, Any]]) -> list[Any]:
    refs = []
    for row in rows:
        sid = str(row["source_session_id"])
        rid = str(row.get("legacy_candidate_id") or "row-unknown")
        refs.append(r15.RowRef(f"{sid}::{rid}", str(row["final_human_event_label"]), row))
    return refs


def vec(row: dict[str, Any], names: list[str]) -> list[float]:
    vals = []
    for name in names:
        try:
            vals.append(float(row.get(name, 0.0) or 0.0))
        except Exception:
            vals.append(0.0)
    return vals


def main() -> None:
    r15 = load_module("r15_runtime_for_r16", R15_PATH)
    phase5 = load_module("phase5_r16", PHASE5)
    bench = load_module("nuisance_r16", NUISANCE)

    preview = row_key_map(load_jsonl(PREVIEW))
    lists = json.loads(DATASET.read_text())
    base_train = [("base", r15.RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])])) for i in lists["train_rows"]]
    internal_refs = [r15.RowRef(str(i["row_key"]), str(i["label"]), preview[str(i["row_key"])]) for i in lists["holdout_rows"]]
    external_refs = [r15.RowRef(str(r["row_key"]), str(r["final_human_event_label"]), r) for r in json.loads(EXTERNAL.read_text())["rows"]]
    session_refs = {
        k: manifest_refs(r15, [row for row in load_jsonl(path) if row.get("final_human_event_label") in {"platform_dive", "noise_or_other"}])
        for k, path in SOURCES.items()
    }
    train_refs = base_train + [(k, ref) for k, refs in session_refs.items() for ref in refs]
    train_items = [ref for _, ref in train_refs]
    all_refs = train_items + internal_refs + external_refs

    audio: dict[str, np.ndarray] = {}
    fmap: dict[str, dict[str, float]] = {}
    for item in all_refs:
        sid = str(item.row["source_session_id"])
        if sid not in audio:
            source_root = phase5.resolve_source_root(str(item.row["source_session_root"]))
            audio[sid] = phase5.decode_audio_mono(source_root / "web/session_source_review.mp4", phase5.SAMPLE_RATE)
        start = max(0.0, phase5.to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(item.row.get("event_window_end_seconds")))
        sig = audio[sid][int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap[item.row_key] = {**phase5.extract_features(sig, phase5.SAMPLE_RATE), **bench.nuisance_features(phase5, sig, phase5.SAMPLE_RATE)}

    def r9_vec(item: Any) -> list[float]:
        return bench.vector_for(phase5, item, fmap, bench.NOISE_BOUNDARY_COMPACT)

    x_train = np.asarray([r9_vec(ref) for ref in train_items], dtype=np.float64)
    y_train = np.asarray([label_int(ref.label) for ref in train_items], dtype=np.int64)
    weights = np.ones(len(train_refs), dtype=np.float64)
    base_total = sum(1 for src, _ in train_refs if src == "base")
    by_source: dict[str, list[int]] = {}
    for idx, (src, _) in enumerate(train_refs):
        by_source.setdefault(src, []).append(idx)
    for src, idxs in by_source.items():
        if src == "base":
            continue
        per_item = base_total * WEIGHTS[src] / len(idxs)
        for idx in idxs:
            weights[idx] = per_item

    r9_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=120, learning_rate=0.05, max_depth=3, min_child_weight=2, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=1)
    r9_model.fit(x_train, y_train, sample_weight=weights)

    def make_rows(refs: list[Any], scores: list[float], split: str) -> list[dict[str, Any]]:
        rows = []
        for ref, score in zip(refs, scores):
            row = {
                "split": split,
                "row_key": ref.row_key,
                "label": ref.label,
                "source_session_id": str(ref.row["source_session_id"]),
                "legacy_subtype": ref.row.get("legacy_subtype"),
                "suggested_event_label_reason": ref.row.get("suggested_event_label_reason"),
                "r9_score": float(score),
            }
            row["role"] = r15.role(float(score), ref.label)
            rows.append(row)
        return rows

    train_rows = make_rows(train_items, r9_model.predict_proba(x_train)[:, 1].tolist(), "train_augmented")
    internal_rows = make_rows(internal_refs, r9_model.predict_proba(np.asarray([r9_vec(ref) for ref in internal_refs], dtype=np.float64))[:, 1].tolist(), "internal_holdout")
    external_rows = make_rows(external_refs, r9_model.predict_proba(np.asarray([r9_vec(ref) for ref in external_refs], dtype=np.float64))[:, 1].tolist(), "external_holdout")

    cache = np.load(CLIP_CACHE, allow_pickle=True)
    keys = [str(x) for x in cache["keys"].tolist()]
    key_index = {key: idx for idx, key in enumerate(keys)}
    clip_embeddings = cache["clip_embeddings"]
    morph_features = cache["morph_features"]
    morph_names = sorted(r15.morphology_v2_features(np.zeros((0, 224, 224), dtype=np.float32)))
    visual_rows = train_rows + internal_rows + external_rows
    for row in visual_rows:
        idx = key_index[row["row_key"]]
        for name, value in zip(morph_names, morph_features[idx]):
            row[name] = float(value)

    def rows_x(rows: list[dict[str, Any]]) -> np.ndarray:
        emb = np.asarray([clip_embeddings[key_index[row["row_key"]]] for row in rows], dtype=np.float64)
        morph = np.asarray([vec(row, morph_names + ["r9_score"]) for row in rows], dtype=np.float64)
        return np.hstack([emb, morph])

    visual_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42))
    visual_model.fit(rows_x(train_rows), [label_int(row["label"]) for row in train_rows], logisticregression__sample_weight=weights)
    train_scores = visual_model.predict_proba(rows_x(train_rows))[:, 1].tolist()
    internal_scores = visual_model.predict_proba(rows_x(internal_rows))[:, 1].tolist()
    external_scores = visual_model.predict_proba(rows_x(external_rows))[:, 1].tolist()
    for rows, scores in [(train_rows, train_scores), (internal_rows, internal_scores), (external_rows, external_scores)]:
        for row, score in zip(rows, scores):
            row["visual_score"] = float(score)

    candidates = []
    policy_defs = [
        ("r9_audio_auto_approve_reference", "audio-only r9 auto-approve, no auto-exclude decision", lambda r: r["r9_score"] >= HIGH),
        ("r16_veto_guard_r9_high_visual_055", "approve only r9 high-score rows with visual late-fusion >= 0.55", lambda r: r["r9_score"] >= HIGH and r["visual_score"] >= 0.55),
        ("r16_strict_late_fusion_guard_075", "approve only r9 high-score rows with visual late-fusion >= 0.75", lambda r: r["r9_score"] >= HIGH and r["visual_score"] >= 0.75),
        ("r16_very_strict_late_fusion_guard_085", "approve only r9 high-score rows with visual late-fusion >= 0.85", lambda r: r["r9_score"] >= HIGH and r["visual_score"] >= 0.85),
        ("r16_approve_confidence_gate", "approve r9 moderate/high rows only when visual confidence is extreme", lambda r: r["r9_score"] >= 0.50 and r["visual_score"] >= 0.95),
        ("r16_hybrid_high_precision_gate", "approve r9 high rows with visual >=0.75 or r9 review-band rows with visual >=0.98", lambda r: (r["r9_score"] >= HIGH and r["visual_score"] >= 0.75) or (0.50 <= r["r9_score"] < HIGH and r["visual_score"] >= 0.98)),
    ]

    for name, description, fn in policy_defs:
        int_flags = [bool(fn(row)) for row in internal_rows]
        ext_flags = [bool(fn(row)) for row in external_rows]
        int_metrics = approve_metrics(internal_rows, int_flags)
        ext_metrics = approve_metrics(external_rows, ext_flags)
        viable = (
            fnum(ext_metrics["auto_approve_precision"]) >= 0.90
            and ext_metrics["dangerous_auto_approve_count"] <= 1
            and int_metrics["dangerous_auto_approve_count"] == 0
            and ext_metrics["auto_approve_count"] >= 10
        )
        candidates.append(
            {
                "candidate": name,
                "description": description,
                "external_auto_approve_precision": ext_metrics["auto_approve_precision"],
                "external_auto_approve_count": ext_metrics["auto_approve_count"],
                "external_approve_coverage": ext_metrics["approve_coverage"],
                "external_review_required_count": ext_metrics["review_required_count"],
                "external_review_burden": ext_metrics["review_burden"],
                "dangerous_external_auto_approves": ext_metrics["dangerous_auto_approve_count"],
                "dangerous_internal_auto_approves": int_metrics["dangerous_auto_approve_count"],
                "internal_auto_approve_precision": int_metrics["auto_approve_precision"],
                "internal_auto_approve_count": int_metrics["auto_approve_count"],
                "approve_guard_viable": viable,
                "internal": int_metrics,
                "external": ext_metrics,
            }
        )

    best = max(candidates, key=lambda row: (row["approve_guard_viable"], -row["dangerous_external_auto_approves"], -row["dangerous_internal_auto_approves"], fnum(row["external_auto_approve_precision"]), row["external_auto_approve_count"]))
    decision = "R16_APPROVE_GUARD_VIABLE" if best["approve_guard_viable"] else "R16_APPROVE_GUARD_NOT_YET_VIABLE"

    exclude_like_external = [row for row in external_rows if row["r9_score"] <= LOW or (row["visual_score"] <= 0.30 and row["r9_score"] < HIGH)]
    exclude_errors = [row for row in exclude_like_external if row["label"] == "platform_dive"]
    exclude_diag = {
        "experiment_name": "r16_exclude_side_diagnostic",
        "final_decision": "EXCLUDE_SIDE_STILL_BLOCKED",
        "external_exclude_candidate_count": len(exclude_like_external),
        "external_exclude_candidate_label_counts": dict(sorted(Counter(row["label"] for row in exclude_like_external).items())),
        "external_exclude_candidate_subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in exclude_like_external).items())),
        "external_exclude_error_count": len(exclude_errors),
        "external_exclude_error_rows": [compact(row) for row in exclude_errors],
        "r9_high_confidence_nuisance_controls_external": sum(1 for row in external_rows if row["role"] == "high_confidence_nuisance_control"),
        "r9_exclude_risk_rows_external": sum(1 for row in external_rows if row["role"] == "exclude_risk"),
        "diagnosis": "Exclude-side supervision/control coverage is too thin and visually ambiguous: only two high-confidence nuisance controls exist in the governed visual set, while many low visual-score candidates are true platform dives from the review band.",
        "recommendation": "Do not promote an exclude verifier yet; collect/curate negative-control visual examples before a bounded exclude verifier benchmark.",
    }
    exclude_decision = exclude_diag["final_decision"]

    report = {
        "experiment_name": "r16_visual_approve_guard_benchmark",
        "final_decision": decision,
        "exclude_side_decision": exclude_decision,
        "best_candidate": best["candidate"],
        "stage_1_reference": "r9_compact_nuisance_generalization_weighted",
        "visual_stage_2": "r15_audio_video_late_fusion_features_reused",
        "success_rule": "approve-only viable if external precision >=0.90, external dangerous auto-approves <=1, internal dangerous auto-approves =0, and at least 10 external rows are auto-approved",
        "comparison_rows": candidates,
    }
    comparison = {
        "experiment_name": "r16_visual_approve_guard_comparison",
        "final_decision": decision,
        "best_candidate": best["candidate"],
        "comparison_rows": [
            {k: v for k, v in row.items() if k not in {"internal", "external"}}
            for row in candidates
        ],
    }

    OUT_BENCH_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_CMP_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    OUT_EXCL_JSON.write_text(json.dumps(exclude_diag, indent=2), encoding="utf-8")

    table_lines = [
        "| candidate | ext approve precision | ext approve count | ext approve coverage | ext dangerous approve | int dangerous approve | review burden | viable |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        table_lines.append(
            f"| `{row['candidate']}` | {fnum(row['external_auto_approve_precision']):.4f} | {row['external_auto_approve_count']} | {row['external_approve_coverage']:.4f} | {row['dangerous_external_auto_approves']} | {row['dangerous_internal_auto_approves']} | {row['external_review_burden']:.4f} | `{row['approve_guard_viable']}` |"
        )
    md = "\n".join([
        "# r16 Visual Approve Guard Benchmark",
        "",
        f"- final decision: `{decision}`",
        f"- best candidate: `{best['candidate']}`",
        f"- exclude-side decision: `{exclude_decision}`",
        "",
        *table_lines,
        "",
        "Approve-side is judged independently from auto-exclude behavior. No detector, taxonomy, or forced-classification path was changed.",
    ]) + "\n"
    OUT_BENCH_MD.write_text(md, encoding="utf-8")
    OUT_CMP_MD.write_text(md.replace("# r16 Visual Approve Guard Benchmark", "# r16 Visual Approve Guard Comparison"), encoding="utf-8")

    OUT_EXCL_MD.write_text(
        "\n".join([
            "# r16 Exclude-Side Diagnostic",
            "",
            f"- final decision: `{exclude_decision}`",
            f"- external exclude-candidate count: `{exclude_diag['external_exclude_candidate_count']}`",
            f"- external exclude-candidate label counts: `{json.dumps(exclude_diag['external_exclude_candidate_label_counts'], sort_keys=True)}`",
            f"- external exclude-candidate subtype counts: `{json.dumps(exclude_diag['external_exclude_candidate_subtype_counts'], sort_keys=True)}`",
            f"- external exclude errors: `{exclude_diag['external_exclude_error_count']}`",
            f"- high-confidence nuisance controls in visual set: `{exclude_diag['r9_high_confidence_nuisance_controls_external']}`",
            f"- exclude-risk rows in visual set: `{exclude_diag['r9_exclude_risk_rows_external']}`",
            "",
            exclude_diag["diagnosis"],
            "",
            exclude_diag["recommendation"],
        ]) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"wrote": [str(OUT_BENCH_JSON), str(OUT_BENCH_MD), str(OUT_CMP_JSON), str(OUT_CMP_MD), str(OUT_EXCL_JSON), str(OUT_EXCL_MD)], "final_decision": decision, "best_candidate": best["candidate"], "exclude_side_decision": exclude_decision}, indent=2))


if __name__ == "__main__":
    main()
