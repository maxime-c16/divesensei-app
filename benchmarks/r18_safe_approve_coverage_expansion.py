from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
R15 = ROOT / "benchmarks/r15_stronger_visual_verifier_benchmark.py"
PHASE5 = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
DATASET = ROOT / "outputs/platform_noise_es4_dataset_rows.json"
EXTERNAL = ROOT / "outputs/external_holdout_slice.json"
PREVIEW = ROOT / "outputs/event_window_manifest_preview.jsonl"
CLIP_CACHE = ROOT / "outputs/r15_clip_frame_embedding_cache.npz"
AST_CACHE = ROOT / "outputs/r13_ast_embedding_cache.npz"

SOURCES = {
    "snmt": ROOT / "outputs/evaluation_SNMT-16min_20260417-131944/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "img_8852": ROOT / "outputs/evaluation_img_8852_rerun_20260406-104430/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "champigny_1704": ROOT / "outputs/evaluation_Champigny-17-04-9min_20260418-065417/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
}
SOURCE_WEIGHTS = {"snmt": 1.0, "img_8852": 1.0, "champigny_1704": 0.3}
LOW = 0.05
HIGH = 0.85

R17_APPROVE_MIN_SCORE = 0.92158
R17_APPROVE_COVERAGE = 0.1717171717171717

OUT_DATASET_JSON = ROOT / "outputs/r18_approve_benchmark_dataset.json"
OUT_DATASET_MD = ROOT / "outputs/r18_approve_benchmark_dataset.md"
OUT_BENCH_JSON = ROOT / "outputs/r18_safe_approve_coverage_expansion.json"
OUT_BENCH_MD = ROOT / "outputs/r18_safe_approve_coverage_expansion.md"
OUT_CMP_JSON = ROOT / "outputs/r18_approve_candidate_comparison.json"
OUT_CMP_MD = ROOT / "outputs/r18_approve_candidate_comparison.md"
OUT_BEST_JSON = ROOT / "outputs/r18_best_approve_policy.json"
OUT_BEST_MD = ROOT / "outputs/r18_best_approve_policy.md"


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


def manifest_refs(r15: Any, rows: list[dict[str, Any]]) -> list[Any]:
    out = []
    for row in rows:
        sid = str(row["source_session_id"])
        rid = str(row.get("legacy_candidate_id") or "row-unknown")
        out.append(r15.RowRef(f"{sid}::{rid}", str(row["final_human_event_label"]), row))
    return out


def label_int(label: str) -> int:
    return 1 if label == "platform_dive" else 0


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def fnum(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "split": row["split"],
        "label": row["label"],
        "r9_score": row["r9_score"],
        "role": row["role"],
        "legacy_subtype": row.get("legacy_subtype"),
        "source_session_id": row["source_session_id"],
    }


def approve_metrics(rows: list[dict[str, Any]], flags: list[bool]) -> dict[str, Any]:
    approved = [row for row, flag in zip(rows, flags) if flag]
    ok = sum(1 for row in approved if row["label"] == "platform_dive")
    errors = [row for row in approved if row["label"] != "platform_dive"]
    return {
        "row_count": len(rows),
        "approve_count": len(approved),
        "approve_coverage": safe_div(len(approved), len(rows)) or 0.0,
        "review_required_count": len(rows) - len(approved),
        "review_burden": safe_div(len(rows) - len(approved), len(rows)) or 0.0,
        "approve_precision": safe_div(ok, len(approved)),
        "dangerous_approve_count": len(errors),
        "approved_label_counts": dict(sorted(Counter(row["label"] for row in approved).items())),
        "approved_role_counts": dict(sorted(Counter(row["role"] for row in approved).items())),
        "dangerous_approve_rows": [compact(row) for row in errors],
    }


def vec(row: dict[str, Any], names: list[str]) -> list[float]:
    vals = []
    for name in names:
        try:
            vals.append(float(row.get(name, 0.0) or 0.0))
        except Exception:
            vals.append(0.0)
    return vals


def candidate_row(name: str, desc: str, internal: dict[str, Any], external: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    interesting = (
        external["dangerous_approve_count"] == 0
        and internal["dangerous_approve_count"] == 0
        and fnum(external["approve_precision"]) >= 0.95
    )
    return {
        "candidate": name,
        "description": desc,
        "policy": policy,
        "external_approve_precision": external["approve_precision"],
        "external_approve_count": external["approve_count"],
        "external_approve_coverage": external["approve_coverage"],
        "external_review_burden": external["review_burden"],
        "dangerous_external_auto_approves": external["dangerous_approve_count"],
        "dangerous_internal_auto_approves": internal["dangerous_approve_count"],
        "internal_approve_precision": internal["approve_precision"],
        "internal_approve_count": internal["approve_count"],
        "crosses_015": external["approve_coverage"] >= 0.15,
        "crosses_020": external["approve_coverage"] >= 0.20,
        "crosses_025": external["approve_coverage"] >= 0.25,
        "crosses_030": external["approve_coverage"] >= 0.30,
        "safe_coverage_gain_vs_r17": interesting and external["approve_coverage"] > R17_APPROVE_COVERAGE,
        "internal": internal,
        "external": external,
    }


def main() -> None:
    r15 = load_module("r15_runtime_for_r17", R15)
    phase5 = load_module("phase5_r17", PHASE5)
    bench = load_module("nuisance_r17", NUISANCE)

    preview = row_key_map(load_jsonl(PREVIEW))
    lists = json.loads(DATASET.read_text())
    base_train = [("base", r15.RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])])) for item in lists["train_rows"]]
    internal_refs = [r15.RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])]) for item in lists["holdout_rows"]]
    external_refs = [r15.RowRef(str(row["row_key"]), str(row["final_human_event_label"]), row) for row in json.loads(EXTERNAL.read_text())["rows"]]
    session_refs = {
        source: manifest_refs(r15, [row for row in load_jsonl(path) if row.get("final_human_event_label") in {"platform_dive", "noise_or_other"}])
        for source, path in SOURCES.items()
    }
    train_refs = base_train + [(source, ref) for source, refs in session_refs.items() for ref in refs]
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
    base_total = sum(1 for source, _ in train_refs if source == "base")
    by_source: dict[str, list[int]] = {}
    for idx, (source, _) in enumerate(train_refs):
        by_source.setdefault(source, []).append(idx)
    for source, idxs in by_source.items():
        if source == "base":
            continue
        per_item = base_total * SOURCE_WEIGHTS[source] / len(idxs)
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

    clip_cache = np.load(CLIP_CACHE, allow_pickle=True)
    clip_keys = [str(x) for x in clip_cache["keys"].tolist()]
    clip_index = {key: idx for idx, key in enumerate(clip_keys)}
    clip_embeddings = clip_cache["clip_embeddings"]
    morph_features = clip_cache["morph_features"]
    morph_names = sorted(r15.morphology_v2_features(np.zeros((0, 224, 224), dtype=np.float32)))
    ast_cache = np.load(AST_CACHE, allow_pickle=True) if AST_CACHE.exists() else None
    ast_index = {str(key): idx for idx, key in enumerate(ast_cache["keys"].tolist())} if ast_cache is not None else {}
    ast_embeddings = ast_cache["embeddings"] if ast_cache is not None else None

    all_rows = train_rows + internal_rows + external_rows
    for row in all_rows:
        idx = clip_index[row["row_key"]]
        for name, value in zip(morph_names, morph_features[idx]):
            row[name] = float(value)

    def clip_x(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([clip_embeddings[clip_index[row["row_key"]]] for row in rows], dtype=np.float64)

    def ast_x(rows: list[dict[str, Any]]) -> np.ndarray:
        if ast_embeddings is None:
            return np.zeros((len(rows), 0), dtype=np.float64)
        return np.asarray([ast_embeddings[ast_index[row["row_key"]]] for row in rows], dtype=np.float64)

    def morph_x(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([vec(row, morph_names + ["r9_score"]) for row in rows], dtype=np.float64)

    feature_sets = {
        "visual_late_fusion": lambda rows: np.hstack([clip_x(rows), morph_x(rows)]),
        "ast_audio_plus_r9": lambda rows: np.hstack([ast_x(rows), np.asarray([[row["r9_score"]] for row in rows], dtype=np.float64)]),
        "audio_visual_ast": lambda rows: np.hstack([clip_x(rows), ast_x(rows), morph_x(rows)]),
    }
    score_bank: dict[str, dict[str, list[float]]] = {
        "r9_score": {
            "train": [row["r9_score"] for row in train_rows],
            "internal": [row["r9_score"] for row in internal_rows],
            "external": [row["r9_score"] for row in external_rows],
        }
    }
    for feature_name, fn in feature_sets.items():
        for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            name = f"{feature_name}_logreg_c{c:g}"
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=c, random_state=42))
            clf.fit(fn(train_rows), y_train, logisticregression__sample_weight=weights)
            score_bank[name] = {
                "train": clf.predict_proba(fn(train_rows))[:, 1].tolist(),
                "internal": clf.predict_proba(fn(internal_rows))[:, 1].tolist(),
                "external": clf.predict_proba(fn(external_rows))[:, 1].tolist(),
            }

    comparison: list[dict[str, Any]] = []

    def evaluate(name: str, desc: str, int_flags: list[bool], ext_flags: list[bool], policy: dict[str, Any]) -> None:
        internal = approve_metrics(internal_rows, int_flags)
        external = approve_metrics(external_rows, ext_flags)
        comparison.append(candidate_row(name, desc, internal, external, policy))

    evaluate(
        "r9_audio_auto_approve_reference",
        "Current audio-only r9 approve lane.",
        [row["r9_score"] >= HIGH for row in internal_rows],
        [row["r9_score"] >= HIGH for row in external_rows],
        {"type": "r9_score>=0.85"},
    )

    evaluate(
        "r17_best_safe_policy_reference",
        "Accepted r17 approve/review policy: r9_score >= 0.92158.",
        [row["r9_score"] >= R17_APPROVE_MIN_SCORE for row in internal_rows],
        [row["r9_score"] >= R17_APPROVE_MIN_SCORE for row in external_rows],
        {"type": "score_gate", "score_key": "r9_score", "score_min": R17_APPROVE_MIN_SCORE},
    )

    # Recreate r16 best-safe point for direct comparison.
    visual_key = "visual_late_fusion_logreg_c0.5"
    evaluate(
        "r16_best_safe_reference",
        "r16 best-safe veto guard: r9 high-score and visual late-fusion >= 0.55.",
        [row["r9_score"] >= HIGH and score >= 0.55 for row, score in zip(internal_rows, score_bank[visual_key]["internal"])],
        [row["r9_score"] >= HIGH and score >= 0.55 for row, score in zip(external_rows, score_bank[visual_key]["external"])],
        {"type": "r9_high_and_visual_min", "r9_min": HIGH, "score_key": visual_key, "score_min": 0.55},
    )

    # Bounded approve-side candidate search. Policies only decide approve vs review.
    for score_key, scores in score_bank.items():
        thresholds = np.unique(np.round(np.quantile(scores["train"], np.linspace(0.50, 0.995, 40)), 6)).tolist()
        thresholds += [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
        for thr in sorted(set(float(t) for t in thresholds if 0.0 <= float(t) <= 1.0)):
            evaluate(
                f"score_gate::{score_key}::{thr:.3f}",
                f"Approve-only confidence gate on {score_key}.",
                [score >= thr for score in scores["internal"]],
                [score >= thr for score in scores["external"]],
                {"type": "score_gate", "score_key": score_key, "score_min": thr},
            )
        for r9_min in [0.30, 0.50, 0.65, 0.75, 0.85]:
            for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]:
                evaluate(
                    f"r9_and_score::{score_key}::{r9_min:.2f}::{thr:.2f}",
                    f"Approve only if r9 >= {r9_min:.2f} and {score_key} >= {thr:.2f}.",
                    [row["r9_score"] >= r9_min and score >= thr for row, score in zip(internal_rows, scores["internal"])],
                    [row["r9_score"] >= r9_min and score >= thr for row, score in zip(external_rows, scores["external"])],
                    {"type": "r9_and_score", "score_key": score_key, "r9_min": r9_min, "score_min": thr},
                )

    # Ensemble/refinement within the same bounded approve-only family.
    ensemble_defs: dict[str, Callable[[dict[str, Any], float, float, float], float]] = {
        "min_r9_visual": lambda row, r9, visual, ast: min(r9, visual),
        "mean_r9_visual": lambda row, r9, visual, ast: 0.5 * r9 + 0.5 * visual,
        "mean_r9_visual_ast": lambda row, r9, visual, ast: (r9 + visual + ast) / 3.0,
        "approve_margin": lambda row, r9, visual, ast: 0.45 * r9 + 0.35 * visual + 0.20 * ast,
    }
    visual_scores = score_bank[visual_key]
    ast_key = "ast_audio_plus_r9_logreg_c0.25" if "ast_audio_plus_r9_logreg_c0.25" in score_bank else "r9_score"
    ast_scores = score_bank[ast_key]
    expansion_keys = [
        visual_key,
        ast_key,
        "audio_visual_ast_logreg_c0.25" if "audio_visual_ast_logreg_c0.25" in score_bank else visual_key,
        "audio_visual_ast_logreg_c0.5" if "audio_visual_ast_logreg_c0.5" in score_bank else visual_key,
    ]
    for score_key in sorted(set(expansion_keys)):
        scores = score_bank[score_key]
        for r9_floor in [0.50, 0.60, 0.70, 0.78, 0.82, 0.86, 0.88, 0.90]:
            for guard_thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.97]:
                evaluate(
                    f"r17_or_guarded_expansion::{score_key}::{r9_floor:.2f}::{guard_thr:.2f}",
                    "Keep the r17 safe r9 gate and add a guarded lower-score approve path.",
                    [
                        row["r9_score"] >= R17_APPROVE_MIN_SCORE
                        or (row["r9_score"] >= r9_floor and score >= guard_thr)
                        for row, score in zip(internal_rows, scores["internal"])
                    ],
                    [
                        row["r9_score"] >= R17_APPROVE_MIN_SCORE
                        or (row["r9_score"] >= r9_floor and score >= guard_thr)
                        for row, score in zip(external_rows, scores["external"])
                    ],
                    {
                        "type": "r17_or_guarded_expansion",
                        "base_score_key": "r9_score",
                        "base_score_min": R17_APPROVE_MIN_SCORE,
                        "guard_score_key": score_key,
                        "r9_floor": r9_floor,
                        "guard_score_min": guard_thr,
                    },
                )
    for name, fn in ensemble_defs.items():
        ens_int = [fn(row, row["r9_score"], v, a) for row, v, a in zip(internal_rows, visual_scores["internal"], ast_scores["internal"])]
        ens_ext = [fn(row, row["r9_score"], v, a) for row, v, a in zip(external_rows, visual_scores["external"], ast_scores["external"])]
        for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            evaluate(
                f"ensemble::{name}::{thr:.2f}",
                f"Approve-only ensemble/margin gate {name}.",
                [score >= thr for score in ens_int],
                [score >= thr for score in ens_ext],
                {"type": "ensemble_gate", "score_key": name, "score_min": thr, "visual_key": visual_key, "ast_key": ast_key},
            )

    interesting = [row for row in comparison if row["dangerous_external_auto_approves"] == 0 and row["dangerous_internal_auto_approves"] == 0 and fnum(row["external_approve_precision"]) >= 0.95]
    best = max(
        interesting or comparison,
        key=lambda row: (
            row["dangerous_external_auto_approves"] == 0 and row["dangerous_internal_auto_approves"] == 0,
            fnum(row["external_approve_precision"]),
            row["external_approve_coverage"],
            row["external_approve_count"],
            -row["external_review_burden"],
        ),
    )
    viable = (
        best["dangerous_external_auto_approves"] == 0
        and best["dangerous_internal_auto_approves"] == 0
        and fnum(best["external_approve_precision"]) >= 0.95
        and best["external_approve_coverage"] > R17_APPROVE_COVERAGE
    )
    decision = "R18_SAFE_APPROVE_COVERAGE_GAIN" if viable else "R18_SAFE_APPROVE_COVERAGE_NO_CLEAR_GAIN"

    benchmark_rows = [row for row in internal_rows + external_rows if row["role"] != "other" or row["r9_score"] >= 0.50]
    dataset = {
        "dataset_name": "r18_approve_benchmark_dataset",
        "row_count": len(benchmark_rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in benchmark_rows).items())),
        "subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in benchmark_rows).items())),
        "source_session_counts": dict(sorted(Counter(row["source_session_id"] for row in benchmark_rows).items())),
        "role_counts": dict(sorted(Counter(row["role"] for row in benchmark_rows).items())),
        "approve_risk_rows": [compact(row) for row in benchmark_rows if row["role"] == "approve_risk"],
        "positive_control_rows": [compact(row) for row in benchmark_rows if row["role"] == "high_confidence_platform_control"],
        "nuisance_control_rows": [compact(row) for row in benchmark_rows if row["role"] == "high_confidence_nuisance_control"],
        "rows": [compact(row) for row in benchmark_rows],
    }
    OUT_DATASET_JSON.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    OUT_DATASET_MD.write_text(
        "\n".join([
            "# r18 Approve Benchmark Dataset",
            "",
            f"- row count: `{dataset['row_count']}`",
            f"- label counts: `{json.dumps(dataset['label_counts'], sort_keys=True)}`",
            f"- subtype counts: `{json.dumps(dataset['subtype_counts'], sort_keys=True)}`",
            f"- source-session counts: `{json.dumps(dataset['source_session_counts'], sort_keys=True)}`",
            f"- role counts: `{json.dumps(dataset['role_counts'], sort_keys=True)}`",
            f"- approve-risk rows: `{len(dataset['approve_risk_rows'])}`",
            f"- positive controls: `{len(dataset['positive_control_rows'])}`",
            f"- nuisance controls: `{len(dataset['nuisance_control_rows'])}`",
        ]) + "\n",
        encoding="utf-8",
    )

    top_rows = sorted(
        comparison,
        key=lambda row: (
            row["dangerous_external_auto_approves"] == 0 and row["dangerous_internal_auto_approves"] == 0,
            fnum(row["external_approve_precision"]),
            row["external_approve_coverage"],
            row["external_approve_count"],
        ),
        reverse=True,
    )[:30]
    report = {
        "experiment_name": "r18_safe_approve_coverage_expansion",
        "final_decision": decision,
        "best_candidate": best["candidate"],
        "best_candidate_metrics": best,
        "candidate_count": len(comparison),
        "safe_candidate_count": len(interesting),
        "comparison_rows": top_rows,
        "coverage_thresholds_crossed": {
            "crosses_015": bool(best["crosses_015"]),
            "crosses_020": bool(best["crosses_020"]),
            "crosses_025": bool(best["crosses_025"]),
            "crosses_030": bool(best["crosses_030"]),
        },
        "r17_reference_approve_coverage": R17_APPROVE_COVERAGE,
        "coverage_delta_vs_r17": best["external_approve_coverage"] - R17_APPROVE_COVERAGE,
        "stopping_rule_result": "safe_approve_coverage_gain_found" if viable else "no_candidate_safely_improved_on_r17_coverage",
    }
    comparison_payload = {
        "experiment_name": "r18_approve_candidate_comparison",
        "final_decision": decision,
        "best_candidate": best["candidate"],
        "candidate_count": len(comparison),
        "safe_candidate_count": len(interesting),
        "comparison_rows": top_rows,
    }
    best_payload = {
        "experiment_name": "r18_best_approve_policy",
        "final_decision": decision,
        "best_candidate": best["candidate"],
        "best_candidate_metrics": best,
        "r17_reference_approve_coverage": R17_APPROVE_COVERAGE,
        "coverage_delta_vs_r17": best["external_approve_coverage"] - R17_APPROVE_COVERAGE,
        "product_interpretation": "Safe approve coverage improved beyond the r17 reference." if viable else "No bounded approve-side candidate safely improved on the r17 coverage point.",
    }
    OUT_BENCH_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_CMP_JSON.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
    OUT_BEST_JSON.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    table = [
        "| candidate | ext precision | ext count | ext coverage | ext dangerous | int dangerous | crosses .15 | crosses .20 | crosses .25 | crosses .30 |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in top_rows[:15]:
        table.append(
            f"| `{row['candidate']}` | {fnum(row['external_approve_precision']):.4f} | {row['external_approve_count']} | {row['external_approve_coverage']:.4f} | {row['dangerous_external_auto_approves']} | {row['dangerous_internal_auto_approves']} | `{row['crosses_015']}` | `{row['crosses_020']}` | `{row['crosses_025']}` | `{row['crosses_030']}` |"
        )
    md = "\n".join([
        "# r18 Safe Approve Coverage Expansion",
        "",
        f"- final decision: `{decision}`",
        f"- best candidate: `{best['candidate']}`",
        f"- candidate count: `{len(comparison)}`",
        f"- safe candidate count: `{len(interesting)}`",
        f"- best external approve precision: `{fnum(best['external_approve_precision']):.4f}`",
        f"- best external approve coverage: `{best['external_approve_coverage']:.4f}`",
        f"- r17 reference approve coverage: `{R17_APPROVE_COVERAGE:.4f}`",
        f"- coverage delta vs r17: `{best['external_approve_coverage'] - R17_APPROVE_COVERAGE:+.4f}`",
        f"- best external dangerous approves: `{best['dangerous_external_auto_approves']}`",
        f"- best internal dangerous approves: `{best['dangerous_internal_auto_approves']}`",
        "",
        *table,
    ]) + "\n"
    OUT_BENCH_MD.write_text(md, encoding="utf-8")
    OUT_CMP_MD.write_text(md.replace("# r18 Safe Approve Coverage Expansion", "# r18 Approve Candidate Comparison"), encoding="utf-8")
    OUT_BEST_MD.write_text(
        "\n".join([
            "# r18 Best Approve Policy",
            "",
            f"- final decision: `{decision}`",
            f"- best candidate: `{best['candidate']}`",
            f"- external approve precision: `{fnum(best['external_approve_precision']):.4f}`",
            f"- external approve count: `{best['external_approve_count']}`",
            f"- external approve coverage: `{best['external_approve_coverage']:.4f}`",
            f"- r17 reference approve coverage: `{R17_APPROVE_COVERAGE:.4f}`",
            f"- coverage delta vs r17: `{best['external_approve_coverage'] - R17_APPROVE_COVERAGE:+.4f}`",
            f"- dangerous external approves: `{best['dangerous_external_auto_approves']}`",
            f"- dangerous internal approves: `{best['dangerous_internal_auto_approves']}`",
            f"- crosses 0.15: `{best['crosses_015']}`",
            f"- crosses 0.20: `{best['crosses_020']}`",
            f"- crosses 0.25: `{best['crosses_025']}`",
            f"- crosses 0.30: `{best['crosses_030']}`",
            "",
            best_payload["product_interpretation"],
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": [str(OUT_DATASET_JSON), str(OUT_DATASET_MD), str(OUT_BENCH_JSON), str(OUT_BENCH_MD), str(OUT_CMP_JSON), str(OUT_CMP_MD), str(OUT_BEST_JSON), str(OUT_BEST_MD)], "final_decision": decision, "best_candidate": best["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
