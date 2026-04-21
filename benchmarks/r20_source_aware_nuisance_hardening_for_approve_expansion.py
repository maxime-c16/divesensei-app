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
R15 = ROOT / "benchmarks/r15_stronger_visual_verifier_benchmark.py"
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
SOURCE_WEIGHTS = {"snmt": 1.0, "img_8852": 1.0, "champigny_1704": 0.3}

V1_R9_MIN = 0.92158
V2_R9_MIN = 0.92158
V2_R9_FLOOR = 0.70
V2_VISUAL_MIN = 0.55

OUT_BANK_JSON = ROOT / "outputs/r20_nuisance_hardening_bank.json"
OUT_BANK_MD = ROOT / "outputs/r20_nuisance_hardening_bank.md"
OUT_HARDENING_JSON = ROOT / "outputs/r20_source_aware_nuisance_hardening.json"
OUT_HARDENING_MD = ROOT / "outputs/r20_source_aware_nuisance_hardening.md"
OUT_COMPARISON_JSON = ROOT / "outputs/r20_approve_candidate_comparison.json"
OUT_COMPARISON_MD = ROOT / "outputs/r20_approve_candidate_comparison.md"
OUT_BEST_JSON = ROOT / "outputs/r20_best_hardened_policy.json"
OUT_BEST_MD = ROOT / "outputs/r20_best_hardened_policy.md"


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


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def fnum(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "split": row["split"],
        "source": row.get("source"),
        "label": row["label"],
        "r9_score": row["r9_score"],
        "visual_score": row.get("visual_score"),
        "legacy_subtype": row.get("legacy_subtype"),
        "source_session_id": row["source_session_id"],
    }


def metrics(rows: list[dict[str, Any]], flags: list[bool]) -> dict[str, Any]:
    approved = [row for row, flag in zip(rows, flags) if flag]
    errors = [row for row in approved if row["label"] != "platform_dive"]
    return {
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in rows).items())),
        "source_session_counts": dict(sorted(Counter(row["source_session_id"] for row in rows).items())),
        "approve_count": len(approved),
        "approve_coverage": safe_div(len(approved), len(rows)) or 0.0,
        "approve_precision": safe_div(sum(1 for row in approved if row["label"] == "platform_dive"), len(approved)),
        "dangerous_approves": len(errors),
        "approved_label_counts": dict(sorted(Counter(row["label"] for row in approved).items())),
        "dangerous_approve_rows": [compact(row) for row in errors],
    }


def flags_v1(rows: list[dict[str, Any]], r9_min: float = V1_R9_MIN) -> list[bool]:
    return [row["r9_score"] >= r9_min for row in rows]


def flags_v2(
    rows: list[dict[str, Any]],
    r9_min: float = V2_R9_MIN,
    r9_floor: float = V2_R9_FLOOR,
    visual_min: float = V2_VISUAL_MIN,
) -> list[bool]:
    return [
        row["r9_score"] >= r9_min
        or (row["r9_score"] >= r9_floor and float(row.get("visual_score") or 0.0) >= visual_min)
        for row in rows
    ]


def vec_from_features(row: dict[str, Any], names: list[str]) -> list[float]:
    vals = []
    for name in names:
        try:
            vals.append(float(row.get(name, 0.0) or 0.0))
        except Exception:
            vals.append(0.0)
    return vals


def xgb_model() -> XGBClassifier:
    return XGBClassifier(
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


def train_weights(train_refs: list[tuple[str, Any]]) -> np.ndarray:
    weights = np.ones(len(train_refs), dtype=np.float64)
    base_total = sum(1 for source, _ in train_refs if source == "base")
    by_source: dict[str, list[int]] = {}
    for idx, (source, _) in enumerate(train_refs):
        by_source.setdefault(source, []).append(idx)
    for source, idxs in by_source.items():
        if source == "base":
            continue
        per_item = base_total * SOURCE_WEIGHTS.get(source, 1.0) / len(idxs)
        for idx in idxs:
            weights[idx] = per_item
    return weights


def main() -> None:
    r15 = load_module("r15_runtime_for_r19", R15)
    phase5 = load_module("phase5_r19", PHASE5)
    bench = load_module("nuisance_r19", NUISANCE)

    preview = row_key_map(load_jsonl(PREVIEW))
    lists = json.loads(DATASET.read_text())
    base_refs = [("base", r15.RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])])) for item in lists["train_rows"]]
    internal_refs = [r15.RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])]) for item in lists["holdout_rows"]]
    external_refs = [r15.RowRef(str(row["row_key"]), str(row["final_human_event_label"]), row) for row in json.loads(EXTERNAL.read_text())["rows"]]
    session_refs = {source: manifest_refs(r15, load_jsonl(path)) for source, path in SOURCES.items()}
    train_refs = base_refs + [(source, ref) for source, refs in session_refs.items() for ref in refs]
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

    clip_cache = np.load(CLIP_CACHE, allow_pickle=True)
    clip_index = {str(key): idx for idx, key in enumerate(clip_cache["keys"].tolist())}
    clip_embeddings = clip_cache["clip_embeddings"]
    morph_features = clip_cache["morph_features"]
    morph_names = sorted(r15.morphology_v2_features(np.zeros((0, 224, 224), dtype=np.float32)))

    def r9_vec(ref: Any) -> list[float]:
        return bench.vector_for(phase5, ref, fmap, bench.NOISE_BOUNDARY_COMPACT)

    def make_rows(refs: list[Any], scores: list[float], split: str, source: str) -> list[dict[str, Any]]:
        rows = []
        for ref, score in zip(refs, scores):
            row = {
                "split": split,
                "source": source,
                "row_key": ref.row_key,
                "label": ref.label,
                "source_session_id": str(ref.row["source_session_id"]),
                "legacy_subtype": ref.row.get("legacy_subtype"),
                "suggested_event_label_reason": ref.row.get("suggested_event_label_reason"),
                "r9_score": float(score),
            }
            idx = clip_index[row["row_key"]]
            for name, value in zip(morph_names, morph_features[idx]):
                row[name] = float(value)
            rows.append(row)
        return rows

    def clip_x(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([clip_embeddings[clip_index[row["row_key"]]] for row in rows], dtype=np.float64)

    def morph_x(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([vec_from_features(row, morph_names + ["r9_score"]) for row in rows], dtype=np.float64)

    def visual_x(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.hstack([clip_x(rows), morph_x(rows)])

    def score_with_train(train_ref_pairs: list[tuple[str, Any]], eval_groups: dict[str, list[Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        train_only = [ref for _, ref in train_ref_pairs]
        x_train = np.asarray([r9_vec(ref) for ref in train_only], dtype=np.float64)
        y_train = np.asarray([label_int(ref.label) for ref in train_only], dtype=np.int64)
        weights = train_weights(train_ref_pairs)
        model = xgb_model()
        model.fit(x_train, y_train, sample_weight=weights)
        train_rows = make_rows(train_only, model.predict_proba(x_train)[:, 1].tolist(), "train_augmented", "train")
        visual = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42))
        visual.fit(visual_x(train_rows), y_train, logisticregression__sample_weight=weights)
        scored: dict[str, list[dict[str, Any]]] = {}
        for name, refs in eval_groups.items():
            if not refs:
                scored[name] = []
                continue
            rows = make_rows(
                refs,
                model.predict_proba(np.asarray([r9_vec(ref) for ref in refs], dtype=np.float64))[:, 1].tolist(),
                name,
                name,
            )
            visual_scores = visual.predict_proba(visual_x(rows))[:, 1].tolist()
            for row, score in zip(rows, visual_scores):
                row["visual_score"] = float(score)
            scored[name] = rows
        return scored, {"train_row_count": len(train_only), "train_source_counts": dict(sorted(Counter(source for source, _ in train_ref_pairs).items()))}

    eval_groups = {
        "internal_official_holdout": internal_refs,
        "corrected_external_holdout": external_refs,
        **{f"source_unit_{source}": refs for source, refs in session_refs.items()},
    }
    full_scored, _ = score_with_train(train_refs, eval_groups)
    loso_scored = {}
    for heldout_source, refs in session_refs.items():
        fold_train = [(source, ref) for source, ref in train_refs if source != heldout_source]
        fold_scored, _ = score_with_train(fold_train, {f"leave_one_source_out_{heldout_source}": refs})
        loso_scored.update(fold_scored)

    def flags_hardened(rows: list[dict[str, Any]], r9_floor: float, visual_min: float, visual_max: float | None = None) -> list[bool]:
        flags = []
        for row in rows:
            visual = float(row.get("visual_score") or 0.0)
            expansion = row["r9_score"] >= r9_floor and visual >= visual_min
            if visual_max is not None and visual > visual_max and row["r9_score"] < V2_R9_MIN:
                expansion = False
            flags.append(row["r9_score"] >= V2_R9_MIN or expansion)
        return flags

    candidates = [
        {"id": "approve_review_v1", "family": "baseline", "r9_floor": None, "visual_min": None, "visual_max": None},
        {"id": "approve_review_v2_candidate", "family": "r18_candidate", "r9_floor": 0.70, "visual_min": 0.55, "visual_max": None},
    ]
    for floor in [0.82, 0.84, 0.86, 0.88, 0.90]:
        for visual_min in [0.55, 0.65, 0.75, 0.85]:
            candidates.append({"id": f"hardened_floor_{floor:.2f}_visual_{visual_min:.2f}", "family": "higher_r9_floor", "r9_floor": floor, "visual_min": visual_min, "visual_max": None})
    for floor in [0.78, 0.82, 0.86]:
        for visual_max in [0.95, 0.98]:
            candidates.append({"id": f"hardened_high_visual_veto_floor_{floor:.2f}_max_{visual_max:.2f}", "family": "high_visual_veto", "r9_floor": floor, "visual_min": 0.55, "visual_max": visual_max})

    def eval_candidate(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        if candidate["id"] == "approve_review_v1":
            return metrics(rows, flags_v1(rows))
        if candidate["id"] == "approve_review_v2_candidate":
            return metrics(rows, flags_v2(rows))
        return metrics(rows, flags_hardened(rows, float(candidate["r9_floor"]), float(candidate["visual_min"]), candidate["visual_max"]))

    comparison = []
    for candidate in candidates:
        fixed_internal = eval_candidate(candidate, full_scored["internal_official_holdout"])
        fixed_external = eval_candidate(candidate, full_scored["corrected_external_holdout"])
        source_metrics = {name: eval_candidate(candidate, rows) for name, rows in loso_scored.items()}
        source_danger = sum(item["dangerous_approves"] for item in source_metrics.values())
        any_danger = fixed_internal["dangerous_approves"] + fixed_external["dangerous_approves"] + source_danger
        v1_external = eval_candidate(candidates[0], full_scored["corrected_external_holdout"])
        v2_external = eval_candidate(candidates[1], full_scored["corrected_external_holdout"])
        comparison.append({
            "candidate": candidate["id"],
            "family": candidate["family"],
            "policy": candidate,
            "fixed_internal": fixed_internal,
            "fixed_external": fixed_external,
            "source_aware_holdout": source_metrics,
            "source_aware_dangerous_approves": source_danger,
            "total_dangerous_approves": any_danger,
            "external_coverage_delta_vs_v1": fixed_external["approve_coverage"] - v1_external["approve_coverage"],
            "external_coverage_recovered_vs_v2_gain": (fixed_external["approve_coverage"] - v1_external["approve_coverage"]) / max(v2_external["approve_coverage"] - v1_external["approve_coverage"], 1e-9),
            "interesting": any_danger == 0 and fnum(fixed_external["approve_precision"]) >= 0.95 and fixed_external["approve_coverage"] > v1_external["approve_coverage"],
        })
    best = max(
        comparison,
        key=lambda row: (
            row["interesting"],
            row["total_dangerous_approves"] == 0,
            fnum(row["fixed_external"]["approve_precision"]),
            row["fixed_external"]["approve_coverage"],
        ),
    )
    decision = "R20_SOURCE_AWARE_HARDENING_GAIN" if best["interesting"] else "R20_SOURCE_AWARE_HARDENING_NO_CLEAR_GAIN"
    rollout = "HARDENED_V2_READY_FOR_SHADOW_MODE" if best["interesting"] else "APPROVE_REVIEW_V1_REMAINS_DEFAULT"

    all_bank_rows = []
    for name, rows in {**full_scored, **loso_scored}.items():
        for row in rows:
            subtype = str(row.get("legacy_subtype") or "none")
            is_nuisance_control = row["label"] == "noise_or_other" and subtype in {"handling_noise", "voice_whistle", "non_dive_splash", "unknown_transient", "none"}
            is_positive_control = row["label"] == "platform_dive" and (row["r9_score"] >= 0.70 or float(row.get("visual_score") or 0.0) >= 0.55)
            is_r19_danger = name == "leave_one_source_out_img_8852" and row["label"] == "noise_or_other" and flags_v2([row])[0]
            if is_nuisance_control or is_positive_control or is_r19_danger:
                item = compact(row)
                item["bank_split"] = name
                item["is_nuisance_negative_hard_control"] = bool(is_nuisance_control or is_r19_danger)
                item["is_positive_approve_control"] = bool(is_positive_control)
                item["is_r19_exposed_dangerous_row"] = bool(is_r19_danger)
                all_bank_rows.append(item)
    bank = {
        "bank_name": "r20_source_aware_nuisance_hardening_bank",
        "row_count": len(all_bank_rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in all_bank_rows).items())),
        "subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in all_bank_rows).items())),
        "source_session_counts": dict(sorted(Counter(row["source_session_id"] for row in all_bank_rows).items())),
        "nuisance_negative_hard_control_count": sum(1 for row in all_bank_rows if row["is_nuisance_negative_hard_control"]),
        "positive_approve_control_count": sum(1 for row in all_bank_rows if row["is_positive_approve_control"]),
        "r19_exposed_dangerous_rows": [row for row in all_bank_rows if row["is_r19_exposed_dangerous_row"]],
        "rows": all_bank_rows,
    }
    payload = {
        "experiment_name": "r20_source_aware_nuisance_hardening_for_approve_expansion",
        "final_decision": decision,
        "rollout_decision": rollout,
        "best_candidate": best["candidate"],
        "best_candidate_metrics": best,
        "candidate_count": len(comparison),
        "interesting_candidate_count": sum(1 for row in comparison if row["interesting"]),
        "comparison_rows": sorted(comparison, key=lambda row: (row["interesting"], row["total_dangerous_approves"] == 0, row["fixed_external"]["approve_coverage"]), reverse=True),
    }
    OUT_BANK_JSON.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    OUT_HARDENING_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_COMPARISON_JSON.write_text(json.dumps({"experiment_name": "r20_approve_candidate_comparison", **payload}, indent=2), encoding="utf-8")
    OUT_BEST_JSON.write_text(json.dumps({"experiment_name": "r20_best_hardened_policy", "final_decision": decision, "rollout_decision": rollout, "best_candidate": best}, indent=2), encoding="utf-8")

    cmp_table = [
        "| candidate | family | ext precision | ext coverage | ext danger | source danger | delta vs v1 | recovered v2 gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison_rows"][:15]:
        cmp_table.append(
            f"| `{row['candidate']}` | `{row['family']}` | {fnum(row['fixed_external']['approve_precision']):.4f} | {row['fixed_external']['approve_coverage']:.4f} | {row['fixed_external']['dangerous_approves']} | {row['source_aware_dangerous_approves']} | {row['external_coverage_delta_vs_v1']:+.4f} | {row['external_coverage_recovered_vs_v2_gain']:.4f} |"
        )
    OUT_BANK_MD.write_text(
        "\n".join([
            "# r20 Nuisance Hardening Bank",
            "",
            f"- row count: `{bank['row_count']}`",
            f"- label counts: `{json.dumps(bank['label_counts'], sort_keys=True)}`",
            f"- subtype counts: `{json.dumps(bank['subtype_counts'], sort_keys=True)}`",
            f"- source-session counts: `{json.dumps(bank['source_session_counts'], sort_keys=True)}`",
            f"- nuisance-negative hard controls: `{bank['nuisance_negative_hard_control_count']}`",
            f"- positive approve controls: `{bank['positive_approve_control_count']}`",
            f"- r19 exposed dangerous rows: `{len(bank['r19_exposed_dangerous_rows'])}`",
        ]) + "\n",
        encoding="utf-8",
    )
    md = "\n".join([
        "# r20 Source-Aware Nuisance Hardening",
        "",
        f"- final decision: `{decision}`",
        f"- rollout decision: `{rollout}`",
        f"- best candidate: `{best['candidate']}`",
        f"- candidate count: `{len(comparison)}`",
        f"- interesting candidates: `{sum(1 for row in comparison if row['interesting'])}`",
        f"- best external precision: `{fnum(best['fixed_external']['approve_precision']):.4f}`",
        f"- best external coverage: `{best['fixed_external']['approve_coverage']:.4f}`",
        f"- best fixed external dangerous approves: `{best['fixed_external']['dangerous_approves']}`",
        f"- best source-aware dangerous approves: `{best['source_aware_dangerous_approves']}`",
        "",
        *cmp_table,
    ]) + "\n"
    OUT_HARDENING_MD.write_text(md, encoding="utf-8")
    OUT_COMPARISON_MD.write_text(md.replace("# r20 Source-Aware Nuisance Hardening", "# r20 Approve Candidate Comparison"), encoding="utf-8")
    OUT_BEST_MD.write_text(
        "\n".join([
            "# r20 Best Hardened Policy",
            "",
            f"- final decision: `{decision}`",
            f"- rollout decision: `{rollout}`",
            f"- best candidate: `{best['candidate']}`",
            f"- family: `{best['family']}`",
            f"- fixed external precision: `{fnum(best['fixed_external']['approve_precision']):.4f}`",
            f"- fixed external coverage: `{best['fixed_external']['approve_coverage']:.4f}`",
            f"- source-aware dangerous approves: `{best['source_aware_dangerous_approves']}`",
            f"- coverage delta vs v1: `{best['external_coverage_delta_vs_v1']:+.4f}`",
            f"- recovered v2 gain fraction: `{best['external_coverage_recovered_vs_v2_gain']:.4f}`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "wrote": [str(OUT_BANK_JSON), str(OUT_BANK_MD), str(OUT_HARDENING_JSON), str(OUT_HARDENING_MD), str(OUT_COMPARISON_JSON), str(OUT_COMPARISON_MD), str(OUT_BEST_JSON), str(OUT_BEST_MD)],
        "final_decision": decision,
        "rollout_decision": rollout,
        "best_candidate": best["candidate"],
    }, indent=2))


if __name__ == "__main__":
    main()
