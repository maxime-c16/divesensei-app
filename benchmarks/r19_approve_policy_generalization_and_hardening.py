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

OUT_GENERALIZATION_JSON = ROOT / "outputs/r19_approve_policy_generalization.json"
OUT_GENERALIZATION_MD = ROOT / "outputs/r19_approve_policy_generalization.md"
OUT_ROBUSTNESS_JSON = ROOT / "outputs/r19_approve_policy_robustness.json"
OUT_ROBUSTNESS_MD = ROOT / "outputs/r19_approve_policy_robustness.md"
OUT_POLICY_JSON = ROOT / "outputs/approve_review_v2_candidate_policy.json"
OUT_POLICY_MD = ROOT / "outputs/approve_review_v2_candidate_policy.md"
OUT_DOC = ROOT / "docs/research/APPROVE_REVIEW_V2_PROMOTION_READINESS.md"


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
    full_scored, full_train_meta = score_with_train(train_refs, eval_groups)

    validations = []
    for split, rows in full_scored.items():
        v1 = metrics(rows, flags_v1(rows))
        v2 = metrics(rows, flags_v2(rows))
        validations.append({
            "validation": f"full_model_on_{split}",
            "validation_type": "fixed_promoted_training_protocol",
            "held_out_source": None,
            "train_meta": full_train_meta,
            "row_count": len(rows),
            "v1": v1,
            "v2_candidate": v2,
            "coverage_delta_v2_minus_v1": v2["approve_coverage"] - v1["approve_coverage"],
            "approve_count_delta_v2_minus_v1": v2["approve_count"] - v1["approve_count"],
        })

    for heldout_source, refs in session_refs.items():
        fold_train = [(source, ref) for source, ref in train_refs if source != heldout_source]
        fold_scored, fold_meta = score_with_train(fold_train, {f"heldout_source_unit_{heldout_source}": refs})
        rows = fold_scored[f"heldout_source_unit_{heldout_source}"]
        v1 = metrics(rows, flags_v1(rows))
        v2 = metrics(rows, flags_v2(rows))
        validations.append({
            "validation": f"leave_one_source_out_{heldout_source}",
            "validation_type": "leave_one_source_unit_out",
            "held_out_source": heldout_source,
            "train_meta": fold_meta,
            "row_count": len(rows),
            "v1": v1,
            "v2_candidate": v2,
            "coverage_delta_v2_minus_v1": v2["approve_coverage"] - v1["approve_coverage"],
            "approve_count_delta_v2_minus_v1": v2["approve_count"] - v1["approve_count"],
        })

    internal_v2 = next(item for item in validations if item["validation"] == "full_model_on_internal_official_holdout")["v2_candidate"]
    external_v2 = next(item for item in validations if item["validation"] == "full_model_on_corrected_external_holdout")["v2_candidate"]
    loso = [item for item in validations if item["validation_type"] == "leave_one_source_unit_out"]
    loso_any_danger = any(item["v2_candidate"]["dangerous_approves"] > 0 for item in loso)
    fixed_any_danger = internal_v2["dangerous_approves"] > 0 or external_v2["dangerous_approves"] > 0
    external_v1 = next(item for item in validations if item["validation"] == "full_model_on_corrected_external_holdout")["v1"]
    promotable = (
        not fixed_any_danger
        and not loso_any_danger
        and fnum(external_v2["approve_precision"]) >= 0.95
        and external_v2["approve_coverage"] > external_v1["approve_coverage"]
    )

    perturbations = []
    external_rows = full_scored["corrected_external_holdout"]
    internal_rows = full_scored["internal_official_holdout"]
    for floor in [0.66, 0.68, 0.70, 0.72, 0.74]:
        for visual_thr in [0.51, 0.53, 0.55, 0.57, 0.59]:
            ext = metrics(external_rows, flags_v2(external_rows, V2_R9_MIN, floor, visual_thr))
            inte = metrics(internal_rows, flags_v2(internal_rows, V2_R9_MIN, floor, visual_thr))
            perturbations.append({
                "r9_floor": floor,
                "visual_min": visual_thr,
                "external_approve_precision": ext["approve_precision"],
                "external_approve_count": ext["approve_count"],
                "external_approve_coverage": ext["approve_coverage"],
                "dangerous_external_approves": ext["dangerous_approves"],
                "dangerous_internal_approves": inte["dangerous_approves"],
                "is_safe": ext["dangerous_approves"] == 0 and inte["dangerous_approves"] == 0,
            })
    safe_perturbations = [row for row in perturbations if row["is_safe"]]
    source_gain_rows = []
    for split, rows in full_scored.items():
        v1_flags = flags_v1(rows)
        v2_flags = flags_v2(rows)
        added = [row for row, old, new in zip(rows, v1_flags, v2_flags) if new and not old]
        source_gain_rows.append({
            "split": split,
            "added_approve_count": len(added),
            "added_label_counts": dict(sorted(Counter(row["label"] for row in added).items())),
            "added_source_session_counts": dict(sorted(Counter(row["source_session_id"] for row in added).items())),
            "added_rows": [compact(row) for row in added],
        })

    robustness = {
        "experiment_name": "r19_approve_policy_robustness",
        "policy_under_audit": "approve_review_v2_candidate",
        "candidate": "r17_or_guarded_expansion::visual_late_fusion_logreg_c0.5::0.70::0.55",
        "threshold_perturbation_count": len(perturbations),
        "safe_perturbation_count": len(safe_perturbations),
        "safe_perturbation_best_coverage": max((row["external_approve_coverage"] for row in safe_perturbations), default=0.0),
        "selected_policy_is_inside_safe_plateau": any(row["r9_floor"] == V2_R9_FLOOR and row["visual_min"] == V2_VISUAL_MIN and row["is_safe"] for row in perturbations),
        "source_gain_rows": source_gain_rows,
        "perturbations": perturbations,
        "robustness_interpretation": "robust_enough_for_flagged_shadow" if promotable and len(safe_perturbations) >= 5 else "not_robust_enough_for_default_promotion",
    }
    final_decision = "R19_APPROVE_V2_PROMOTABLE" if promotable else "R19_APPROVE_V2_NOT_YET_PROMOTABLE"
    rollout_decision = "APPROVE_REVIEW_V2_READY_FOR_FLAGGED_ROLLOUT" if promotable else "APPROVE_REVIEW_V1_REMAINS_DEFAULT"
    generalization = {
        "experiment_name": "r19_approve_policy_generalization_and_hardening",
        "final_decision": final_decision,
        "rollout_decision": rollout_decision,
        "policies": {
            "approve_review_v1": {"r9_score_min": V1_R9_MIN},
            "approve_review_v2_candidate": {
                "r9_score_min": V2_R9_MIN,
                "or_guard": {"r9_score_min": V2_R9_FLOOR, "visual_late_fusion_logreg_c0.5_min": V2_VISUAL_MIN},
            },
        },
        "validation_rows": validations,
        "promotion_criteria": {
            "dangerous_external_approves_zero": external_v2["dangerous_approves"] == 0,
            "dangerous_internal_approves_zero": internal_v2["dangerous_approves"] == 0,
            "leave_one_source_out_danger_zero": not loso_any_danger,
            "external_precision_high": fnum(external_v2["approve_precision"]) >= 0.95,
            "external_coverage_above_v1": external_v2["approve_coverage"] > external_v1["approve_coverage"],
        },
    }
    policy_payload = {
        "policy_id": "approve_review_v2_candidate",
        "status": "promotable_flagged_shadow" if promotable else "experimental_not_default",
        "base_policy": "approve_review_v1",
        "model_ref": "r9_compact_nuisance_generalization_weighted",
        "visual_guard_ref": "visual_late_fusion_logreg_c0.5",
        "logic": {
            "approve_if_any": [
                {"r9_score_gte": V2_R9_MIN},
                {"r9_score_gte": V2_R9_FLOOR, "visual_late_fusion_logreg_c0.5_gte": V2_VISUAL_MIN},
            ],
            "otherwise": "needs_review",
        },
        "validated_metrics": {
            "fixed_internal": internal_v2,
            "fixed_corrected_external": external_v2,
        },
        "decision": final_decision,
        "rollout_decision": rollout_decision,
        "default_behavior": "v1 remains default; v2 may run in shadow mode only" if promotable else "v1 remains default; v2 remains experimental",
    }

    OUT_GENERALIZATION_JSON.write_text(json.dumps(generalization, indent=2), encoding="utf-8")
    OUT_ROBUSTNESS_JSON.write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    OUT_POLICY_JSON.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    validation_table = [
        "| validation | rows | v1 precision | v1 coverage | v1 danger | v2 precision | v2 coverage | v2 danger | delta coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validations:
        validation_table.append(
            f"| `{row['validation']}` | {row['row_count']} | {fnum(row['v1']['approve_precision']):.4f} | {row['v1']['approve_coverage']:.4f} | {row['v1']['dangerous_approves']} | {fnum(row['v2_candidate']['approve_precision']):.4f} | {row['v2_candidate']['approve_coverage']:.4f} | {row['v2_candidate']['dangerous_approves']} | {row['coverage_delta_v2_minus_v1']:+.4f} |"
        )
    OUT_GENERALIZATION_MD.write_text(
        "\n".join([
            "# r19 Approve Policy Generalization",
            "",
            f"- final decision: `{final_decision}`",
            f"- rollout decision: `{rollout_decision}`",
            f"- fixed external v2 precision: `{fnum(external_v2['approve_precision']):.4f}`",
            f"- fixed external v2 coverage: `{external_v2['approve_coverage']:.4f}`",
            f"- fixed external v2 dangerous approves: `{external_v2['dangerous_approves']}`",
            f"- fixed internal v2 dangerous approves: `{internal_v2['dangerous_approves']}`",
            f"- leave-one-source-out any v2 danger: `{loso_any_danger}`",
            "",
            *validation_table,
        ]) + "\n",
        encoding="utf-8",
    )

    pert_table = [
        "| r9 floor | visual min | precision | count | coverage | ext danger | int danger | safe |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(perturbations, key=lambda item: (item["is_safe"], item["external_approve_coverage"]), reverse=True)[:15]:
        pert_table.append(
            f"| {row['r9_floor']:.2f} | {row['visual_min']:.2f} | {fnum(row['external_approve_precision']):.4f} | {row['external_approve_count']} | {row['external_approve_coverage']:.4f} | {row['dangerous_external_approves']} | {row['dangerous_internal_approves']} | `{row['is_safe']}` |"
        )
    OUT_ROBUSTNESS_MD.write_text(
        "\n".join([
            "# r19 Approve Policy Robustness",
            "",
            f"- candidate: `r17_or_guarded_expansion::visual_late_fusion_logreg_c0.5::0.70::0.55`",
            f"- threshold perturbations tested: `{len(perturbations)}`",
            f"- safe perturbations: `{len(safe_perturbations)}`",
            f"- selected policy inside safe plateau: `{robustness['selected_policy_is_inside_safe_plateau']}`",
            f"- interpretation: `{robustness['robustness_interpretation']}`",
            "",
            *pert_table,
        ]) + "\n",
        encoding="utf-8",
    )
    OUT_POLICY_MD.write_text(
        "\n".join([
            "# approve_review_v2_candidate Policy",
            "",
            f"- status: `{policy_payload['status']}`",
            "- rule: approve if `r9_score >= 0.92158`",
            "- expansion rule: or approve if `r9_score >= 0.70` and `visual_late_fusion_logreg_c0.5 >= 0.55`",
            "- otherwise: `needs_review`",
            f"- decision: `{final_decision}`",
            f"- rollout decision: `{rollout_decision}`",
            "",
            "This policy must not replace `approve_review_v1` as the default unless the required visual guard score is generated in the app workflow and the flagged rollout is accepted.",
        ]) + "\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(
        "\n".join([
            "# Approve Review v2 Promotion Readiness",
            "",
            "`approve_review_v2_candidate` is the r18 approve-side expansion candidate.",
            "",
            "## Candidate Logic",
            "",
            "- approve if `r9_score >= 0.92158`",
            "- or approve if `r9_score >= 0.70` and `visual_late_fusion_logreg_c0.5 >= 0.55`",
            "- otherwise keep the row in `Needs review`",
            "",
            "## r19 Decision",
            "",
            f"- generalization decision: `{final_decision}`",
            f"- rollout decision: `{rollout_decision}`",
            "",
            "## Product Guidance",
            "",
            "- `approve_review_v1` remains the visible default.",
            "- `approve_review_v2_candidate` can be prepared for flagged shadow-mode evaluation only if the visual guard score is available in manifests.",
            "- Do not introduce an auto-excluded lane.",
            "- Do not replace v1 silently.",
            "",
            "## Evidence",
            "",
            *validation_table,
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "wrote": [str(OUT_GENERALIZATION_JSON), str(OUT_GENERALIZATION_MD), str(OUT_ROBUSTNESS_JSON), str(OUT_ROBUSTNESS_MD), str(OUT_POLICY_JSON), str(OUT_POLICY_MD), str(OUT_DOC)],
        "final_decision": final_decision,
        "rollout_decision": rollout_decision,
        "promotable": promotable,
    }, indent=2))


if __name__ == "__main__":
    main()
