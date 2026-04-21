from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
R15 = ROOT / "benchmarks/r15_stronger_visual_verifier_benchmark.py"
PHASE5 = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
PREVIEW = ROOT / "outputs/event_window_manifest_preview.jsonl"
DATASET = ROOT / "outputs/platform_noise_es4_dataset_rows.json"
EXTERNAL = ROOT / "outputs/external_holdout_slice.json"

RUNTIME_ROOTS = [
    ROOT / "outputs/evaluation_r27_scorepath_insep_quick_v2",
    ROOT / "outputs/evaluation_r27_scorepath_champigny_proxy",
]
SOURCE_MANIFESTS = {
    "snmt": ROOT / "outputs/evaluation_SNMT-16min_20260417-131944/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "img_8852": ROOT / "outputs/evaluation_img_8852_rerun_20260406-104430/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "champigny_1704": ROOT / "outputs/evaluation_Champigny-17-04-9min_20260418-065417/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
}
SOURCE_WEIGHTS = {"snmt": 1.0, "img_8852": 1.0, "champigny_1704": 0.3}
V1_THRESHOLD = 0.92158

OUT_ALIGNMENT_JSON = ROOT / "outputs/r29_runtime_offline_score_alignment.json"
OUT_ALIGNMENT_MD = ROOT / "outputs/r29_runtime_offline_score_alignment.md"
OUT_AUDIT_JSON = ROOT / "outputs/r29_runtime_offline_equivalence_audit.json"
OUT_AUDIT_MD = ROOT / "outputs/r29_runtime_offline_equivalence_audit.md"
OUT_RECOMMENDATION_JSON = ROOT / "outputs/r29_runtime_score_recommendation.json"
OUT_RECOMMENDATION_MD = ROOT / "outputs/r29_runtime_score_recommendation.md"
OUT_DOC = ROOT / "docs/research/R29_RUNTIME_OFFLINE_SCORE_EQUIVALENCE.md"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        rid = str(row.get("legacy_candidate_id") or f"row-{counts[sid]:04d}")
        out[f"{sid}::{rid}"] = row
    return out


def manifest_refs(r15: Any, rows: list[dict[str, Any]], preview: dict[str, dict[str, Any]] | None = None) -> list[Any]:
    refs = []
    for row in rows:
        label = row.get("final_human_event_label") or row.get("label")
        if label not in {"platform_dive", "noise_or_other"}:
            continue
        sid = str(row["source_session_id"])
        rid = str(row.get("legacy_candidate_id") or "row-unknown")
        key = f"{sid}::{rid}"
        merged = dict(preview.get(key, {}) if preview else {})
        merged.update(row)
        refs.append(r15.RowRef(key, str(label), merged))
    return refs


def label_int(label: str) -> int:
    return 1 if label == "platform_dive" else 0


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


def build_runtime_rows() -> list[dict[str, Any]]:
    rows = []
    for root in RUNTIME_ROOTS:
        manifest_path = root / "ui_session_manifest.json"
        review_path = root / "evaluation_review.json"
        if not manifest_path.exists() or not review_path.exists():
            continue
        manifest = load_json(manifest_path)
        review = load_json(review_path)
        decisions = {
            str(item.get("detectionId")): item
            for item in review.get("decisions", [])
            if item.get("eventLabel") in {"platform_dive", "noise_or_other"}
        }
        for detection in manifest.get("detections", []):
            detection_id = str(detection.get("id"))
            decision = decisions.get(detection_id)
            if decision is None:
                continue
            original_session_id = str(decision.get("analysisRunId") or root.name)
            original_detection_id = str(
                decision.get("_replayedFromDetectionId") or decision.get("id", "").split(":")[-1] or detection_id
            )
            scores = dict(detection.get("scores", {}) or {})
            features = dict(detection.get("features", {}) or {})
            rows.append(
                {
                    "runtime_session_id": root.name,
                    "runtime_detection_id": detection_id,
                    "original_session_id": original_session_id,
                    "original_detection_id": original_detection_id,
                    "row_key": f"{original_session_id}::{original_detection_id}",
                    "label": str(decision.get("eventLabel")),
                    "runtime_governed_r9_score": scores.get("governed_r9_score"),
                    "runtime_visual_late_fusion_logreg_c0.5": features.get("visual_late_fusion_logreg_c0.5"),
                    "runtime_timestamp_seconds": detection.get("timestamp_seconds"),
                    "runtime_approved_v1": float(scores.get("governed_r9_score") or 0.0) >= V1_THRESHOLD,
                }
            )
    return rows


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row.get("row_key"),
        "runtime_session_id": row.get("runtime_session_id"),
        "runtime_detection_id": row.get("runtime_detection_id"),
        "label": row.get("label"),
        "runtime_governed_r9_score": row.get("runtime_governed_r9_score"),
        "offline_governed_r9_score": row.get("offline_governed_r9_score"),
        "runtime_approved_v1": row.get("runtime_approved_v1"),
        "offline_approved_v1": row.get("offline_approved_v1"),
        "runtime_visual_late_fusion_logreg_c0.5": row.get("runtime_visual_late_fusion_logreg_c0.5"),
    }


def table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    out = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key)
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            elif val is None:
                vals.append("n/a")
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    r15 = load_module("r15_runtime_for_r29", R15)
    phase5 = load_module("phase5_r29", PHASE5)
    nuisance = load_module("nuisance_r29", NUISANCE)

    preview_rows = load_jsonl(PREVIEW)
    preview = row_key_map(preview_rows)
    dataset = load_json(DATASET)
    base_train = [("base", r15.RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])])) for item in dataset["train_rows"]]
    internal_refs = [r15.RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])]) for item in dataset["holdout_rows"]]
    external_refs = [
        r15.RowRef(str(row["row_key"]), str(row["final_human_event_label"]), row)
        for row in load_json(EXTERNAL)["rows"]
        if row.get("final_human_event_label") in {"platform_dive", "noise_or_other"}
    ]
    source_refs = {
        source: manifest_refs(r15, load_jsonl(path), preview)
        for source, path in SOURCE_MANIFESTS.items()
        if path.exists()
    }
    train_refs = base_train + [(source, ref) for source, refs in source_refs.items() for ref in refs]
    train_items = [ref for _, ref in train_refs]

    runtime_rows = build_runtime_rows()
    target_keys = {row["row_key"] for row in runtime_rows}
    target_refs = []
    for key in sorted(target_keys):
        if key not in preview:
            continue
        row = dict(preview[key])
        # Use reviewed labels from runtime replay for the original row.
        labels = {r["label"] for r in runtime_rows if r["row_key"] == key}
        if len(labels) != 1:
            continue
        target_refs.append(r15.RowRef(key, labels.pop(), row))

    all_refs = train_items + internal_refs + external_refs + target_refs
    audio: dict[str, np.ndarray] = {}
    fmap: dict[str, dict[str, float]] = {}
    for ref in all_refs:
        sid = str(ref.row["source_session_id"])
        if sid not in audio:
            source_root = phase5.resolve_source_root(str(ref.row["source_session_root"]))
            audio[sid] = phase5.decode_audio_mono(source_root / "web/session_source_review.mp4", phase5.SAMPLE_RATE)
        start = max(0.0, phase5.to_float(ref.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(ref.row.get("event_window_end_seconds")))
        sig = audio[sid][int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap[ref.row_key] = {
            **phase5.extract_features(sig, phase5.SAMPLE_RATE),
            **nuisance.nuisance_features(phase5, sig, phase5.SAMPLE_RATE),
        }

    def vec(ref: Any) -> list[float]:
        return nuisance.vector_for(phase5, ref, fmap, nuisance.NOISE_BOUNDARY_COMPACT)

    x_train = np.asarray([vec(ref) for ref in train_items], dtype=np.float64)
    y_train = np.asarray([label_int(ref.label) for ref in train_items], dtype=np.int64)
    weights = train_weights(train_refs)
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
    model.fit(x_train, y_train, sample_weight=weights)
    offline_scores = {
        ref.row_key: float(score)
        for ref, score in zip(target_refs, model.predict_proba(np.asarray([vec(ref) for ref in target_refs], dtype=np.float64))[:, 1])
    }

    joined = []
    unmatched = []
    for row in runtime_rows:
        if row["row_key"] not in offline_scores:
            unmatched.append(row)
            continue
        out = dict(row)
        out["offline_governed_r9_score"] = offline_scores[row["row_key"]]
        out["offline_approved_v1"] = offline_scores[row["row_key"]] >= V1_THRESHOLD
        out["score_delta_runtime_minus_offline"] = float(row["runtime_governed_r9_score"]) - offline_scores[row["row_key"]]
        out["threshold_agrees_v1"] = bool(out["runtime_approved_v1"] == out["offline_approved_v1"])
        joined.append(out)

    runtime_scores = [float(row["runtime_governed_r9_score"]) for row in joined]
    offline = [float(row["offline_governed_r9_score"]) for row in joined]
    pearson = corr(runtime_scores, offline)
    spearman = corr(rankdata(np.asarray(runtime_scores)).tolist(), rankdata(np.asarray(offline)).tolist())
    runtime_approved = {row["row_key"] for row in joined if row["runtime_approved_v1"]}
    offline_approved = {row["row_key"] for row in joined if row["offline_approved_v1"]}
    overlap = runtime_approved & offline_approved
    union = runtime_approved | offline_approved
    disagreements = [row for row in joined if not row["threshold_agrees_v1"]]
    dangerous_runtime = [row for row in joined if row["runtime_approved_v1"] and row["label"] != "platform_dive"]
    dangerous_offline = [row for row in joined if row["offline_approved_v1"] and row["label"] != "platform_dive"]
    r28_dangerous = [
        row for row in joined
        if row["runtime_session_id"] == "evaluation_r27_scorepath_insep_quick_v2" and row["runtime_detection_id"] == "det-0039"
    ]

    alignment = {
        "experiment_name": "r29_runtime_offline_score_alignment",
        "runtime_roots": [root.name for root in RUNTIME_ROOTS],
        "row_counts": {
            "runtime_rows": len(runtime_rows),
            "matched_rows": len(joined),
            "unmatched_rows": len(unmatched),
            "target_offline_refs": len(target_refs),
        },
        "label_counts": dict(sorted(Counter(row["label"] for row in joined).items())),
        "source_counts": dict(sorted(Counter(row["original_session_id"] for row in joined).items())),
        "joined_rows": [compact(row) | {"score_delta_runtime_minus_offline": row["score_delta_runtime_minus_offline"]} for row in joined],
        "unmatched_rows": [compact(row) for row in unmatched],
        "dangerous_r28_row": [compact(row) | {"score_delta_runtime_minus_offline": row["score_delta_runtime_minus_offline"]} for row in r28_dangerous],
    }

    audit = {
        "experiment_name": "r29_runtime_offline_equivalence_audit",
        "equivalence_metrics": {
            "matched_rows": len(joined),
            "pearson_score_correlation": pearson,
            "spearman_rank_correlation": spearman,
            "mean_abs_score_delta": float(np.mean(np.abs(np.asarray(runtime_scores) - np.asarray(offline)))) if joined else None,
            "median_abs_score_delta": float(np.median(np.abs(np.asarray(runtime_scores) - np.asarray(offline)))) if joined else None,
            "max_abs_score_delta": float(np.max(np.abs(np.asarray(runtime_scores) - np.asarray(offline)))) if joined else None,
            "threshold_agreement_count": sum(1 for row in joined if row["threshold_agrees_v1"]),
            "threshold_agreement_rate": safe_div(sum(1 for row in joined if row["threshold_agrees_v1"]), len(joined)),
            "runtime_approved_count": len(runtime_approved),
            "offline_approved_count": len(offline_approved),
            "approve_overlap_count": len(overlap),
            "approve_union_count": len(union),
            "approve_jaccard": safe_div(len(overlap), len(union)),
            "runtime_dangerous_approve_count": len(dangerous_runtime),
            "offline_dangerous_approve_count": len(dangerous_offline),
        },
        "disagreement_rows": [compact(row) | {"score_delta_runtime_minus_offline": row["score_delta_runtime_minus_offline"]} for row in disagreements],
        "dangerous_runtime_rows": [compact(row) | {"score_delta_runtime_minus_offline": row["score_delta_runtime_minus_offline"]} for row in dangerous_runtime],
        "dangerous_offline_rows": [compact(row) for row in dangerous_offline],
        "root_cause_audit": {
            "model_identity": {
                "runtime": ".divesensei-runtime/models/governed_r9_audio_candidate_model.json bootstrapped AudioCandidateModel/logistic proxy",
                "offline_reference": "r9_compact_nuisance_generalization_weighted XGBoost protocol reconstructed from r20/r17 scripts",
                "mismatch": "different model family and different persisted model artifact",
            },
            "feature_ordering": {
                "runtime": "MODEL_FEATURES from divesensei.detection.audio_model plus simple candidate details",
                "offline_reference": "phase5 ES4 + r12/r9 compact nuisance feature vector",
                "mismatch": "runtime lacks the exact offline compact nuisance feature vector and uses fewer/different fields",
            },
            "scaling_normalization": {
                "runtime": "bootstrapped logistic model means/stds in JSON",
                "offline_reference": "XGBoost raw tree model; no same scaler",
                "mismatch": "scores are not calibrated onto the same probability scale",
            },
            "training_source": {
                "runtime": "auto-bootstrapped from all reviewed output sessions available locally",
                "offline_reference": "governed source-weighted protocol with base train rows plus selected nuisance source weights",
                "mismatch": "training set and source weights differ",
            },
            "window_extraction": {
                "runtime": "candidate-time details from live evaluate-session path",
                "offline_reference": "event-window manifest windows and source-root decoded review proxy audio",
                "mismatch": "candidate anchor/window can differ from event-window manifest representation",
            },
            "candidate_matching": {
                "method": "r27 replay metadata maps runtime detection IDs to original source detection IDs via _replayedFromDetectionId",
                "matched": len(joined),
                "unmatched": len(unmatched),
            },
            "runtime_score_paths_approximation": (
                "runtime_score_paths.py explicitly bootstraps a runtime model from reviewed rows when the exact governed model is absent; "
                "this is a plumbing proxy, not the exact governed r9 model."
            ),
        },
    }

    equivalence_confirmed = (
        pearson is not None
        and spearman is not None
        and pearson >= 0.95
        and spearman >= 0.95
        and audit["equivalence_metrics"]["threshold_agreement_rate"] == 1.0
        and len(dangerous_runtime) == len(dangerous_offline)
    )
    recommendation = {
        "experiment_name": "r29_runtime_score_recommendation",
        "classification": "runtime_scorer_should_load_exact_governed_offline_model",
        "blocked_by": ["model_identity_mismatch", "feature_vector_mismatch", "score_calibration_mismatch"],
        "options_assessment": {
            "runtime_close_enough_needs_calibration": False,
            "load_exact_governed_offline_model": True,
            "runtime_too_approximate_should_not_drive_live_approval": True,
        },
        "dangerous_r28_row_offline_status": {
            "found": bool(r28_dangerous),
            "runtime_dangerous": bool(r28_dangerous and r28_dangerous[0]["runtime_approved_v1"] and r28_dangerous[0]["label"] != "platform_dive"),
            "offline_dangerous": bool(r28_dangerous and r28_dangerous[0]["offline_approved_v1"] and r28_dangerous[0]["label"] != "platform_dive"),
            "row": compact(r28_dangerous[0]) if r28_dangerous else None,
        },
        "required_next_step": (
            "Replace the runtime bootstrapped governed_r9_score proxy with exact governed r9 model loading and exact feature extraction, "
            "then rerun r28/r29 before using runtime scores for live approval decisions."
        ),
        "final_decisions": [
            "R29_RUNTIME_OFFLINE_EQUIVALENCE_CONFIRMED" if equivalence_confirmed else "R29_RUNTIME_OFFLINE_EQUIVALENCE_NOT_CONFIRMED",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }

    OUT_ALIGNMENT_JSON.write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    OUT_AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    OUT_RECOMMENDATION_JSON.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    summary_rows = [
        {
            "metric": "matched rows",
            "value": len(joined),
        },
        {"metric": "pearson", "value": pearson},
        {"metric": "spearman", "value": spearman},
        {"metric": "threshold agreement", "value": audit["equivalence_metrics"]["threshold_agreement_rate"]},
        {"metric": "runtime approved", "value": len(runtime_approved)},
        {"metric": "offline approved", "value": len(offline_approved)},
        {"metric": "approve overlap", "value": len(overlap)},
        {"metric": "runtime dangerous", "value": len(dangerous_runtime)},
        {"metric": "offline dangerous", "value": len(dangerous_offline)},
    ]

    OUT_ALIGNMENT_MD.write_text(
        "# R29 Runtime/Offline Score Alignment\n\n"
        f"- Runtime rows: `{len(runtime_rows)}`\n"
        f"- Matched rows: `{len(joined)}`\n"
        f"- Unmatched rows: `{len(unmatched)}`\n"
        f"- Labels: `{json.dumps(alignment['label_counts'], sort_keys=True)}`\n"
        f"- Sources: `{json.dumps(alignment['source_counts'], sort_keys=True)}`\n\n"
        "## Dangerous R28 Row\n\n"
        f"```json\n{json.dumps(alignment['dangerous_r28_row'], indent=2)}\n```\n",
        encoding="utf-8",
    )
    OUT_AUDIT_MD.write_text(
        "# R29 Runtime/Offline Equivalence Audit\n\n"
        "## Metrics\n\n"
        + table(summary_rows, ["metric", "value"])
        + "\n\n## Threshold Disagreements\n\n"
        + f"```json\n{json.dumps(audit['disagreement_rows'][:20], indent=2)}\n```\n\n"
        "## Root Cause\n\n"
        "- Runtime uses a bootstrapped logistic proxy model.\n"
        "- Offline governed reference uses the source-weighted r9 XGBoost compact nuisance model.\n"
        "- Feature vectors, scaling, model family, and training source are not identical.\n",
        encoding="utf-8",
    )
    OUT_RECOMMENDATION_MD.write_text(
        "# R29 Runtime Score Recommendation\n\n"
        f"- Classification: `{recommendation['classification']}`\n"
        f"- Blocked by: `{', '.join(recommendation['blocked_by'])}`\n"
        f"- Dangerous r28 row offline dangerous: `{recommendation['dangerous_r28_row_offline_status']['offline_dangerous']}`\n\n"
        "## Recommendation\n\n"
        f"{recommendation['required_next_step']}\n\n"
        "## Decisions\n\n"
        + "\n".join(f"- `{item}`" for item in recommendation["final_decisions"])
        + "\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(
        "# R29 Runtime/Offline Score Equivalence\n\n"
        "R29 compared repaired runtime `governed_r9_score` values against the governed offline r9 reference on rows replayed into r27 runtime sessions.\n\n"
        f"- Matched rows: `{len(joined)}`\n"
        f"- Pearson correlation: `{pearson}`\n"
        f"- Spearman correlation: `{spearman}`\n"
        f"- Threshold agreement: `{audit['equivalence_metrics']['threshold_agreement_rate']}`\n"
        f"- Runtime dangerous approvals: `{len(dangerous_runtime)}`\n"
        f"- Offline dangerous approvals: `{len(dangerous_offline)}`\n\n"
        "The runtime scorer is not equivalent to the governed offline reference. It should be replaced with exact governed model loading and exact feature extraction before any live approve decision uses widened runtime scoring.\n\n"
        "Decisions:\n\n"
        + "\n".join(f"- `{item}`" for item in recommendation["final_decisions"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "matched_rows": len(joined),
        "pearson": pearson,
        "spearman": spearman,
        "threshold_agreement": audit["equivalence_metrics"]["threshold_agreement_rate"],
        "runtime_approved": len(runtime_approved),
        "offline_approved": len(offline_approved),
        "runtime_dangerous": len(dangerous_runtime),
        "offline_dangerous": len(dangerous_offline),
        "final_decisions": recommendation["final_decisions"],
    }, indent=2))


if __name__ == "__main__":
    main()
