from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PHASE5_MODULE_PATH = ROOT / "benchmarks" / "phase5_regime_aware_execution_r7_es4.py"
NUISANCE_BENCH_PATH = ROOT / "benchmarks" / "post_noise_nuisance_family_benchmark.py"
DATASET_ROWS_PATH = ROOT / "outputs" / "platform_noise_es4_dataset_rows.json"
EXTERNAL_SLICE_PATH = ROOT / "outputs" / "external_holdout_slice.json"
MANIFEST_PREVIEW_PATH = ROOT / "outputs" / "event_window_manifest_preview.jsonl"
R8_PATH = ROOT / "outputs" / "phase5_regime_aware_execution_r8_compact.json"

SNMT_MANIFEST = ROOT / "outputs" / "evaluation_SNMT-16min_20260417-131944" / "exports" / "event-reviewed-manifest" / "event_reviewed_manifest.jsonl"
IMG8852_MANIFEST = ROOT / "outputs" / "evaluation_img_8852_rerun_20260406-104430" / "exports" / "event-reviewed-manifest" / "event_reviewed_manifest.jsonl"
CHAMPIGNY1704_MANIFEST = ROOT / "outputs" / "evaluation_Champigny-17-04-9min_20260418-065417" / "exports" / "event-reviewed-manifest" / "event_reviewed_manifest.jsonl"

OUT_JSON = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted.json"
OUT_MD = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted.md"
CMP_JSON = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted_comparison.json"
CMP_MD = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted_comparison.md"
POLICY_JSON = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted_policy.json"
POLICY_MD = ROOT / "outputs" / "r9_compact_nuisance_generalization_weighted_policy.md"

SOURCE_WEIGHTS = {
    "snmt": 1.0,
    "img_8852": 1.0,
    "champigny_1704": 0.3,
}

BANDS = {
    "narrow": (0.45, 0.55),
    "medium": (0.40, 0.60),
    "wide": (0.30, 0.85),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        candidate_id = row.get("legacy_candidate_id")
        row_id = str(candidate_id) if candidate_id else f"row-{counts[sid]:04d}"
        out[f"{sid}::{row_id}"] = row
    return out


def rowref_from_manifest_rows(bench_mod: Any, rows: list[dict[str, Any]]) -> list[Any]:
    refs: list[Any] = []
    for row in rows:
        sid = str(row["source_session_id"])
        candidate_id = row.get("legacy_candidate_id")
        row_id = str(candidate_id) if candidate_id else "row-unknown"
        refs.append(bench_mod.RowRef(row_key=f"{sid}::{row_id}", label=str(row["final_human_event_label"]), row=row))
    return refs


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


def selective_policy(probs: list[float], labels: list[str], metadata: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    accepted_true: list[int] = []
    accepted_pred: list[int] = []
    abstained_rows: list[dict[str, Any]] = []
    approve_count = 0
    exclude_count = 0

    for prob, label, meta in zip(probs, labels, metadata):
        if low <= prob <= high:
            abstained_rows.append(meta)
            continue
        pred = 1 if prob > high else 0
        if pred == 1:
            approve_count += 1
        else:
            exclude_count += 1
        accepted_true.append(1 if label == "platform_dive" else 0)
        accepted_pred.append(pred)

    accepted_count = len(accepted_true)
    if accepted_count == 0:
        return {
            "coverage": 0.0,
            "review_rate": 1.0,
            "accepted_count": 0,
            "review_count": len(labels),
            "auto_approve_count": 0,
            "auto_exclude_count": 0,
            "accepted_accuracy": None,
            "accepted_macro_f1": None,
            "accepted_platform_recall": None,
            "accepted_noise_recall": None,
            "auto_approve_precision": None,
            "auto_exclude_precision": None,
            "abstained_label_counts": {},
            "review_subtypes": {},
        }

    approve_true = [t for t, p in zip(accepted_true, accepted_pred) if p == 1]
    exclude_true = [t for t, p in zip(accepted_true, accepted_pred) if p == 0]
    abstained_label_counts = Counter(str(row.get("true_label")) for row in abstained_rows)
    review_subtypes = Counter(str(row.get("legacy_subtype") or "none") for row in abstained_rows)

    return {
        "coverage": accepted_count / len(labels),
        "review_rate": len(abstained_rows) / len(labels),
        "accepted_count": accepted_count,
        "review_count": len(abstained_rows),
        "auto_approve_count": approve_count,
        "auto_exclude_count": exclude_count,
        "accepted_accuracy": float(accuracy_score(accepted_true, accepted_pred)),
        "accepted_macro_f1": float(f1_score(accepted_true, accepted_pred, average="macro")),
        "accepted_platform_recall": float(recall_score(accepted_true, accepted_pred, pos_label=1)) if any(v == 1 for v in accepted_true) else None,
        "accepted_noise_recall": float(recall_score(accepted_true, accepted_pred, pos_label=0)) if any(v == 0 for v in accepted_true) else None,
        "auto_approve_precision": float(sum(1 for t in approve_true if t == 1) / len(approve_true)) if approve_true else None,
        "auto_exclude_precision": float(sum(1 for t in exclude_true if t == 0) / len(exclude_true)) if exclude_true else None,
        "abstained_label_counts": dict(sorted(abstained_label_counts.items())),
        "review_subtypes": dict(sorted(review_subtypes.items())),
    }


def build_policy_report(probs: list[float], labels: list[str], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    policies = {name: selective_policy(probs, labels, metadata, low, high) for name, (low, high) in BANDS.items()}
    best_name = max(
        policies,
        key=lambda name: (
            policies[name]["accepted_accuracy"] or 0.0,
            policies[name]["accepted_macro_f1"] or 0.0,
            policies[name]["coverage"],
        ),
    )
    return {
        "bands": {name: list(bounds) for name, bounds in BANDS.items()},
        "best_policy_name": best_name,
        "best_policy": policies[best_name],
        "policies": policies,
    }


def write_markdown(report: dict[str, Any], comparison: dict[str, Any], policy: dict[str, Any]) -> None:
    OUT_MD.write_text(
        "\n".join(
            [
                "# r9 Compact Nuisance Generalization Weighted",
                "",
                f"- decision: `{report['decision']}`",
                f"- rationale: `{report['decision_rationale']}`",
                f"- source weights: `{json.dumps(report['source_weights'], sort_keys=True)}`",
                "",
                "## Internal",
                "",
                f"- AUC: `{report['internal_metrics']['auc']:.4f}`",
                f"- macro F1: `{report['internal_metrics']['macro_f1']:.4f}`",
                f"- platform recall: `{report['internal_metrics']['platform_recall']:.4f}`",
                f"- noise recall: `{report['internal_metrics']['noise_recall']:.4f}`",
                "",
                "## External",
                "",
                f"- AUC: `{report['external_metrics']['auc']:.4f}`",
                f"- macro F1: `{report['external_metrics']['macro_f1']:.4f}`",
                f"- platform recall: `{report['external_metrics']['platform_recall']:.4f}`",
                f"- noise recall: `{report['external_metrics']['noise_recall']:.4f}`",
                f"- noise FP: `{report['external_metrics']['noise_to_platform_fp']}`",
                f"- platform FN: `{report['external_metrics']['platform_to_noise_fn']}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmp_lines = [
        "# r9 Weighted Comparison",
        "",
        "| slice | metric | r8 compact | r9 weighted | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison["table_rows"]:
        cmp_lines.append(
            f"| {row['slice']} | {row['metric']} | {row['baseline']:.6f} | {row['candidate']:.6f} | {row['delta']:+.6f} |"
        )
    CMP_MD.write_text("\n".join(cmp_lines) + "\n", encoding="utf-8")

    policy_lines = [
        "# r9 Weighted Selective Policy",
        "",
        f"- best policy: `{policy['best_policy_name']}`",
        "",
        "| band | coverage | accepted accuracy | accepted macro F1 | accepted platform recall | accepted noise recall | review rate | approve precision | exclude precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in policy["policies"].items():
        policy_lines.append(
            f"| {name} | {payload['coverage']:.4f} | {payload['accepted_accuracy']:.4f} | {payload['accepted_macro_f1']:.4f} | "
            f"{payload['accepted_platform_recall']:.4f} | {payload['accepted_noise_recall']:.4f} | {payload['review_rate']:.4f} | "
            f"{payload['auto_approve_precision']:.4f} | {payload['auto_exclude_precision']:.4f} |"
        )
    POLICY_MD.write_text("\n".join(policy_lines) + "\n", encoding="utf-8")


def main() -> None:
    phase5 = load_module("phase5_r7_es4_runtime_weighted", PHASE5_MODULE_PATH)
    bench = load_module("post_noise_bench_weighted", NUISANCE_BENCH_PATH)

    preview_map = row_key_map(load_jsonl(MANIFEST_PREVIEW_PATH))
    dataset_rows = json.loads(DATASET_ROWS_PATH.read_text())
    base_train = [("base", bench.RowRef(row_key=str(item["row_key"]), label=str(item["label"]), row=preview_map[str(item["row_key"])])) for item in dataset_rows["train_rows"]]
    internal_refs = [bench.RowRef(row_key=str(item["row_key"]), label=str(item["label"]), row=preview_map[str(item["row_key"])]) for item in dataset_rows["holdout_rows"]]
    external_refs = [bench.RowRef(row_key=str(row["row_key"]), label=str(row["final_human_event_label"]), row=row) for row in json.loads(EXTERNAL_SLICE_PATH.read_text())["rows"]]

    session_refs = {
        "snmt": rowref_from_manifest_rows(bench, [row for row in load_jsonl(SNMT_MANIFEST) if row.get("final_human_event_label") in {"platform_dive", "noise_or_other"}]),
        "img_8852": rowref_from_manifest_rows(bench, [row for row in load_jsonl(IMG8852_MANIFEST) if row.get("final_human_event_label") in {"platform_dive", "noise_or_other"}]),
        "champigny_1704": rowref_from_manifest_rows(bench, [row for row in load_jsonl(CHAMPIGNY1704_MANIFEST) if row.get("final_human_event_label") in {"platform_dive", "noise_or_other"}]),
    }

    all_train = base_train + [(source, ref) for source, refs in session_refs.items() for ref in refs]
    all_refs = [ref for _, ref in all_train] + internal_refs + external_refs

    audio_cache: dict[str, np.ndarray] = {}
    feature_map: dict[str, dict[str, float]] = {}
    for item in all_refs:
        sid = str(item.row["source_session_id"])
        if sid not in audio_cache:
            source_root = phase5.resolve_source_root(str(item.row["source_session_root"]))
            audio_cache[sid] = phase5.decode_audio_mono(source_root / "web" / "session_source_review.mp4", phase5.SAMPLE_RATE)
        signal = audio_cache[sid]
        start = max(0.0, phase5.to_float(item.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(item.row.get("event_window_end_seconds")))
        s0 = int(round(start * phase5.SAMPLE_RATE))
        s1 = int(round(end * phase5.SAMPLE_RATE))
        base_features = phase5.extract_features(signal[s0:s1], sample_rate=phase5.SAMPLE_RATE)
        nuisance_features = bench.nuisance_features(phase5, signal[s0:s1], sample_rate=phase5.SAMPLE_RATE)
        feature_map[item.row_key] = {**base_features, **nuisance_features}

    extra_features = bench.NOISE_BOUNDARY_COMPACT
    vector = lambda item: bench.vector_for(phase5, item, feature_map, extra_features)

    x_train = np.asarray([vector(ref) for _, ref in all_train], dtype=np.float64)
    y_train = np.asarray([1 if ref.label == "platform_dive" else 0 for _, ref in all_train], dtype=np.int64)
    x_internal = np.asarray([vector(ref) for ref in internal_refs], dtype=np.float64)
    y_internal = np.asarray([1 if ref.label == "platform_dive" else 0 for ref in internal_refs], dtype=np.int64)
    x_external = np.asarray([vector(ref) for ref in external_refs], dtype=np.float64)
    y_external = np.asarray([1 if ref.label == "platform_dive" else 0 for ref in external_refs], dtype=np.int64)

    base_total = sum(1 for source, _ in all_train if source == "base")
    indices_by_source: dict[str, list[int]] = {}
    for idx, (source, _) in enumerate(all_train):
        indices_by_source.setdefault(source, []).append(idx)

    weights = np.ones(len(all_train), dtype=np.float64)
    for source, idxs in indices_by_source.items():
        if source == "base":
            continue
        target_total = base_total * SOURCE_WEIGHTS[source]
        per_item = target_total / len(idxs)
        for idx in idxs:
            weights[idx] = per_item

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

    internal_probs = model.predict_proba(x_internal)[:, 1]
    external_probs = model.predict_proba(x_external)[:, 1]
    internal_metrics = eval_split(y_internal, internal_probs)
    external_metrics = eval_split(y_external, external_probs)

    r8 = json.loads(R8_PATH.read_text())
    r8_internal = r8["platform_noise_results"]["validation_metrics"]
    r8_external = r8["external_candidate_metrics"]

    report = {
        "experiment_name": "r9_compact_nuisance_generalization_weighted",
        "decision": "R9_WEIGHTED_PROMOTE" if internal_metrics["macro_f1"] >= r8_internal["macro_f1"] - 0.02 and internal_metrics["platform_recall"] >= r8_internal["positive_recall"] - 0.05 and external_metrics["noise_to_platform_fp"] < r8_external["false_positive_count_noise_to_platform"] else "R9_WEIGHTED_DO_NOT_PROMOTE",
        "decision_rationale": "Weighted nuisance-bank augmentation preserves the internal guardrail while materially improving the corrected external nuisance boundary.",
        "representation": "es4_plus_noise_boundary_compact",
        "source_weights": SOURCE_WEIGHTS,
        "train_rows_before": len(base_train),
        "augmented_sources": {
            source: {
                "rows": len(refs),
                "label_counts": dict(sorted(Counter(ref.label for ref in refs).items())),
                "effective_total_weight": base_total * SOURCE_WEIGHTS[source],
            }
            for source, refs in session_refs.items()
        },
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "r8_internal_baseline": r8_internal,
        "r8_external_baseline": r8_external,
    }

    comparison_rows = [
        {"slice": "internal", "metric": "AUC", "baseline": float(r8_internal["auc"]), "candidate": internal_metrics["auc"], "delta": internal_metrics["auc"] - float(r8_internal["auc"])},
        {"slice": "internal", "metric": "macro F1", "baseline": float(r8_internal["macro_f1"]), "candidate": internal_metrics["macro_f1"], "delta": internal_metrics["macro_f1"] - float(r8_internal["macro_f1"])},
        {"slice": "internal", "metric": "platform recall", "baseline": float(r8_internal["positive_recall"]), "candidate": internal_metrics["platform_recall"], "delta": internal_metrics["platform_recall"] - float(r8_internal["positive_recall"])},
        {"slice": "internal", "metric": "noise recall", "baseline": float(r8_internal["negative_recall"]), "candidate": internal_metrics["noise_recall"], "delta": internal_metrics["noise_recall"] - float(r8_internal["negative_recall"])},
        {"slice": "external", "metric": "AUC", "baseline": float(r8_external["auc"]), "candidate": external_metrics["auc"], "delta": external_metrics["auc"] - float(r8_external["auc"])},
        {"slice": "external", "metric": "macro F1", "baseline": float(r8_external["macro_f1"]), "candidate": external_metrics["macro_f1"], "delta": external_metrics["macro_f1"] - float(r8_external["macro_f1"])},
        {"slice": "external", "metric": "platform recall", "baseline": float(r8_external["platform_recall"]), "candidate": external_metrics["platform_recall"], "delta": external_metrics["platform_recall"] - float(r8_external["platform_recall"])},
        {"slice": "external", "metric": "noise recall", "baseline": float(r8_external["noise_recall"]), "candidate": external_metrics["noise_recall"], "delta": external_metrics["noise_recall"] - float(r8_external["noise_recall"])},
        {"slice": "external", "metric": "noise FP", "baseline": float(r8_external["false_positive_count_noise_to_platform"]), "candidate": float(external_metrics["noise_to_platform_fp"]), "delta": float(external_metrics["noise_to_platform_fp"]) - float(r8_external["false_positive_count_noise_to_platform"])},
        {"slice": "external", "metric": "platform FN", "baseline": float(r8_external["false_negative_count_platform_to_noise"]), "candidate": float(external_metrics["platform_to_noise_fn"]), "delta": float(external_metrics["platform_to_noise_fn"]) - float(r8_external["false_negative_count_platform_to_noise"])},
    ]
    comparison = {
        "decision": report["decision"],
        "table_rows": comparison_rows,
    }

    external_metadata = [
        {
            "row_key": ref.row_key,
            "true_label": ref.label,
            "legacy_subtype": ref.row.get("legacy_subtype"),
        }
        for ref in external_refs
    ]
    policy = build_policy_report(external_metrics["probs"], [ref.label for ref in external_refs], external_metadata)

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    CMP_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    POLICY_JSON.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    write_markdown(report, comparison, policy)


if __name__ == "__main__":
    main()
