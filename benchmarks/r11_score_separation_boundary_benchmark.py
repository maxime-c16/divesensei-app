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
OUT_DATASET_JSON = ROOT / "outputs/r11_score_separation_boundary_dataset.json"
OUT_DATASET_MD = ROOT / "outputs/r11_score_separation_boundary_dataset.md"
OUT_BENCH_JSON = ROOT / "outputs/r11_score_separation_boundary_benchmark.json"
OUT_BENCH_MD = ROOT / "outputs/r11_score_separation_boundary_benchmark.md"
OUT_POLICY_JSON = ROOT / "outputs/r11_score_separation_boundary_policy.json"
OUT_POLICY_MD = ROOT / "outputs/r11_score_separation_boundary_policy.md"
COMPACT = ["dominant_frequency_hz_post_std", "spectral_rolloff_90_post_mean", "zero_crossing_rate_post_mean"]
BOUNDARY = [
    "r9_score",
    "spectral_rolloff_90_post_mean",
    "dominant_frequency_hz_post_std",
    "whistle_band_energy_fraction_post",
    "spectral_contrast_low_high_slope_post",
    "impact_peak_prominence_db",
    "impact_peak_to_window_rms_ratio",
    "post_impact_early_to_late_rms_ratio",
    "audio_score",
    "spectral_flatness_post_mean",
    "zero_crossing_rate_post_mean",
    "spectral_entropy_post_mean",
]


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
    return [float(row.get(name, 0.0) or 0.0) for name in names]


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
        fmap[item.row_key] = {**phase5.extract_features(sig, phase5.SAMPLE_RATE), **bench.nuisance_features(phase5, sig, phase5.SAMPLE_RATE)}

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
            for name in set(phase5.ALL_FEATURE_NAMES + bench.NOISE_BOUNDARY_COMPACT + BOUNDARY):
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
    OUT_DATASET_JSON.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    compact_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
    boundary_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
    compact_model.fit(np.asarray([vec(r, ["r9_score"] + COMPACT) for r in train_rows], dtype=np.float64), [label_int(r["label"]) for r in train_rows])
    boundary_model.fit(np.asarray([vec(r, BOUNDARY) for r in train_rows], dtype=np.float64), [label_int(r["label"]) for r in train_rows])

    def scores(rows: list[dict[str, Any]], cand: str) -> list[float]:
        if cand == "r9_reference":
            return [float(r["r9_score"]) for r in rows]
        if cand == "compact_nuisance_verifier":
            return compact_model.predict_proba(np.asarray([vec(r, ["r9_score"] + COMPACT) for r in rows], dtype=np.float64))[:, 1].tolist()
        return boundary_model.predict_proba(np.asarray([vec(r, BOUNDARY) for r in rows], dtype=np.float64))[:, 1].tolist()

    def queues(rows: list[dict[str, Any]], cand: str, sc: list[float]) -> list[str]:
        out = []
        for row, score in zip(rows, sc):
            r9 = float(row["r9_score"])
            if cand == "r9_reference":
                out.append("auto_approved" if r9 >= HIGH else ("auto_excluded" if r9 <= LOW else "needs_review"))
            elif cand == "compact_nuisance_verifier":
                out.append("auto_approved" if r9 >= HIGH and score >= 0.50 else ("auto_excluded" if r9 <= 0.20 and score <= 0.30 else "needs_review"))
            elif cand == "boundary_logistic_score":
                out.append("auto_approved" if score >= HIGH else ("auto_excluded" if score <= LOW else "needs_review"))
            elif cand == "approve_safety_verifier":
                out.append("auto_approved" if r9 >= HIGH and score >= 0.70 else ("auto_excluded" if r9 <= LOW else "needs_review"))
            elif cand == "reject_option_boundary_scorer":
                out.append("auto_approved" if r9 >= HIGH and score >= 0.70 else ("auto_excluded" if r9 <= 0.20 and score <= 0.30 else "needs_review"))
        return out

    candidates = [
        ("r9_reference", "Promoted r9 weighted reference with r10 thresholds."),
        ("compact_nuisance_verifier", "Second-stage logistic verifier using r9 score plus compact nuisance features."),
        ("boundary_logistic_score", "Boundary-specific logistic score with fixed 0.05/0.85 triage thresholds."),
        ("approve_safety_verifier", "Conservative auto-approve verifier layered on r9."),
        ("reject_option_boundary_scorer", "Dual verifier: conservative auto-approve and expanded conservative auto-exclude."),
    ]
    reports, table = [], []
    for name, desc in candidates:
        int_scores, ext_scores = scores(internal_rows, name), scores(external_rows, name)
        int_triage, ext_triage = triage(internal_rows, queues(internal_rows, name, int_scores)), triage(external_rows, queues(external_rows, name, ext_scores))
        int_forced, ext_forced = forced_metrics([r["label"] for r in internal_rows], int_scores), forced_metrics([r["label"] for r in external_rows], ext_scores)
        viable = fnum(ext_triage["auto_approve_precision"]) >= PRECISION_TARGET and fnum(ext_triage["auto_exclude_precision"]) >= PRECISION_TARGET and ext_triage["coverage"] >= COVERAGE_TARGET and int_triage["auto_approve_error_count"] == 0
        row = {
            "candidate": name,
            "internal_macro_f1": int_forced["macro_f1"],
            "external_macro_f1": ext_forced["macro_f1"],
            "external_auto_approve_precision": ext_triage["auto_approve_precision"],
            "external_auto_exclude_precision": ext_triage["auto_exclude_precision"],
            "external_coverage": ext_triage["coverage"],
            "external_review_required_count": ext_triage["review_required_count"],
            "internal_auto_approve_precision": int_triage["auto_approve_precision"],
            "internal_auto_approve_error_count": int_triage["auto_approve_error_count"],
            "product_viable": viable,
        }
        table.append(row)
        reports.append({"name": name, "description": desc, "forced_classification": {"internal": int_forced, "external": ext_forced}, "triage_policy": {"internal": int_triage, "external": ext_triage}, "product_viable": viable})
    ref = next(r for r in table if r["candidate"] == "r9_reference")
    best = max([r for r in table if r["candidate"] != "r9_reference"], key=lambda r: (r["product_viable"], fnum(r["external_auto_approve_precision"]) >= PRECISION_TARGET, fnum(r["external_auto_exclude_precision"]) >= PRECISION_TARGET, r["external_coverage"], -r["internal_auto_approve_error_count"], r["external_macro_f1"]))
    improves = (
        fnum(best["external_auto_approve_precision"]) >= fnum(ref["external_auto_approve_precision"])
        and fnum(best["external_auto_exclude_precision"]) >= fnum(ref["external_auto_exclude_precision"])
        and best["internal_auto_approve_error_count"] <= ref["internal_auto_approve_error_count"]
        and (
            best["external_coverage"] > ref["external_coverage"]
            or best["internal_auto_approve_error_count"] < ref["internal_auto_approve_error_count"]
        )
    )
    decision = "R11_BOUNDARY_IMPROVES_TRIAGE" if improves else "R11_BOUNDARY_NO_CLEAR_GAIN"
    benchmark = {"experiment_name": "r11_score_separation_boundary_benchmark", "final_decision": decision, "best_candidate": best["candidate"], "three_queue_coach_mode_viable": bool(best["product_viable"]), "comparison_rows": table, "candidate_reports": reports, "r9_reference_forced": {"internal": r9_reference["internal_metrics"], "external": r9_reference["external_metrics"]}}
    policy = {"experiment_name": "r11_score_separation_boundary_policy", "final_decision": decision, "recommended_candidate": best["candidate"], "three_queue_coach_mode_viable": bool(best["product_viable"]), "recommended_candidate_metrics": next(r for r in reports if r["name"] == best["candidate"])}
    OUT_BENCH_JSON.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    OUT_POLICY_JSON.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    OUT_DATASET_MD.write_text("\n".join([
        "# r11 Score-Separation Boundary Dataset", "",
        f"- row count: `{dataset['row_count']}`",
        f"- label counts: `{json.dumps(dataset['label_counts'], sort_keys=True)}`",
        f"- subtype counts: `{json.dumps(dataset['subtype_counts'], sort_keys=True)}`",
        f"- score-band counts: `{json.dumps(dataset['score_band_counts'], sort_keys=True)}`",
        f"- boundary-role counts: `{json.dumps(dataset['boundary_role_counts'], sort_keys=True)}`",
        f"- approve-risk rows: `{len(dataset['approve_risk_rows'])}`",
        f"- exclude-risk rows: `{len(dataset['exclude_risk_rows'])}`",
    ]) + "\n", encoding="utf-8")
    bench_lines = ["# r11 Score-Separation Boundary Benchmark", "", f"- final decision: `{decision}`", f"- best candidate: `{best['candidate']}`", f"- three-queue coach mode viable: `{bool(best['product_viable'])}`", "", "| candidate | internal macro F1 | external macro F1 | external approve precision | external exclude precision | external coverage | external review rows | internal approve precision | internal approve errors | product viable |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in table:
        bench_lines.append(f"| `{r['candidate']}` | {r['internal_macro_f1']:.4f} | {r['external_macro_f1']:.4f} | {fnum(r['external_auto_approve_precision']):.4f} | {fnum(r['external_auto_exclude_precision']):.4f} | {r['external_coverage']:.4f} | {r['external_review_required_count']} | {fnum(r['internal_auto_approve_precision']):.4f} | {r['internal_auto_approve_error_count']} | `{r['product_viable']}` |")
    OUT_BENCH_MD.write_text("\n".join(bench_lines) + "\n", encoding="utf-8")
    OUT_POLICY_MD.write_text("\n".join(["# r11 Score-Separation Boundary Policy", "", f"- final decision: `{decision}`", f"- recommended candidate: `{best['candidate']}`", f"- three-queue coach mode viable: `{bool(best['product_viable'])}`", "", "The bounded second-stage benchmark did not alter detector behavior, taxonomy, or the promoted r9 reference."]) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_DATASET_JSON), str(OUT_DATASET_MD), str(OUT_BENCH_JSON), str(OUT_BENCH_MD), str(OUT_POLICY_JSON), str(OUT_POLICY_MD)], "final_decision": decision, "best_candidate": best["candidate"], "three_queue_coach_mode_viable": bool(best["product_viable"])}, indent=2))


if __name__ == "__main__":
    main()
