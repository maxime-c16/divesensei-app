from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
PREVIEW = ROOT / "outputs/event_window_manifest_preview.jsonl"
DATASET = ROOT / "outputs/platform_noise_es4_dataset_rows.json"
EXTERNAL = ROOT / "outputs/external_holdout_slice.json"
MODEL_DIR = ROOT / ".divesensei-runtime/models/r9_compact_nuisance_weighted"
MODEL_PATH = MODEL_DIR / "xgboost_model.json"
CONTRACT_PATH = MODEL_DIR / "contract.json"
V1_THRESHOLD = 0.92158
GOVERNED_PLATFORM_NOISE_WINDOW_PRE_SECONDS = 0.75
GOVERNED_PLATFORM_NOISE_WINDOW_POST_SECONDS = 2.25

SOURCE_MANIFESTS = {
    "snmt": ROOT / "outputs/evaluation_SNMT-16min_20260417-131944/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "img_8852": ROOT / "outputs/evaluation_img_8852_rerun_20260406-104430/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
    "champigny_1704": ROOT / "outputs/evaluation_Champigny-17-04-9min_20260418-065417/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl",
}
SOURCE_WEIGHTS = {"snmt": 1.0, "img_8852": 1.0, "champigny_1704": 0.3}

OUT_CONTRACT_JSON = ROOT / "outputs/r30_governed_model_contract.json"
OUT_CONTRACT_MD = ROOT / "outputs/r30_governed_model_contract.md"
OUT_PARITY_JSON = ROOT / "outputs/r30_runtime_model_parity.json"
OUT_PARITY_MD = ROOT / "outputs/r30_runtime_model_parity.md"
OUT_RECHECK_JSON = ROOT / "outputs/r30_runtime_offline_recheck.json"
OUT_RECHECK_MD = ROOT / "outputs/r30_runtime_offline_recheck.md"
OUT_DOC = ROOT / "docs/research/R30_EXACT_GOVERNED_RUNTIME_MODEL_PARITY.md"


@dataclass(frozen=True)
class RowRef:
    row_key: str
    label: str
    row: dict[str, Any]


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


def row_key_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        rid = str(row.get("legacy_candidate_id") or f"row-{counts[sid]:04d}")
        out[f"{sid}::{rid}"] = row
    return out


def manifest_refs(rows: list[dict[str, Any]], preview: dict[str, dict[str, Any]] | None = None) -> list[RowRef]:
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
        refs.append(RowRef(key, str(label), merged))
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


def build_governed_training_contract() -> tuple[Any, Any, list[tuple[str, RowRef]], list[RowRef], dict[str, Any]]:
    phase5 = load_module("phase5_r30", PHASE5)
    nuisance = load_module("nuisance_r30", NUISANCE)
    preview = row_key_map(load_jsonl(PREVIEW))
    dataset = load_json(DATASET)
    base_train = [("base", RowRef(str(item["row_key"]), str(item["label"]), preview[str(item["row_key"])])) for item in dataset["train_rows"]]
    source_refs = {
        source: manifest_refs(load_jsonl(path), preview)
        for source, path in SOURCE_MANIFESTS.items()
        if path.exists()
    }
    train_refs = base_train + [(source, ref) for source, refs in source_refs.items() for ref in refs]
    train_items = [ref for _, ref in train_refs]
    return phase5, nuisance, train_refs, train_items, preview


def feature_map_for_refs(phase5: Any, nuisance: Any, refs: list[Any]) -> dict[str, dict[str, float]]:
    audio: dict[str, np.ndarray] = {}
    fmap: dict[str, dict[str, float]] = {}
    for ref in refs:
        sid = str(ref.row["source_session_id"])
        if sid not in audio:
            source_root = phase5.resolve_source_root(str(ref.row["source_session_root"]))
            audio[sid] = phase5.decode_audio_mono(source_root / "web/session_source_review.mp4", phase5.SAMPLE_RATE)
        start = max(0.0, phase5.to_float(ref.row.get("event_window_start_seconds")))
        end = max(start + 0.05, phase5.to_float(ref.row.get("event_window_end_seconds")))
        signal = audio[sid][int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap[ref.row_key] = {
            **phase5.extract_features(signal, phase5.SAMPLE_RATE),
            **nuisance.nuisance_features(phase5, signal, phase5.SAMPLE_RATE),
        }
    return fmap


def vector_for(phase5: Any, nuisance: Any, ref: Any, fmap: dict[str, dict[str, float]]) -> list[float]:
    return nuisance.vector_for(phase5, ref, fmap, nuisance.NOISE_BOUNDARY_COMPACT)


def export_model() -> dict[str, Any]:
    phase5, nuisance, train_refs, train_items, _ = build_governed_training_contract()
    fmap = feature_map_for_refs(phase5, nuisance, train_items)
    x_train = np.asarray([vector_for(phase5, nuisance, ref, fmap) for ref in train_items], dtype=np.float64)
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    feature_names = (
        phase5.BASELINE_FEATURE_NAMES
        + phase5.PROBE1_FEATURE_NAMES
        + phase5.R2_FEATURE_NAMES
        + phase5.R4_FEATURE_NAMES
        + nuisance.NOISE_BOUNDARY_COMPACT
    )
    contract = {
        "model_id": "r9_compact_nuisance_generalization_weighted",
        "model_family": "xgboost.XGBClassifier",
        "model_path": str(MODEL_PATH),
        "serialization_format": "xgboost_json",
        "feature_contract_sources": [str(PHASE5), str(NUISANCE)],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "base_feature_names": phase5.BASELINE_FEATURE_NAMES,
        "probe1_feature_names": phase5.PROBE1_FEATURE_NAMES,
        "r2_feature_names": phase5.R2_FEATURE_NAMES,
        "r4_feature_names": phase5.R4_FEATURE_NAMES,
        "compact_nuisance_feature_names": nuisance.NOISE_BOUNDARY_COMPACT,
        "preprocessing": "raw numeric feature vector; no sklearn scaler; XGBoost handles raw values",
        "window_contract_offline": "event_window_start_seconds to event_window_end_seconds from event-window manifest",
        "window_contract_runtime": "proposal timestamp with 0.75s pre + 2.25s post governed platform/noise event window",
        "training": {
            "train_rows": len(train_items),
            "positive_rows": int(np.sum(y_train)),
            "negative_rows": int(len(y_train) - np.sum(y_train)),
            "source_weights": SOURCE_WEIGHTS,
            "base_rows": sum(1 for source, _ in train_refs if source == "base"),
            "source_row_counts": dict(sorted(Counter(source for source, _ in train_refs).items())),
        },
        "inference_threshold": V1_THRESHOLD,
    }
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    OUT_CONTRACT_JSON.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    OUT_CONTRACT_MD.write_text(
        "# R30 Governed Model Contract\n\n"
        f"- Model id: `{contract['model_id']}`\n"
        f"- Model family: `{contract['model_family']}`\n"
        f"- Model artifact: `{MODEL_PATH}`\n"
        f"- Feature count: `{contract['feature_count']}`\n"
        f"- Runtime threshold: `{V1_THRESHOLD}`\n\n"
        "## Feature Ordering\n\n"
        + "\n".join(f"{idx + 1}. `{name}`" for idx, name in enumerate(feature_names))
        + "\n",
        encoding="utf-8",
    )
    return contract


def score_runtime_manifest_with_exact_contract(session_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase5 = load_module(f"phase5_r30_recheck_{session_root.name}", PHASE5)
    nuisance = load_module(f"nuisance_r30_recheck_{session_root.name}", NUISANCE)
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    manifest = load_json(session_root / "ui_session_manifest.json")
    review_path = session_root / "evaluation_review.json"
    review = load_json(review_path) if review_path.exists() else {"decisions": []}
    decisions = {
        str(item.get("detectionId")): item
        for item in review.get("decisions", [])
        if item.get("eventLabel") in {"platform_dive", "noise_or_other"}
    }
    detections = manifest.get("detections", [])
    source_video_path = Path(str(manifest.get("session", {}).get("source_video_path")))
    audio = phase5.decode_audio_mono(source_video_path, phase5.SAMPLE_RATE)
    rows = []
    x_rows = []
    for detection in detections:
        scores = dict(detection.get("scores", {}) or {})
        features = dict(detection.get("features", {}) or {})
        timestamp = phase5.to_float(detection.get("timestamp_seconds"))
        start = max(0.0, timestamp - GOVERNED_PLATFORM_NOISE_WINDOW_PRE_SECONDS)
        end = max(start + 0.05, timestamp + GOVERNED_PLATFORM_NOISE_WINDOW_POST_SECONDS)
        signal = audio[int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap = {"runtime": {**phase5.extract_features(signal, phase5.SAMPLE_RATE), **nuisance.nuisance_features(phase5, signal, phase5.SAMPLE_RATE)}}
        row = {
            "audio_score": scores.get("audio"),
            "audio_clip_probability": scores.get("audio_clip_probability"),
            "event_anchor_timestamp_seconds": detection.get("timestamp_seconds"),
            "is_false_negative_window": False,
        }
        ref = nuisance.RowRef("runtime", "unknown", row)
        x_rows.append(nuisance.vector_for(phase5, ref, fmap, nuisance.NOISE_BOUNDARY_COMPACT))
        rows.append(
            {
                "session_id": session_root.name,
                "detection_id": detection.get("id"),
                "label": (decisions.get(str(detection.get("id"))) or {}).get("eventLabel"),
                "runtime_governed_r9_score": scores.get("governed_r9_score"),
                "runtime_score_source": scores.get("governed_r9_score_source") or features.get("governed_r9_score_source"),
                "timestamp_seconds": detection.get("timestamp_seconds"),
            }
        )
    exact_scores = model.predict_proba(np.asarray(x_rows, dtype=np.float64))[:, 1] if x_rows else []
    for row, exact in zip(rows, exact_scores):
        row["offline_recomputed_same_window_score"] = float(exact)
        row["abs_delta"] = abs(float(row["runtime_governed_r9_score"] or 0.0) - float(exact))
        row["runtime_approved_v1"] = float(row["runtime_governed_r9_score"] or 0.0) >= V1_THRESHOLD
        row["offline_same_window_approved_v1"] = float(exact) >= V1_THRESHOLD
    meta = {
        "session_id": session_root.name,
        "detections": len(rows),
        "source_video_path": str(source_video_path),
        "runtime_score_source_counts": dict(sorted(Counter(str(row.get("runtime_score_source")) for row in rows).items())),
    }
    return rows, meta


def corr(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2:
        return None
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks.tolist()


def recheck(session_roots: list[Path]) -> dict[str, Any]:
    all_rows = []
    session_meta = []
    for root in session_roots:
        rows, meta = score_runtime_manifest_with_exact_contract(root)
        all_rows.extend(rows)
        session_meta.append(meta)
    runtime_scores = [float(row["runtime_governed_r9_score"] or 0.0) for row in all_rows]
    offline_scores = [float(row["offline_recomputed_same_window_score"]) for row in all_rows]
    threshold_agree = [row["runtime_approved_v1"] == row["offline_same_window_approved_v1"] for row in all_rows]
    runtime_approved = {f"{row['session_id']}::{row['detection_id']}" for row in all_rows if row["runtime_approved_v1"]}
    offline_approved = {f"{row['session_id']}::{row['detection_id']}" for row in all_rows if row["offline_same_window_approved_v1"]}
    union = runtime_approved | offline_approved
    dangerous_runtime = [row for row in all_rows if row.get("label") == "noise_or_other" and row["runtime_approved_v1"]]
    dangerous_offline = [row for row in all_rows if row.get("label") == "noise_or_other" and row["offline_same_window_approved_v1"]]
    formerly_dangerous = [
        row for row in all_rows
        if row["session_id"] == "evaluation_r30_exact_scorepath_insep_quick" and row["detection_id"] == "det-0039"
    ]
    report = {
        "experiment_name": "r30_runtime_offline_recheck",
        "session_meta": session_meta,
        "row_count": len(all_rows),
        "pearson": corr(runtime_scores, offline_scores),
        "spearman": corr(rankdata(runtime_scores), rankdata(offline_scores)),
        "mean_abs_delta": float(np.mean([row["abs_delta"] for row in all_rows])) if all_rows else None,
        "median_abs_delta": float(np.median([row["abs_delta"] for row in all_rows])) if all_rows else None,
        "max_abs_delta": float(np.max([row["abs_delta"] for row in all_rows])) if all_rows else None,
        "threshold_agreement_rate": float(np.mean(threshold_agree)) if threshold_agree else None,
        "runtime_approved_count": len(runtime_approved),
        "offline_approved_count": len(offline_approved),
        "approve_overlap_count": len(runtime_approved & offline_approved),
        "approve_jaccard": (len(runtime_approved & offline_approved) / len(union)) if union else None,
        "runtime_dangerous_approve_count": len(dangerous_runtime),
        "offline_same_window_dangerous_approve_count": len(dangerous_offline),
        "formerly_dangerous_r28_runtime_row": formerly_dangerous[0] if formerly_dangerous else None,
        "disagreement_rows": [row for row in all_rows if row["runtime_approved_v1"] != row["offline_same_window_approved_v1"]][:20],
    }
    OUT_RECHECK_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_RECHECK_MD.write_text(
        "# R30 Runtime/Offline Recheck\n\n"
        f"- Rows: `{report['row_count']}`\n"
        f"- Pearson: `{report['pearson']}`\n"
        f"- Spearman: `{report['spearman']}`\n"
        f"- Mean absolute delta: `{report['mean_abs_delta']}`\n"
        f"- Threshold agreement: `{report['threshold_agreement_rate']}`\n"
        f"- Approve Jaccard: `{report['approve_jaccard']}`\n\n"
        f"- Runtime dangerous approvals: `{report['runtime_dangerous_approve_count']}`\n"
        f"- Offline same-window dangerous approvals: `{report['offline_same_window_dangerous_approve_count']}`\n"
        f"- Formerly dangerous r28 row: `{json.dumps(report['formerly_dangerous_r28_runtime_row'], sort_keys=True)}`\n\n"
        "## Disagreements\n\n"
        f"```json\n{json.dumps(report['disagreement_rows'], indent=2)}\n```\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--recheck", nargs="*", default=[])
    args = parser.parse_args()
    contract = export_model() if args.export or not CONTRACT_PATH.exists() else load_json(CONTRACT_PATH)
    parity = {
        "experiment_name": "r30_runtime_model_parity",
        "model_artifact_created": MODEL_PATH.exists(),
        "contract_artifact_created": CONTRACT_PATH.exists(),
        "runtime_score_path": "src/divesensei/workflows/runtime_score_paths.py",
        "runtime_exact_model_path": str(MODEL_PATH),
        "runtime_contract_path": str(CONTRACT_PATH),
        "runtime_uses_exact_model_when_available": True,
        "fallback_behavior": "If exact model artifacts or xgboost/feature extraction are unavailable, runtime records fallback and uses bootstrapped proxy score.",
        "remaining_parity_risk": "springboard-specific event-window policy is outside this platform/noise scorer; platform/noise runtime scoring now uses the governed 0.75s pre + 2.25s post event window.",
        "contract": contract,
    }
    recheck_report = None
    if args.recheck:
        recheck_report = recheck([Path(item).resolve() for item in args.recheck])
        parity["recheck"] = recheck_report
    classification = "partial_parity_but_not_yet_sufficient"
    if recheck_report and recheck_report["mean_abs_delta"] is not None and recheck_report["mean_abs_delta"] < 1e-9 and recheck_report["threshold_agreement_rate"] == 1.0:
        classification = "exact_governed_runtime_parity_achieved"
    parity["classification"] = classification
    parity["final_decisions"] = ["R30_GOVERNED_RUNTIME_PARITY_PROGRESS", "APPROVE_REVIEW_V1_REMAINS_DEFAULT"]
    OUT_PARITY_JSON.write_text(json.dumps(parity, indent=2), encoding="utf-8")
    OUT_PARITY_MD.write_text(
        "# R30 Runtime Model Parity\n\n"
        f"- Exact model artifact: `{MODEL_PATH}`\n"
        f"- Contract artifact: `{CONTRACT_PATH}`\n"
        f"- Classification: `{classification}`\n"
        f"- Runtime exact path enabled: `{parity['runtime_uses_exact_model_when_available']}`\n"
        f"- Remaining risk: {parity['remaining_parity_risk']}\n\n"
        "## Decisions\n\n"
        + "\n".join(f"- `{item}`" for item in parity["final_decisions"])
        + "\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(
        "# R30 Exact Governed Runtime Model Parity\n\n"
        "R30 persisted the governed r9 compact nuisance weighted XGBoost model and wired runtime scoring to prefer this exact artifact over the prior bootstrapped proxy.\n\n"
        f"- Model artifact: `{MODEL_PATH}`\n"
        f"- Contract artifact: `{CONTRACT_PATH}`\n"
        f"- Classification: `{classification}`\n\n"
        "The remaining risk is window parity: live runtime scoring currently uses candidate windows, while the governed offline training contract used event-window manifest windows.\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "model_path": str(MODEL_PATH), "recheck": recheck_report, "final_decisions": parity["final_decisions"]}, indent=2))


if __name__ == "__main__":
    main()
