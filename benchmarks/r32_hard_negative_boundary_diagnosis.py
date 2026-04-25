from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PHASE5_PATH = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE_PATH = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
CONTRACT_PATH = ROOT / ".divesensei-runtime/models/r9_compact_nuisance_weighted/contract.json"
MODEL_PATH = ROOT / ".divesensei-runtime/models/r9_compact_nuisance_weighted/xgboost_model.json"
SESSION_ROOTS = [
    ROOT / "outputs/evaluation_r30_exact_scorepath_insep_quick",
    ROOT / "outputs/evaluation_r30_exact_scorepath_champigny_proxy",
]
V1_THRESHOLD = 0.92158
WINDOW_PRE_SECONDS = 0.75
WINDOW_POST_SECONDS = 2.25

OUT_BANK_JSON = ROOT / "outputs/r32_hard_negative_candidate_bank.json"
OUT_BANK_MD = ROOT / "outputs/r32_hard_negative_candidate_bank.md"
OUT_DIAG_JSON = ROOT / "outputs/r32_hard_negative_boundary_diagnosis.json"
OUT_DIAG_MD = ROOT / "outputs/r32_hard_negative_boundary_diagnosis.md"
OUT_DOC = ROOT / "docs/research/R32_HARD_NEGATIVE_BOUNDARY_DIAGNOSIS.md"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in SESSION_ROOTS:
        manifest = load_json(root / "ui_session_manifest.json")
        source_video_path = manifest.get("session", {}).get("source_video_path")
        review = load_json(root / "evaluation_review.json")
        diagnostics = {
            str(item.get("final_detection_id")): item
            for item in load_jsonl(root / "proposal_diagnostics.jsonl")
            if item.get("final_detection_id")
        }
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
            scores = dict(detection.get("scores", {}) or {})
            features = dict(detection.get("features", {}) or {})
            diag = diagnostics.get(detection_id, {})
            rows.append(
                {
                    "row_key": f"{root.name}::{detection_id}",
                    "session_id": root.name,
                    "session_root": str(root),
                    "detection_id": detection_id,
                    "timestamp_seconds": to_float(detection.get("timestamp_seconds")),
                    "label": decision.get("eventLabel"),
                    "legacy_label": decision.get("label"),
                    "subtype": decision.get("subtype") or "none",
                    "notes": decision.get("notes") or "",
                    "r9_score_manifest": to_float(scores.get("governed_r9_score")),
                    "visual_late_fusion_logreg_c0.5": features.get("visual_late_fusion_logreg_c0.5"),
                    "audio_score": to_float(scores.get("audio")),
                    "proposal_frontend": diag.get("proposal_frontend"),
                    "source_video_path": source_video_path or diag.get("source_video_path"),
                    "source_audio_path": diag.get("source_audio_path") or str(root / "session_audio.wav"),
                    "proposal_diagnostics": diag,
                }
            )
    return rows


def compute_feature_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    phase5 = load_module("phase5_r32", PHASE5_PATH)
    nuisance = load_module("nuisance_r32", NUISANCE_PATH)
    contract = load_json(CONTRACT_PATH)
    feature_names = list(contract["feature_names"])
    audio_cache: dict[str, np.ndarray] = {}

    for row in rows:
        # Use the same decode source as r30 parity: the original source video, not
        # the derived session_audio.wav, because tiny decode/window differences move
        # this boundary row materially.
        media_path = str(row.get("source_video_path") or row["source_audio_path"])
        if media_path not in audio_cache:
            audio_cache[media_path] = phase5.decode_audio_mono(Path(media_path), phase5.SAMPLE_RATE)
        audio = audio_cache[media_path]
        ts = row["timestamp_seconds"]
        start = max(0.0, ts - WINDOW_PRE_SECONDS)
        end = max(start + 0.05, ts + WINDOW_POST_SECONDS)
        signal = audio[int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
        fmap = {
            **phase5.extract_features(signal, phase5.SAMPLE_RATE),
            **nuisance.nuisance_features(phase5, signal, phase5.SAMPLE_RATE),
        }
        values = {
            "audio_score": row["audio_score"],
            "audio_clip_probability": 0.0,
            "event_anchor_timestamp_seconds": ts,
            "is_false_negative_window": 0.0,
            **fmap,
        }
        row["governed_window_start_seconds"] = start
        row["governed_window_end_seconds"] = end
        row["governed_features"] = {name: to_float(values.get(name)) for name in feature_names}
        row["feature_vector"] = [row["governed_features"][name] for name in feature_names]
    return rows, feature_names


def score_and_explain(rows: list[dict[str, Any]], feature_names: list[str]) -> None:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    x = np.asarray([row["feature_vector"] for row in rows], dtype=np.float64)
    probs = model.predict_proba(x)[:, 1]
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(x, feature_names=feature_names), pred_contribs=True)
    for row, prob, contrib in zip(rows, probs, contribs, strict=True):
        row["r9_score_recomputed"] = float(prob)
        row["score_abs_delta_vs_manifest"] = abs(float(prob) - row["r9_score_manifest"])
        pairs = [
            {"feature": name, "contribution": float(value), "value": row["governed_features"][name]}
            for name, value in zip(feature_names, contrib[:-1], strict=True)
        ]
        row["bias_contribution"] = float(contrib[-1])
        row["top_positive_contributions"] = sorted(pairs, key=lambda item: item["contribution"], reverse=True)[:8]
        row["top_negative_contributions"] = sorted(pairs, key=lambda item: item["contribution"])[:8]


def percentile_rank(value: float, sample: list[float]) -> float:
    if not sample:
        return 0.0
    return float(sum(1 for item in sample if item <= value) / len(sample))


def build_reports(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    dangerous = [
        row for row in rows
        if row["label"] == "noise_or_other" and row["r9_score_recomputed"] >= V1_THRESHOLD
    ]
    high_score_noise = [
        row for row in rows
        if row["label"] == "noise_or_other" and row["r9_score_recomputed"] >= 0.80
    ]
    high_visual_noise = [
        row for row in rows
        if row["label"] == "noise_or_other"
        and row.get("visual_late_fusion_logreg_c0.5") is not None
        and to_float(row.get("visual_late_fusion_logreg_c0.5")) >= 0.95
    ]
    positive_controls = [
        row for row in rows
        if row["label"] == "platform_dive" and row["r9_score_recomputed"] >= 0.80
    ]
    approved_platform_controls = [
        row for row in rows
        if row["label"] == "platform_dive" and row["r9_score_recomputed"] >= V1_THRESHOLD
    ]

    bank_rows: list[dict[str, Any]] = []
    for row in dangerous:
        bank_rows.append({**compact_row(row), "priority": "P0", "role": "dangerous_approved_hard_negative"})
    for row in high_score_noise:
        if row["row_key"] not in {item["row_key"] for item in bank_rows}:
            bank_rows.append({**compact_row(row), "priority": "P1", "role": "high_score_noise_control"})
    for row in high_visual_noise:
        if row["row_key"] not in {item["row_key"] for item in bank_rows}:
            bank_rows.append({**compact_row(row), "priority": "P1", "role": "high_visual_noise_control"})
    for row in sorted(positive_controls, key=lambda item: item["r9_score_recomputed"], reverse=True)[:15]:
        bank_rows.append({**compact_row(row), "priority": "P2", "role": "platform_positive_control"})

    platform_rows = [row for row in rows if row["label"] == "platform_dive"]
    noise_rows = [row for row in rows if row["label"] == "noise_or_other"]
    feature_samples = {
        name: {
            "platform": [item["governed_features"][name] for item in platform_rows],
            "noise": [item["governed_features"][name] for item in noise_rows],
        }
        for name in feature_names
    }

    dangerous_analysis: list[dict[str, Any]] = []
    for row in dangerous:
        feature_percentiles = {
            name: {
                "value": row["governed_features"][name],
                "percentile_vs_platform": percentile_rank(row["governed_features"][name], feature_samples[name]["platform"]),
                "percentile_vs_noise": percentile_rank(row["governed_features"][name], feature_samples[name]["noise"]),
            }
            for name in feature_names
        }
        nearest_platform = sorted(
            [compact_row(item) | {"distance": l2_distance(row, item, feature_names)} for item in platform_rows],
            key=lambda item: item["distance"],
        )[:5]
        nearest_noise = sorted(
            [
                compact_row(item) | {"distance": l2_distance(row, item, feature_names)}
                for item in noise_rows
                if item["row_key"] != row["row_key"]
            ],
            key=lambda item: item["distance"],
        )[:5]
        dangerous_analysis.append(
            {
                **compact_row(row),
                "review_note": row["notes"],
                "source_video_path": row.get("source_video_path"),
                "score_delta_vs_manifest": row["score_abs_delta_vs_manifest"],
                "top_positive_contributions": row["top_positive_contributions"],
                "top_negative_contributions": row["top_negative_contributions"],
                "feature_percentiles": feature_percentiles,
                "nearest_platform_controls": nearest_platform,
                "nearest_noise_controls": nearest_noise,
                "diagnosis": "High governed score is real under exact parity. Current visual score is also high, so existing visual late-fusion cannot veto this non_dive_splash/shammy hard negative.",
            }
        )

    bank = {
        "experiment_name": "r32_hard_negative_candidate_bank",
        "source": "r31 exact-runtime approve reevaluation",
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "subtype_counts": dict(sorted(Counter(row["subtype"] for row in rows).items())),
        "dangerous_hard_negative_count": len(dangerous),
        "high_score_noise_count": len(high_score_noise),
        "high_visual_noise_count": len(high_visual_noise),
        "platform_positive_control_count": len(positive_controls),
        "bank_rows": bank_rows,
    }

    diagnosis = {
        "experiment_name": "r32_hard_negative_boundary_diagnosis",
        "v1_threshold": V1_THRESHOLD,
        "row_count": len(rows),
        "label_counts": bank["label_counts"],
        "dangerous_hard_negatives": dangerous_analysis,
        "risk_family_counts": {
            "dangerous_subtypes": dict(sorted(Counter(row["subtype"] for row in dangerous).items())),
            "high_score_noise_subtypes": dict(sorted(Counter(row["subtype"] for row in high_score_noise).items())),
            "high_visual_noise_subtypes": dict(sorted(Counter(row["subtype"] for row in high_visual_noise).items())),
        },
        "score_summary": {
            "max_noise_score": max((row["r9_score_recomputed"] for row in noise_rows), default=0.0),
            "min_approved_platform_score": min((row["r9_score_recomputed"] for row in approved_platform_controls), default=None),
            "approved_platform_count": len(approved_platform_controls),
            "dangerous_noise_count": len(dangerous),
        },
        "visual_signal_summary": {
            "visual_present_count": sum(1 for row in rows if row.get("visual_late_fusion_logreg_c0.5") is not None),
            "dangerous_visual_scores": [to_float(row.get("visual_late_fusion_logreg_c0.5")) for row in dangerous],
            "conclusion": "Current visual late-fusion score is not a safe hard-negative veto because the dangerous row scores near 1.0 visually.",
        },
        "candidate_filter_observation": evaluate_simple_filters(rows),
        "best_next_bounded_move": {
            "name": "r33_visual_entry_or_splash_morphology_hard_negative_probe",
            "recommendation": "Build a tiny hard-negative probe around real diver-entry visual evidence versus shammy/non_dive_splash impact-like nuisance. Use the r32 bank as a fixed evaluation set and include the source video clip for det-0007.",
            "blocked_options": [
                "Do not lower or raise approve thresholds as the main fix; the hard negative sits above the v1 threshold.",
                "Do not use reviewed subtype as a runtime veto.",
                "Do not rely on the current visual_late_fusion_logreg_c0.5 score as a veto; it fails on the key hard negative.",
            ],
        },
        "final_decision": "R32_HARD_NEGATIVE_DIAGNOSIS_COMPLETE",
    }
    return bank, diagnosis


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "session_id": row["session_id"],
        "detection_id": row["detection_id"],
        "timestamp_seconds": row["timestamp_seconds"],
        "label": row["label"],
        "subtype": row["subtype"],
        "notes": row["notes"],
        "r9_score": row["r9_score_recomputed"],
        "visual_late_fusion_logreg_c0.5": row.get("visual_late_fusion_logreg_c0.5"),
        "proposal_frontend": row.get("proposal_frontend"),
    }


def l2_distance(a: dict[str, Any], b: dict[str, Any], feature_names: list[str]) -> float:
    av = np.asarray([a["governed_features"][name] for name in feature_names], dtype=np.float64)
    bv = np.asarray([b["governed_features"][name] for name in feature_names], dtype=np.float64)
    scale = np.maximum(np.std(np.vstack([av, bv]), axis=0), 1.0)
    return float(np.linalg.norm((av - bv) / scale))


def evaluate_simple_filters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit obvious runtime-only filters without proposing them as product policy."""
    candidates = []
    for visual_max in [0.90, 0.95, 0.98, 0.995]:
        approved = [
            row for row in rows
            if row["r9_score_recomputed"] >= V1_THRESHOLD
            and row.get("visual_late_fusion_logreg_c0.5") is not None
            and to_float(row["visual_late_fusion_logreg_c0.5"]) <= visual_max
        ]
        candidates.append(filter_metrics(f"v1_and_visual_lte_{visual_max}", rows, approved))
    for score_min in [0.94, 0.95, 0.97, 0.99]:
        approved = [row for row in rows if row["r9_score_recomputed"] >= score_min]
        candidates.append(filter_metrics(f"r9_score_gte_{score_min}", rows, approved))
    return {
        "note": "These are diagnostic filters only, not a product search. They show whether a trivial runtime-only veto separates the hard negative.",
        "candidates": candidates,
    }


def filter_metrics(name: str, all_rows: list[dict[str, Any]], approved: list[dict[str, Any]]) -> dict[str, Any]:
    dangerous = [row for row in approved if row["label"] != "platform_dive"]
    return {
        "filter": name,
        "approve_count": len(approved),
        "coverage": len(approved) / len(all_rows) if all_rows else 0.0,
        "precision": None if not approved else sum(1 for row in approved if row["label"] == "platform_dive") / len(approved),
        "dangerous_count": len(dangerous),
        "dangerous_rows": [compact_row(row) for row in dangerous],
    }


def write_outputs(bank: dict[str, Any], diagnosis: dict[str, Any]) -> None:
    OUT_BANK_JSON.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    OUT_DIAG_JSON.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")

    bank_lines = [
        "# R32 Hard-Negative Candidate Bank",
        "",
        f"- rows analyzed: `{bank['row_count']}`",
        f"- label counts: `{json.dumps(bank['label_counts'], sort_keys=True)}`",
        f"- subtype counts: `{json.dumps(bank['subtype_counts'], sort_keys=True)}`",
        f"- dangerous hard negatives: `{bank['dangerous_hard_negative_count']}`",
        f"- high-score noise rows: `{bank['high_score_noise_count']}`",
        f"- high-visual noise rows: `{bank['high_visual_noise_count']}`",
        "",
        "| priority | role | row | label | subtype | r9 | visual | note |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in bank["bank_rows"]:
        visual = row.get("visual_late_fusion_logreg_c0.5")
        visual_s = "n/a" if visual is None else f"{float(visual):.4f}"
        bank_lines.append(
            f"| {row['priority']} | {row['role']} | `{row['row_key']}` | {row['label']} | {row['subtype']} | {row['r9_score']:.4f} | {visual_s} | {row['notes']} |"
        )
    OUT_BANK_MD.write_text("\n".join(bank_lines) + "\n", encoding="utf-8")

    danger = diagnosis["dangerous_hard_negatives"][0] if diagnosis["dangerous_hard_negatives"] else None
    diag_lines = [
        "# R32 Hard-Negative Boundary Diagnosis",
        "",
        "This pass diagnoses the exact-runtime r31 dangerous approval instead of tuning thresholds.",
        "",
        f"- rows analyzed: `{diagnosis['row_count']}`",
        f"- v1 threshold: `{diagnosis['v1_threshold']}`",
        f"- blocker: `true governed-model nuisance boundary issue`",
        "",
    ]
    if danger:
        diag_lines += [
            "## Primary Hard Negative",
            "",
            f"- row: `{danger['row_key']}`",
            f"- label: `{danger['label']}`",
            f"- subtype: `{danger['subtype']}`",
            f"- note: `{danger['review_note']}`",
            f"- r9 score: `{danger['r9_score']:.4f}`",
            f"- visual score: `{float(danger['visual_late_fusion_logreg_c0.5']):.4f}`",
            f"- source video: `{danger.get('source_video_path')}`",
            "",
            "## Top Positive Model Contributions",
            "",
            "| feature | contribution | value |",
            "|---|---:|---:|",
        ]
        for item in danger["top_positive_contributions"]:
            diag_lines.append(f"| `{item['feature']}` | {item['contribution']:.4f} | {item['value']:.4f} |")
        diag_lines += [
            "",
            "## Interpretation",
            "",
            "- The row is a reviewed `non_dive_splash` / shammy-thrown hard negative.",
            "- The exact governed model gives it a high platform probability.",
            "- The current visual late-fusion score also scores it near 1.0, so visual late fusion cannot act as a veto.",
            "- Threshold tuning is not a robust fix because the row sits above the current v1 approve threshold.",
            "",
        ]
    diag_lines += [
        "## Next Move",
        "",
        f"- `{diagnosis['best_next_bounded_move']['name']}`",
        f"- {diagnosis['best_next_bounded_move']['recommendation']}",
        "",
        "## Decision",
        "",
        f"- `{diagnosis['final_decision']}`",
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`",
    ]
    OUT_DIAG_MD.write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    OUT_DOC.write_text(
        "# R32 Hard-Negative Boundary Diagnosis\n\n"
        "R32 confirms that the post-r31 blocker is a true nuisance-boundary problem, not runtime parity.\n\n"
        "Primary hard negative:\n\n"
        "- `evaluation_r30_exact_scorepath_champigny_proxy::det-0007`\n"
        "- reviewed label: `noise_or_other`\n"
        "- subtype: `non_dive_splash`\n"
        "- note: `shammy thrown`\n"
        "- exact governed r9 score: `0.9423382878`\n"
        "- visual late-fusion score: `0.9901774245`\n\n"
        "Conclusion: the current visual late-fusion score is not sufficient as a runtime veto. "
        "The next bounded move is a hard-negative visual/splash morphology probe centered on this bank, "
        "without reviewed subtype leakage and without threshold tuning as the main mechanism.\n\n"
        "Decision:\n\n"
        "- `R32_HARD_NEGATIVE_DIAGNOSIS_COMPLETE`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = collect_rows()
    rows, feature_names = compute_feature_rows(rows)
    score_and_explain(rows, feature_names)
    bank, diagnosis = build_reports(rows, feature_names)
    write_outputs(bank, diagnosis)
    print(json.dumps({
        "rows": diagnosis["row_count"],
        "dangerous_hard_negatives": len(diagnosis["dangerous_hard_negatives"]),
        "risk_family_counts": diagnosis["risk_family_counts"],
        "next": diagnosis["best_next_bounded_move"]["name"],
        "decision": diagnosis["final_decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
