from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R33_PATH = ROOT / "benchmarks/r33_visual_entry_splash_morphology_probe.py"
SESSION_ROOTS = [
    ROOT / "outputs/evaluation_r30_exact_scorepath_insep_quick",
    ROOT / "outputs/evaluation_r30_exact_scorepath_champigny_proxy",
]
V1_THRESHOLD = 0.92158
CLIP_PRE_SECONDS = 3.0
CLIP_POST_SECONDS = 5.0
TARGET_FPS = 8.0

OUT_DATASET_JSON = ROOT / "outputs/r34_legacy_splash_morphology_expanded_dataset.json"
OUT_DATASET_MD = ROOT / "outputs/r34_legacy_splash_morphology_expanded_dataset.md"
OUT_PROBE_JSON = ROOT / "outputs/r34_legacy_splash_morphology_expanded_probe.json"
OUT_PROBE_MD = ROOT / "outputs/r34_legacy_splash_morphology_expanded_probe.md"
OUT_DOC = ROOT / "docs/research/R34_LEGACY_SPLASH_MORPHOLOGY_EXPANDED_PROBE.md"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R33 = load_module("r33_features", R33_PATH)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in SESSION_ROOTS:
        manifest = load_json(root / "ui_session_manifest.json")
        review = load_json(root / "evaluation_review.json")
        source_video_path = manifest.get("session", {}).get("source_video_path")
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
            rows.append(
                {
                    "row_key": f"{root.name}::{detection_id}",
                    "session_id": root.name,
                    "session_root": str(root),
                    "detection_id": detection_id,
                    "timestamp_seconds": to_float(detection.get("timestamp_seconds")),
                    "label": decision.get("eventLabel"),
                    "subtype": decision.get("subtype") or "none",
                    "notes": decision.get("notes") or "",
                    "r9_score": to_float(scores.get("governed_r9_score")),
                    "visual_late_fusion_logreg_c0.5": features.get("visual_late_fusion_logreg_c0.5"),
                    "source_video_path": source_video_path,
                }
            )
    return rows


def decode_event_frames(video_path: Path, timestamp: float) -> tuple[list[np.ndarray], str | None]:
    if not video_path.exists():
        return [], f"missing source video: {video_path}"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], f"cannot open source video: {video_path}"
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / TARGET_FPS)))
    start_frame = max(0, int(round((timestamp - CLIP_PRE_SECONDS) * source_fps)))
    end_frame = int(round((timestamp + CLIP_POST_SECONDS) * source_fps))
    if frame_count:
        end_frame = min(frame_count - 1, end_frame)
    frames: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_idx = start_frame
    while source_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (source_idx - start_frame) % stride == 0:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        source_idx += 1
    cap.release()
    if len(frames) < 3:
        return frames, "too few decoded frames"
    return frames, None


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        frames, error = decode_event_frames(Path(str(row["source_video_path"])), float(row["timestamp_seconds"]))
        row["visual_feature_decode_error"] = error
        row["visual_feature_frame_count"] = len(frames)
        row["features"] = R33.clip_features_from_frames(frames) if error is None else {name: 0.0 for name in R33.FEATURE_NAMES}
    return rows


def compact(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features", {})
    return {
        "row_key": row["row_key"],
        "session_id": row["session_id"],
        "detection_id": row["detection_id"],
        "timestamp_seconds": row["timestamp_seconds"],
        "label": row["label"],
        "subtype": row["subtype"],
        "notes": row["notes"],
        "r9_score": row["r9_score"],
        "visual_late_fusion_logreg_c0.5": row["visual_late_fusion_logreg_c0.5"],
        "legacy_v1_video_score": features.get("legacy_v1_video_score"),
        "legacy_v1_splash_ratio_pre": features.get("legacy_v1_splash_ratio_pre"),
        "legacy_v1_diver_ratio_pre": features.get("legacy_v1_diver_ratio_pre"),
        "waterline_motion_fraction": features.get("waterline_motion_fraction"),
        "downward_flow_strength": features.get("downward_flow_strength"),
        "decode_error": row.get("visual_feature_decode_error"),
    }


def feature_thresholds(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    platform = [row for row in rows if row["label"] == "platform_dive" and not row.get("visual_feature_decode_error")]
    thresholds: dict[str, dict[str, float]] = {}
    for name in R33.FEATURE_NAMES:
        values = np.asarray([row["features"][name] for row in platform], dtype=np.float64)
        if values.size == 0:
            thresholds[name] = {"q25": 0.0, "median": 0.0, "q75": 0.0}
        else:
            thresholds[name] = {
                "q25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "q75": float(np.percentile(values, 75)),
            }
    return thresholds


def evaluate_policy(rows: list[dict[str, Any]], name: str, fn: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    approved = [row for row in rows if fn(row)]
    dangerous = [row for row in approved if row["label"] != "platform_dive"]
    platforms = [row for row in approved if row["label"] == "platform_dive"]
    return {
        "candidate": name,
        "approve_count": len(approved),
        "approve_coverage": len(approved) / len(rows) if rows else 0.0,
        "approve_precision": None if not approved else len(platforms) / len(approved),
        "dangerous_count": len(dangerous),
        "safe": len(dangerous) == 0,
        "platform_approvals": len(platforms),
        "dangerous_rows": [compact(row) for row in dangerous],
        "approved_source_counts": dict(sorted(Counter(row["session_id"] for row in approved).items())),
        "approved_subtype_counts": dict(sorted(Counter(row["subtype"] for row in approved).items())),
    }


def feature_separation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in R33.FEATURE_NAMES:
        pos = [row["features"][name] for row in rows if row["label"] == "platform_dive" and not row.get("visual_feature_decode_error")]
        neg = [row["features"][name] for row in rows if row["label"] != "platform_dive" and not row.get("visual_feature_decode_error")]
        if not pos or not neg:
            continue
        pooled = float(np.std(pos + neg) or 1.0)
        out.append(
            {
                "feature": name,
                "platform_mean": float(np.mean(pos)),
                "nuisance_mean": float(np.mean(neg)),
                "separation": float((np.mean(pos) - np.mean(neg)) / pooled),
            }
        )
    return sorted(out, key=lambda item: abs(item["separation"]), reverse=True)


def build_candidates(rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    def v1(row: dict[str, Any]) -> bool:
        return row["r9_score"] >= V1_THRESHOLD

    def gate(row: dict[str, Any], feature: str, q: str) -> bool:
        return v1(row) and not row.get("visual_feature_decode_error") and row["features"][feature] >= thresholds[feature][q]

    def vote(row: dict[str, Any], q: str, minimum: int) -> bool:
        if not v1(row) or row.get("visual_feature_decode_error"):
            return False
        checks = [
            row["features"]["legacy_v1_diver_ratio_pre"] >= thresholds["legacy_v1_diver_ratio_pre"][q],
            row["features"]["waterline_motion_fraction"] >= thresholds["waterline_motion_fraction"][q],
            row["features"]["downward_flow_strength"] >= thresholds["downward_flow_strength"][q],
            row["features"]["center_motion_fraction"] >= thresholds["center_motion_fraction"][q],
        ]
        return sum(checks) >= minimum

    definitions: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("approve_review_v1", v1),
        ("v1_and_legacy_diver_ratio_gte_platform_q25", lambda row: gate(row, "legacy_v1_diver_ratio_pre", "q25")),
        ("v1_and_legacy_diver_ratio_gte_platform_median", lambda row: gate(row, "legacy_v1_diver_ratio_pre", "median")),
        ("v1_and_waterline_motion_gte_platform_q25", lambda row: gate(row, "waterline_motion_fraction", "q25")),
        ("v1_and_downward_flow_gte_platform_q25", lambda row: gate(row, "downward_flow_strength", "q25")),
        ("v1_and_center_motion_gte_platform_q25", lambda row: gate(row, "center_motion_fraction", "q25")),
        ("v1_and_entry_legacy_vote_2of4_q25", lambda row: vote(row, "q25", 2)),
        ("v1_and_entry_legacy_vote_3of4_q25", lambda row: vote(row, "q25", 3)),
        ("v1_and_entry_legacy_vote_2of4_median", lambda row: vote(row, "median", 2)),
    ]
    return [evaluate_policy(rows, name, fn) for name, fn in definitions]


def write_outputs(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], separations: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> None:
    v1 = candidates[0]
    safe = [item for item in candidates if item["safe"] and item["approve_count"] > 0]
    best = max(safe, key=lambda item: (item["platform_approvals"], item["approve_count"]), default=None)
    hard_rows = [
        compact(row)
        for row in rows
        if row["label"] != "platform_dive" and row["r9_score"] >= V1_THRESHOLD
    ]
    dataset = {
        "experiment_name": "r34_legacy_splash_morphology_expanded_dataset",
        "input_sessions": [str(root) for root in SESSION_ROOTS],
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "subtype_counts": dict(sorted(Counter(row["subtype"] for row in rows).items())),
        "session_counts": dict(sorted(Counter(row["session_id"] for row in rows).items())),
        "decode_error_count": sum(1 for row in rows if row.get("visual_feature_decode_error")),
        "feature_names": R33.FEATURE_NAMES,
        "hard_negative_rows": hard_rows,
        "rows": [compact(row) | {"features": row["features"]} for row in rows],
    }
    probe = {
        "experiment_name": "r34_legacy_splash_morphology_expanded_probe",
        "purpose": "Expand r33 recovered legacy-v1 splash/diver cues and direct morphology features from 7 clips to all exact-runtime reviewed rows.",
        "baseline_v1": v1,
        "candidate_comparison": candidates,
        "best_safe_candidate": best,
        "feature_separation_top": separations[:20],
        "platform_feature_thresholds": {
            name: thresholds[name]
            for name in [
                "legacy_v1_diver_ratio_pre",
                "legacy_v1_video_score",
                "waterline_motion_fraction",
                "downward_flow_strength",
                "center_motion_fraction",
            ]
        },
        "interpretation": {
            "legacy_v1_splash_gate": "Not used as a final rule because r33 showed the legacy splash gate approved every tiny-bank clip, including nuisance.",
            "focus": "v1-guarded rules that keep exact r9 approve candidates but require runtime visual entry/diver evidence.",
            "promotion_status": "offline_diagnostic_only",
        },
        "final_decisions": [
            "R34_EXPANDED_VISUAL_HARD_NEGATIVE_DIAGNOSIS_COMPLETE",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }
    OUT_DATASET_JSON.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    OUT_PROBE_JSON.write_text(json.dumps(probe, indent=2), encoding="utf-8")

    dataset_lines = [
        "# R34 Legacy Splash Morphology Expanded Dataset",
        "",
        f"- rows: `{len(rows)}`",
        f"- labels: `{json.dumps(dataset['label_counts'], sort_keys=True)}`",
        f"- subtypes: `{json.dumps(dataset['subtype_counts'], sort_keys=True)}`",
        f"- sessions: `{json.dumps(dataset['session_counts'], sort_keys=True)}`",
        f"- decode errors: `{dataset['decode_error_count']}`",
        "",
        "| row | session | label | subtype | r9 | old visual | legacy v1 score | legacy diver ratio | waterline | notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: item["r9_score"], reverse=True)[:30]:
        features = row["features"]
        dataset_lines.append(
            f"| `{row['row_key']}` | `{row['session_id']}` | {row['label']} | {row['subtype']} | {row['r9_score']:.4f} | {to_float(row['visual_late_fusion_logreg_c0.5']):.4f} | {features['legacy_v1_video_score']:.4f} | {features['legacy_v1_diver_ratio_pre']:.4f} | {features['waterline_motion_fraction']:.4f} | {row['notes']} |"
        )
    OUT_DATASET_MD.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    probe_lines = [
        "# R34 Legacy Splash Morphology Expanded Probe",
        "",
        "R34 expands the r33 recovered legacy-v1 splash/diver ROI cues across the exact-runtime reviewed rows from the r30 INSEP and Champigny proxy sessions.",
        "",
        "## Candidate Comparison",
        "",
        "| candidate | approvals | platform approvals | coverage | precision | dangerous | safe |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in sorted(candidates, key=lambda row: (row["dangerous_count"], -row["platform_approvals"], -row["approve_count"])):
        precision = "n/a" if item["approve_precision"] is None else f"{item['approve_precision']:.4f}"
        probe_lines.append(
            f"| `{item['candidate']}` | {item['approve_count']} | {item['platform_approvals']} | {item['approve_coverage']:.4f} | {precision} | {item['dangerous_count']} | `{item['safe']}` |"
        )
    probe_lines += [
        "",
        "## Top Feature Separations",
        "",
        "| feature | platform mean | nuisance mean | separation |",
        "|---|---:|---:|---:|",
    ]
    for item in separations[:12]:
        probe_lines.append(
            f"| `{item['feature']}` | {item['platform_mean']:.4f} | {item['nuisance_mean']:.4f} | {item['separation']:.4f} |"
        )
    probe_lines += [
        "",
        "## Decision",
        "",
        "- `R34_EXPANDED_VISUAL_HARD_NEGATIVE_DIAGNOSIS_COMPLETE`",
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`",
    ]
    OUT_PROBE_MD.write_text("\n".join(probe_lines) + "\n", encoding="utf-8")
    OUT_DOC.write_text(
        "# R34 Legacy Splash Morphology Expanded Probe\n\n"
        "R34 expands the r33 recovered legacy-v1 splash/diver cues from the seven-clip diagnostic bank to all reviewed exact-runtime rows in the r30 INSEP and Champigny proxy sessions.\n\n"
        "The key product framing remains unchanged: this is offline diagnosis only, not a runtime policy promotion. The legacy splash gate itself is too permissive, but legacy pre-diver ROI motion and direct entry morphology are useful candidate veto signals to test on a larger bank.\n\n"
        "Decisions:\n\n"
        "- `R34_EXPANDED_VISUAL_HARD_NEGATIVE_DIAGNOSIS_COMPLETE`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows),
        "labels": dataset["label_counts"],
        "v1": {key: v1[key] for key in ["approve_count", "approve_precision", "dangerous_count"]},
        "best_safe_candidate": None if best is None else {key: best[key] for key in ["candidate", "approve_count", "approve_precision", "dangerous_count", "platform_approvals"]},
        "decode_error_count": dataset["decode_error_count"],
    }, indent=2))


def main() -> None:
    rows = enrich_rows(collect_rows())
    thresholds = feature_thresholds(rows)
    candidates = build_candidates(rows, thresholds)
    separations = feature_separation(rows)
    write_outputs(rows, candidates, separations, thresholds)


if __name__ == "__main__":
    main()
