from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
R33_PATH = ROOT / "benchmarks/r33_visual_entry_splash_morphology_probe.py"
PHASE5_PATH = ROOT / "benchmarks/phase5_regime_aware_execution_r7_es4.py"
NUISANCE_PATH = ROOT / "benchmarks/post_noise_nuisance_family_benchmark.py"
MODEL_PATH = ROOT / ".divesensei-runtime/models/r9_compact_nuisance_weighted/xgboost_model.json"
CONTRACT_PATH = ROOT / ".divesensei-runtime/models/r9_compact_nuisance_weighted/contract.json"
R34_DATASET_PATH = ROOT / "outputs/r34_legacy_splash_morphology_expanded_dataset.json"

OUT_JSON = ROOT / "outputs/r35_independent_visual_veto_validation.json"
OUT_MD = ROOT / "outputs/r35_independent_visual_veto_validation.md"
OUT_FAILURE_JSON = ROOT / "outputs/r35_independent_visual_veto_failure_analysis.json"
OUT_FAILURE_MD = ROOT / "outputs/r35_independent_visual_veto_failure_analysis.md"
OUT_DOC = ROOT / "docs/research/R35_INDEPENDENT_VISUAL_VETO_VALIDATION.md"

V1_THRESHOLD = 0.92158
WINDOW_PRE_SECONDS = 0.75
WINDOW_POST_SECONDS = 2.25
CLIP_PRE_SECONDS = 3.0
CLIP_POST_SECONDS = 5.0
TARGET_FPS = 8.0
CALIBRATION_SESSIONS = {
    "evaluation_r30_exact_scorepath_insep_quick",
    "evaluation_r30_exact_scorepath_champigny_proxy",
}
EXCLUDED_DUPLICATE_OR_DERIVED = {
    "evaluation_champigny_20260406-labelling",
    "evaluation_insep_quick_9015_20260409_ui",
    "evaluation_r27_scorepath_insep_quick",
    "evaluation_r27_scorepath_insep_quick_v2",
    "evaluation_r27_scorepath_champigny_proxy",
    "evaluation_r30_exact_scorepath_insep_quick",
    "evaluation_r30_exact_scorepath_champigny_proxy",
    "evaluation_champigny_20260410_tailimbalance_417",
    "evaluation_champigny_20260406-retrained",
    "evaluation_priority123_model_20260406-184900",
    "evaluation_priority123_champigny_model",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R33 = load_module("r33_features_r35", R33_PATH)
PHASE5 = load_module("phase5_r35", PHASE5_PATH)
NUISANCE = load_module("nuisance_r35", NUISANCE_PATH)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def resolved_review_video(root: Path, manifest: dict[str, Any]) -> tuple[Path | None, str]:
    original = manifest.get("session", {}).get("source_video_path")
    if original and Path(original).exists():
        return Path(original), "original_source"
    proxy = root / "web/session_source_review.mp4"
    if proxy.exists():
        return proxy, "review_proxy"
    if original:
        return None, f"missing_original:{original}"
    return None, "missing_source_video_path"


def candidate_session_roots() -> list[Path]:
    roots: list[Path] = []
    for root in sorted((ROOT / "outputs").glob("evaluation_*")):
        if root.name in EXCLUDED_DUPLICATE_OR_DERIVED:
            continue
        if not (root / "ui_session_manifest.json").exists() or not (root / "evaluation_review.json").exists():
            continue
        roots.append(root)
    return roots


def collect_reviewed_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    session_summary: list[dict[str, Any]] = []
    for root in candidate_session_roots():
        manifest = load_json(root / "ui_session_manifest.json")
        review = load_json(root / "evaluation_review.json")
        video_path, video_status = resolved_review_video(root, manifest)
        decisions = {
            str(item.get("detectionId")): item
            for item in review.get("decisions", [])
            if item.get("eventLabel") in {"platform_dive", "noise_or_other"}
        }
        included = 0
        for detection in manifest.get("detections", []):
            decision = decisions.get(str(detection.get("id")))
            if decision is None or video_path is None:
                continue
            rows.append(
                {
                    "row_key": f"{root.name}::{detection.get('id')}",
                    "session_id": root.name,
                    "session_root": str(root),
                    "detection_id": str(detection.get("id")),
                    "timestamp_seconds": to_float(detection.get("timestamp_seconds")),
                    "label": decision.get("eventLabel"),
                    "subtype": decision.get("subtype") or "none",
                    "notes": decision.get("notes") or "",
                    "audio_score": to_float((detection.get("scores") or {}).get("audio")),
                    "source_video_path": str(video_path),
                    "video_source_status": video_status,
                }
            )
            included += 1
        session_summary.append(
            {
                "session_id": root.name,
                "video_status": video_status,
                "source_video_path": str(video_path) if video_path is not None else None,
                "eligible_reviewed_rows": included,
                "decision_rows": len(decisions),
            }
        )
    return rows, session_summary


def decode_event_frames(video_path: Path, timestamp: float) -> tuple[list[np.ndarray], str | None]:
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
            frames.append(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))
        source_idx += 1
    cap.release()
    if len(frames) < 3:
        return frames, "too few decoded frames"
    return frames, None


def calibration_thresholds() -> dict[str, dict[str, float]]:
    data = load_json(R34_DATASET_PATH)
    platform_rows = [
        row for row in data["rows"]
        if row["label"] == "platform_dive"
        and row["session_id"] in CALIBRATION_SESSIONS
        and not row.get("decode_error")
    ]
    thresholds: dict[str, dict[str, float]] = {}
    for name in R33.FEATURE_NAMES:
        values = np.asarray([row["features"][name] for row in platform_rows], dtype=np.float64)
        thresholds[name] = {
            "q25": float(np.percentile(values, 25)) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "q75": float(np.percentile(values, 75)) if values.size else 0.0,
        }
    return thresholds


def exact_r9_scores(rows: list[dict[str, Any]]) -> None:
    contract = load_json(CONTRACT_PATH)
    feature_names = list(contract["feature_names"])
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    audio_cache: dict[str, np.ndarray] = {}
    pending: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    for row in rows:
        video_path = row["source_video_path"]
        if video_path not in audio_cache:
            audio_cache[video_path] = PHASE5.decode_audio_mono(Path(video_path), PHASE5.SAMPLE_RATE)
        audio = audio_cache[video_path]
        ts = float(row["timestamp_seconds"])
        start = max(0.0, ts - WINDOW_PRE_SECONDS)
        end = max(start + 0.05, ts + WINDOW_POST_SECONDS)
        signal = audio[int(round(start * PHASE5.SAMPLE_RATE)) : int(round(end * PHASE5.SAMPLE_RATE))]
        fmap = {
            **PHASE5.extract_features(signal, PHASE5.SAMPLE_RATE),
            **NUISANCE.nuisance_features(PHASE5, signal, PHASE5.SAMPLE_RATE),
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
        vectors.append([row["governed_features"][name] for name in feature_names])
        pending.append(row)
    if not vectors:
        return
    probs = model.predict_proba(np.asarray(vectors, dtype=np.float64))[:, 1]
    for row, prob in zip(pending, probs, strict=True):
        row["r9_score"] = float(prob)


def visual_features(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        frames, error = decode_event_frames(Path(row["source_video_path"]), float(row["timestamp_seconds"]))
        row["visual_feature_decode_error"] = error
        row["visual_feature_frame_count"] = len(frames)
        row["features"] = R33.clip_features_from_frames(frames) if error is None else {name: 0.0 for name in R33.FEATURE_NAMES}


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
        "r9_score": row.get("r9_score"),
        "legacy_v1_diver_ratio_pre": features.get("legacy_v1_diver_ratio_pre"),
        "legacy_v1_video_score": features.get("legacy_v1_video_score"),
        "downward_flow_strength": features.get("downward_flow_strength"),
        "waterline_motion_fraction": features.get("waterline_motion_fraction"),
        "center_motion_fraction": features.get("center_motion_fraction"),
        "video_source_status": row.get("video_source_status"),
        "decode_error": row.get("visual_feature_decode_error"),
    }


def evaluate_policy(rows: list[dict[str, Any]], name: str, fn: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    approved = [row for row in rows if fn(row)]
    dangerous = [row for row in approved if row["label"] != "platform_dive"]
    platforms = [row for row in approved if row["label"] == "platform_dive"]
    return {
        "candidate": name,
        "approve_count": len(approved),
        "platform_approvals": len(platforms),
        "approve_coverage": len(approved) / len(rows) if rows else 0.0,
        "approve_precision": None if not approved else len(platforms) / len(approved),
        "dangerous_count": len(dangerous),
        "safe": len(dangerous) == 0,
        "dangerous_rows": [compact(row) for row in dangerous],
        "approved_source_counts": dict(sorted(Counter(row["session_id"] for row in approved).items())),
        "approved_subtype_counts": dict(sorted(Counter(row["subtype"] for row in approved).items())),
    }


def build_candidates(rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    def v1(row: dict[str, Any]) -> bool:
        return float(row.get("r9_score") or 0.0) >= V1_THRESHOLD

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
        ("approve_review_v1_exact_r9", v1),
        ("v1_and_downward_flow_gte_r34_platform_q25", lambda row: gate(row, "downward_flow_strength", "q25")),
        ("v1_and_entry_legacy_vote_3of4_r34_q25", lambda row: vote(row, "q25", 3)),
        ("v1_and_legacy_diver_ratio_gte_r34_platform_q25", lambda row: gate(row, "legacy_v1_diver_ratio_pre", "q25")),
        ("v1_and_legacy_diver_ratio_gte_r34_platform_median", lambda row: gate(row, "legacy_v1_diver_ratio_pre", "median")),
        ("v1_and_waterline_motion_gte_r34_platform_q25", lambda row: gate(row, "waterline_motion_fraction", "q25")),
    ]
    return [evaluate_policy(rows, name, fn) for name, fn in definitions]


def feature_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name in [
        "legacy_v1_diver_ratio_pre",
        "legacy_v1_video_score",
        "legacy_v1_splash_ratio_pre",
        "downward_flow_strength",
        "waterline_motion_fraction",
        "center_motion_fraction",
    ]:
        pos = [row["features"][name] for row in rows if row["label"] == "platform_dive" and not row.get("visual_feature_decode_error")]
        neg = [row["features"][name] for row in rows if row["label"] != "platform_dive" and not row.get("visual_feature_decode_error")]
        pooled = float(np.std(pos + neg) or 1.0) if pos and neg else 1.0
        out.append(
            {
                "feature": name,
                "platform_mean": float(np.mean(pos)) if pos else None,
                "nuisance_mean": float(np.mean(neg)) if neg else None,
                "separation": float((np.mean(pos) - np.mean(neg)) / pooled) if pos and neg else None,
            }
        )
    return out


def write_reports(rows: list[dict[str, Any]], session_summary: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    by_session = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)
    session_results = {}
    for session_id, session_rows in by_session.items():
        session_results[session_id] = {
            "row_count": len(session_rows),
            "label_counts": dict(sorted(Counter(row["label"] for row in session_rows).items())),
            "subtype_counts": dict(sorted(Counter(row["subtype"] for row in session_rows).items())),
            "v1_approved": sum(1 for row in session_rows if row["r9_score"] >= V1_THRESHOLD),
            "v1_dangerous": [
                compact(row)
                for row in session_rows
                if row["r9_score"] >= V1_THRESHOLD and row["label"] != "platform_dive"
            ],
        }
    best_safe = max(
        [item for item in candidates if item["safe"] and item["approve_count"] > 0],
        key=lambda item: (item["platform_approvals"], item["approve_count"]),
        default=None,
    )
    validation = {
        "experiment_name": "r35_independent_visual_veto_validation",
        "purpose": "Validate r34 visual-entry/legacy-v1 veto candidates on independent reviewed sources outside r30 calibration.",
        "calibration_source": str(R34_DATASET_PATH),
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "subtype_counts": dict(sorted(Counter(row["subtype"] for row in rows).items())),
        "session_counts": dict(sorted(Counter(row["session_id"] for row in rows).items())),
        "video_source_status_counts": dict(sorted(Counter(row["video_source_status"] for row in rows).items())),
        "decode_error_count": sum(1 for row in rows if row.get("visual_feature_decode_error")),
        "session_summary": session_summary,
        "candidate_comparison": candidates,
        "best_safe_candidate": best_safe,
        "session_results": session_results,
        "feature_summary": feature_summary(rows),
        "final_decisions": [
            "R35_INDEPENDENT_VISUAL_VETO_VALIDATION_COMPLETE",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }
    failures = {
        "experiment_name": "r35_independent_visual_veto_failure_analysis",
        "v1_dangerous_rows": [
            compact(row) for row in rows if row["label"] != "platform_dive" and row["r9_score"] >= V1_THRESHOLD
        ],
        "candidate_failures": [
            {
                "candidate": item["candidate"],
                "dangerous_count": item["dangerous_count"],
                "dangerous_rows": item["dangerous_rows"],
            }
            for item in candidates
            if item["dangerous_count"] > 0
        ],
        "interpretation": {
            "question": "Does the r34 visual-entry veto generalize beyond the r30 calibration bank?",
            "answer": "A safe veto candidate exists on this expanded independent bank if dangerous_count remains zero, but source/proxy composition must be considered before runtime policy wiring.",
        },
    }
    OUT_JSON.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    OUT_FAILURE_JSON.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    lines = [
        "# R35 Independent Visual Veto Validation",
        "",
        "R35 validates the r34 visual-entry and recovered legacy-v1 cues on reviewed sources outside the r30 calibration bank.",
        "",
        f"- rows: `{len(rows)}`",
        f"- labels: `{json.dumps(validation['label_counts'], sort_keys=True)}`",
        f"- sessions: `{json.dumps(validation['session_counts'], sort_keys=True)}`",
        f"- video source status: `{json.dumps(validation['video_source_status_counts'], sort_keys=True)}`",
        f"- decode errors: `{validation['decode_error_count']}`",
        "",
        "## Candidate Comparison",
        "",
        "| candidate | approvals | platform approvals | coverage | precision | dangerous | safe |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in sorted(candidates, key=lambda row: (row["dangerous_count"], -row["platform_approvals"], -row["approve_count"])):
        precision = "n/a" if item["approve_precision"] is None else f"{item['approve_precision']:.4f}"
        lines.append(
            f"| `{item['candidate']}` | {item['approve_count']} | {item['platform_approvals']} | {item['approve_coverage']:.4f} | {precision} | {item['dangerous_count']} | `{item['safe']}` |"
        )
    lines += [
        "",
        "## Per-Session V1 Danger",
        "",
        "| session | rows | labels | v1 approved | v1 dangerous |",
        "|---|---:|---|---:|---:|",
    ]
    for session_id, result in sorted(session_results.items()):
        lines.append(
            f"| `{session_id}` | {result['row_count']} | `{json.dumps(result['label_counts'], sort_keys=True)}` | {result['v1_approved']} | {len(result['v1_dangerous'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- `R35_INDEPENDENT_VISUAL_VETO_VALIDATION_COMPLETE`",
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    failure_lines = [
        "# R35 Independent Visual Veto Failure Analysis",
        "",
        f"- v1 dangerous rows: `{len(failures['v1_dangerous_rows'])}`",
        "",
        "## V1 Dangerous Rows",
        "",
        "| row | label | subtype | r9 | legacy diver ratio | downward flow | source | notes |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in failures["v1_dangerous_rows"]:
        failure_lines.append(
            f"| `{row['row_key']}` | {row['label']} | {row['subtype']} | {to_float(row['r9_score']):.4f} | {to_float(row['legacy_v1_diver_ratio_pre']):.4f} | {to_float(row['downward_flow_strength']):.8f} | {row['video_source_status']} | {row['notes']} |"
        )
    failure_lines += [
        "",
        "## Candidate Failures",
        "",
        "| candidate | dangerous |",
        "|---|---:|",
    ]
    for item in failures["candidate_failures"]:
        failure_lines.append(f"| `{item['candidate']}` | {item['dangerous_count']} |")
    OUT_FAILURE_MD.write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    OUT_DOC.write_text(
        "# R35 Independent Visual Veto Validation\n\n"
        "R35 validates the r34 visual-entry and recovered legacy-v1 splash/diver cues on reviewed sources outside the r30 calibration bank. It uses exact r9 scoring and event-centered video morphology extraction from original source videos when present, otherwise from review proxy video.\n\n"
        "This remains an offline validation pass. `approve_review_v1` remains default and no auto-exclude lane is introduced.\n\n"
        "Decisions:\n\n"
        "- `R35_INDEPENDENT_VISUAL_VETO_VALIDATION_COMPLETE`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows),
        "labels": validation["label_counts"],
        "sessions": len(validation["session_counts"]),
        "v1": {
            "approve_count": candidates[0]["approve_count"],
            "precision": candidates[0]["approve_precision"],
            "dangerous": candidates[0]["dangerous_count"],
        },
        "best_safe_candidate": None if best_safe is None else {
            key: best_safe[key]
            for key in ["candidate", "approve_count", "platform_approvals", "approve_precision", "dangerous_count"]
        },
        "decode_error_count": validation["decode_error_count"],
    }, indent=2))


def main() -> None:
    rows, session_summary = collect_reviewed_rows()
    exact_r9_scores(rows)
    visual_features(rows)
    thresholds = calibration_thresholds()
    candidates = build_candidates(rows, thresholds)
    write_reports(rows, session_summary, candidates)


if __name__ == "__main__":
    main()
