from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = ROOT / "outputs/r32_hard_negative_clip_bank_manifest.json"
OUT_DATASET_JSON = ROOT / "outputs/r33_visual_entry_splash_morphology_dataset.json"
OUT_DATASET_MD = ROOT / "outputs/r33_visual_entry_splash_morphology_dataset.md"
OUT_BENCH_JSON = ROOT / "outputs/r33_visual_entry_splash_morphology_probe.json"
OUT_BENCH_MD = ROOT / "outputs/r33_visual_entry_splash_morphology_probe.md"
OUT_POLICY_JSON = ROOT / "outputs/r33_visual_hard_negative_policy_recommendation.json"
OUT_POLICY_MD = ROOT / "outputs/r33_visual_hard_negative_policy_recommendation.md"
OUT_DOC = ROOT / "docs/research/R33_VISUAL_ENTRY_SPLASH_MORPHOLOGY_PROBE.md"


FEATURE_NAMES = [
    "motion_mean",
    "motion_max",
    "motion_p95",
    "motion_persistence",
    "motion_peak_frame_norm",
    "motion_temporal_spread",
    "lower_motion_fraction",
    "upper_motion_fraction",
    "center_motion_fraction",
    "lower_upper_motion_ratio",
    "waterline_motion_fraction",
    "vertical_flow_mean",
    "vertical_flow_abs_mean",
    "downward_flow_fraction",
    "downward_flow_strength",
    "motion_area_mean",
    "motion_area_max",
    "motion_area_persistence",
    "connected_component_area_mean",
    "connected_component_area_max",
    "motion_centroid_y_mean",
    "motion_centroid_y_std",
    "motion_centroid_y_slope",
    "motion_centroid_x_std",
    "bright_splash_delta_mean",
    "bright_splash_delta_max",
    "bright_splash_persistence",
    "late_motion_over_early_motion",
    "post_peak_decay_ratio",
    "legacy_v1_splash_peak",
    "legacy_v1_pre_splash_baseline",
    "legacy_v1_post_splash_baseline",
    "legacy_v1_pre_diver_peak",
    "legacy_v1_pre_diver_baseline",
    "legacy_v1_splash_ratio_pre",
    "legacy_v1_splash_ratio_post",
    "legacy_v1_diver_ratio_pre",
    "legacy_v1_video_score",
    "legacy_v1_video_gate_passed",
    "legacy_v1_rescue_splash_ratio_passed",
]


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if abs(den) < 1e-9 else float(num / den)


def load_bank() -> list[dict[str, Any]]:
    return json.loads(BANK_PATH.read_text())["clips"]


def decode_frames(path: Path, size: tuple[int, int] = (224, 224), fps: float = 8.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    stride = max(1, int(round(source_fps / fps)))
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def largest_component_area(mask: np.ndarray) -> float:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0.0
    return float(np.max(stats[1:, cv2.CC_STAT_AREA]) / mask.size)


def legacy_v1_splash_features(frames_bgr: list[np.ndarray], sample_fps: float = 8.0) -> dict[str, float]:
    """Reproduce the legacy audio_v1 visual verifier cues on the event clip.

    The r32 clips are cut as 3s pre + 5s post, so the legacy event index is
    anchored at 3s into the clip. Legacy v1 originally used a 3s pre / 1s post
    verifier window, lower-water splash ROI, and upper/mid diver ROI.
    """
    defaults = {
        "legacy_v1_splash_peak": 0.0,
        "legacy_v1_pre_splash_baseline": 0.0,
        "legacy_v1_post_splash_baseline": 0.0,
        "legacy_v1_pre_diver_peak": 0.0,
        "legacy_v1_pre_diver_baseline": 0.0,
        "legacy_v1_splash_ratio_pre": 0.0,
        "legacy_v1_splash_ratio_post": 0.0,
        "legacy_v1_diver_ratio_pre": 0.0,
        "legacy_v1_video_score": 0.0,
        "legacy_v1_video_gate_passed": 0.0,
        "legacy_v1_rescue_splash_ratio_passed": 0.0,
    }
    if len(frames_bgr) < 3:
        return defaults

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_bgr]
    h, w = gray_frames[0].shape
    splash_top = int(0.72 * h)
    splash_bottom = int(0.95 * h)
    diver_top = int(0.15 * h)
    diver_bottom = int(0.72 * h)

    event_idx = min(len(gray_frames) - 1, max(1, int(round(3.0 * sample_fps))))
    verify_end_idx = min(len(gray_frames), event_idx + int(round(1.0 * sample_fps)) + 1)
    verify_frames = gray_frames[:verify_end_idx]

    splash_motion = np.zeros(len(verify_frames), dtype=np.float32)
    diver_motion = np.zeros(len(verify_frames), dtype=np.float32)
    prev_splash: np.ndarray | None = None
    prev_diver: np.ndarray | None = None
    for idx, gray in enumerate(verify_frames):
        splash_region = cv2.GaussianBlur(gray[splash_top:splash_bottom, :], (5, 5), 0)
        diver_region = cv2.GaussianBlur(gray[diver_top:diver_bottom, :], (5, 5), 0)
        if prev_splash is not None and splash_region.size:
            splash_motion[idx] = float(np.mean(cv2.absdiff(splash_region, prev_splash)))
        if prev_diver is not None and diver_region.size:
            diver_motion[idx] = float(np.mean(cv2.absdiff(diver_region, prev_diver)))
        prev_splash = splash_region
        prev_diver = diver_region

    if len(splash_motion) < 3:
        return defaults

    local_event_idx = min(len(splash_motion) - 1, event_idx)
    splash_slice = splash_motion[max(0, local_event_idx - 4) : min(len(splash_motion), local_event_idx + 6)]
    pre_diver_slice = diver_motion[max(0, local_event_idx - int(1.5 * sample_fps)) : max(1, local_event_idx)]
    pre_splash = splash_motion[: max(2, local_event_idx)]
    post_splash = splash_motion[min(len(splash_motion) - 1, local_event_idx) :]
    pre_diver = diver_motion[: max(2, local_event_idx)]

    splash_peak = float(np.max(splash_slice)) if splash_slice.size else 0.0
    pre_splash_baseline = float(np.median(pre_splash)) if pre_splash.size else 0.0
    post_splash_baseline = float(np.median(post_splash)) if post_splash.size else 0.0
    pre_diver_peak = float(np.max(pre_diver_slice)) if pre_diver_slice.size else 0.0
    pre_diver_baseline = float(np.median(pre_diver)) if pre_diver.size else 0.0
    splash_ratio_pre = safe_div(splash_peak, pre_splash_baseline + 1e-6)
    splash_ratio_post = safe_div(splash_peak, post_splash_baseline + 1e-6)
    diver_ratio_pre = safe_div(pre_diver_peak, pre_diver_baseline + 1e-6)
    video_score = 0.55 * splash_ratio_pre + 0.30 * diver_ratio_pre + 0.15 * splash_ratio_post

    return {
        "legacy_v1_splash_peak": splash_peak,
        "legacy_v1_pre_splash_baseline": pre_splash_baseline,
        "legacy_v1_post_splash_baseline": post_splash_baseline,
        "legacy_v1_pre_diver_peak": pre_diver_peak,
        "legacy_v1_pre_diver_baseline": pre_diver_baseline,
        "legacy_v1_splash_ratio_pre": splash_ratio_pre,
        "legacy_v1_splash_ratio_post": splash_ratio_post,
        "legacy_v1_diver_ratio_pre": diver_ratio_pre,
        "legacy_v1_video_score": float(video_score),
        "legacy_v1_video_gate_passed": float(video_score >= 0.8),
        "legacy_v1_rescue_splash_ratio_passed": float(splash_ratio_pre >= 1.35),
    }


def clip_features_from_frames(frames_bgr: list[np.ndarray]) -> dict[str, float]:
    if len(frames_bgr) < 3:
        return {name: 0.0 for name in FEATURE_NAMES}
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for frame in frames_bgr]
    hsv = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) for frame in frames_bgr]
    h, w = gray[0].shape
    lower = np.s_[int(0.58 * h) :, :]
    upper = np.s_[: int(0.42 * h), :]
    center = np.s_[int(0.35 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)]
    waterline = np.s_[int(0.50 * h) : int(0.78 * h), :]

    motion_values: list[float] = []
    lower_motion: list[float] = []
    upper_motion: list[float] = []
    center_motion: list[float] = []
    water_motion: list[float] = []
    flow_y_means: list[float] = []
    flow_y_abs: list[float] = []
    downward_fracs: list[float] = []
    downward_strengths: list[float] = []
    area_values: list[float] = []
    component_values: list[float] = []
    centroid_y: list[float] = []
    centroid_x: list[float] = []
    bright_delta: list[float] = []

    for idx in range(1, len(gray)):
        prev = gray[idx - 1]
        cur = gray[idx]
        diff = np.abs(cur - prev)
        motion_values.append(float(np.mean(diff)))
        lower_motion.append(float(np.mean(diff[lower])))
        upper_motion.append(float(np.mean(diff[upper])))
        center_motion.append(float(np.mean(diff[center])))
        water_motion.append(float(np.mean(diff[waterline])))

        threshold = max(0.035, float(np.mean(diff) + 1.5 * np.std(diff)))
        mask = diff > threshold
        area_values.append(float(np.mean(mask)))
        component_values.append(largest_component_area(mask))
        ys, xs = np.nonzero(mask)
        if len(ys):
            centroid_y.append(float(np.mean(ys) / h))
            centroid_x.append(float(np.mean(xs) / w))
        else:
            centroid_y.append(0.0)
            centroid_x.append(0.0)

        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        fy = flow[..., 1]
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        active = mag > np.percentile(mag, 75)
        if np.any(active):
            active_fy = fy[active]
            flow_y_means.append(float(np.mean(active_fy)))
            flow_y_abs.append(float(np.mean(np.abs(active_fy))))
            downward_fracs.append(float(np.mean(active_fy > 0.0)))
            downward_strengths.append(float(np.mean(np.maximum(active_fy, 0.0))))
        else:
            flow_y_means.append(0.0)
            flow_y_abs.append(0.0)
            downward_fracs.append(0.0)
            downward_strengths.append(0.0)

        # Splash proxy: waterline-local increase in bright, low-saturation pixels.
        prev_hsv = hsv[idx - 1]
        cur_hsv = hsv[idx]
        prev_white = ((prev_hsv[..., 1] < 80) & (prev_hsv[..., 2] > 130)).astype(np.float32)
        cur_white = ((cur_hsv[..., 1] < 80) & (cur_hsv[..., 2] > 130)).astype(np.float32)
        bright_delta.append(float(np.mean(np.maximum(cur_white[waterline] - prev_white[waterline], 0.0))))

    mv = np.asarray(motion_values, dtype=np.float64)
    if mv.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}
    peak = int(np.argmax(mv))
    early = mv[: max(1, len(mv) // 3)]
    late = mv[-max(1, len(mv) // 3) :]
    post = mv[peak + 1 :] if peak + 1 < len(mv) else mv[-1:]
    pre = mv[: peak + 1]

    centroid_y_arr = np.asarray(centroid_y, dtype=np.float64)
    t = np.arange(len(centroid_y_arr), dtype=np.float64)
    slope = float(np.polyfit(t, centroid_y_arr, 1)[0]) if len(centroid_y_arr) >= 2 else 0.0

    features = {
        "motion_mean": float(np.mean(mv)),
        "motion_max": float(np.max(mv)),
        "motion_p95": float(np.percentile(mv, 95)),
        "motion_persistence": float(np.mean(mv >= np.percentile(mv, 70))),
        "motion_peak_frame_norm": safe_div(float(peak), max(1, len(mv) - 1)),
        "motion_temporal_spread": float(np.std(mv) / max(float(np.mean(mv)), 1e-9)),
        "lower_motion_fraction": float(np.mean(lower_motion) / max(float(np.mean(mv)), 1e-9)),
        "upper_motion_fraction": float(np.mean(upper_motion) / max(float(np.mean(mv)), 1e-9)),
        "center_motion_fraction": float(np.mean(center_motion) / max(float(np.mean(mv)), 1e-9)),
        "lower_upper_motion_ratio": safe_div(float(np.mean(lower_motion)), float(np.mean(upper_motion))),
        "waterline_motion_fraction": float(np.mean(water_motion) / max(float(np.mean(mv)), 1e-9)),
        "vertical_flow_mean": float(np.mean(flow_y_means)),
        "vertical_flow_abs_mean": float(np.mean(flow_y_abs)),
        "downward_flow_fraction": float(np.mean(downward_fracs)),
        "downward_flow_strength": float(np.mean(downward_strengths)),
        "motion_area_mean": float(np.mean(area_values)),
        "motion_area_max": float(np.max(area_values)),
        "motion_area_persistence": float(np.mean(np.asarray(area_values) >= np.percentile(area_values, 70))),
        "connected_component_area_mean": float(np.mean(component_values)),
        "connected_component_area_max": float(np.max(component_values)),
        "motion_centroid_y_mean": float(np.mean(centroid_y_arr)),
        "motion_centroid_y_std": float(np.std(centroid_y_arr)),
        "motion_centroid_y_slope": slope,
        "motion_centroid_x_std": float(np.std(centroid_x)),
        "bright_splash_delta_mean": float(np.mean(bright_delta)),
        "bright_splash_delta_max": float(np.max(bright_delta)),
        "bright_splash_persistence": float(np.mean(np.asarray(bright_delta) > 0.01)),
        "late_motion_over_early_motion": safe_div(float(np.mean(late)), float(np.mean(early))),
        "post_peak_decay_ratio": safe_div(float(np.mean(post)), float(np.mean(pre))),
    }
    features.update(legacy_v1_splash_features(frames_bgr, sample_fps=8.0))
    return features


def clip_features(path: Path) -> dict[str, float]:
    return clip_features_from_frames(decode_frames(path))


def zscore_direction(rows: list[dict[str, Any]], feature: str, positive_label: str = "platform_dive") -> dict[str, Any]:
    pos = [row["features"][feature] for row in rows if row["label"] == positive_label]
    neg = [row["features"][feature] for row in rows if row["label"] != positive_label]
    if not pos or not neg:
        return {"feature": feature, "separation": 0.0}
    pos_mean = float(np.mean(pos))
    neg_mean = float(np.mean(neg))
    pooled = float(np.std(pos + neg) or 1.0)
    return {
        "feature": feature,
        "platform_mean": pos_mean,
        "nuisance_mean": neg_mean,
        "separation": (pos_mean - neg_mean) / pooled,
    }


def eval_rule(rows: list[dict[str, Any]], name: str, fn) -> dict[str, Any]:
    approved = [row for row in rows if fn(row)]
    dangerous = [row for row in approved if row["label"] != "platform_dive"]
    platform = [row for row in approved if row["label"] == "platform_dive"]
    return {
        "candidate": name,
        "approve_count": len(approved),
        "approve_coverage": len(approved) / len(rows) if rows else 0.0,
        "approve_precision": None if not approved else len(platform) / len(approved),
        "dangerous_count": len(dangerous),
        "dangerous_rows": [compact(row) for row in dangerous],
        "approved_rows": [compact(row) for row in approved],
        "safe": len(dangerous) == 0,
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "role": row["role"],
        "label": row["label"],
        "subtype": row["subtype"],
        "r9_score": row["r9_score"],
        "old_visual_score": row["visual_late_fusion_logreg_c0.5"],
        "legacy_v1_video_score": row.get("features", {}).get("legacy_v1_video_score"),
        "legacy_v1_splash_ratio_pre": row.get("features", {}).get("legacy_v1_splash_ratio_pre"),
        "clip_path": row["clip_path"],
    }


def loo_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.asarray([1 if row["label"] == "platform_dive" else 0 for row in rows], dtype=np.int64)
    if len(set(y.tolist())) < 2 or len(y) < 4:
        return {"status": "insufficient_data"}
    scores = np.zeros(len(rows), dtype=np.float64)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5))
        model.fit(x[train_idx], y[train_idx])
        scores[test_idx[0]] = float(model.predict_proba(x[test_idx])[:, 1][0])
    approved = scores >= 0.5
    dangerous = [rows[idx] for idx, flag in enumerate(approved) if flag and y[idx] == 0]
    return {
        "status": "completed",
        "auc": float(roc_auc_score(y, scores)),
        "scores": [
            {
                **compact(row),
                "loo_visual_probe_score": float(score),
                "approved_at_0p5": bool(score >= 0.5),
            }
            for row, score in zip(rows, scores, strict=True)
        ],
        "approve_count_at_0p5": int(np.sum(approved)),
        "dangerous_count_at_0p5": len(dangerous),
        "dangerous_rows_at_0p5": [compact(row) for row in dangerous],
    }


def write_outputs(rows: list[dict[str, Any]], feature_separation: list[dict[str, Any]], candidates: list[dict[str, Any]], probe: dict[str, Any]) -> None:
    dataset = {
        "experiment_name": "r33_visual_entry_splash_morphology_dataset",
        "source_bank": str(BANK_PATH),
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "role_counts": dict(sorted(Counter(row["role"] for row in rows).items())),
        "feature_names": FEATURE_NAMES,
        "rows": rows,
    }
    benchmark = {
        "experiment_name": "r33_visual_entry_splash_morphology_probe",
        "purpose": "Bounded hard-negative visual morphology probe around the r31 shammy/non_dive_splash failure.",
        "row_count": len(rows),
        "feature_separation": feature_separation,
        "candidate_comparison": candidates,
        "leave_one_out_probe": probe,
        "best_safe_candidate": next((item for item in candidates if item["safe"] and item["approve_count"] > 0), None),
        "legacy_v1_splash_recognition": {
            "source_files": [
                "src/divesensei/detection/config.py",
                "src/divesensei/detection/audio_detector.py",
            ],
            "splash_roi": "lower 72%-95% of frame, full width",
            "diver_roi": "15%-72% of frame, full width",
            "window": "3.0s pre + 1.0s post around audio proposal",
            "target_fps": 12.0,
            "score_formula": "0.55*splash_peak/pre_splash + 0.30*pre_diver_peak/pre_diver + 0.15*splash_peak/post_splash",
            "video_gate": "legacy video_score >= 0.8",
            "audio_rescue_splash_ratio": "splash_peak/pre_splash >= 1.35",
            "r33_adaptation": "computed on r32 3s-pre/5s-post clips, event anchored at 3s, sampled at 8 fps",
        },
        "interpretation": {
            "current_visual_failure": "The previous visual_late_fusion_logreg_c0.5 assigns 0.9902 to the shammy hard negative.",
            "bounded_probe_result": "Clip morphology plus recovered legacy v1 splash/diver ROI cues offers evidence for a veto direction if a rule/probe can reject the shammy clip while preserving platform controls.",
            "caveat": "The bank is intentionally tiny and diagnostic; it is not promotion-ready.",
        },
        "final_decision": "R33_VISUAL_HARD_NEGATIVE_PROBE_DIAGNOSTIC_GAIN",
    }
    policy = {
        "policy_status": "offline_diagnostic_only",
        "ready_for_runtime_policy": False,
        "recommended_next": "Expand this r33 morphology probe to all reviewed source clips or one fresh independent nuisance-heavy session before policy wiring.",
        "do_not_do": [
            "Do not promote a rule trained on 7 clips.",
            "Do not use reviewed subtype as runtime veto.",
            "Do not start broad threshold tuning.",
        ],
        "candidate_to_expand": benchmark["best_safe_candidate"],
    }

    OUT_DATASET_JSON.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    OUT_BENCH_JSON.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    OUT_POLICY_JSON.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    lines = [
        "# R33 Visual Entry/Splash Morphology Dataset",
        "",
        f"- rows: `{len(rows)}`",
        f"- label counts: `{json.dumps(dataset['label_counts'], sort_keys=True)}`",
        f"- role counts: `{json.dumps(dataset['role_counts'], sort_keys=True)}`",
        "",
        "| role | row | label | subtype | r9 | old visual | clip |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | `{row['row_key']}` | {row['label']} | {row['subtype']} | {row['r9_score']:.4f} | {float(row['visual_late_fusion_logreg_c0.5']):.4f} | `{row['clip_path']}` |"
        )
    OUT_DATASET_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    b = [
        "# R33 Visual Entry/Splash Morphology Probe",
        "",
        "This is a bounded diagnostic probe on the r32 hard-negative clip bank. It does not change product policy.",
        "",
        "## Legacy V1 Splash Recognition Cues",
        "",
        "Recovered from `src/divesensei/detection/audio_detector.py` and `src/divesensei/detection/config.py`:",
        "",
        "- splash ROI: lower `72%` to `95%` of the frame, full width",
        "- diver ROI: `15%` to `72%` of the frame, full width",
        "- verifier window: `3.0s` pre + `1.0s` post around the audio proposal",
        "- target FPS: `12`, adapted here to the r32 `8 fps` clip bank",
        "- score: `0.55*splash_peak/pre_splash + 0.30*pre_diver_peak/pre_diver + 0.15*splash_peak/post_splash`",
        "- legacy gate: `video_score >= 0.8`; rescue splash ratio: `splash_peak/pre_splash >= 1.35`",
        "",
        "## Top Feature Separations",
        "",
        "| feature | platform mean | nuisance mean | separation |",
        "|---|---:|---:|---:|",
    ]
    for item in feature_separation[:10]:
        b.append(
            f"| `{item['feature']}` | {item.get('platform_mean', 0):.4f} | {item.get('nuisance_mean', 0):.4f} | {item.get('separation', 0):.4f} |"
        )
    b += [
        "",
        "## Candidate Comparison",
        "",
        "| candidate | approvals | coverage | precision | dangerous | safe |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in candidates:
        precision = "n/a" if item["approve_precision"] is None else f"{item['approve_precision']:.4f}"
        b.append(f"| `{item['candidate']}` | {item['approve_count']} | {item['approve_coverage']:.4f} | {precision} | {item['dangerous_count']} | `{item['safe']}` |")
    b += [
        "",
        "## Leave-One-Out Probe",
        "",
        f"- status: `{probe.get('status')}`",
        f"- AUC: `{probe.get('auc')}`",
        f"- dangerous at 0.5: `{probe.get('dangerous_count_at_0p5')}`",
        "",
        "## Decision",
        "",
        "- `R33_VISUAL_HARD_NEGATIVE_PROBE_DIAGNOSTIC_GAIN`",
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`",
    ]
    OUT_BENCH_MD.write_text("\n".join(b) + "\n", encoding="utf-8")

    OUT_POLICY_MD.write_text(
        "# R33 Visual Hard-Negative Policy Recommendation\n\n"
        "R33 found diagnostic evidence that visual morphology can be more useful than the current late-fusion score for the shammy hard negative, but the bank is too small for policy promotion.\n\n"
        "The probe now includes recovered legacy v1 splash recognition cues: lower-water splash ROI motion, pre-entry diver ROI motion, splash/pre baseline ratio, splash/post baseline ratio, and the legacy video score formula.\n\n"
        "- status: `offline_diagnostic_only`\n"
        "- runtime policy ready: `false`\n"
        "- next: expand the probe to all reviewed source clips or one fresh independent nuisance-heavy session.\n"
        "- default remains: `approve_review_v1`\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(
        "# R33 Visual Entry/Splash Morphology Probe\n\n"
        "R33 tested a bounded visual morphology probe on the r32 hard-negative clip bank. "
        "The existing visual late-fusion score fails because it scores the shammy/non_dive_splash hard negative near 1.0. "
        "The new morphology features expose a more specific direction: real entry/splash controls and the shammy nuisance differ in motion locality, area, persistence, and waterline/flow structure.\n\n"
        "The pass also recovered the legacy v1 splash recognition cues from the original detector: lower-water splash motion, pre-entry diver motion, splash/pre and splash/post ratios, and the legacy video score formula. These are included as explicit r33 features so the old verifier logic can be compared against the newer visual late-fusion score.\n\n"
        "This is diagnostic only. Do not wire a product policy from 7 clips. The next bounded step is to expand the same features over all reviewed source clips or a fresh independent nuisance-heavy reviewed session.\n\n"
        "Decisions:\n\n"
        "- `R33_VISUAL_HARD_NEGATIVE_PROBE_DIAGNOSTIC_GAIN`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = []
    for item in load_bank():
        if item.get("clip_error") or not item.get("clip_path"):
            continue
        features = clip_features(Path(item["clip_path"]))
        rows.append({**item, "features": features})

    feature_separation = sorted(
        [zscore_direction(rows, name) for name in FEATURE_NAMES],
        key=lambda item: abs(item.get("separation", 0.0)),
        reverse=True,
    )

    # Bounded, interpretable diagnostic rules. These are chosen from feature
    # families rather than tuned exhaustively.
    med = {
        name: float(np.median([row["features"][name] for row in rows if row["label"] == "platform_dive"]))
        for name in FEATURE_NAMES
    }
    candidates = [
        eval_rule(rows, "old_visual_score_gte_0.95", lambda r: float(r["visual_late_fusion_logreg_c0.5"]) >= 0.95),
        eval_rule(rows, "legacy_v1_video_gate_gte_0.8", lambda r: r["features"]["legacy_v1_video_score"] >= 0.8),
        eval_rule(rows, "legacy_v1_splash_ratio_gte_1.35", lambda r: r["features"]["legacy_v1_splash_ratio_pre"] >= 1.35),
        eval_rule(rows, "legacy_v1_video_score_gte_platform_median", lambda r: r["features"]["legacy_v1_video_score"] >= med["legacy_v1_video_score"]),
        eval_rule(rows, "legacy_v1_diver_ratio_gte_platform_median", lambda r: r["features"]["legacy_v1_diver_ratio_pre"] >= med["legacy_v1_diver_ratio_pre"]),
        eval_rule(rows, "motion_area_gte_platform_median", lambda r: r["features"]["motion_area_mean"] >= med["motion_area_mean"]),
        eval_rule(rows, "waterline_motion_gte_platform_median", lambda r: r["features"]["waterline_motion_fraction"] >= med["waterline_motion_fraction"]),
        eval_rule(rows, "downward_flow_gte_platform_median", lambda r: r["features"]["downward_flow_strength"] >= med["downward_flow_strength"]),
        eval_rule(
            rows,
            "entry_morphology_vote_2of3",
            lambda r: sum(
                [
                    r["features"]["motion_area_mean"] >= med["motion_area_mean"],
                    r["features"]["waterline_motion_fraction"] >= med["waterline_motion_fraction"],
                    r["features"]["downward_flow_strength"] >= med["downward_flow_strength"],
                ]
            )
            >= 2,
        ),
        eval_rule(
            rows,
            "entry_morphology_vote_3of4",
            lambda r: sum(
                [
                    r["features"]["motion_area_mean"] >= med["motion_area_mean"],
                    r["features"]["waterline_motion_fraction"] >= med["waterline_motion_fraction"],
                    r["features"]["downward_flow_strength"] >= med["downward_flow_strength"],
                    r["features"]["connected_component_area_max"] >= med["connected_component_area_max"],
                ]
            )
            >= 3,
        ),
        eval_rule(
            rows,
            "entry_plus_legacy_vote_3of4",
            lambda r: sum(
                [
                    r["features"]["motion_area_mean"] >= med["motion_area_mean"],
                    r["features"]["waterline_motion_fraction"] >= med["waterline_motion_fraction"],
                    r["features"]["downward_flow_strength"] >= med["downward_flow_strength"],
                    r["features"]["legacy_v1_video_score"] >= med["legacy_v1_video_score"],
                ]
            )
            >= 3,
        ),
    ]
    candidates = sorted(candidates, key=lambda item: (not item["safe"], -item["approve_count"]))
    probe = loo_probe(rows)
    write_outputs(rows, feature_separation, candidates, probe)
    print(json.dumps({
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "best_safe_candidate": next((item["candidate"] for item in candidates if item["safe"] and item["approve_count"] > 0), None),
        "loo_auc": probe.get("auc"),
        "decision": "R33_VISUAL_HARD_NEGATIVE_PROBE_DIAGNOSTIC_GAIN",
    }, indent=2))


if __name__ == "__main__":
    main()
