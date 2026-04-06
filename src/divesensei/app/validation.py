#!/usr/bin/env python3
"""
Dataset-driven validation runner for baseline and advanced dive detectors.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

from divesensei.detection.audio_detector import AudioVisualDiveDetector
from divesensei.detection.config import DetectionConfig
from divesensei.io.logging_utils import StructuredLogger
from divesensei.profiles import apply_named_profile


DEFAULT_DATASET_ROOTS = [
    Path("/Volumes/Videos/Eindhoven 2026"),
    Path("/srv/nas/video/eindhoven"),
    Path("/srv/nas/video/Eindhoven 2026"),
    Path("/srv/nas/videos/Eindhoven 2026"),
]


def candidate_dataset_roots() -> List[Path]:
    roots: List[Path] = []
    env_roots = os.environ.get("DIVESENSEI_DATASET_ROOTS", "")
    for raw_root in env_roots.split(os.pathsep):
        raw_root = raw_root.strip()
        if raw_root:
            roots.append(Path(raw_root))
    roots.extend(DEFAULT_DATASET_ROOTS)
    return roots


def resolve_video_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    basename = path.name
    for root in candidate_dataset_roots():
        candidate = root / basename
        if candidate.exists():
            return candidate
    return path


def build_config(entry: Dict[str, Any], detector: Dict[str, Any]) -> DetectionConfig:
    merged_entry = {**entry, **detector.get("config", {})}
    profile = str(merged_entry.get("profile", "reviewed"))
    defaults = apply_named_profile(
        {
            "audio_peak_threshold": 4.0,
            "audio_min_score": 4.5,
            "audio_pattern_min_score": 0.4,
            "audio_peak_separation": 4.0,
            "audio_visual_merge_seconds": 2.0,
            "audio_clip_model_min_probability": 0.5,
            "audio_decode_timeout_seconds": 20.0,
        },
        profile,
        str(detector["id"]),
    )
    return DetectionConfig(
        detector_id=str(detector["id"]),
        method="audio_visual",
        splash_zone_top_norm=merged_entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[0],
        splash_zone_bottom_norm=merged_entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[1],
        splash_zone_left_norm=merged_entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[2],
        splash_zone_right_norm=merged_entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[3],
        min_extraction_score=float(merged_entry.get("min_extraction_score", 15.0)),
        high_confidence_threshold=float(merged_entry.get("high_confidence_threshold", 20.0)),
        audio_peak_threshold=float(merged_entry.get("audio_peak_threshold", defaults["audio_peak_threshold"])),
        audio_peak_min_separation_seconds=float(merged_entry.get("audio_peak_separation", defaults["audio_peak_separation"])),
        audio_ignore_before_seconds=float(merged_entry.get("audio_ignore_before_seconds", 0.35)),
        audio_min_score=float(merged_entry.get("audio_min_score", defaults["audio_min_score"])),
        audio_min_hf_ratio=float(merged_entry.get("audio_min_hf_ratio", 0.115)),
        audio_early_peak_score=float(merged_entry.get("audio_early_peak_score", 4.0)),
        audio_early_peak_max_seconds=float(merged_entry.get("audio_early_peak_max_seconds", 0.8)),
        audio_early_peak_max_hf_ratio=float(merged_entry.get("audio_early_peak_max_hf_ratio", 0.6)),
        audio_early_peak_max_centroid_hz=float(merged_entry.get("audio_early_peak_max_centroid_hz", 2200.0)),
        audio_early_peak_max_flatness=float(merged_entry.get("audio_early_peak_max_flatness", 0.45)),
        audio_pattern_min_score=float(merged_entry.get("audio_pattern_min_score", defaults["audio_pattern_min_score"])),
        audio_noise_max_peak_count=int(merged_entry.get("audio_noise_max_peak_count", 5)),
        audio_noise_max_top_ratio=float(merged_entry.get("audio_noise_max_top_ratio", 1.8)),
        audio_long_session_seconds=float(merged_entry.get("audio_long_session_seconds", 120.0)),
        audio_long_session_max_candidates=int(merged_entry.get("audio_long_session_max_candidates", 120)),
        audio_model_path=str(merged_entry.get("audio_model_path", "")),
        audio_model_min_probability=float(merged_entry.get("audio_model_min_probability", 0.0)),
        audio_clip_model_path=str(merged_entry.get("audio_clip_model_path", "")),
        audio_clip_model_min_probability=float(merged_entry.get("audio_clip_model_min_probability", defaults["audio_clip_model_min_probability"])),
        audio_clip_classifier_window_seconds=float(merged_entry.get("audio_clip_classifier_window_seconds", 3.0)),
        audio_clip_classifier_ambiguity_low=float(merged_entry.get("audio_clip_classifier_ambiguity_low", 0.35)),
        audio_clip_classifier_ambiguity_high=float(merged_entry.get("audio_clip_classifier_ambiguity_high", 0.65)),
        audio_pcen_threshold=float(merged_entry.get("audio_pcen_threshold", 2.4)),
        audio_pcen_merge_weight=float(merged_entry.get("audio_pcen_merge_weight", 0.65)),
        audio_duplicate_suppress_window_seconds=float(merged_entry.get("audio_duplicate_suppress_window_seconds", 0.9)),
        audio_duplicate_leader_min_score=float(merged_entry.get("audio_duplicate_leader_min_score", 12.0)),
        audio_duplicate_leader_min_prominence=float(merged_entry.get("audio_duplicate_leader_min_prominence", 10.0)),
        audio_duplicate_follower_max_score_ratio=float(merged_entry.get("audio_duplicate_follower_max_score_ratio", 0.55)),
        audio_decode_timeout_seconds=float(merged_entry.get("audio_decode_timeout_seconds", defaults["audio_decode_timeout_seconds"])),
        ffmpeg_threads=int(merged_entry.get("ffmpeg_threads", 0)),
        opencv_threads=int(merged_entry.get("opencv_threads", 0)),
        audio_only_pre_seconds=float(merged_entry.get("audio_only_pre_seconds", 6.0)),
        audio_only_post_seconds=float(merged_entry.get("audio_only_post_seconds", 3.0)),
        audio_visual_verify_pre_seconds=float(merged_entry.get("audio_verify_pre", 3.0)),
        audio_visual_verify_post_seconds=float(merged_entry.get("audio_verify_post", 1.0)),
        audio_visual_skip_video_verification=bool(merged_entry.get("audio_visual_skip_video_verification", detector["id"] != "audio_v2_hybrid_video")),
        audio_visual_verify_target_fps=float(merged_entry.get("audio_visual_verify_target_fps", 12.0)),
        audio_visual_max_proposals=int(merged_entry.get("audio_visual_max_proposals", 4)),
        audio_visual_min_video_score=float(merged_entry.get("audio_visual_min_video_score", 0.8)),
        audio_visual_hard_video_floor=float(merged_entry.get("audio_visual_hard_video_floor", 0.2)),
        audio_visual_audio_rescue_score=float(merged_entry.get("audio_visual_audio_rescue_score", 4.0)),
        audio_visual_rescue_splash_ratio=float(merged_entry.get("audio_visual_rescue_splash_ratio", 1.35)),
        audio_visual_min_combined_score=float(merged_entry.get("audio_visual_min_combined_score", 3.8)),
        audio_visual_merge_seconds=float(merged_entry.get("audio_visual_merge_seconds", defaults["audio_visual_merge_seconds"])),
        audio_visual_max_verify_width=int(merged_entry.get("audio_visual_max_verify_width", 640)),
        audio_priority_weight=float(merged_entry.get("audio_priority_weight", 0.85)),
        enable_debug_plots=False,
    )


def match_timestamps(predicted: List[float], expected: List[float], tolerance_seconds: float) -> Dict[str, Any]:
    unmatched_pred = set(range(len(predicted)))
    unmatched_exp = set(range(len(expected)))
    matches = []
    pairs = []
    for exp_idx, exp_ts in enumerate(expected):
        for pred_idx, pred_ts in enumerate(predicted):
            delta = abs(pred_ts - exp_ts)
            if delta <= tolerance_seconds:
                pairs.append((delta, exp_idx, pred_idx))
    pairs.sort(key=lambda item: item[0])
    for delta, exp_idx, pred_idx in pairs:
        if exp_idx not in unmatched_exp or pred_idx not in unmatched_pred:
            continue
        unmatched_exp.remove(exp_idx)
        unmatched_pred.remove(pred_idx)
        matches.append(
            {
                "expected_timestamp": expected[exp_idx],
                "predicted_timestamp": predicted[pred_idx],
                "delta_seconds": delta,
            }
        )
    return {
        "matches": matches,
        "matched_count": len(matches),
        "unmatched_expected": [expected[idx] for idx in sorted(unmatched_exp)],
        "unmatched_predicted": [predicted[idx] for idx in sorted(unmatched_pred)],
    }


def evaluate_entry(entry: Dict[str, Any], detector: Dict[str, Any]) -> Dict[str, Any]:
    path = resolve_video_path(entry["path"])
    base_result: Dict[str, Any] = {
        "path": entry["path"],
        "resolved_path": str(path),
        "label": entry.get("label", "unknown"),
        "detector_id": detector["id"],
        "detector_label": detector.get("label", detector["id"]),
        "notes": entry.get("notes"),
    }
    if not path.exists():
        return {
            **base_result,
            "exists": False,
            "passed": False,
            "error": "file not found",
        }

    start = time.time()
    detector_instance = AudioVisualDiveDetector(build_config(entry, detector))
    try:
        candidates = detector_instance.detect(str(path))
    except Exception as exc:
        return {
            **base_result,
            "exists": True,
            "passed": False,
            "error": str(exc),
        }
    runtime_seconds = time.time() - start
    predicted = [float(candidate.timestamp) for candidate in candidates]
    result: Dict[str, Any] = {
        **base_result,
        "exists": True,
        "runtime_seconds": runtime_seconds,
        "actual_event_count": len(candidates),
        "confidence_counts": {
            "high": sum(1 for c in candidates if c.confidence == "high"),
            "medium": sum(1 for c in candidates if c.confidence == "medium"),
            "low": sum(1 for c in candidates if c.confidence == "low"),
        },
        "predicted_timestamps": predicted,
        "candidates": [asdict(candidate) for candidate in candidates],
    }

    expected_events = [float(item) for item in entry.get("expected_events", [])]
    forbidden_events = [float(item) for item in entry.get("forbidden_events", [])]
    tolerance_seconds = float(entry.get("tolerance_seconds", 0.75))
    if expected_events:
        matching = match_timestamps(predicted, expected_events, tolerance_seconds)
        forbidden_matching = match_timestamps(predicted, forbidden_events, tolerance_seconds) if forbidden_events else {"matches": []}
        false_positives = len(matching["unmatched_predicted"])
        max_false_positives = int(entry.get("max_false_positives", 0))
        passed = (
            matching["matched_count"] == len(expected_events)
            and false_positives <= max_false_positives
            and len(forbidden_matching["matches"]) == 0
        )
        result.update(
            {
                "passed": passed,
                "expected_events": expected_events,
                "forbidden_events": forbidden_events,
                "tolerance_seconds": tolerance_seconds,
                "matching": matching,
                "forbidden_hits": forbidden_matching["matches"],
                "true_positive_count": matching["matched_count"],
                "false_positive_count": false_positives,
                "false_negative_count": len(matching["unmatched_expected"]),
                "precision": (matching["matched_count"] / len(predicted)) if predicted else 0.0,
                "recall": matching["matched_count"] / len(expected_events),
            }
        )
        return result

    min_events = int(entry.get("min_events", 0))
    max_events = int(entry.get("max_events", 999999))
    result.update(
        {
            "passed": min_events <= len(candidates) <= max_events,
            "expected": {
                "min_events": min_events,
                "max_events": max_events,
            },
        }
    )
    return result


def summarise_detector(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing = [r for r in results if r.get("exists")]
    passed = [r for r in existing if r.get("passed")]
    timestamp_cases = [r for r in existing if "true_positive_count" in r]
    tp = sum(int(r.get("true_positive_count", 0)) for r in timestamp_cases)
    fp = sum(int(r.get("false_positive_count", 0)) for r in timestamp_cases)
    fn = sum(int(r.get("false_negative_count", 0)) for r in timestamp_cases)
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    return {
        "case_count": len(results),
        "existing_cases": len(existing),
        "pass_rate": (len(passed) / len(existing)) if existing else None,
        "timestamp_case_count": len(timestamp_cases),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "precision": precision,
        "recall": recall,
        "mean_detected_events": mean(r["actual_event_count"] for r in existing) if existing else 0.0,
        "mean_runtime_seconds": mean(r["runtime_seconds"] for r in existing) if existing else 0.0,
    }


def write_markdown_report(manifest_path: Path, summaries: Dict[str, Dict[str, Any]], results: List[Dict[str, Any]]) -> Path:
    report_path = manifest_path.with_suffix(".report.md")
    lines = [
        "# DiveSensei Detector Validation",
        "",
        f"Manifest: `{manifest_path}`",
        "",
        "## Detector Summary",
        "",
    ]
    for detector_id, summary in summaries.items():
        lines.extend(
            [
                f"### `{detector_id}`",
                "",
                f"- Existing cases: {summary['existing_cases']}",
                f"- Pass rate: {summary['pass_rate'] if summary['pass_rate'] is not None else 'n/a'}",
                f"- Precision: {summary['precision'] if summary['precision'] is not None else 'n/a'}",
                f"- Recall: {summary['recall'] if summary['recall'] is not None else 'n/a'}",
                f"- Mean runtime seconds: {summary['mean_runtime_seconds']:.3f}",
                "",
            ]
        )
    lines.extend(["## Cases", ""])
    for result in results:
        lines.extend(
            [
                f"### {'PASS' if result.get('passed') else 'FAIL'} - `{Path(result['path']).name}` [{result['detector_id']}]",
                "",
                f"- Detected events: `{result.get('actual_event_count', 'n/a')}`",
                f"- Runtime seconds: `{result.get('runtime_seconds', 'n/a')}`",
                f"- Predicted timestamps: `{result.get('predicted_timestamps', [])}`",
                f"- Notes: {result.get('notes') or 'n/a'}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines))
    return report_path


def manifest_detectors(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    detectors = manifest.get("detectors")
    if detectors:
        return detectors
    return [{"id": "audio_v1_heuristic", "label": "Baseline heuristic", "config": {}}]


def manifest_cases(manifest: Dict[str, Any], manifest_path: Path) -> List[Dict[str, Any]]:
    cases = manifest.get("cases")
    if cases:
        return cases
    source_manifest = str(manifest.get("source_manifest", "")).strip()
    if not source_manifest:
        return []
    source_path = (manifest_path.parent / source_manifest).resolve()
    source = json.loads(source_path.read_text())
    return list(source.get("cases", []))


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python -m divesensei.app.validation <manifest.json>")
        return 1

    manifest_path = Path(argv[0]).resolve()
    manifest = json.loads(manifest_path.read_text())
    detectors = manifest_detectors(manifest)
    cases = manifest_cases(manifest, manifest_path)
    logger = StructuredLogger(manifest_path.with_suffix(".runlog.jsonl"))
    logger.log("validation_start", manifest=manifest_path, total_cases=len(cases), detector_count=len(detectors))

    results: List[Dict[str, Any]] = []
    for detector in detectors:
        for idx, case in enumerate(cases, start=1):
            result = evaluate_entry(case, detector)
            results.append(result)
            logger.log(
                "case_result",
                index=idx,
                total_cases=len(cases),
                path=result["path"],
                detector_id=result["detector_id"],
                passed=result.get("passed"),
                actual_event_count=result.get("actual_event_count"),
                runtime_seconds=result.get("runtime_seconds"),
                error=result.get("error"),
            )
            print(
                f"[{result['detector_id']}] [{idx}/{len(cases)}] {Path(result['path']).name}: "
                f"detected={result.get('actual_event_count', 'n/a')} passed={result.get('passed')}"
            )

    summaries = {
        detector["id"]: summarise_detector([result for result in results if result["detector_id"] == detector["id"]])
        for detector in detectors
    }
    output = {
        "manifest": str(manifest_path),
        "detectors": detectors,
        "summaries": summaries,
        "results": results,
    }
    json_path = manifest_path.with_suffix(".results.json")
    json_path.write_text(json.dumps(output, indent=2))
    report_path = write_markdown_report(manifest_path, summaries, results)
    logger.log("validation_complete", summaries=summaries, json_report=json_path, markdown_report=report_path)
    print(json.dumps({"summaries": summaries, "json_report": str(json_path), "markdown_report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
