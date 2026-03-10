#!/usr/bin/env python3
"""
Dataset-driven validation runner for the production dive detector.

Usage:
  python -m divesensei.app.validation benchmarks/manifests/reviewed_audio.json
"""

from __future__ import annotations

import json
import os
import sys
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


def build_config(entry: Dict[str, Any]) -> DetectionConfig:
    defaults = apply_named_profile(
        {
            "audio_min_score": 4.5,
            "audio_pattern_min_score": 0.4,
            "audio_decode_timeout_seconds": 20.0,
        },
        str(entry.get("profile", "reviewed")),
    )
    config = DetectionConfig(
        method="audio_visual",
        splash_zone_top_norm=entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[0],
        splash_zone_bottom_norm=entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[1],
        splash_zone_left_norm=entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[2],
        splash_zone_right_norm=entry.get("bbox", [0.72, 0.95, 0.0, 1.0])[3],
        min_extraction_score=float(entry.get("min_extraction_score", 15.0)),
        high_confidence_threshold=float(entry.get("high_confidence_threshold", 20.0)),
        audio_peak_threshold=float(entry.get("audio_peak_threshold", 4.0)),
        audio_peak_min_separation_seconds=float(entry.get("audio_peak_separation", 4.0)),
        audio_ignore_before_seconds=float(entry.get("audio_ignore_before_seconds", 0.35)),
        audio_min_score=float(entry.get("audio_min_score", defaults["audio_min_score"])),
        audio_min_hf_ratio=float(entry.get("audio_min_hf_ratio", 0.115)),
        audio_early_peak_score=float(entry.get("audio_early_peak_score", 4.0)),
        audio_early_peak_max_seconds=float(entry.get("audio_early_peak_max_seconds", 0.8)),
        audio_early_peak_max_hf_ratio=float(entry.get("audio_early_peak_max_hf_ratio", 0.6)),
        audio_early_peak_max_centroid_hz=float(entry.get("audio_early_peak_max_centroid_hz", 2200.0)),
        audio_early_peak_max_flatness=float(entry.get("audio_early_peak_max_flatness", 0.45)),
        audio_pattern_min_score=float(entry.get("audio_pattern_min_score", defaults["audio_pattern_min_score"])),
        audio_noise_max_peak_count=int(entry.get("audio_noise_max_peak_count", 5)),
        audio_noise_max_top_ratio=float(entry.get("audio_noise_max_top_ratio", 1.8)),
        audio_long_session_seconds=float(entry.get("audio_long_session_seconds", 120.0)),
        audio_long_session_max_candidates=int(entry.get("audio_long_session_max_candidates", 120)),
        audio_model_path=str(entry.get("audio_model_path", "")),
        audio_model_min_probability=float(entry.get("audio_model_min_probability", 0.0)),
        audio_decode_timeout_seconds=float(entry.get("audio_decode_timeout_seconds", defaults["audio_decode_timeout_seconds"])),
        ffmpeg_threads=int(entry.get("ffmpeg_threads", 1)),
        opencv_threads=int(entry.get("opencv_threads", 1)),
        audio_only_pre_seconds=float(entry.get("audio_only_pre_seconds", 3.0)),
        audio_only_post_seconds=float(entry.get("audio_only_post_seconds", 1.0)),
        audio_visual_verify_pre_seconds=float(entry.get("audio_verify_pre", 3.0)),
        audio_visual_verify_post_seconds=float(entry.get("audio_verify_post", 1.0)),
        audio_visual_skip_video_verification=bool(entry.get("audio_visual_skip_video_verification", False)),
        audio_visual_verify_target_fps=float(entry.get("audio_visual_verify_target_fps", 12.0)),
        audio_visual_max_proposals=int(entry.get("audio_visual_max_proposals", 4)),
        audio_visual_min_video_score=float(entry.get("audio_visual_min_video_score", 0.8)),
        audio_visual_hard_video_floor=float(entry.get("audio_visual_hard_video_floor", 0.2)),
        audio_visual_audio_rescue_score=float(entry.get("audio_visual_audio_rescue_score", 4.0)),
        audio_visual_rescue_splash_ratio=float(entry.get("audio_visual_rescue_splash_ratio", 1.35)),
        audio_visual_min_combined_score=float(entry.get("audio_visual_min_combined_score", 3.8)),
        audio_visual_max_verify_width=int(entry.get("audio_visual_max_verify_width", 640)),
        audio_priority_weight=float(entry.get("audio_priority_weight", 0.85)),
        enable_debug_plots=False,
    )
    return config


def evaluate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    path = resolve_video_path(entry["path"])
    if not path.exists():
        return {
            "path": entry["path"],
            "resolved_path": str(path),
            "exists": False,
            "passed": False,
            "error": "file not found",
            "label": entry.get("label", "unknown"),
        }

    detector = AudioVisualDiveDetector(build_config(entry))
    try:
        candidates = detector.detect(str(path))
    except Exception as exc:
        return {
            "path": entry["path"],
            "resolved_path": str(path),
            "exists": True,
            "passed": False,
            "error": str(exc),
            "label": entry.get("label", "unknown"),
            "notes": entry.get("notes"),
        }

    label = entry.get("label", "unknown")
    min_events = int(entry.get("min_events", 0))
    max_events = int(entry.get("max_events", 999999))
    passed = min_events <= len(candidates) <= max_events

    return {
        "path": entry["path"],
        "resolved_path": str(path),
        "exists": True,
        "label": label,
        "passed": passed,
        "expected": {
            "min_events": min_events,
            "max_events": max_events,
        },
        "actual_event_count": len(candidates),
        "confidence_counts": {
            "high": sum(1 for c in candidates if c.confidence == "high"),
            "medium": sum(1 for c in candidates if c.confidence == "medium"),
            "low": sum(1 for c in candidates if c.confidence == "low"),
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "notes": entry.get("notes"),
    }


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing = [r for r in results if r.get("exists")]
    labeled = [r for r in existing if r.get("label") in {"positive", "negative"}]
    positives = [r for r in labeled if r["label"] == "positive"]
    negatives = [r for r in labeled if r["label"] == "negative"]

    return {
        "total_cases": len(results),
        "existing_cases": len(existing),
        "labeled_cases": len(labeled),
        "pass_rate": (sum(1 for r in labeled if r["passed"]) / len(labeled)) if labeled else None,
        "positive_recall_proxy": (sum(1 for r in positives if r["passed"]) / len(positives)) if positives else None,
        "negative_rejection_proxy": (sum(1 for r in negatives if r["passed"]) / len(negatives)) if negatives else None,
        "mean_detected_events": mean(r["actual_event_count"] for r in existing) if existing else 0.0,
    }


def write_markdown_report(manifest_path: Path, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> Path:
    report_path = manifest_path.with_suffix(".report.md")
    lines = [
        "# Audio-Visual Dive Detector Validation",
        "",
        f"Manifest: `{manifest_path}`",
        "",
        "## Summary",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Existing cases: {summary['existing_cases']}",
        f"- Labeled cases: {summary['labeled_cases']}",
        f"- Pass rate: {summary['pass_rate'] if summary['pass_rate'] is not None else 'n/a'}",
        f"- Positive recall proxy: {summary['positive_recall_proxy'] if summary['positive_recall_proxy'] is not None else 'n/a'}",
        f"- Negative rejection proxy: {summary['negative_rejection_proxy'] if summary['negative_rejection_proxy'] is not None else 'n/a'}",
        "",
        "## Cases",
        "",
    ]

    for result in results:
        status = "PASS" if result.get("passed") else "FAIL"
        lines.extend(
            [
                f"### {status} - `{Path(result['path']).name}`",
                "",
                f"- Label: `{result.get('label')}`",
                f"- Exists: `{result.get('exists')}`",
                f"- Detected events: `{result.get('actual_event_count', 'n/a')}`",
                f"- Expected range: `{result.get('expected', {}).get('min_events', 'n/a')}..{result.get('expected', {}).get('max_events', 'n/a')}`",
                f"- Notes: {result.get('notes') or 'n/a'}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines))
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python -m divesensei.app.validation <manifest.json>")
        return 1

    manifest_path = Path(argv[0]).resolve()
    manifest = json.loads(manifest_path.read_text())
    logger = StructuredLogger(manifest_path.with_suffix(".runlog.jsonl"))
    cases = manifest.get("cases", [])
    results = []
    total_cases = len(cases)
    logger.log("validation_start", manifest=manifest_path, total_cases=total_cases)
    for idx, case in enumerate(cases, start=1):
        result = evaluate_entry(case)
        results.append(result)
        logger.log(
            "case_result",
            index=idx,
            total_cases=total_cases,
            path=result["path"],
            resolved_path=result.get("resolved_path"),
            label=result.get("label"),
            passed=result.get("passed"),
            actual_event_count=result.get("actual_event_count"),
            error=result.get("error"),
        )
        print(
            f"[{idx}/{total_cases}] {Path(result['path']).name}: "
            f"detected={result.get('actual_event_count', 'n/a')} "
            f"passed={result.get('passed')} label={result.get('label')}"
        )
    summary = summarise(results)

    output = {
        "manifest": str(manifest_path),
        "summary": summary,
        "results": results,
    }
    json_path = manifest_path.with_suffix(".results.json")
    json_path.write_text(json.dumps(output, indent=2))
    report_path = write_markdown_report(manifest_path, results, summary)
    logger.log("validation_complete", summary=summary, json_report=json_path, markdown_report=report_path)

    print(json.dumps({"summary": summary, "json_report": str(json_path), "markdown_report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
