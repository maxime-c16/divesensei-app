#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from divesensei.app.session_pipeline import (
    apply_profile_overrides,
    build_config,
    print_human_summary,
    write_candidates_csv,
    write_session_outputs,
)
from divesensei.io.logging_utils import StructuredLogger, build_candidate_debug_summary
from divesensei.io.media_io import extract_audio_wav_ffmpeg, generate_review_proxy_ffmpeg, probe_media_duration_seconds
from divesensei.workflows.evaluation_session_support import build_proposal_diagnostics, write_json, write_jsonl
from divesensei.workflows.runtime_score_paths import enrich_candidates_with_runtime_scores


def default_output_dir(video_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "outputs" / f"evaluation_{video_path.stem}_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    from divesensei.app.session_pipeline import build_parser as build_detect_parser

    parser = build_detect_parser()
    parser.prog = "divesensei evaluate-session"
    parser.description = "Prepare an audio-first evaluation session with cached audio and a review proxy."
    parser.add_argument(
        "--prepared-audio-path",
        default="",
        help="Optional pre-extracted mono 16k PCM WAV file. If omitted, the workflow extracts it once with ffmpeg.",
    )
    parser.add_argument(
        "--audio-output-name",
        default="session_audio.wav",
        help=argparse.SUPPRESS,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    raw_argv = list(argv or [])
    args = parser.parse_args(raw_argv)
    if not args.output_dir:
        args.output_dir = str(default_output_dir(Path(args.video_path).resolve()))
    args.review_only = True
    args.no_extract = True
    if args.with_video_verification:
        args.skip_video_verification = False
    if args.detector_id == "audio_v2_hybrid_video" and "--skip-video-verification" not in raw_argv:
        args.skip_video_verification = False
    explicit_flags = {item for item in raw_argv if item.startswith("--")}
    return apply_profile_overrides(args, explicit_flags)


def build_report(
    *,
    video_path: Path,
    audio_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    config: Any,
    candidates: Sequence[Any],
    detect_seconds: float,
    audio_extract_seconds: float,
    review_proxy_path: Path,
    review_proxy_status: str,
    review_proxy_error: str | None,
) -> dict[str, Any]:
    debug_summary = build_candidate_debug_summary(candidates)
    return {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "profile": args.profile,
        "detector_id": args.detector_id,
        "session_mode": "evaluation",
        "session_created_at": datetime.now(timezone.utc).isoformat(),
        "session_name": (args.session_name.strip() if getattr(args, "session_name", "") else "") or f"{video_path.stem} evaluation",
        "detector_seconds": detect_seconds,
        "audio_extract_seconds": audio_extract_seconds,
        "config": config.__dict__,
        "candidate_count": len(candidates),
        "session_estimated_duration_seconds": probe_media_duration_seconds(video_path) or max((candidate.end_time for candidate in candidates), default=0.0),
        "debug_summary": debug_summary,
        "candidates": [asdict(candidate) for candidate in candidates],
        "source_audio_path": str(audio_path),
        "review_proxy_path": str(review_proxy_path) if review_proxy_status == "ready" else None,
        "review_proxy_status": review_proxy_status,
        "review_proxy_error": review_proxy_error,
        "evaluation_review_path": str(output_dir / "evaluation_review.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(output_dir / "session_pipeline.log.jsonl")
    logger.log("session_start", video_path=video_path, output_dir=output_dir, profile=args.profile, session_mode="evaluation")

    prepared_audio_path = Path(args.prepared_audio_path).resolve() if str(args.prepared_audio_path).strip() else output_dir / args.audio_output_name
    if str(args.prepared_audio_path).strip() and not prepared_audio_path.exists():
        print(f"Prepared audio not found: {prepared_audio_path}")
        return 1

    audio_extract_started = time.time()
    if prepared_audio_path.exists() and not str(args.prepared_audio_path).strip():
        logger.log("audio_extract_reused", audio_path=prepared_audio_path)
    elif str(args.prepared_audio_path).strip():
        logger.log("audio_extract_reused", audio_path=prepared_audio_path, source="user")
    else:
        logger.log("audio_extract_start", output_path=prepared_audio_path)
        extract_audio_wav_ffmpeg(
            video_path=video_path,
            output_path=prepared_audio_path,
            sample_rate=16000,
            ffmpeg_threads=int(args.ffmpeg_threads),
        )
        logger.log("audio_extract_complete", output_path=prepared_audio_path, elapsed_seconds=round(time.time() - audio_extract_started, 3))
    audio_extract_seconds = time.time() - audio_extract_started

    config = build_config(args)
    from divesensei.detection.audio_detector import AudioVisualDiveDetector

    detector = AudioVisualDiveDetector(
        config,
        progress_callback=lambda payload: logger.log(
            str(payload["event"]),
            **{key: value for key, value in payload.items() if key != "event"},
        ),
    )

    detect_started = time.time()
    logger.log("detection_start", profile=args.profile, audio_path=prepared_audio_path, source_video_path=video_path)
    candidates = detector.detect_from_audio_file(str(prepared_audio_path), video_path=str(video_path))
    detect_seconds = time.time() - detect_started
    runtime_score_enrichment = enrich_candidates_with_runtime_scores(candidates=list(candidates), source_video_path=video_path)
    logger.log(
        "detection_complete",
        detector_seconds=detect_seconds,
        candidate_count=len(candidates),
        debug_summary=build_candidate_debug_summary(candidates),
        runtime_score_enrichment=runtime_score_enrichment,
    )

    csv_path = write_candidates_csv(output_dir, candidates)
    review_proxy_path = output_dir / "web" / "session_source_review.mp4"
    review_proxy_status = "skipped" if args.skip_review_proxy else "pending"
    review_proxy_error = None

    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    evaluation_review_path = output_dir / "evaluation_review.json"
    if not evaluation_review_path.exists():
        evaluation_review_path.write_text(json.dumps({"schemaVersion": "1.0.0", "decisions": [], "falseNegatives": []}, indent=2))
    report = build_report(
        video_path=video_path,
        audio_path=prepared_audio_path,
        output_dir=output_dir,
        args=args,
        config=config,
        candidates=candidates,
        detect_seconds=detect_seconds,
        audio_extract_seconds=audio_extract_seconds,
        review_proxy_path=review_proxy_path,
        review_proxy_status=review_proxy_status,
        review_proxy_error=review_proxy_error,
    )
    report["detections_csv"] = str(csv_path)
    report["runtime_score_enrichment"] = runtime_score_enrichment
    report["extract_seconds"] = 0.0
    report["peak_rss_kb"] = peak_rss_kb
    report["manifest_ready_seconds"] = time.time() - audio_extract_started
    report["total_runtime_seconds"] = report["manifest_ready_seconds"]
    report["extraction_error_count"] = 0
    report["extraction_errors"] = []
    report["extracted_paths"] = []

    report_path, ui_manifest_path = write_session_outputs(
        video_path=video_path,
        output_dir=output_dir,
        profile=args.profile,
        report=report,
        candidates=candidates,
        extracted_paths=[],
        status_override="evaluation_ready",
        session_mode="evaluation",
        source_audio_path=str(prepared_audio_path),
        review_proxy_path=None,
        evaluation_review_path=str(evaluation_review_path),
    )
    logger.log(
        "evaluation_manifest_ready",
        report_path=report_path,
        ui_manifest_path=ui_manifest_path,
        detections_csv=csv_path,
        candidate_count=len(candidates),
        manifest_ready_seconds=report["manifest_ready_seconds"],
        peak_rss_kb=peak_rss_kb,
        review_proxy_status=review_proxy_status,
        session_mode="evaluation",
    )

    if not args.skip_review_proxy:
        logger.log("review_proxy_start", output_path=review_proxy_path)
        try:
            generate_review_proxy_ffmpeg(
                video_path=video_path,
                output_path=review_proxy_path,
                preset="ultrafast" if args.quality == "fast" else "veryfast",
                ffmpeg_threads=args.ffmpeg_threads,
            )
            review_proxy_status = "ready"
            logger.log("review_proxy_ready", output_path=review_proxy_path)
        except Exception as exc:
            review_proxy_status = "failed"
            review_proxy_error = str(exc)
            logger.log("review_proxy_error", output_path=review_proxy_path, error=review_proxy_error)

    report = build_report(
        video_path=video_path,
        audio_path=prepared_audio_path,
        output_dir=output_dir,
        args=args,
        config=config,
        candidates=candidates,
        detect_seconds=detect_seconds,
        audio_extract_seconds=audio_extract_seconds,
        review_proxy_path=review_proxy_path,
        review_proxy_status=review_proxy_status,
        review_proxy_error=review_proxy_error,
    )
    report["detections_csv"] = str(csv_path)
    report["runtime_score_enrichment"] = runtime_score_enrichment
    proposal_trace = build_proposal_diagnostics(
        detector=detector,
        audio_path=prepared_audio_path,
        source_video_path=video_path,
        candidates=candidates,
        session_id=output_dir.name,
    )
    proposal_rows = list(proposal_trace.get("final_proposals", []))
    proposal_diagnostics_path = write_jsonl(output_dir / "proposal_diagnostics.jsonl", proposal_rows)
    proposal_transient_peaks_path = write_jsonl(output_dir / "proposal_transient_peaks.jsonl", proposal_trace.get("transient_peaks", []))
    proposal_raw_peaks_path = write_jsonl(output_dir / "proposal_raw_peaks.jsonl", proposal_trace.get("raw_peaks", []))
    proposal_frontend_candidates_path = write_jsonl(output_dir / "proposal_frontend_candidates.jsonl", proposal_trace.get("frontend_candidates", []))
    proposal_suppression_events_path = write_jsonl(output_dir / "proposal_suppression_events.jsonl", proposal_trace.get("suppression_events", []))
    proposal_frontend_stage_summary_path = write_json(output_dir / "proposal_frontend_stage_summary.json", proposal_trace.get("frontend_stage_summaries", []))
    proposal_summary = dict(proposal_trace.get("summary", {}) or {})
    proposal_summary.update(
        {
            "session_id": output_dir.name,
            "proposal_count": len(proposal_rows),
            "selected_count": sum(1 for row in proposal_rows if row.get("pipeline_selected")),
            "classifier_rejected_count": sum(1 for row in proposal_rows if row.get("pipeline_stage") == "classifier_rejected"),
            "threshold_rejected_count": sum(1 for row in proposal_rows if row.get("pipeline_stage") == "threshold_rejected"),
            "ambiguous_count": sum(1 for row in proposal_rows if row.get("pipeline_stage") == "ambiguous_case"),
        }
    )
    proposal_summary_path = write_json(output_dir / "proposal_diagnostics_summary.json", proposal_summary)
    report["proposal_diagnostics_path"] = str(proposal_diagnostics_path)
    report["proposal_transient_peaks_path"] = str(proposal_transient_peaks_path)
    report["proposal_raw_peaks_path"] = str(proposal_raw_peaks_path)
    report["proposal_frontend_candidates_path"] = str(proposal_frontend_candidates_path)
    report["proposal_suppression_events_path"] = str(proposal_suppression_events_path)
    report["proposal_frontend_stage_summary_path"] = str(proposal_frontend_stage_summary_path)
    report["proposal_diagnostics_summary_path"] = str(proposal_summary_path)
    report["extract_seconds"] = 0.0
    report["peak_rss_kb"] = peak_rss_kb
    report["manifest_ready_seconds"] = time.time() - audio_extract_started
    report["total_runtime_seconds"] = report["manifest_ready_seconds"]
    report["extraction_error_count"] = 0
    report["extraction_errors"] = []
    report["extracted_paths"] = []

    report_path, ui_manifest_path = write_session_outputs(
        video_path=video_path,
        output_dir=output_dir,
        profile=args.profile,
        report=report,
        candidates=candidates,
        extracted_paths=[],
        status_override="evaluation_ready" if review_proxy_status != "failed" else "evaluation_proxy_error",
        session_mode="evaluation",
        source_audio_path=str(prepared_audio_path),
        review_proxy_path=str(review_proxy_path) if review_proxy_path.exists() else None,
        evaluation_review_path=str(evaluation_review_path),
    )
    logger.log(
        "review_ready",
        report_path=report_path,
        ui_manifest_path=ui_manifest_path,
        detections_csv=csv_path,
        proposal_diagnostics_path=str(proposal_diagnostics_path),
        proposal_transient_peaks_path=str(proposal_transient_peaks_path),
        proposal_raw_peaks_path=str(proposal_raw_peaks_path),
        proposal_frontend_candidates_path=str(proposal_frontend_candidates_path),
        proposal_suppression_events_path=str(proposal_suppression_events_path),
        proposal_frontend_stage_summary_path=str(proposal_frontend_stage_summary_path),
        candidate_count=len(candidates),
        extracted_count=0,
        extraction_error_count=0,
        manifest_ready_seconds=report["manifest_ready_seconds"],
        peak_rss_kb=peak_rss_kb,
        session_mode="evaluation",
    )

    total_runtime_seconds = time.time() - audio_extract_started
    report["manifest_ready_seconds"] = total_runtime_seconds
    report["total_runtime_seconds"] = total_runtime_seconds
    report["runtime_score_enrichment"] = runtime_score_enrichment
    report_path, ui_manifest_path = write_session_outputs(
        video_path=video_path,
        output_dir=output_dir,
        profile=args.profile,
        report=report,
        candidates=candidates,
        extracted_paths=[],
        status_override="evaluation_ready" if review_proxy_status != "failed" else "evaluation_proxy_error",
        session_mode="evaluation",
        source_audio_path=str(prepared_audio_path),
        review_proxy_path=str(review_proxy_path) if review_proxy_path.exists() else None,
        evaluation_review_path=str(evaluation_review_path),
    )
    logger.log(
        "session_complete",
        report_path=report_path,
        ui_manifest_path=ui_manifest_path,
        detections_csv=csv_path,
        proposal_diagnostics_path=str(proposal_diagnostics_path),
        proposal_transient_peaks_path=str(proposal_transient_peaks_path),
        proposal_raw_peaks_path=str(proposal_raw_peaks_path),
        proposal_frontend_candidates_path=str(proposal_frontend_candidates_path),
        proposal_suppression_events_path=str(proposal_suppression_events_path),
        proposal_frontend_stage_summary_path=str(proposal_frontend_stage_summary_path),
        candidate_count=len(candidates),
        extracted_count=0,
        extraction_error_count=0,
        review_proxy_path=str(review_proxy_path) if review_proxy_path.exists() else None,
        review_proxy_error=review_proxy_error,
        review_proxy_status=review_proxy_status,
        manifest_ready_seconds=report["manifest_ready_seconds"],
        extract_seconds=0.0,
        total_runtime_seconds=total_runtime_seconds,
        peak_rss_kb=peak_rss_kb,
        session_mode="evaluation",
    )

    summary = {
        "video_path": str(video_path),
        "audio_path": str(prepared_audio_path),
        "candidate_count": len(candidates),
        "detector_seconds": detect_seconds,
        "audio_extract_seconds": audio_extract_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "peak_rss_kb": peak_rss_kb,
        "report_path": str(report_path),
        "ui_manifest_path": str(ui_manifest_path),
        "detections_csv": str(csv_path),
        "review_proxy_status": review_proxy_status,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_human_summary(
            {
                "video_path": str(video_path),
                "candidate_count": len(candidates),
                "detector_seconds": detect_seconds,
                "extract_seconds": 0.0,
                "total_runtime_seconds": total_runtime_seconds,
                "peak_rss_kb": peak_rss_kb,
                "report_path": str(report_path),
                "ui_manifest_path": str(ui_manifest_path),
                "detections_csv": str(csv_path),
                "extracted_count": 0,
            }
        )
        if args.debug:
            print(f"Prepared audio: {prepared_audio_path}")
            print(f"Run log: {output_dir / 'session_pipeline.log.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
