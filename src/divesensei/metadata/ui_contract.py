from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


UI_SCHEMA_VERSION = "1.0.0"
REVIEW_PRE_SECONDS = 2.0
REVIEW_POST_SECONDS = 2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip_browser_path(clip_path: str | None) -> str | None:
    if not clip_path:
        return None
    candidate = Path(clip_path).parent / "web" / Path(clip_path).name
    return str(candidate) if candidate.exists() else clip_path


def build_ui_session_manifest(
    *,
    video_path: Path,
    output_dir: Path,
    profile: str,
    report: dict[str, Any],
    candidates: Sequence[Any],
    extracted_paths: Sequence[str],
    status_override: str | None = None,
    session_mode: str = "standard",
    source_audio_path: str | None = None,
    review_proxy_path: str | None = None,
    evaluation_review_path: str | None = None,
) -> dict[str, Any]:
    debug_summary = report.get("debug_summary", {})
    created_at = str(report.get("session_created_at") or _utc_now())
    updated_at = _utc_now()
    session_name = str(report.get("session_name") or f"{video_path.stem} · {created_at[0:16].replace('T', ' ')}")
    detections = []
    session_duration_seconds = float(report.get("session_estimated_duration_seconds") or 0.0)
    for idx, candidate in enumerate(candidates, start=1):
        clip_path = extracted_paths[idx - 1] if idx - 1 < len(extracted_paths) else None
        details = dict(getattr(candidate, "details", {}) or {})
        review_start_seconds = max(0.0, float(candidate.timestamp) - REVIEW_PRE_SECONDS)
        review_end_seconds = max(review_start_seconds + 0.25, float(candidate.timestamp) + REVIEW_POST_SECONDS)
        if session_duration_seconds > 0.0:
            review_end_seconds = min(session_duration_seconds, review_end_seconds)
            review_start_seconds = min(review_start_seconds, max(0.0, review_end_seconds - 0.25))
        detections.append(
            {
                "id": f"det-{idx:04d}",
                "index": idx,
                "timestamp_seconds": float(candidate.timestamp),
                "start_time_seconds": float(candidate.start_time),
                "end_time_seconds": float(candidate.end_time),
                "duration_seconds": float(candidate.end_time - candidate.start_time),
                "review_start_seconds": review_start_seconds,
                "review_end_seconds": review_end_seconds,
                "review_duration_seconds": float(review_end_seconds - review_start_seconds),
                "confidence": candidate.confidence,
                "scores": {
                    "audio": float(candidate.audio_score),
                    "video": float(candidate.video_score),
                    "combined": float(candidate.combined_score),
                    "governed_r9_score": details.get("governed_r9_score"),
                    "audio_model_probability": float(details.get("audio_model_probability", 0.0) or 0.0),
                    "audio_clip_probability": float(details.get("audio_clip_probability", 0.0) or 0.0),
                },
                "features": {
                    "spectral_flux": details.get("spectral_flux"),
                    "rms": details.get("rms"),
                    "hf_ratio": details.get("hf_ratio"),
                    "spectral_centroid_hz": details.get("spectral_centroid_hz"),
                    "spectral_flatness": details.get("spectral_flatness"),
                    "post_flux_ratio": details.get("post_flux_ratio"),
                    "post_rms_ratio": details.get("post_rms_ratio"),
                    "local_prominence": details.get("local_prominence"),
                    "nearby_peaks_8s": details.get("nearby_peaks_8s"),
                    "visual_late_fusion_logreg_c0.5": details.get("visual_late_fusion_logreg_c0.5"),
                },
                "clip": {
                    "path": clip_path,
                    "browser_path": _clip_browser_path(clip_path),
                    "filename": Path(clip_path).name if clip_path else None,
                },
            }
        )

    return {
        "schema_version": UI_SCHEMA_VERSION,
        "kind": "divesensei.ui-session",
        "generated_at": _utc_now(),
        "session": {
            "id": output_dir.name,
            "title": video_path.stem,
            "session_name": session_name,
            "mode": session_mode,
            "profile": profile,
            "detector_id": report.get("detector_id"),
            "source_video_path": str(video_path),
            "output_dir": str(output_dir),
            "status": status_override or ("complete" if report.get("extraction_error_count", 0) == 0 else "complete_with_errors"),
            "created_at": created_at,
            "updated_at": updated_at,
            "session_duration_seconds": session_duration_seconds,
            "candidate_count": report.get("candidate_count", 0),
            "extracted_count": len(extracted_paths),
            "timestamp_range": debug_summary.get("timestamp_range", {}),
            "telemetry": {
                "detector_seconds": report.get("detector_seconds"),
                "extract_seconds": report.get("extract_seconds"),
                "total_runtime_seconds": report.get("total_runtime_seconds"),
                "peak_rss_kb": report.get("peak_rss_kb"),
            },
        },
        "artifacts": {
            "session_pipeline_report": str(output_dir / "session_pipeline_report.json"),
            "session_debug_summary": str(output_dir / "session_debug_summary.json"),
            "session_pipeline_log": str(output_dir / "session_pipeline.log.jsonl"),
            "detections_csv": report.get("detections_csv"),
            "source_audio": source_audio_path,
            "review_proxy": review_proxy_path,
            "evaluation_review": evaluation_review_path,
            "event_review_support": report.get("event_review_support_path"),
            "event_review_support_summary": report.get("event_review_support_summary_path"),
            "proposal_diagnostics": report.get("proposal_diagnostics_path"),
            "proposal_diagnostics_summary": report.get("proposal_diagnostics_summary_path"),
            "proposal_transient_peaks": report.get("proposal_transient_peaks_path"),
            "proposal_raw_peaks": report.get("proposal_raw_peaks_path"),
            "proposal_frontend_candidates": report.get("proposal_frontend_candidates_path"),
            "proposal_frontend_stage_summary": report.get("proposal_frontend_stage_summary_path"),
            "proposal_suppression_events": report.get("proposal_suppression_events_path"),
            "evaluation_export_summary": report.get("evaluation_export_summary_path"),
            "reviewed_candidates": report.get("reviewed_candidates_path"),
            "hard_negative_manifest": report.get("hard_negative_manifest_path"),
            "hard_negative_commands": report.get("hard_negative_commands_path"),
            "candidate_diagnostics": report.get("candidate_diagnostics_path"),
            "ranking_reports": report.get("ranking_reports_path"),
        },
        "detections": detections,
    }


def write_ui_session_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(manifest, indent=2))
    return output_path


def build_ui_library_index(session_manifests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sessions = []
    for manifest in session_manifests:
        session = manifest["session"]
        sessions.append(
            {
                "id": session["id"],
                "title": session["title"],
                "session_name": session.get("session_name"),
                "mode": session.get("mode", "standard"),
                "profile": session["profile"],
                "source_video_path": session["source_video_path"],
                "output_dir": session["output_dir"],
                "status": session["status"],
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "session_duration_seconds": session.get("session_duration_seconds"),
                "candidate_count": session["candidate_count"],
                "extracted_count": session["extracted_count"],
                "timestamp_range": session.get("timestamp_range", {}),
                "telemetry": session.get("telemetry", {}),
                "detector_id": session.get("detector_id"),
                "manifest_path": manifest["artifacts"]["session_pipeline_report"].replace("session_pipeline_report.json", "ui_session_manifest.json"),
            }
        )
    sessions.sort(key=lambda item: item["title"])
    return {
        "schema_version": UI_SCHEMA_VERSION,
        "kind": "divesensei.ui-library",
        "generated_at": _utc_now(),
        "session_count": len(sessions),
        "sessions": sessions,
    }
