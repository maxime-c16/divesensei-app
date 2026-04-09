from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from divesensei.detection.audio_features import AUDIO_CLIP_FEATURES


NON_DIVE_SUBTYPES = (
    "board_rebound",
    "board_slap",
    "non_dive_splash",
    "voice_whistle",
    "handling_noise",
    "unknown_transient",
)


def normalize_non_dive_subtype(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    return raw if raw in NON_DIVE_SUBTYPES else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def resolve_evaluation_session_paths(raw_path: str | Path) -> dict[str, Path]:
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        output_dir = path
        manifest_path = output_dir / "ui_session_manifest.json"
        report_path = output_dir / "session_pipeline_report.json"
    elif path.name == "ui_session_manifest.json":
        manifest_path = path
        output_dir = path.parent
        report_path = output_dir / "session_pipeline_report.json"
    elif path.name == "session_pipeline_report.json":
        report_path = path
        output_dir = path.parent
        manifest_path = output_dir / "ui_session_manifest.json"
    else:
        raise FileNotFoundError(f"Could not resolve an evaluation session from {path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")

    manifest = read_json(manifest_path)
    artifacts = manifest.get("artifacts", {})
    review_path = Path(str(artifacts.get("evaluation_review") or output_dir / "evaluation_review.json")).resolve()
    proposal_path = Path(str(artifacts.get("proposal_diagnostics") or output_dir / "proposal_diagnostics.jsonl")).resolve()
    raw_peaks_path = Path(str(artifacts.get("proposal_raw_peaks") or output_dir / "proposal_raw_peaks.jsonl")).resolve()
    transient_peaks_path = Path(str(artifacts.get("proposal_transient_peaks") or output_dir / "proposal_transient_peaks.jsonl")).resolve()
    frontend_candidates_path = Path(str(artifacts.get("proposal_frontend_candidates") or output_dir / "proposal_frontend_candidates.jsonl")).resolve()
    frontend_stage_summary_path = Path(str(artifacts.get("proposal_frontend_stage_summary") or output_dir / "proposal_frontend_stage_summary.json")).resolve()
    suppression_events_path = Path(str(artifacts.get("proposal_suppression_events") or output_dir / "proposal_suppression_events.jsonl")).resolve()
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "report_path": report_path,
        "review_path": review_path,
        "proposal_path": proposal_path,
        "raw_peaks_path": raw_peaks_path,
        "transient_peaks_path": transient_peaks_path,
        "frontend_candidates_path": frontend_candidates_path,
        "frontend_stage_summary_path": frontend_stage_summary_path,
        "suppression_events_path": suppression_events_path,
    }


def infer_domain_metadata(source_video_path: str) -> dict[str, Any]:
    path = Path(source_video_path)
    parent_names = [part for part in path.parts[:-1] if part not in {"/", "Users", "Volumes", "private", "var"}]
    stem_tokens = [token for token in re.split(r"[^A-Za-z0-9]+", path.stem.lower()) if token]
    parent_tokens = [token for token in re.split(r"[^A-Za-z0-9]+", " ".join(parent_names[-3:]).lower()) if token]
    return {
        "source_session_id": path.stem,
        "source_filename": path.name,
        "source_parent_directory": path.parent.name,
        "path_context": parent_names[-3:],
        "filename_tokens": stem_tokens,
        "path_tokens": parent_tokens,
    }


def _greedy_timestamp_matches(
    left_values: Sequence[float],
    right_values: Sequence[float],
    tolerance_seconds: float,
) -> list[tuple[int, int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for left_idx, left_value in enumerate(left_values):
        for right_idx, right_value in enumerate(right_values):
            delta = abs(float(left_value) - float(right_value))
            if delta <= tolerance_seconds:
                pairs.append((delta, left_idx, right_idx))
    pairs.sort(key=lambda item: item[0])
    taken_left: set[int] = set()
    taken_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for delta, left_idx, right_idx in pairs:
        if left_idx in taken_left or right_idx in taken_right:
            continue
        taken_left.add(left_idx)
        taken_right.add(right_idx)
        matches.append((left_idx, right_idx, delta))
    return matches


def build_proposal_diagnostics(
    *,
    detector: Any,
    audio_path: Path,
    source_video_path: Path,
    candidates: Sequence[Any],
    session_id: str,
    tolerance_seconds: float = 0.75,
) -> dict[str, Any]:
    pipeline = detector.inspect_audio_proposal_pipeline_from_audio_file(str(audio_path))
    proposal_rows = list(pipeline.get("final_proposals", []))
    candidate_timestamps = [float(candidate.timestamp) for candidate in candidates]
    proposal_timestamps = [float(row.get("timestamp", 0.0)) for row in proposal_rows]
    matches = _greedy_timestamp_matches(proposal_timestamps, candidate_timestamps, tolerance_seconds)
    candidate_meta = {
        idx: {
            "detection_id": f"det-{idx + 1:04d}",
            "confidence": getattr(candidate, "confidence", None),
            "combined_score": float(getattr(candidate, "combined_score", 0.0) or 0.0),
        }
        for idx, candidate in enumerate(candidates)
    }
    proposal_match_map = {proposal_idx: (candidate_idx, delta) for proposal_idx, candidate_idx, delta in matches}
    domain = infer_domain_metadata(str(source_video_path))
    enriched_rows: list[dict[str, Any]] = []
    for index, row in enumerate(proposal_rows, start=1):
        details = dict(row.get("details", {}) or {})
        match = proposal_match_map.get(index - 1)
        classifier_bucket = str(details.get("audio_clip_bucket", row.get("classifier_bucket", "unclassified")))
        detection_id = None
        match_delta = None
        if match is not None:
            detection_id = candidate_meta[match[0]]["detection_id"]
            match_delta = match[1]
            pipeline_stage = "final_selected"
        elif classifier_bucket == "ambiguous":
            pipeline_stage = "ambiguous_case"
        elif classifier_bucket in {"accepted", "accepted_no_model"}:
            pipeline_stage = "threshold_rejected"
        else:
            pipeline_stage = "classifier_rejected"

        payload = {
            "proposal_id": f"prop-{index:04d}",
            "session_id": session_id,
            "source_video_path": str(source_video_path),
            "source_audio_path": str(audio_path),
            "source_file": source_video_path.name,
            "timestamp": float(row.get("timestamp", 0.0) or 0.0),
            "proposal_frontend": str(row.get("proposal_frontend", details.get("proposal_frontend", "unknown"))),
            "raw_proposal_score": float(row.get("raw_proposal_score", details.get("audio_score", 0.0)) or 0.0),
            "audio_model_probability": details.get("audio_model_probability"),
            "audio_clip_probability": details.get("audio_clip_probability"),
            "classifier_bucket": classifier_bucket,
            "classifier_decision": row.get("classifier_decision", "dive" if classifier_bucket in {"accepted", "accepted_no_model"} else "non-dive"),
            "pipeline_selected": bool(match is not None),
            "pipeline_stage": pipeline_stage,
            "final_detection_id": detection_id,
            "final_match_delta_seconds": match_delta,
            "final_confidence": candidate_meta[match[0]]["confidence"] if match is not None else None,
            "final_combined_score": candidate_meta[match[0]]["combined_score"] if match is not None else None,
            "clip_feature_window_seconds": details.get("clip_feature_window_seconds"),
            "details": details,
            **domain,
        }
        for feature_name in AUDIO_CLIP_FEATURES:
            payload[feature_name] = details.get(feature_name)
        enriched_rows.append(payload)
    raw_peaks = list(pipeline.get("raw_peaks", []))
    frontend_candidates = list(pipeline.get("frontend_candidates", []))
    suppression_events = list(pipeline.get("suppression_events", []))
    summary = {
        "session_id": session_id,
        "final_proposal_count": len(enriched_rows),
        "transient_peak_count": len(pipeline.get("transient_peaks", [])),
        "raw_peak_count": len(raw_peaks),
        "frontend_candidate_count": len(frontend_candidates),
        "suppression_event_count": len(suppression_events),
        "evidence_threshold_promoted_count": sum(1 for row in raw_peaks if row.get("evidence_threshold_promoted")),
        "proposal_evidence_boosted_count": sum(
            1 for row in enriched_rows if float((row.get("details") or {}).get("proposal_evidence_boost", 0.0) or 0.0) > 0.0
        ),
        "frontend_stage_summaries": list(pipeline.get("frontend_stage_summaries", [])),
        "raw_peak_rejection_counts": {
            stage: sum(1 for row in raw_peaks if row.get("rejection_stage") == stage)
            for stage in [
                "accepted",
                "below_threshold",
                "ignored_before_start",
                "low_hf_ratio",
                "low_audio_score",
                "sustained_noise_reject",
                "weak_pattern_score",
                "audio_model_rejected",
            ]
        },
        "local_rescue_survivor_count": sum(1 for row in raw_peaks if row.get("pre_candidate_loss_stage") == "local_rescue_survivor"),
        "protected_survivor_count": sum(1 for row in raw_peaks if row.get("pre_candidate_loss_stage") == "protected_survivor"),
        "suppression_event_counts": {
            event_type: sum(1 for row in suppression_events if row.get("event_type") == event_type)
            for event_type in [
                "merged_replaced_by_stronger_neighbor",
                "merged_into_stronger_neighbor",
                "suppressed_rebound_precursor",
                "suppressed_duplicate_follower",
            ]
        },
    }
    return {
        "final_proposals": enriched_rows,
        "transient_peaks": list(pipeline.get("transient_peaks", [])),
        "raw_peaks": raw_peaks,
        "frontend_candidates": frontend_candidates,
        "frontend_stage_summaries": list(pipeline.get("frontend_stage_summaries", [])),
        "suppression_events": suppression_events,
        "summary": summary,
    }


def build_detector_from_report(report: dict[str, Any], progress_callback: Any = None) -> Any:
    from divesensei.detection.audio_detector import AudioVisualDiveDetector
    from divesensei.detection.config import DetectionConfig

    config = DetectionConfig()
    for key, value in dict(report.get("config", {}) or {}).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return AudioVisualDiveDetector(config, progress_callback=progress_callback)


def load_evaluation_review_data(review_path: Path) -> dict[str, Any]:
    if not review_path.exists():
        return {"schemaVersion": "1.0.0", "decisions": [], "falseNegatives": []}
    payload = read_json(review_path)
    payload.setdefault("decisions", [])
    payload.setdefault("falseNegatives", [])
    return payload


def update_session_artifacts(session_paths: dict[str, Path], artifact_updates: dict[str, Any]) -> None:
    manifest = read_json(session_paths["manifest_path"])
    report = read_json(session_paths["report_path"])
    manifest_artifacts = manifest.setdefault("artifacts", {})
    for key, value in artifact_updates.items():
        manifest_artifacts[key] = value
        report[f"{key}_path"] = value
    if manifest.get("session"):
        manifest["session"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(session_paths["manifest_path"], manifest)
    write_json(session_paths["report_path"], report)
