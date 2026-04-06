#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from divesensei.detection.audio_detector import AudioVisualDiveDetector
from divesensei.detection.audio_features import AUDIO_CLIP_FEATURES, extract_clip_feature_map
from divesensei.detection.config import DetectionConfig
from divesensei.profiles import apply_named_profile


VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


@dataclass(slots=True)
class LabelRecord:
    source_key: str
    timestamp_seconds: float
    label: str
    source_keys: tuple[str, ...] = ()
    source_video_path: str = ""
    source_file: str = ""
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei evaluate-audio-pcen",
        description="Evaluate the audio_v2_pcen_classifier proposal path and sweep audio thresholds.",
    )
    parser.add_argument("video_root", help="Folder containing session videos to evaluate")
    parser.add_argument(
        "--labels-path",
        action="append",
        default=[],
        help="Path to a labels.jsonl, review CSV, or directory containing label files. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default=".divesensei-runtime/audio-eval",
        help="Directory for proposal exports, summaries, and hard-negative artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=["classifier", "pipeline", "both"],
        default="both",
        help="Which evaluation summary to compute.",
    )
    parser.add_argument("--detector-id", default="audio_v2_pcen_classifier", help="Detector identifier to evaluate")
    parser.add_argument("--profile", default="long-session", help="Named profile to apply before threshold overrides")
    parser.add_argument("--audio-clip-model-path", default=".divesensei-runtime/models/audio_clip_model.json")
    parser.add_argument("--audio-pcen-threshold", type=float, default=2.4)
    parser.add_argument("--audio-pcen-merge-weight", type=float, default=0.65)
    parser.add_argument("--audio-clip-model-min-probability", type=float, default=0.5)
    parser.add_argument("--audio-clip-classifier-ambiguity-low", type=float, default=0.35)
    parser.add_argument("--audio-clip-classifier-ambiguity-high", type=float, default=0.65)
    parser.add_argument("--audio-clip-classifier-window-seconds", type=float, default=3.0)
    parser.add_argument("--audio-peak-threshold", type=float, default=4.0)
    parser.add_argument("--audio-peak-separation", type=float, default=4.0)
    parser.add_argument("--audio-min-score", type=float, default=4.5)
    parser.add_argument("--audio-pattern-min-score", type=float, default=0.4)
    parser.add_argument("--audio-min-hf-ratio", type=float, default=0.115)
    parser.add_argument("--audio-decode-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--audio-visual-merge-seconds", type=float, default=2.0)
    parser.add_argument("--audio-duplicate-suppress-window-seconds", type=float, default=0.9)
    parser.add_argument("--audio-duplicate-leader-min-score", type=float, default=12.0)
    parser.add_argument("--audio-duplicate-leader-min-prominence", type=float, default=10.0)
    parser.add_argument("--audio-duplicate-follower-max-score-ratio", type=float, default=0.55)
    parser.add_argument("--audio-visual-max-proposals", type=int, default=4)
    parser.add_argument("--audio-visual-min-video-score", type=float, default=0.8)
    parser.add_argument("--audio-visual-hard-video-floor", type=float, default=0.2)
    parser.add_argument("--audio-visual-audio-rescue-score", type=float, default=4.0)
    parser.add_argument("--audio-visual-rescue-splash-ratio", type=float, default=1.35)
    parser.add_argument("--audio-visual-min-combined-score", type=float, default=3.8)
    parser.add_argument("--audio-visual-max-verify-width", type=int, default=640)
    parser.add_argument("--audio-priority-weight", type=float, default=0.85)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-frame-length", type=int, default=1024)
    parser.add_argument("--audio-hop-length", type=int, default=256)
    parser.add_argument("--audio-visual-skip-video-verification", action="store_true", default=True)
    parser.add_argument("--with-video-verification", action="store_true", help="Enable video verification for the full pipeline evaluation")
    parser.add_argument("--tolerance-seconds", type=float, default=0.75, help="Timestamp tolerance for joining labels")
    parser.add_argument("--recall-floor", type=float, default=0.9, help="Recall floor for the precision-optimized sweep recommendation")
    parser.add_argument("--top-k", type=int, default=10, help="How many false positives/negatives to print")
    parser.add_argument("--proposal-jsonl", default="", help="Write one row per proposal to this JSONL file")
    parser.add_argument("--summary-json", default="", help="Write the evaluation summary to this JSON file")
    parser.add_argument("--hard-negative-manifest", default="", help="Write false positives as a non-dive JSONL manifest")
    parser.add_argument("--hard-negative-commands", default="", help="Write label-audio shell commands for false positives")
    parser.add_argument("--hard-negative-pre-seconds", type=float, default=2.0, help="Seconds before the hard negative timestamp for label-audio")
    parser.add_argument("--hard-negative-post-seconds", type=float, default=2.0, help="Seconds after the hard negative timestamp for label-audio")
    parser.add_argument("--sweep-audio-pcen-thresholds", default="", help="Comma-separated audio_pcen_threshold values to sweep")
    parser.add_argument("--sweep-audio-pcen-merge-weights", default="", help="Comma-separated audio_pcen_merge_weight values to sweep")
    parser.add_argument("--sweep-audio-clip-model-min-probabilities", default="", help="Comma-separated audio_clip_model_min_probability values to sweep")
    parser.add_argument("--sweep-audio-clip-ambiguity-low", default="", help="Comma-separated audio_clip_classifier_ambiguity_low values to sweep")
    parser.add_argument("--sweep-audio-clip-ambiguity-high", default="", help="Comma-separated audio_clip_classifier_ambiguity_high values to sweep")
    return parser


def _split_csv_floats(raw: str, default: float) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return [float(default)]
    return [float(item) for item in values]


def _iter_video_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    return sorted(files)


def _source_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field_name in ("source_video_path", "source_file", "file", "path"):
        raw_value = str(row.get(field_name, "") or "").strip()
        if not raw_value:
            continue
        candidate = Path(raw_value)
        keys.add(candidate.name.lower())
        keys.add(candidate.stem.lower())
    return keys


def _parse_label_row(row: dict[str, Any], source_hint: str = "") -> LabelRecord | None:
    raw_label = str(row.get("label") or row.get("review_label") or "").strip().lower()
    if raw_label not in {"dive", "non-dive"}:
        return None
    timestamp_raw = row.get("timestamp_seconds", row.get("timestamp", row.get("time_seconds", 0.0)))
    try:
        timestamp_seconds = float(timestamp_raw)
    except (TypeError, ValueError):
        return None
    source_keys = _source_keys(row)
    if not source_keys and source_hint:
        candidate = Path(source_hint)
        source_keys = {candidate.name.lower(), candidate.stem.lower()}
    source_video_path = str(row.get("source_video_path", row.get("video_path", row.get("source_path", ""))) or "")
    source_file = str(row.get("source_file", row.get("file", Path(source_video_path).name if source_video_path else "")) or "")
    notes = str(row.get("notes", row.get("comment", "")) or "")
    if not source_keys and source_file:
        candidate = Path(source_file)
        source_keys = {candidate.name.lower(), candidate.stem.lower()}
    if not source_keys:
        return None
    source_key_list = tuple(sorted(source_keys))
    return LabelRecord(
        source_key=source_key_list[0],
        timestamp_seconds=timestamp_seconds,
        label=raw_label,
        source_keys=source_key_list,
        source_video_path=source_video_path,
        source_file=source_file,
        notes=notes,
        raw=row,
    )


def _load_label_file(path: Path) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".csv", ".jsonl", ".json"}:
                records.extend(_load_label_file(child))
        return records

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = _parse_label_row(row, source_hint=path.stem)
                if parsed is not None:
                    records.append(parsed)
        return records

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = _parse_label_row(json.loads(line), source_hint=path.stem)
            if parsed is not None:
                records.append(parsed)
    return records


def load_labels(paths: Sequence[str]) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue
        records.extend(_load_label_file(path))
    return records


def index_labels(records: Sequence[LabelRecord]) -> dict[str, list[LabelRecord]]:
    index: dict[str, list[LabelRecord]] = {}
    for record in records:
        keys = record.source_keys or (record.source_key,)
        for key in keys:
            index.setdefault(key, []).append(record)
    for rows in index.values():
        rows.sort(key=lambda item: item.timestamp_seconds)
    return index


def build_config(args: argparse.Namespace) -> DetectionConfig:
    defaults = apply_named_profile(
        {
            "audio_peak_threshold": args.audio_peak_threshold,
            "audio_min_score": args.audio_min_score,
            "audio_pattern_min_score": args.audio_pattern_min_score,
            "audio_peak_separation": args.audio_peak_separation,
            "audio_visual_merge_seconds": args.audio_visual_merge_seconds,
            "audio_clip_model_min_probability": args.audio_clip_model_min_probability,
            "audio_decode_timeout_seconds": args.audio_decode_timeout_seconds,
        },
        args.profile,
        args.detector_id,
    )
    audio_visual_skip_video_verification = bool(args.audio_visual_skip_video_verification)
    if args.with_video_verification:
        audio_visual_skip_video_verification = False
    return DetectionConfig(
        detector_id=args.detector_id,
        method="audio_visual",
        splash_zone_top_norm=0.72,
        splash_zone_bottom_norm=0.95,
        splash_zone_left_norm=0.0,
        splash_zone_right_norm=1.0,
        audio_sample_rate=args.audio_sample_rate,
        audio_frame_length=args.audio_frame_length,
        audio_hop_length=args.audio_hop_length,
        audio_peak_threshold=float(defaults["audio_peak_threshold"]),
        audio_peak_min_separation_seconds=float(defaults["audio_peak_separation"]),
        audio_min_score=float(defaults["audio_min_score"]),
        audio_min_hf_ratio=args.audio_min_hf_ratio,
        audio_pattern_min_score=float(defaults["audio_pattern_min_score"]),
        audio_noise_max_peak_count=5,
        audio_noise_max_top_ratio=1.8,
        audio_long_session_seconds=120.0,
        audio_long_session_max_candidates=120,
        audio_model_path="",
        audio_model_min_probability=0.0,
        audio_clip_model_path=args.audio_clip_model_path,
        audio_clip_model_min_probability=float(defaults["audio_clip_model_min_probability"]),
        audio_clip_classifier_window_seconds=args.audio_clip_classifier_window_seconds,
        audio_clip_classifier_ambiguity_low=args.audio_clip_classifier_ambiguity_low,
        audio_clip_classifier_ambiguity_high=args.audio_clip_classifier_ambiguity_high,
        audio_pcen_threshold=args.audio_pcen_threshold,
        audio_pcen_merge_weight=args.audio_pcen_merge_weight,
        audio_duplicate_suppress_window_seconds=args.audio_duplicate_suppress_window_seconds,
        audio_duplicate_leader_min_score=args.audio_duplicate_leader_min_score,
        audio_duplicate_leader_min_prominence=args.audio_duplicate_leader_min_prominence,
        audio_duplicate_follower_max_score_ratio=args.audio_duplicate_follower_max_score_ratio,
        audio_decode_timeout_seconds=float(defaults["audio_decode_timeout_seconds"]),
        ffmpeg_threads=0,
        opencv_threads=0,
        audio_only_pre_seconds=3.0,
        audio_only_post_seconds=1.0,
        audio_visual_skip_video_verification=audio_visual_skip_video_verification,
        audio_visual_verify_pre_seconds=3.0,
        audio_visual_verify_post_seconds=1.0,
        audio_visual_verify_target_fps=12.0,
        audio_visual_max_proposals=args.audio_visual_max_proposals,
        audio_visual_min_video_score=args.audio_visual_min_video_score,
        audio_visual_hard_video_floor=args.audio_visual_hard_video_floor,
        audio_visual_audio_rescue_score=args.audio_visual_audio_rescue_score,
        audio_visual_rescue_splash_ratio=args.audio_visual_rescue_splash_ratio,
        audio_visual_min_combined_score=args.audio_visual_min_combined_score,
        audio_visual_merge_seconds=float(defaults["audio_visual_merge_seconds"]),
        audio_visual_max_verify_width=args.audio_visual_max_verify_width,
        audio_priority_weight=args.audio_priority_weight,
        enable_debug_plots=False,
    )


def _match_timestamps(predicted: Sequence[dict[str, Any]], labels: Sequence[LabelRecord], tolerance_seconds: float) -> tuple[list[dict[str, Any]], list[LabelRecord], list[dict[str, Any]]]:
    pairs: list[tuple[float, int, int]] = []
    for row_idx, row in enumerate(predicted):
        timestamp = float(row["timestamp"])
        for label_idx, label in enumerate(labels):
            delta = abs(timestamp - label.timestamp_seconds)
            if delta <= tolerance_seconds:
                pairs.append((delta, row_idx, label_idx))
    pairs.sort(key=lambda item: item[0])
    matched_rows: set[int] = set()
    matched_labels: set[int] = set()
    matches: list[dict[str, Any]] = []
    for delta, row_idx, label_idx in pairs:
        if row_idx in matched_rows or label_idx in matched_labels:
            continue
        matched_rows.add(row_idx)
        matched_labels.add(label_idx)
        matches.append(
            {
                "row_index": row_idx,
                "label_index": label_idx,
                "delta_seconds": float(delta),
                "row": predicted[row_idx],
                "label": labels[label_idx],
            }
        )
    unmatched_rows = [predicted[idx] for idx in range(len(predicted)) if idx not in matched_rows]
    unmatched_labels = [labels[idx] for idx in range(len(labels)) if idx not in matched_labels]
    return matches, unmatched_labels, unmatched_rows


def _evaluate_rows(rows: list[dict[str, Any]], labels_by_source: dict[str, list[LabelRecord]], *, prediction_field: str, tolerance_seconds: float) -> dict[str, Any]:
    annotated_rows: list[dict[str, Any]] = []
    all_tp = all_fp = all_tn = all_fn = 0
    unmatched_labels_total = 0
    unlabeled_rows_total = 0
    false_positive_rows: list[dict[str, Any]] = []
    false_negative_rows: list[dict[str, Any]] = []

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row.get("source_file", "")), []).append(row)

    for source_key, source_rows in grouped_rows.items():
        source_path = Path(source_key)
        candidate_keys = {source_key.lower(), source_path.stem.lower()}
        source_labels: list[LabelRecord] = []
        seen_labels: set[int] = set()
        for candidate_key in candidate_keys:
            for label in labels_by_source.get(candidate_key, []):
                if id(label) in seen_labels:
                    continue
                seen_labels.add(id(label))
                source_labels.append(label)
        source_labels.sort(key=lambda item: item.timestamp_seconds)
        if not source_labels:
            for row in source_rows:
                row = dict(row)
                row["human_label"] = None
                row["label_match_delta_seconds"] = None
                row["label_notes"] = None
                row["prediction"] = str(row.get(prediction_field, "non-dive"))
                row["evaluation_state"] = "unlabeled"
                annotated_rows.append(row)
                unlabeled_rows_total += 1
            continue

        matches, unmatched_labels, unmatched_rows = _match_timestamps(source_rows, source_labels, tolerance_seconds)
        matched_row_indices = {match["row_index"] for match in matches}
        for match in matches:
            row = dict(match["row"])
            label = match["label"]
            truth = label.label
            prediction = str(row.get(prediction_field, "non-dive"))
            row["human_label"] = truth
            row["label_match_delta_seconds"] = match["delta_seconds"]
            row["label_notes"] = label.notes
            row["prediction"] = prediction
            row["evaluation_state"] = "matched"
            row["matched_label_timestamp"] = label.timestamp_seconds
            row["matched_label_source_key"] = label.source_key
            annotated_rows.append(row)
            if truth == "dive" and prediction == "dive":
                all_tp += 1
            elif truth == "non-dive" and prediction == "dive":
                all_fp += 1
                false_positive_rows.append(row)
            elif truth == "dive" and prediction == "non-dive":
                all_fn += 1
                false_negative_rows.append(row)
            elif truth == "non-dive" and prediction == "non-dive":
                all_tn += 1

        for row in unmatched_rows:
            annotated = dict(row)
            annotated["human_label"] = "non-dive"
            annotated["label_match_delta_seconds"] = None
            annotated["label_notes"] = "background"
            annotated["prediction"] = str(annotated.get(prediction_field, "non-dive"))
            annotated["evaluation_state"] = "background_unmatched"
            annotated_rows.append(annotated)
            if annotated["prediction"] == "dive":
                all_fp += 1
                false_positive_rows.append(annotated)
            else:
                all_tn += 1

        for label in unmatched_labels:
            synthetic_row = {
                "source_video_path": label.source_video_path,
                "source_file": Path(label.source_video_path or label.source_file or source_key).name or source_key,
                "timestamp": float(label.timestamp_seconds),
                "human_label": label.label,
                "label_match_delta_seconds": None,
                "label_notes": label.notes,
                "prediction": "non-dive",
                "evaluation_state": "unmatched_label",
            }
            annotated_rows.append(synthetic_row)
            if label.label == "dive":
                all_fn += 1
                false_negative_rows.append(synthetic_row)
            else:
                all_tn += 1
            unmatched_labels_total += 1

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "rows": annotated_rows,
        "tp": all_tp,
        "fp": all_fp,
        "tn": all_tn,
        "fn": all_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unlabeled_rows": unlabeled_rows_total,
        "unmatched_labels": unmatched_labels_total,
        "false_positive_rows": false_positive_rows,
        "false_negative_rows": false_negative_rows,
    }


def _format_row_for_jsonl(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    details = dict(payload.get("details", {}) or {})
    for name in AUDIO_CLIP_FEATURES:
        payload[name] = details.get(name)
    payload["details"] = details
    return payload


def _annotate_proposals(detector: AudioVisualDiveDetector, video_path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    signal, sample_rate = detector._extract_audio_signal(str(video_path))
    proposal_rows = detector.inspect_audio_proposals(str(video_path))
    for row in proposal_rows:
        features = extract_clip_feature_map(
            signal,
            sample_rate,
            float(row["timestamp"]),
            window_seconds=float(getattr(detector.config, "audio_clip_classifier_window_seconds", 3.0)),
            frame_length=int(getattr(detector.config, "audio_frame_length", 1024)),
            hop_length=int(getattr(detector.config, "audio_hop_length", 256)),
        )
        row["clip_feature_window_seconds"] = float(getattr(detector.config, "audio_clip_classifier_window_seconds", 3.0))
        row["details"] = {**dict(row.get("details", {}) or {}), **features}
        for feature_name, feature_value in features.items():
            row[feature_name] = feature_value
    final_candidates = detector.detect(str(video_path))
    return proposal_rows, list(final_candidates)


def _apply_pipeline_matches(rows: list[dict[str, Any]], final_candidates: Sequence[Any], tolerance_seconds: float) -> list[dict[str, Any]]:
    pairs: list[tuple[float, int, int]] = []
    candidate_timestamps = [float(candidate.timestamp) for candidate in final_candidates]
    for row_idx, row in enumerate(rows):
        timestamp = float(row.get("timestamp", 0.0))
        for candidate_idx, candidate_timestamp in enumerate(candidate_timestamps):
            delta = abs(timestamp - candidate_timestamp)
            if delta <= tolerance_seconds:
                pairs.append((delta, row_idx, candidate_idx))
    pairs.sort(key=lambda item: item[0])
    matched_rows: set[int] = set()
    matched_candidates: set[int] = set()
    for delta, row_idx, candidate_idx in pairs:
        if row_idx in matched_rows or candidate_idx in matched_candidates:
            continue
        matched_rows.add(row_idx)
        matched_candidates.add(candidate_idx)
    updated_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        enriched = dict(row)
        enriched["pipeline_selected"] = idx in matched_rows
        enriched["pipeline_decision"] = "dive" if enriched["pipeline_selected"] else "non-dive"
        updated_rows.append(enriched)
    return updated_rows


def _top_rows(rows: Sequence[dict[str, Any]], *, label: str, limit: int) -> list[dict[str, Any]]:
    if label == "fp":
        filtered = [row for row in rows if row.get("human_label") == "non-dive" and row.get("prediction") == "dive"]
        key = lambda row: (float(row.get("audio_clip_probability", 0.0) or 0.0), float(row.get("raw_proposal_score", 0.0) or 0.0))
        reverse = True
    else:
        filtered = [row for row in rows if row.get("human_label") == "dive" and row.get("prediction") == "non-dive"]
        key = lambda row: (float(row.get("audio_clip_probability", 1.0) or 1.0), float(row.get("raw_proposal_score", 0.0) or 0.0))
        reverse = False
    return sorted(filtered, key=key, reverse=reverse)[:limit]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_format_row_for_jsonl(row)) + "\n")


def _write_hard_negative_manifest(path: Path, rows: Sequence[dict[str, Any]], *, pre_seconds: float, post_seconds: float) -> None:
    selected = [row for row in rows if row.get("human_label") == "non-dive" and row.get("prediction") == "dive"]
    payload_rows = []
    for row in selected:
        payload_rows.append(
            {
                "video_path": row.get("source_video_path"),
                "source_video_path": row.get("source_video_path"),
                "source_file": row.get("source_file"),
                "timestamp_seconds": float(row.get("timestamp", 0.0)),
                "label": "non-dive",
                "notes": "hard negative mined from evaluation",
                "pre_seconds": float(pre_seconds),
                "post_seconds": float(post_seconds),
            }
        )
    _write_jsonl(path, payload_rows)


def _write_hard_negative_commands(path: Path, rows: Sequence[dict[str, Any]], *, pre_seconds: float, post_seconds: float) -> None:
    selected = [row for row in rows if row.get("human_label") == "non-dive" and row.get("prediction") == "dive"]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in selected:
        lines.append(
            "divesensei label-audio "
            f'"{row.get("source_video_path")}" '
            f'{float(row.get("timestamp", 0.0)):.3f} '
            f'--label non-dive --pre-seconds {float(pre_seconds):.3f} --post-seconds {float(post_seconds):.3f}'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parameter_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    grids = [
        ("audio_pcen_threshold", _split_csv_floats(args.sweep_audio_pcen_thresholds, args.audio_pcen_threshold)),
        ("audio_pcen_merge_weight", _split_csv_floats(args.sweep_audio_pcen_merge_weights, args.audio_pcen_merge_weight)),
        ("audio_clip_model_min_probability", _split_csv_floats(args.sweep_audio_clip_model_min_probabilities, args.audio_clip_model_min_probability)),
        ("audio_clip_classifier_ambiguity_low", _split_csv_floats(args.sweep_audio_clip_ambiguity_low, args.audio_clip_classifier_ambiguity_low)),
        ("audio_clip_classifier_ambiguity_high", _split_csv_floats(args.sweep_audio_clip_ambiguity_high, args.audio_clip_classifier_ambiguity_high)),
    ]
    combos: list[dict[str, float]] = []
    for values in itertools.product(*[items for _, items in grids]):
        combo = {name: float(value) for (name, _), value in zip(grids, values, strict=True)}
        combos.append(combo)
    return combos


def _config_with_overrides(base: argparse.Namespace, overrides: dict[str, float]) -> argparse.Namespace:
    merged = argparse.Namespace(**vars(base))
    for key, value in overrides.items():
        setattr(merged, key, value)
    return merged


def _evaluate_dataset(args: argparse.Namespace, video_paths: Sequence[Path], labels_index: dict[str, list[LabelRecord]]) -> dict[str, Any]:
    config = build_config(args)
    detector = AudioVisualDiveDetector(config)
    proposal_rows: list[dict[str, Any]] = []
    pipeline_rows: list[dict[str, Any]] = []
    for video_path in video_paths:
        rows, final_candidates = _annotate_proposals(detector, video_path)
        proposal_rows.extend(rows)
        pipeline_rows.extend(_apply_pipeline_matches(rows, final_candidates, args.tolerance_seconds))

    classifier_summary = _evaluate_rows(proposal_rows, labels_index, prediction_field="classifier_decision", tolerance_seconds=args.tolerance_seconds)
    pipeline_summary = _evaluate_rows(pipeline_rows, labels_index, prediction_field="pipeline_decision", tolerance_seconds=args.tolerance_seconds)
    return {
        "config": {
            "audio_pcen_threshold": args.audio_pcen_threshold,
            "audio_pcen_merge_weight": args.audio_pcen_merge_weight,
            "audio_clip_model_min_probability": args.audio_clip_model_min_probability,
            "audio_clip_classifier_ambiguity_low": args.audio_clip_classifier_ambiguity_low,
            "audio_clip_classifier_ambiguity_high": args.audio_clip_classifier_ambiguity_high,
        },
        "proposal_rows": proposal_rows,
        "classifier": classifier_summary,
        "pipeline": pipeline_summary,
    }


def _better_precision_candidate(candidate: dict[str, Any] | None, current: dict[str, Any], recall_floor: float) -> dict[str, Any]:
    if current["classifier"]["recall"] < recall_floor:
        return candidate or current
    if candidate is None:
        return current
    candidate_metrics = candidate["classifier"]
    current_metrics = current["classifier"]
    if current_metrics["precision"] > candidate_metrics["precision"]:
        return current
    if current_metrics["precision"] < candidate_metrics["precision"]:
        return candidate
    if current_metrics["f1"] > candidate_metrics["f1"]:
        return current
    return candidate


def _better_f1_candidate(candidate: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if candidate is None:
        return current
    candidate_metrics = candidate["classifier"]
    current_metrics = current["classifier"]
    if current_metrics["f1"] > candidate_metrics["f1"]:
        return current
    if current_metrics["f1"] < candidate_metrics["f1"]:
        return candidate
    if current_metrics["precision"] > candidate_metrics["precision"]:
        return current
    return candidate


def _print_summary(title: str, summary: dict[str, Any], *, top_k: int) -> None:
    print(f"{title}: TP={summary['tp']} FP={summary['fp']} TN={summary['tn']} FN={summary['fn']} precision={summary['precision']:.3f} recall={summary['recall']:.3f} F1={summary['f1']:.3f}")
    if summary["false_positive_rows"]:
        print("Top false positives:")
        for row in _top_rows(summary["rows"], label="fp", limit=top_k):
            print(
                f"  {row.get('source_file')} @ {float(row.get('timestamp', 0.0)):.3f}s "
                f"prob={float(row.get('audio_clip_probability', 0.0) or 0.0):.3f} "
                f"score={float(row.get('raw_proposal_score', 0.0) or 0.0):.3f}"
            )
    if summary["false_negative_rows"]:
        print("Top false negatives:")
        for row in _top_rows(summary["rows"], label="fn", limit=top_k):
            print(
                f"  {row.get('source_file')} @ {float(row.get('timestamp', 0.0)):.3f}s "
                f"prob={float(row.get('audio_clip_probability', 0.0) or 0.0):.3f} "
                f"score={float(row.get('raw_proposal_score', 0.0) or 0.0):.3f}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    video_root = Path(args.video_root).expanduser().resolve()
    if not video_root.exists():
        print(f"Video root not found: {video_root}")
        return 1

    video_paths = _iter_video_files(video_root)
    if not video_paths:
        print(f"No videos found under: {video_root}")
        return 1

    label_records = load_labels(args.labels_path)
    label_index = index_labels(label_records)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_args = args
    baseline_result = _evaluate_dataset(baseline_args, video_paths, label_index)
    proposal_rows = baseline_result["proposal_rows"]
    if args.proposal_jsonl:
        _write_jsonl(Path(args.proposal_jsonl).expanduser().resolve(), proposal_rows)

    summary: dict[str, Any] = {
        "video_root": str(video_root),
        "video_count": len(video_paths),
        "label_count": len(label_records),
        "mode": args.mode,
        "baseline": {
            "config": baseline_result["config"],
            "classifier": baseline_result["classifier"],
            "pipeline": baseline_result["pipeline"],
        },
    }

    if args.mode in {"classifier", "both"}:
        _print_summary("Classifier", baseline_result["classifier"], top_k=args.top_k)
    if args.mode in {"pipeline", "both"}:
        _print_summary("Pipeline", baseline_result["pipeline"], top_k=args.top_k)

    if args.hard_negative_manifest:
        _write_hard_negative_manifest(Path(args.hard_negative_manifest).expanduser().resolve(), baseline_result["classifier"]["rows"], pre_seconds=args.hard_negative_pre_seconds, post_seconds=args.hard_negative_post_seconds)
    if args.hard_negative_commands:
        _write_hard_negative_commands(Path(args.hard_negative_commands).expanduser().resolve(), baseline_result["classifier"]["rows"], pre_seconds=args.hard_negative_pre_seconds, post_seconds=args.hard_negative_post_seconds)

    grid = _parameter_grid(args)
    sweep_results: list[dict[str, Any]] = []
    best_precision: dict[str, Any] | None = None
    best_f1: dict[str, Any] | None = None
    if len(grid) > 1 or any(len(values) > 1 for values in [
        _split_csv_floats(args.sweep_audio_pcen_thresholds, args.audio_pcen_threshold),
        _split_csv_floats(args.sweep_audio_pcen_merge_weights, args.audio_pcen_merge_weight),
        _split_csv_floats(args.sweep_audio_clip_model_min_probabilities, args.audio_clip_model_min_probability),
        _split_csv_floats(args.sweep_audio_clip_ambiguity_low, args.audio_clip_classifier_ambiguity_low),
        _split_csv_floats(args.sweep_audio_clip_ambiguity_high, args.audio_clip_classifier_ambiguity_high),
    ]):
        for combo in grid:
            combo_args = _config_with_overrides(args, combo)
            combo_result = _evaluate_dataset(combo_args, video_paths, label_index)
            sweep_entry = {
                "config": combo,
                "classifier": combo_result["classifier"],
                "pipeline": combo_result["pipeline"],
            }
            sweep_results.append(sweep_entry)
            if combo_result["classifier"]["recall"] >= args.recall_floor:
                best_precision = _better_precision_candidate(best_precision, sweep_entry, args.recall_floor)
            best_f1 = _better_f1_candidate(best_f1, sweep_entry)

    summary["sweep_results"] = sweep_results
    if best_precision is not None:
        summary["best_precision_with_recall_floor"] = best_precision
        print(
            "Best precision with recall floor: "
            f"precision={best_precision['classifier']['precision']:.3f} "
            f"recall={best_precision['classifier']['recall']:.3f} "
            f"F1={best_precision['classifier']['f1']:.3f} config={best_precision['config']}"
        )
    if best_f1 is not None:
        summary["best_f1"] = best_f1
        print(
            "Best F1: "
            f"precision={best_f1['classifier']['precision']:.3f} "
            f"recall={best_f1['classifier']['recall']:.3f} "
            f"F1={best_f1['classifier']['f1']:.3f} config={best_f1['config']}"
        )

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
