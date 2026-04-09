#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from divesensei.workflows.evaluation_session_support import load_evaluation_review_data, read_json, resolve_evaluation_session_paths, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei replay-evaluation-review",
        description="Replay a reviewed evaluation session onto another evaluation session by matching detections on timestamp.",
    )
    parser.add_argument("source_session")
    parser.add_argument("target_session")
    parser.add_argument("--tolerance-seconds", type=float, default=0.01)
    parser.add_argument("--copy-false-negatives", action="store_true", default=True)
    parser.add_argument("--no-copy-false-negatives", dest="copy_false_negatives", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    source_paths = resolve_evaluation_session_paths(args.source_session)
    target_paths = resolve_evaluation_session_paths(args.target_session)
    source_manifest = read_json(source_paths["manifest_path"])
    target_manifest = read_json(target_paths["manifest_path"])
    source_review = load_evaluation_review_data(source_paths["review_path"])
    target_review = load_evaluation_review_data(target_paths["review_path"])

    source_detections = source_manifest.get("detections", [])
    target_detections = target_manifest.get("detections", [])
    remaining_target_detections = list(target_detections)

    mapped_decisions = []
    unmatched_decisions = []
    for decision in source_review.get("decisions", []):
        detection_id = str(decision.get("detectionId"))
        source_detection = next((row for row in source_detections if str(row.get("id")) == detection_id), None)
        if source_detection is None:
            unmatched_decisions.append({"detectionId": detection_id, "reason": "missing_source_detection"})
            continue
        source_timestamp = float(source_detection.get("timestamp_seconds", 0.0))
        candidate_matches = []
        for index, target_row in enumerate(remaining_target_detections):
            target_timestamp = float(target_row.get("timestamp_seconds", 0.0))
            delta = abs(target_timestamp - source_timestamp)
            if delta <= float(args.tolerance_seconds):
                candidate_matches.append((delta, index, target_row))
        if not candidate_matches:
            unmatched_decisions.append({"detectionId": detection_id, "timestamp_seconds": source_timestamp, "reason": "no_target_match"})
            continue
        candidate_matches.sort(key=lambda item: (item[0], float(item[2].get("timestamp_seconds", 0.0))))
        matched_delta, matched_index, matched_target_row = candidate_matches[0]
        matched_target_id = str(matched_target_row.get("id"))
        remaining_target_detections.pop(matched_index)
        mapped = dict(decision)
        mapped["detectionId"] = matched_target_id
        mapped["_replayedFromDetectionId"] = detection_id
        mapped["_replayedDeltaSeconds"] = matched_delta
        mapped_decisions.append(mapped)

    target_review["decisions"] = mapped_decisions
    if args.copy_false_negatives:
        target_review["falseNegatives"] = list(source_review.get("falseNegatives", []))
    mapping_deltas = [float(item.get("_replayedDeltaSeconds", 0.0) or 0.0) for item in mapped_decisions]
    mapping_coverage = len(mapped_decisions) / max(1, len(source_review.get("decisions", [])))
    if mapping_coverage >= 0.9:
        mapping_quality = "high"
    elif mapping_coverage >= 0.7:
        mapping_quality = "medium"
    else:
        mapping_quality = "degraded"
    target_review["replayMetadata"] = {
        "source_session_id": source_manifest.get("session", {}).get("id"),
        "target_session_id": target_manifest.get("session", {}).get("id"),
        "tolerance_seconds": float(args.tolerance_seconds),
        "source_decision_count": len(source_review.get("decisions", [])),
        "target_detection_count": len(target_detections),
        "mapped_decision_count": len(mapped_decisions),
        "unmatched_decision_count": len(unmatched_decisions),
        "mapping_coverage": mapping_coverage,
        "median_delta_seconds": sorted(mapping_deltas)[len(mapping_deltas) // 2] if mapping_deltas else None,
        "max_delta_seconds": max(mapping_deltas) if mapping_deltas else None,
        "mapping_quality": mapping_quality,
    }
    write_json(target_paths["review_path"], target_review)

    print(
        json.dumps(
            {
                "source_session_id": source_manifest.get("session", {}).get("id"),
                "target_session_id": target_manifest.get("session", {}).get("id"),
                "mapped_decision_count": len(mapped_decisions),
                "unmatched_decision_count": len(unmatched_decisions),
                "false_negative_count_copied": len(target_review.get("falseNegatives", [])),
                "mapping_coverage": mapping_coverage,
                "mapping_quality": mapping_quality,
                "unmatched_decisions": unmatched_decisions[:20],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
