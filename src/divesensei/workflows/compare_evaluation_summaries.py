#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei compare-evaluation-summaries",
        description="Compare two evaluation export summaries on identical or overlapping reviewed sessions.",
    )
    parser.add_argument("baseline_summary")
    parser.add_argument("candidate_summary")
    parser.add_argument("--output-json", default="")
    return parser


def _read_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path).expanduser().resolve()
    return json.loads(summary_path.read_text())


def _session_metrics_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("session_id")): item
        for item in summary.get("per_session_metrics", [])
        if item.get("session_id")
    }


def _delta(current: Any, baseline: Any) -> Any:
    if isinstance(current, (int, float)) and isinstance(baseline, (int, float)):
        return current - baseline
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    baseline = _read_summary(args.baseline_summary)
    candidate = _read_summary(args.candidate_summary)
    baseline_sessions = _session_metrics_map(baseline)
    candidate_sessions = _session_metrics_map(candidate)
    shared_session_ids = sorted(set(baseline_sessions) & set(candidate_sessions))
    per_session = []
    for session_id in shared_session_ids:
        before = baseline_sessions[session_id]
        after = candidate_sessions[session_id]
        per_session.append(
            {
                "session_id": session_id,
                "reviewed_non_dive_delta": _delta(after.get("reviewed_non_dive_count"), before.get("reviewed_non_dive_count")),
                "false_negative_delta": _delta(after.get("false_negative_count"), before.get("false_negative_count")),
                "reviewed_false_positives_per_minute_delta": _delta(
                    after.get("reviewed_false_positives_per_minute"),
                    before.get("reviewed_false_positives_per_minute"),
                ),
                "reviewed_false_negatives_per_minute_delta": _delta(
                    after.get("reviewed_false_negatives_per_minute"),
                    before.get("reviewed_false_negatives_per_minute"),
                ),
            }
        )

    same_source_video = baseline.get("source_video_path") == candidate.get("source_video_path")
    if not per_session and same_source_video:
        per_session.append(
            {
                "source_video_path": baseline.get("source_video_path"),
                "reviewed_non_dive_delta": _delta(
                    ((candidate.get("per_session_metrics") or [{}])[0]).get("reviewed_non_dive_count"),
                    ((baseline.get("per_session_metrics") or [{}])[0]).get("reviewed_non_dive_count"),
                ),
                "false_negative_delta": _delta(
                    ((candidate.get("per_session_metrics") or [{}])[0]).get("false_negative_count"),
                    ((baseline.get("per_session_metrics") or [{}])[0]).get("false_negative_count"),
                ),
                "reviewed_false_positives_per_minute_delta": _delta(
                    ((candidate.get("per_session_metrics") or [{}])[0]).get("reviewed_false_positives_per_minute"),
                    ((baseline.get("per_session_metrics") or [{}])[0]).get("reviewed_false_positives_per_minute"),
                ),
                "reviewed_false_negatives_per_minute_delta": _delta(
                    ((candidate.get("per_session_metrics") or [{}])[0]).get("reviewed_false_negatives_per_minute"),
                    ((baseline.get("per_session_metrics") or [{}])[0]).get("reviewed_false_negatives_per_minute"),
                ),
            }
        )

    result = {
        "baseline_session_id": baseline.get("session_id"),
        "candidate_session_id": candidate.get("session_id"),
        "baseline_source_video_path": baseline.get("source_video_path"),
        "candidate_source_video_path": candidate.get("source_video_path"),
        "same_source_video": same_source_video,
        "shared_session_ids": shared_session_ids,
        "reviewed_candidate_count_delta": _delta(candidate.get("reviewed_candidate_count"), baseline.get("reviewed_candidate_count")),
        "hard_negative_count_delta": _delta(candidate.get("hard_negative_count"), baseline.get("hard_negative_count")),
        "false_negative_count_delta": _delta(candidate.get("false_negative_count"), baseline.get("false_negative_count")),
        "threshold_recommendation_delta": {
            "best_precision_threshold_delta": _delta(
                ((candidate.get("threshold_recommendation") or {}).get("best_precision_under_recall_floor") or {}).get("threshold"),
                ((baseline.get("threshold_recommendation") or {}).get("best_precision_under_recall_floor") or {}).get("threshold"),
            ),
            "best_f1_threshold_delta": _delta(
                ((candidate.get("threshold_recommendation") or {}).get("best_f1") or {}).get("threshold"),
                ((baseline.get("threshold_recommendation") or {}).get("best_f1") or {}).get("threshold"),
            ),
            "best_f1_delta": _delta(
                ((candidate.get("threshold_recommendation") or {}).get("best_f1") or {}).get("f1"),
                ((baseline.get("threshold_recommendation") or {}).get("best_f1") or {}).get("f1"),
            ),
        },
        "failure_attribution_delta": {
            key: _delta(
                (candidate.get("failure_attribution") or {}).get(key, 0),
                (baseline.get("failure_attribution") or {}).get(key, 0),
            )
            for key in sorted(set((baseline.get("failure_attribution") or {}).keys()) | set((candidate.get("failure_attribution") or {}).keys()))
        },
        "per_session": per_session,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
