from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from divesensei.workflows.visual_vlm_proposals import (
    VisualProposalConfig,
    _merged_proposal_rows,
    frame_predictions_to_intervals,
    intervals_to_proposals,
)


TOLERANCE_SECONDS = 2.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def probe_duration_seconds(video_path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
        ).strip()
    )


def load_ground_truth(session_root: Path) -> dict[str, Any]:
    reviewed = read_jsonl(session_root / "exports/evaluation-review/reviewed_candidates.jsonl")
    false_negatives = read_jsonl(session_root / "exports/evaluation-review/false_negatives.jsonl")
    audio_rows = read_jsonl(session_root / "proposal_diagnostics.jsonl")
    anchors = [
        {
            "timestamp": row["timestamp_seconds"],
            "kind": "audio_reviewed_dive",
            "source": row.get("source_candidate_id") or row.get("proposal_id"),
        }
        for row in reviewed
        if row.get("review_label") == "dive"
    ]
    anchors.extend(
        {
            "timestamp": row["timestamp_seconds"],
            "kind": "false_negative",
            "source": row.get("review_annotation_id") or row.get("proposal_id"),
        }
        for row in false_negatives
    )
    anchors.sort(key=lambda row: row["timestamp"])
    duration_seconds = probe_duration_seconds(session_root / "web/session_source_review.mp4")
    return {
        "reviewed_candidates": reviewed,
        "false_negatives": false_negatives,
        "audio_rows": audio_rows,
        "anchors": anchors,
        "duration_seconds": duration_seconds,
    }


def nearest_delta(timestamp: float, anchors: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    anchor = min(anchors, key=lambda row: abs(float(row["timestamp"]) - timestamp))
    return timestamp - float(anchor["timestamp"]), anchor


def evaluate_visual_run(
    *,
    session_root: Path,
    ground_truth: dict[str, Any],
    visual_rows: list[dict[str, Any]],
    roi_mode: str,
    label: str,
) -> dict[str, Any]:
    anchors = ground_truth["anchors"]
    anchor_timestamps = [float(row["timestamp"]) for row in anchors]
    audio_rows = ground_truth["audio_rows"]
    audio_timestamps = [float(row.get("timestamp", 0.0) or 0.0) for row in audio_rows]
    visual_timestamps = [float(row["timestamp"]) for row in visual_rows]

    def matched(ts_list: list[float]) -> list[float]:
        return [anchor for anchor in anchor_timestamps if any(abs(ts - anchor) <= TOLERANCE_SECONDS for ts in ts_list)]

    audio_matched = matched(audio_timestamps)
    visual_matched = matched(visual_timestamps)
    union_matched = matched(audio_timestamps + visual_timestamps)

    recovered = sorted({round(ts, 3) for ts in union_matched} - {round(ts, 3) for ts in audio_matched})

    visual_only_vs_audio: list[dict[str, Any]] = []
    overlap_with_audio: list[dict[str, Any]] = []
    matched_visual_deltas: list[float] = []
    unmatched_visual: list[dict[str, Any]] = []
    for row in visual_rows:
        ts = float(row["timestamp"])
        audio_delta = min((ts - audio_ts for audio_ts in audio_timestamps), key=abs)
        target = {
            "timestamp": round(ts, 3),
            "nearest_audio_delta": round(audio_delta, 3),
        }
        if abs(audio_delta) <= TOLERANCE_SECONDS:
            overlap_with_audio.append(target)
        else:
            visual_only_vs_audio.append(target)

        delta, nearest_anchor = nearest_delta(ts, anchors)
        if abs(delta) <= TOLERANCE_SECONDS:
            matched_visual_deltas.append(delta)
        else:
            unmatched_visual.append(
                {
                    "timestamp": round(ts, 3),
                    "nearest_anchor": round(float(nearest_anchor["timestamp"]), 3),
                    "nearest_anchor_kind": nearest_anchor["kind"],
                    "delta_seconds": round(delta, 3),
                }
            )

    interval_lengths = [float(row["end_seconds"]) - float(row["start_seconds"]) for row in visual_rows]
    merged_rows = _merged_proposal_rows(session_root, visual_rows)
    duration_minutes = float(ground_truth["duration_seconds"]) / 60.0

    return {
        "label": label,
        "roi_mode": roi_mode,
        "anchor_count": len(anchors),
        "audio_baseline_matched_anchor_count": len(audio_matched),
        "audio_baseline_recall": round(len(audio_matched) / len(anchors), 4),
        "visual_proposal_count": len(visual_rows),
        "visual_matched_anchor_count": len(visual_matched),
        "visual_recall": round(len(visual_matched) / len(anchors), 4),
        "union_matched_anchor_count": len(union_matched),
        "union_recall": round(len(union_matched) / len(anchors), 4),
        "recovered_anchor_count": len(recovered),
        "recovered_anchor_timestamps": recovered,
        "overlap_with_audio_count": len(overlap_with_audio),
        "visual_only_vs_audio_count": len(visual_only_vs_audio),
        "visual_only_vs_audio": visual_only_vs_audio,
        "unmatched_visual_count": len(unmatched_visual),
        "unmatched_visual": unmatched_visual,
        "false_visual_proposals_per_minute": round(len(unmatched_visual) / duration_minutes, 3),
        "interval_length_min_seconds": round(min(interval_lengths), 3) if interval_lengths else 0.0,
        "interval_length_median_seconds": round(statistics.median(interval_lengths), 3) if interval_lengths else 0.0,
        "interval_length_max_seconds": round(max(interval_lengths), 3) if interval_lengths else 0.0,
        "merged_proposal_count": len(merged_rows),
        "matched_visual_timing_delta_median_seconds": round(statistics.median(matched_visual_deltas), 3) if matched_visual_deltas else None,
    }


def split_long_intervals(
    intervals: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    *,
    max_interval_length_seconds: float,
    split_grouping_threshold_seconds: float,
    duration_seconds: float,
    config: VisualProposalConfig,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for interval in intervals:
        length = float(interval["end_seconds"]) - float(interval["start_seconds"])
        if length <= max_interval_length_seconds:
            out.append(dict(interval))
            continue
        positives = [
            row
            for row in frame_rows
            if row.get("prompt_id") == interval.get("prompt_id")
            and row.get("decision_rule") == interval.get("decision_rule")
            and bool(row.get("is_positive"))
            and float(interval["start_seconds"]) <= float(row["timestamp_seconds"]) <= float(interval["end_seconds"])
        ]
        if not positives:
            out.append(dict(interval))
            continue
        split_config = replace(config, grouping_threshold_seconds=split_grouping_threshold_seconds, merge_gap_seconds=0.0)
        split_intervals = frame_predictions_to_intervals(
            positives,
            config=split_config,
            decision_rule=str(interval.get("decision_rule")),
            prompt_id=str(interval.get("prompt_id")),
            duration_seconds=duration_seconds,
        )
        if len(split_intervals) <= 1:
            out.append(dict(interval))
            continue
        out.extend(split_intervals)
    out.sort(key=lambda row: float(row["start_seconds"]))
    for idx, row in enumerate(out, start=1):
        row["visual_interval_id"] = f"vis-int-{idx:04d}"
    return out


def run_ablation(
    *,
    session_root: Path,
    ground_truth: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    base_config: VisualProposalConfig,
) -> list[dict[str, Any]]:
    variants = [
        {
            "name": "default_center_pool",
            "config": base_config,
            "max_interval_length_seconds": None,
            "split_grouping_threshold_seconds": None,
        },
        {
            "name": "tighter_grouping_buffers",
            "config": replace(
                base_config,
                grouping_threshold_seconds=1.0,
                buffer_start_seconds=0.5,
                buffer_end_seconds=1.0,
                merge_gap_seconds=1.0,
            ),
            "max_interval_length_seconds": None,
            "split_grouping_threshold_seconds": None,
        },
        {
            "name": "tighter_grouping_plus_cap12",
            "config": replace(
                base_config,
                grouping_threshold_seconds=1.0,
                buffer_start_seconds=0.5,
                buffer_end_seconds=1.0,
                merge_gap_seconds=1.0,
            ),
            "max_interval_length_seconds": 12.0,
            "split_grouping_threshold_seconds": 0.75,
        },
        {
            "name": "tightest_gap_cap8",
            "config": replace(
                base_config,
                grouping_threshold_seconds=0.75,
                buffer_start_seconds=0.5,
                buffer_end_seconds=0.75,
                merge_gap_seconds=0.75,
            ),
            "max_interval_length_seconds": 8.0,
            "split_grouping_threshold_seconds": 0.5,
        },
    ]

    reports: list[dict[str, Any]] = []
    for variant in variants:
        config = variant["config"]
        intervals = frame_predictions_to_intervals(
            frame_rows,
            config=config,
            decision_rule="yes_no_first_token_margin",
            prompt_id="diving_attempt",
            duration_seconds=float(ground_truth["duration_seconds"]),
        )
        if variant["max_interval_length_seconds"] is not None:
            intervals = split_long_intervals(
                intervals,
                frame_rows,
                max_interval_length_seconds=float(variant["max_interval_length_seconds"]),
                split_grouping_threshold_seconds=float(variant["split_grouping_threshold_seconds"]),
                duration_seconds=float(ground_truth["duration_seconds"]),
                config=config,
            )
        proposals = intervals_to_proposals(
            intervals,
            session_id=session_root.name,
            source_video_path=str(session_root / "web/session_source_review.mp4"),
        )
        report = evaluate_visual_run(
            session_root=session_root,
            ground_truth=ground_truth,
            visual_rows=proposals,
            roi_mode=str(config.roi_mode),
            label=variant["name"],
        )
        report["grouping_threshold_seconds"] = config.grouping_threshold_seconds
        report["buffer_start_seconds"] = config.buffer_start_seconds
        report["buffer_end_seconds"] = config.buffer_end_seconds
        report["merge_gap_seconds"] = config.merge_gap_seconds
        report["max_interval_length_seconds"] = variant["max_interval_length_seconds"]
        reports.append(report)
    return reports


def render_md(
    *,
    full_frame: dict[str, Any],
    center_pool: dict[str, Any],
    ablation_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    full_md = "\n".join(
        [
            "# R42 Full-Frame Control",
            "",
            f"- visual proposals: `{full_frame['visual_proposal_count']}`",
            f"- visual recall: `{full_frame['visual_recall']}`",
            f"- union recall: `{full_frame['union_recall']}`",
            f"- recovered anchors over audio baseline: `{full_frame['recovered_anchor_count']}`",
            f"- unmatched visual proposals: `{full_frame['unmatched_visual_count']}`",
            f"- false visual proposals / min: `{full_frame['false_visual_proposals_per_minute']}`",
            f"- interval length min / median / max: `{full_frame['interval_length_min_seconds']}` / `{full_frame['interval_length_median_seconds']}` / `{full_frame['interval_length_max_seconds']}`",
        ]
    )
    comparison_lines = [
        "# R42 ROI Comparison",
        "",
        "| ROI | Visual proposals | Visual recall | Union recall | Recovered anchors | Unmatched visual | False visual/min | Interval min / med / max |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in (full_frame, center_pool):
        comparison_lines.append(
            f"| {row['roi_mode']} | {row['visual_proposal_count']} | {row['visual_recall']:.4f} | {row['union_recall']:.4f} | {row['recovered_anchor_count']} | {row['unmatched_visual_count']} | {row['false_visual_proposals_per_minute']:.3f} | {row['interval_length_min_seconds']:.1f} / {row['interval_length_median_seconds']:.1f} / {row['interval_length_max_seconds']:.1f} |"
        )
    ablation_lines = [
        "# R42 Interval-Shaping Ablation",
        "",
        "| Variant | Visual proposals | Visual recall | Union recall | Recovered anchors | Unmatched visual | False visual/min | Interval min / med / max |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ablation_rows:
        ablation_lines.append(
            f"| {row['label']} | {row['visual_proposal_count']} | {row['visual_recall']:.4f} | {row['union_recall']:.4f} | {row['recovered_anchor_count']} | {row['unmatched_visual_count']} | {row['false_visual_proposals_per_minute']:.3f} | {row['interval_length_min_seconds']:.1f} / {row['interval_length_median_seconds']:.1f} / {row['interval_length_max_seconds']:.1f} |"
        )
    return full_md + "\n", "\n".join(comparison_lines) + "\n", "\n".join(ablation_lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="R42 ROI vs interval shaping ablation.")
    parser.add_argument("--session-root", default="outputs/evaluation_insep_plateform_mixed_sound")
    parser.add_argument("--full-frame-root", required=True)
    parser.add_argument("--center-pool-root", default="/tmp/r41_bundle_inspect_2/r41_remote_gpu_results_center_pool")
    args = parser.parse_args()

    session_root = Path(args.session_root).resolve()
    full_root = Path(args.full_frame_root).resolve()
    center_root = Path(args.center_pool_root).resolve()
    ground_truth = load_ground_truth(session_root)

    full_rows = read_jsonl(full_root / "audio_gated_full_frame_1p0fps" / "visual_proposals.jsonl")
    center_rows = read_jsonl(center_root / "audio_gated_center_pool_1p0fps" / "visual_proposals.jsonl")
    center_frame_rows = read_jsonl(center_root / "audio_gated_center_pool_1p0fps" / "visual_frame_predictions.jsonl")

    full_report = evaluate_visual_run(
        session_root=session_root,
        ground_truth=ground_truth,
        visual_rows=full_rows,
        roi_mode="full_frame",
        label="full_frame_control",
    )
    center_report = evaluate_visual_run(
        session_root=session_root,
        ground_truth=ground_truth,
        visual_rows=center_rows,
        roi_mode="center_pool",
        label="center_pool_baseline",
    )

    base_config = VisualProposalConfig(
        mode="audio-gated",
        roi_mode="center_pool",
        confidence_threshold=0.845,
        grouping_threshold_seconds=2.5,
        buffer_start_seconds=1.5,
        buffer_end_seconds=3.0,
        merge_gap_seconds=3.5,
        prompt_ids=("diving_attempt",),
        decision_rules=("yes_no_first_token_margin",),
    )
    ablation_rows = run_ablation(
        session_root=session_root,
        ground_truth=ground_truth,
        frame_rows=center_frame_rows,
        base_config=base_config,
    )

    outputs = Path("outputs")
    write_json(outputs / "r42_visual_full_frame_control.json", full_report)
    write_json(outputs / "r42_visual_roi_comparison.json", {"full_frame": full_report, "center_pool": center_report})
    write_json(outputs / "r42_interval_shaping_ablation.json", {"variants": ablation_rows})

    full_md, comparison_md, ablation_md = render_md(full_frame=full_report, center_pool=center_report, ablation_rows=ablation_rows)
    write_md(outputs / "r42_visual_full_frame_control.md", full_md)
    write_md(outputs / "r42_visual_roi_comparison.md", comparison_md)
    write_md(outputs / "r42_interval_shaping_ablation.md", ablation_md)

    best_ablation = max(
        ablation_rows,
        key=lambda row: (
            row["union_recall"],
            -row["unmatched_visual_count"],
            -row["recovered_anchor_count"],
            -row["false_visual_proposals_per_minute"],
        ),
    )
    interval_shaping_primary = (
        best_ablation["union_recall"] > center_report["union_recall"]
        and best_ablation["interval_length_max_seconds"] < center_report["interval_length_max_seconds"]
    )
    conclusion = {
        "full_frame": full_report,
        "center_pool": center_report,
        "best_interval_variant": best_ablation,
        "roi_value_confirmed": center_report["union_recall"] > full_report["union_recall"],
        "interval_shaping_primary_bottleneck": interval_shaping_primary,
    }
    write_md(
        Path("docs/research/R42_VISUAL_ROI_AND_INTERVAL_SHAPING.md"),
        "\n".join(
            [
                "# R42 Visual ROI And Interval Shaping",
                "",
                "## ROI Comparison",
                "",
                f"- full-frame recovered anchors: `{full_report['recovered_anchor_count']}`",
                f"- full-frame union recall: `{full_report['union_recall']}`",
                f"- full-frame unmatched visual proposals: `{full_report['unmatched_visual_count']}`",
                f"- center-pool recovered anchors: `{center_report['recovered_anchor_count']}`",
                f"- center-pool union recall: `{center_report['union_recall']}`",
                f"- center-pool unmatched visual proposals: `{center_report['unmatched_visual_count']}`",
                "",
                "## Interval Shaping",
                "",
                f"- best interval variant: `{best_ablation['label']}`",
                f"- best interval variant union recall: `{best_ablation['union_recall']}`",
                f"- best interval variant recovered anchors: `{best_ablation['recovered_anchor_count']}`",
                f"- best interval variant unmatched visual proposals: `{best_ablation['unmatched_visual_count']}`",
                f"- best interval variant max interval seconds: `{best_ablation['interval_length_max_seconds']}`",
                "",
                "## Interpretation",
                "",
                f"- ROI value confirmed: `{conclusion['roi_value_confirmed']}`",
                f"- interval shaping primary bottleneck: `{conclusion['interval_shaping_primary_bottleneck']}`",
                "- conclusion: `full_frame` is the better ROI on this session; `center_pool` is not the main lever.",
                "- conclusion: interval shaping changes recall and interval width materially, but the current tighter variants buy recall by adding too many extra review items.",
            ]
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
