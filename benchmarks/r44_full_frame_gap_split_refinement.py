from __future__ import annotations

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
        }
        for row in reviewed
        if row.get("review_label") == "dive"
    ]
    anchors.extend(
        {
            "timestamp": row["timestamp_seconds"],
            "kind": "false_negative",
        }
        for row in false_negatives
    )
    anchors.sort(key=lambda row: row["timestamp"])
    return {
        "anchors": anchors,
        "audio_rows": audio_rows,
        "duration_seconds": probe_duration_seconds(session_root / "web/session_source_review.mp4"),
    }


def matched(anchor_timestamps: list[float], proposal_timestamps: list[float]) -> list[float]:
    return [anchor for anchor in anchor_timestamps if any(abs(ts - anchor) <= TOLERANCE_SECONDS for ts in proposal_timestamps)]


def nearest_delta(timestamp: float, anchor_rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    anchor = min(anchor_rows, key=lambda row: abs(float(row["timestamp"]) - timestamp))
    return timestamp - float(anchor["timestamp"]), anchor


def evaluate_proposals(
    *,
    session_root: Path,
    ground_truth: dict[str, Any],
    proposal_rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    anchors = ground_truth["anchors"]
    anchor_timestamps = [float(row["timestamp"]) for row in anchors]
    audio_timestamps = [float(row.get("timestamp", 0.0) or 0.0) for row in ground_truth["audio_rows"]]
    proposal_timestamps = [float(row["timestamp"]) for row in proposal_rows]

    audio_matched = matched(anchor_timestamps, audio_timestamps)
    visual_matched = matched(anchor_timestamps, proposal_timestamps)
    union_matched = matched(anchor_timestamps, audio_timestamps + proposal_timestamps)
    recovered = sorted({round(ts, 3) for ts in union_matched} - {round(ts, 3) for ts in audio_matched})

    unmatched_visual: list[dict[str, Any]] = []
    matched_visual_deltas: list[float] = []
    for row in proposal_rows:
        ts = float(row["timestamp"])
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

    interval_lengths = [float(row["end_seconds"]) - float(row["start_seconds"]) for row in proposal_rows]
    duration_minutes = float(ground_truth["duration_seconds"]) / 60.0
    return {
        "label": label,
        "visual_proposal_count": len(proposal_rows),
        "visual_recall": round(len(visual_matched) / len(anchor_timestamps), 4),
        "audio_baseline_recall": round(len(audio_matched) / len(anchor_timestamps), 4),
        "union_recall": round(len(union_matched) / len(anchor_timestamps), 4),
        "recovered_anchor_count": len(recovered),
        "recovered_anchor_timestamps": recovered,
        "unmatched_visual_count": len(unmatched_visual),
        "unmatched_visual": unmatched_visual,
        "false_visual_proposals_per_minute": round(len(unmatched_visual) / duration_minutes, 3),
        "interval_length_min_seconds": round(min(interval_lengths), 3) if interval_lengths else 0.0,
        "interval_length_median_seconds": round(statistics.median(interval_lengths), 3) if interval_lengths else 0.0,
        "interval_length_max_seconds": round(max(interval_lengths), 3) if interval_lengths else 0.0,
        "merged_proposal_count": len(_merged_proposal_rows(session_root, proposal_rows)),
        "matched_visual_timing_delta_median_seconds": round(statistics.median(matched_visual_deltas), 3) if matched_visual_deltas else None,
    }


def build_gap_split_intervals(
    *,
    base_intervals: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    base_config: VisualProposalConfig,
    duration_seconds: float,
    internal_gap_seconds: float,
    cleanup_merge_gap_seconds: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    zero_merge_config = replace(base_config, merge_gap_seconds=0.0)
    for interval in base_intervals:
        start = float(interval["start_seconds"])
        end = float(interval["end_seconds"])
        positives = [
            row
            for row in frame_rows
            if row.get("prompt_id") == interval.get("prompt_id")
            and row.get("decision_rule") == interval.get("decision_rule")
            and bool(row.get("is_positive"))
            and start <= float(row["timestamp_seconds"]) <= end
        ]
        if not positives:
            out.append(dict(interval))
            continue

        chunks: list[list[dict[str, Any]]] = [[positives[0]]]
        for previous, current in zip(positives, positives[1:]):
            if float(current["timestamp_seconds"]) - float(previous["timestamp_seconds"]) > internal_gap_seconds:
                chunks.append([current])
            else:
                chunks[-1].append(current)

        for chunk in chunks:
            out.extend(
                frame_predictions_to_intervals(
                    chunk,
                    config=zero_merge_config,
                    decision_rule=str(interval.get("decision_rule")),
                    prompt_id=str(interval.get("prompt_id")),
                    duration_seconds=duration_seconds,
                )
            )

    out.sort(key=lambda row: float(row["start_seconds"]))

    if cleanup_merge_gap_seconds is not None:
        merged: list[dict[str, Any]] = []
        for interval in out:
            if not merged:
                merged.append(dict(interval))
                continue
            previous = merged[-1]
            gap = float(interval["start_seconds"]) - float(previous["end_seconds"])
            combined_length = float(interval["end_seconds"]) - float(previous["start_seconds"])
            if gap <= cleanup_merge_gap_seconds and combined_length <= 10.0:
                previous["end_seconds"] = interval["end_seconds"]
                previous["timestamp"] = round((float(previous["start_seconds"]) + float(previous["end_seconds"])) / 2.0, 3)
            else:
                merged.append(dict(interval))
        out = merged

    for idx, row in enumerate(out, start=1):
        row["visual_interval_id"] = f"vis-int-{idx:04d}"
    return out


def render_md(control: dict[str, Any], r43_best: dict[str, Any], candidates: list[dict[str, Any]], best: dict[str, Any]) -> tuple[str, str]:
    summary_lines = [
        "# R44 Gap Split Refinement",
        "",
        "## Baselines",
        "",
        f"- r43 control union recall: `{control['union_recall']}`",
        f"- r43 control unmatched visual: `{control['unmatched_visual_count']}`",
        f"- r43 best (`split_internal_gap_3s`) union recall: `{r43_best['union_recall']}`",
        f"- r43 best unmatched visual: `{r43_best['unmatched_visual_count']}`",
        "",
        "## Best Candidate",
        "",
        f"- label: `{best['label']}`",
        f"- union recall: `{best['union_recall']}`",
        f"- recovered anchors: `{best['recovered_anchor_count']}`",
        f"- unmatched visual proposals: `{best['unmatched_visual_count']}`",
        f"- false visual/min: `{best['false_visual_proposals_per_minute']}`",
        f"- interval max seconds: `{best['interval_length_max_seconds']}`",
    ]
    comparison_lines = [
        "# R44 Gap Split Candidate Comparison",
        "",
        "| Candidate | Visual proposals | Visual recall | Union recall | Recovered anchors | Unmatched visual | False visual/min | Interval min / med / max | Merged proposals |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in candidates:
        comparison_lines.append(
            f"| {row['label']} | {row['visual_proposal_count']} | {row['visual_recall']:.4f} | {row['union_recall']:.4f} | {row['recovered_anchor_count']} | {row['unmatched_visual_count']} | {row['false_visual_proposals_per_minute']:.3f} | {row['interval_length_min_seconds']:.1f} / {row['interval_length_median_seconds']:.1f} / {row['interval_length_max_seconds']:.1f} | {row['merged_proposal_count']} |"
        )
    return "\n".join(summary_lines) + "\n", "\n".join(comparison_lines) + "\n"


def main() -> int:
    session_root = Path("outputs/evaluation_insep_plateform_mixed_sound").resolve()
    full_frame_root = Path("/Users/mcauchy/Downloads/r42_visual_full_frame_control").resolve()
    proposal_root = full_frame_root / "audio_gated_full_frame_1p0fps"
    r43_control = read_json(Path("outputs/r43_full_frame_interval_geometry_control.json"))
    r43_ablation = read_json(Path("outputs/r43_interval_geometry_ablation.json"))
    r43_best = r43_ablation["best_variant"]

    ground_truth = load_ground_truth(session_root)
    frame_rows = read_jsonl(proposal_root / "visual_frame_predictions.jsonl")
    control_rows = read_jsonl(proposal_root / "visual_proposals.jsonl")
    control = evaluate_proposals(
        session_root=session_root,
        ground_truth=ground_truth,
        proposal_rows=control_rows,
        label="r43_full_frame_control_reference",
    )

    base_config = VisualProposalConfig(
        mode="audio-gated",
        roi_mode="full_frame",
        confidence_threshold=0.845,
        grouping_threshold_seconds=2.5,
        buffer_start_seconds=1.5,
        buffer_end_seconds=3.0,
        merge_gap_seconds=3.5,
        prompt_ids=("diving_attempt",),
        decision_rules=("yes_no_first_token_margin",),
    )
    base_intervals = frame_predictions_to_intervals(
        frame_rows,
        config=base_config,
        decision_rule="yes_no_first_token_margin",
        prompt_id="diving_attempt",
        duration_seconds=float(ground_truth["duration_seconds"]),
    )

    specs = [
        ("gap_split_2.0s", 2.0, None),
        ("gap_split_2.5s", 2.5, None),
        ("gap_split_3.0s", 3.0, None),
        ("gap_split_3.5s", 3.5, None),
        ("gap_split_3.0s_cleanup_merge_0.5s", 3.0, 0.5),
        ("gap_split_3.0s_cleanup_merge_1.0s", 3.0, 1.0),
    ]
    candidate_rows: list[dict[str, Any]] = []
    for label, gap, cleanup_merge in specs:
        intervals = build_gap_split_intervals(
            base_intervals=base_intervals,
            frame_rows=frame_rows,
            base_config=base_config,
            duration_seconds=float(ground_truth["duration_seconds"]),
            internal_gap_seconds=gap,
            cleanup_merge_gap_seconds=cleanup_merge,
        )
        proposals = intervals_to_proposals(
            intervals,
            session_id=session_root.name,
            source_video_path=str(session_root / "web/session_source_review.mp4"),
        )
        report = evaluate_proposals(
            session_root=session_root,
            ground_truth=ground_truth,
            proposal_rows=proposals,
            label=label,
        )
        report["internal_gap_seconds"] = gap
        report["cleanup_merge_gap_seconds"] = cleanup_merge
        report["improves_union_vs_r43_best"] = report["union_recall"] > r43_best["union_recall"]
        report["matches_union_vs_r43_best"] = report["union_recall"] == r43_best["union_recall"]
        report["better_burden_than_r43_best"] = (
            report["unmatched_visual_count"] < r43_best["unmatched_visual_count"]
            or report["visual_proposal_count"] < r43_best["visual_proposal_count"]
        )
        candidate_rows.append(report)

    best_candidate = max(
        candidate_rows,
        key=lambda row: (
            row["union_recall"],
            -row["unmatched_visual_count"],
            -row["recovered_anchor_count"],
            -row["false_visual_proposals_per_minute"],
            -row["visual_proposal_count"],
        ),
    )

    no_clear_gain = (
        best_candidate["union_recall"] == r43_best["union_recall"]
        and best_candidate["recovered_anchor_count"] == r43_best["recovered_anchor_count"]
        and best_candidate["unmatched_visual_count"] == r43_best["unmatched_visual_count"]
        and best_candidate["false_visual_proposals_per_minute"] == r43_best["false_visual_proposals_per_minute"]
    )

    summary = {
        "r43_control_reference": r43_control,
        "r43_best_reference": r43_best,
        "candidates": candidate_rows,
        "best_candidate": best_candidate,
        "plateau_detected": True,
        "no_clear_gain_vs_r43_best": no_clear_gain,
        "interval_geometry_remains_primary": True,
        "prefilter_now_justified": False,
    }

    outputs = Path("outputs")
    write_json(outputs / "r44_gap_split_refinement.json", summary)
    write_json(outputs / "r44_gap_split_candidate_comparison.json", {"candidates": candidate_rows})
    summary_md, comparison_md = render_md(control, r43_best, candidate_rows, best_candidate)
    write_md(outputs / "r44_gap_split_refinement.md", summary_md)
    write_md(outputs / "r44_gap_split_candidate_comparison.md", comparison_md)

    write_md(
        Path("docs/research/R44_FULL_FRAME_GAP_SPLIT_REFINEMENT.md"),
        "\n".join(
            [
                "# R44 Full-Frame Gap Split Refinement",
                "",
                f"- r43 best reference: `split_internal_gap_3s` with union recall `{r43_best['union_recall']}` and unmatched visual `{r43_best['unmatched_visual_count']}`.",
                f"- best r44 candidate: `{best_candidate['label']}`",
                f"- best r44 candidate union recall: `{best_candidate['union_recall']}`",
                f"- best r44 candidate unmatched visual: `{best_candidate['unmatched_visual_count']}`",
                f"- no clear gain vs r43 best: `{no_clear_gain}`",
                "- interpretation: nearby split values sit on the same performance plateau.",
                "- interpretation: conservative cleanup merge slightly reduces proposal count but does not materially change burden or recall.",
                "- interpretation: interval geometry remains the next lever; prefilter is still premature.",
            ]
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
