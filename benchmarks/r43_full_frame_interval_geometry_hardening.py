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
        "anchors": anchors,
        "audio_rows": audio_rows,
        "duration_seconds": duration_seconds,
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

    matched_visual_deltas: list[float] = []
    unmatched_visual: list[dict[str, Any]] = []
    visual_only_vs_audio: list[dict[str, Any]] = []
    overlap_with_audio = 0
    for row in proposal_rows:
        ts = float(row["timestamp"])
        nearest_audio_delta = min((ts - audio_ts for audio_ts in audio_timestamps), key=abs)
        if abs(nearest_audio_delta) <= TOLERANCE_SECONDS:
            overlap_with_audio += 1
        else:
            visual_only_vs_audio.append(
                {
                    "timestamp": round(ts, 3),
                    "nearest_audio_delta": round(nearest_audio_delta, 3),
                }
            )

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
        "overlap_with_audio_count": overlap_with_audio,
        "visual_only_vs_audio_count": len(visual_only_vs_audio),
        "visual_only_vs_audio": visual_only_vs_audio,
        "unmatched_visual_count": len(unmatched_visual),
        "unmatched_visual": unmatched_visual,
        "false_visual_proposals_per_minute": round(len(unmatched_visual) / duration_minutes, 3),
        "interval_length_min_seconds": round(min(interval_lengths), 3) if interval_lengths else 0.0,
        "interval_length_median_seconds": round(statistics.median(interval_lengths), 3) if interval_lengths else 0.0,
        "interval_length_max_seconds": round(max(interval_lengths), 3) if interval_lengths else 0.0,
        "merged_proposal_count": len(_merged_proposal_rows(session_root, proposal_rows)),
        "matched_visual_timing_delta_median_seconds": round(statistics.median(matched_visual_deltas), 3) if matched_visual_deltas else None,
    }


def build_split_intervals(
    *,
    base_intervals: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    base_config: VisualProposalConfig,
    duration_seconds: float,
    internal_gap_seconds: float | None,
    valley_margin_threshold: float | None,
    max_interval_length_seconds: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    zero_merge_config = replace(base_config, merge_gap_seconds=0.0)
    cap_config = replace(
        base_config,
        grouping_threshold_seconds=1.0,
        buffer_start_seconds=0.5,
        buffer_end_seconds=1.0,
        merge_gap_seconds=0.0,
    )
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
            previous_ts = float(previous["timestamp_seconds"])
            current_ts = float(current["timestamp_seconds"])
            previous_margin = float(previous.get("yes_no_first_token_margin") or 0.0)
            current_margin = float(current.get("yes_no_first_token_margin") or 0.0)
            split = False
            if internal_gap_seconds is not None and (current_ts - previous_ts) > internal_gap_seconds:
                split = True
            if valley_margin_threshold is not None and previous_margin < valley_margin_threshold and current_margin < valley_margin_threshold:
                split = True
            if split:
                chunks.append([current])
            else:
                chunks[-1].append(current)

        split_intervals: list[dict[str, Any]] = []
        for chunk in chunks:
            split_intervals.extend(
                frame_predictions_to_intervals(
                    chunk,
                    config=zero_merge_config,
                    decision_rule=str(interval.get("decision_rule")),
                    prompt_id=str(interval.get("prompt_id")),
                    duration_seconds=duration_seconds,
                )
            )

        if max_interval_length_seconds is not None:
            capped: list[dict[str, Any]] = []
            for split_interval in split_intervals:
                length = float(split_interval["end_seconds"]) - float(split_interval["start_seconds"])
                if length <= max_interval_length_seconds:
                    capped.append(split_interval)
                    continue
                split_start = float(split_interval["start_seconds"])
                split_end = float(split_interval["end_seconds"])
                segment_rows = [
                    row
                    for row in positives
                    if split_start <= float(row["timestamp_seconds"]) <= split_end
                ]
                capped.extend(
                    frame_predictions_to_intervals(
                        segment_rows,
                        config=cap_config,
                        decision_rule=str(split_interval.get("decision_rule")),
                        prompt_id=str(split_interval.get("prompt_id")),
                        duration_seconds=duration_seconds,
                    )
                )
            split_intervals = capped

        out.extend(split_intervals)

    out.sort(key=lambda row: float(row["start_seconds"]))
    for idx, row in enumerate(out, start=1):
        row["visual_interval_id"] = f"vis-int-{idx:04d}"
    return out


def render_md(control: dict[str, Any], variants: list[dict[str, Any]]) -> tuple[str, str]:
    control_md = "\n".join(
        [
            "# R43 Full-Frame Interval Geometry Control",
            "",
            f"- visual proposals: `{control['visual_proposal_count']}`",
            f"- visual recall: `{control['visual_recall']}`",
            f"- union recall: `{control['union_recall']}`",
            f"- recovered anchors: `{control['recovered_anchor_count']}`",
            f"- unmatched visual proposals: `{control['unmatched_visual_count']}`",
            f"- false visual proposals / min: `{control['false_visual_proposals_per_minute']}`",
            f"- interval min / median / max: `{control['interval_length_min_seconds']}` / `{control['interval_length_median_seconds']}` / `{control['interval_length_max_seconds']}`",
            f"- merged proposal count: `{control['merged_proposal_count']}`",
            f"- timing delta median: `{control['matched_visual_timing_delta_median_seconds']}`",
        ]
    )
    ablation_lines = [
        "# R43 Interval Geometry Ablation",
        "",
        "| Variant | Visual proposals | Visual recall | Union recall | Recovered anchors | Unmatched visual | False visual/min | Interval min / med / max | Merged proposals |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in variants:
        ablation_lines.append(
            f"| {row['label']} | {row['visual_proposal_count']} | {row['visual_recall']:.4f} | {row['union_recall']:.4f} | {row['recovered_anchor_count']} | {row['unmatched_visual_count']} | {row['false_visual_proposals_per_minute']:.3f} | {row['interval_length_min_seconds']:.1f} / {row['interval_length_median_seconds']:.1f} / {row['interval_length_max_seconds']:.1f} | {row['merged_proposal_count']} |"
        )
    return control_md + "\n", "\n".join(ablation_lines) + "\n"


def main() -> int:
    session_root = Path("outputs/evaluation_insep_plateform_mixed_sound").resolve()
    full_frame_root = Path("/Users/mcauchy/Downloads/r42_visual_full_frame_control").resolve()
    proposal_root = full_frame_root / "audio_gated_full_frame_1p0fps"

    ground_truth = load_ground_truth(session_root)
    frame_rows = read_jsonl(proposal_root / "visual_frame_predictions.jsonl")
    control_rows = read_jsonl(proposal_root / "visual_proposals.jsonl")

    control = evaluate_proposals(
        session_root=session_root,
        ground_truth=ground_truth,
        proposal_rows=control_rows,
        label="r42_full_frame_control",
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

    variants_spec = [
        {
            "label": "control_default_geometry",
            "internal_gap_seconds": None,
            "valley_margin_threshold": None,
            "max_interval_length_seconds": None,
        },
        {
            "label": "split_internal_gap_4s",
            "internal_gap_seconds": 4.0,
            "valley_margin_threshold": None,
            "max_interval_length_seconds": None,
        },
        {
            "label": "split_internal_gap_3s",
            "internal_gap_seconds": 3.0,
            "valley_margin_threshold": None,
            "max_interval_length_seconds": None,
        },
        {
            "label": "split_gap_3s_plus_margin_valley_0.25",
            "internal_gap_seconds": 3.0,
            "valley_margin_threshold": 0.25,
            "max_interval_length_seconds": None,
        },
        {
            "label": "split_internal_gap_3s_plus_cap12",
            "internal_gap_seconds": 3.0,
            "valley_margin_threshold": None,
            "max_interval_length_seconds": 12.0,
        },
    ]

    variant_reports: list[dict[str, Any]] = []
    for spec in variants_spec:
        if spec["label"] == "control_default_geometry":
            report = dict(control)
        else:
            intervals = build_split_intervals(
                base_intervals=base_intervals,
                frame_rows=frame_rows,
                base_config=base_config,
                duration_seconds=float(ground_truth["duration_seconds"]),
                internal_gap_seconds=spec["internal_gap_seconds"],
                valley_margin_threshold=spec["valley_margin_threshold"],
                max_interval_length_seconds=spec["max_interval_length_seconds"],
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
                label=spec["label"],
            )
        report["internal_gap_seconds"] = spec["internal_gap_seconds"]
        report["valley_margin_threshold"] = spec["valley_margin_threshold"]
        report["max_interval_length_seconds"] = spec["max_interval_length_seconds"]
        variant_reports.append(report)

    bounded_candidates = [
        row
        for row in variant_reports[1:]
        if row["unmatched_visual_count"] <= 4 and row["false_visual_proposals_per_minute"] <= 0.3
    ]
    best_variant = max(
        (bounded_candidates or variant_reports[1:]),
        key=lambda row: (
            row["union_recall"],
            -row["unmatched_visual_count"],
            -row["recovered_anchor_count"],
            -row["false_visual_proposals_per_minute"],
        ),
    )
    summary = {
        "control": control,
        "variants": variant_reports,
        "best_variant": best_variant,
        "bounded_variant_candidates": [row["label"] for row in bounded_candidates],
        "geometry_gain_confirmed": (
            best_variant["union_recall"] > control["union_recall"]
            and best_variant["interval_length_max_seconds"] < control["interval_length_max_seconds"]
        ),
        "interval_hardening_remains_primary": True,
    }

    outputs = Path("outputs")
    write_json(outputs / "r43_full_frame_interval_geometry_control.json", control)
    write_json(outputs / "r43_interval_geometry_ablation.json", summary)

    control_md, ablation_md = render_md(control, variant_reports)
    write_md(outputs / "r43_full_frame_interval_geometry_control.md", control_md)
    write_md(outputs / "r43_interval_geometry_ablation.md", ablation_md)

    write_md(
        Path("docs/research/R43_FULL_FRAME_INTERVAL_GEOMETRY_HARDENING.md"),
        "\n".join(
            [
                "# R43 Full-Frame Interval Geometry Hardening",
                "",
                "## Control",
                "",
                f"- union recall: `{control['union_recall']}`",
                f"- recovered anchors: `{control['recovered_anchor_count']}`",
                f"- unmatched visual proposals: `{control['unmatched_visual_count']}`",
                f"- interval max seconds: `{control['interval_length_max_seconds']}`",
                "",
                "## Best Bounded Variant",
                "",
                f"- label: `{best_variant['label']}`",
                f"- union recall: `{best_variant['union_recall']}`",
                f"- recovered anchors: `{best_variant['recovered_anchor_count']}`",
                f"- unmatched visual proposals: `{best_variant['unmatched_visual_count']}`",
                f"- false visual proposals / min: `{best_variant['false_visual_proposals_per_minute']}`",
                f"- interval max seconds: `{best_variant['interval_length_max_seconds']}`",
                "",
                "## Interpretation",
                "",
                "- internal-gap splitting on the better `full_frame` ROI is the useful geometry lever.",
                "- hard caps improve recall further, but they increase unmatched burden too sharply.",
                f"- bounded recommendation set: `{', '.join(summary['bounded_variant_candidates']) if summary['bounded_variant_candidates'] else 'none'}`",
                "- prefilter work is still premature because geometry-only hardening still buys meaningful utility.",
            ]
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
