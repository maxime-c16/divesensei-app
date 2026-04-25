from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from divesensei.workflows.visual_vlm_proposals import (
    VisualProposalConfig,
    _merged_proposal_rows,
    frame_predictions_to_intervals,
    intervals_to_proposals,
)


TOLERANCE_SECONDS = 2.0
INTERNAL_GAP_SECONDS = 3.0


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


@dataclass(frozen=True)
class FragmentGeometrySpec:
    label: str
    center_strategy: str
    support_strategy: str
    min_positive_frames: int = 1
    min_peak_score: float = 0.845
    min_margin_sum: float = 0.0


def split_interval_chunks(
    *,
    interval: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    internal_gap_seconds: float,
) -> list[list[dict[str, Any]]]:
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
        return []
    positives.sort(key=lambda row: float(row["timestamp_seconds"]))
    chunks: list[list[dict[str, Any]]] = [[positives[0]]]
    for previous, current in zip(positives, positives[1:]):
        if float(current["timestamp_seconds"]) - float(previous["timestamp_seconds"]) > internal_gap_seconds:
            chunks.append([current])
        else:
            chunks[-1].append(current)
    return chunks


def choose_center_timestamp(chunk: list[dict[str, Any]], strategy: str) -> float:
    if strategy == "median_timestamp":
        return float(chunk[int(len(chunk) / 2)]["timestamp_seconds"])
    if strategy == "highest_score_frame":
        top = max(chunk, key=lambda row: float(row.get("score", 0.0) or 0.0))
        return float(top["timestamp_seconds"])
    if strategy == "support_weighted":
        weighted = []
        for row in chunk:
            score = float(row.get("score", 0.0) or 0.0)
            margin = float(row.get("yes_no_first_token_margin", 0.0) or 0.0)
            weight = max(0.0, margin) + max(0.0, score - 0.845) + 1e-6
            weighted.append((float(row["timestamp_seconds"]), weight))
        total_weight = sum(weight for _, weight in weighted)
        return float(sum(ts * weight for ts, weight in weighted) / max(total_weight, 1e-6))
    if strategy == "densest_cluster":
        if len(chunk) <= 2:
            return float(chunk[int(len(chunk) / 2)]["timestamp_seconds"])
        best_idx = 0
        best_span = float("inf")
        for idx in range(len(chunk) - 2):
            t0 = float(chunk[idx]["timestamp_seconds"])
            t2 = float(chunk[idx + 2]["timestamp_seconds"])
            span = t2 - t0
            if span < best_span:
                best_span = span
                best_idx = idx + 1
        return float(chunk[best_idx]["timestamp_seconds"])
    raise ValueError(f"Unsupported center strategy: {strategy}")


def fragment_support_ok(chunk: list[dict[str, Any]], spec: FragmentGeometrySpec) -> bool:
    frame_count = len(chunk)
    peak_score = max(float(row.get("score", 0.0) or 0.0) for row in chunk)
    margin_sum = sum(max(0.0, float(row.get("yes_no_first_token_margin", 0.0) or 0.0)) for row in chunk)
    if spec.support_strategy == "none":
        return True
    if spec.support_strategy == "density_aware":
        return frame_count >= spec.min_positive_frames or peak_score >= spec.min_peak_score
    if spec.support_strategy == "density_and_margin":
        return (
            frame_count >= spec.min_positive_frames
            and peak_score >= spec.min_peak_score
            and margin_sum >= spec.min_margin_sum
        )
    raise ValueError(f"Unsupported support strategy: {spec.support_strategy}")


def build_fragment_geometry_proposals(
    *,
    session_root: Path,
    frame_rows: list[dict[str, Any]],
    base_intervals: list[dict[str, Any]],
    base_config: VisualProposalConfig,
    duration_seconds: float,
    spec: FragmentGeometrySpec,
) -> list[dict[str, Any]]:
    zero_merge_config = replace(base_config, merge_gap_seconds=0.0)
    out_intervals: list[dict[str, Any]] = []
    for interval in base_intervals:
        chunks = split_interval_chunks(interval=interval, frame_rows=frame_rows, internal_gap_seconds=INTERNAL_GAP_SECONDS)
        if not chunks:
            continue
        for chunk in chunks:
            if not fragment_support_ok(chunk, spec):
                continue
            split_intervals = frame_predictions_to_intervals(
                chunk,
                config=zero_merge_config,
                decision_rule=str(interval.get("decision_rule")),
                prompt_id=str(interval.get("prompt_id")),
                duration_seconds=duration_seconds,
            )
            for split_interval in split_intervals:
                center_ts = choose_center_timestamp(chunk, spec.center_strategy)
                split_interval["anchor_timestamp_seconds"] = round(center_ts, 3)
                out_intervals.append(split_interval)
    out_intervals.sort(key=lambda row: float(row["start_seconds"]))
    for idx, row in enumerate(out_intervals, start=1):
        row["visual_interval_id"] = f"vis-int-{idx:04d}"
    return intervals_to_proposals(
        out_intervals,
        session_id=session_root.name,
        source_video_path=str(session_root / "web/session_source_review.mp4"),
    )


def render_md(reference: dict[str, Any], candidates: list[dict[str, Any]], best: dict[str, Any]) -> tuple[str, str]:
    summary = [
        "# R45 Fragment Scoring And Event-Center Geometry",
        "",
        "## Reference",
        "",
        f"- label: `{reference['label']}`",
        f"- visual proposals: `{reference['visual_proposal_count']}`",
        f"- visual recall: `{reference['visual_recall']}`",
        f"- union recall: `{reference['union_recall']}`",
        f"- recovered anchors: `{reference['recovered_anchor_count']}`",
        f"- unmatched visual proposals: `{reference['unmatched_visual_count']}`",
        f"- false visual/min: `{reference['false_visual_proposals_per_minute']}`",
        "",
        "## Best Candidate",
        "",
        f"- label: `{best['label']}`",
        f"- center strategy: `{best['center_strategy']}`",
        f"- support strategy: `{best['support_strategy']}`",
        f"- visual proposals: `{best['visual_proposal_count']}`",
        f"- visual recall: `{best['visual_recall']}`",
        f"- union recall: `{best['union_recall']}`",
        f"- recovered anchors: `{best['recovered_anchor_count']}`",
        f"- unmatched visual proposals: `{best['unmatched_visual_count']}`",
        f"- false visual/min: `{best['false_visual_proposals_per_minute']}`",
    ]
    comparison = [
        "# R45 Fragment Geometry Candidate Comparison",
        "",
        "| Candidate | Center strategy | Support strategy | Visual proposals | Visual recall | Union recall | Recovered anchors | Unmatched visual | False visual/min | Interval min / med / max | Merged proposals |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in candidates:
        comparison.append(
            f"| {row['label']} | `{row['center_strategy']}` | `{row['support_strategy']}` | {row['visual_proposal_count']} | {row['visual_recall']:.4f} | {row['union_recall']:.4f} | {row['recovered_anchor_count']} | {row['unmatched_visual_count']} | {row['false_visual_proposals_per_minute']:.3f} | {row['interval_length_min_seconds']:.1f} / {row['interval_length_median_seconds']:.1f} / {row['interval_length_max_seconds']:.1f} | {row['merged_proposal_count']} |"
        )
    return "\n".join(summary) + "\n", "\n".join(comparison) + "\n"


def main() -> int:
    session_root = Path("outputs/evaluation_insep_plateform_mixed_sound").resolve()
    full_frame_root = Path("/Users/mcauchy/Downloads/r42_visual_full_frame_control").resolve()
    proposal_root = full_frame_root / "audio_gated_full_frame_1p0fps"
    r44_summary = read_json(Path("outputs/r44_gap_split_refinement.json"))
    reference = r44_summary["r43_best_reference"]

    ground_truth = load_ground_truth(session_root)
    frame_rows = read_jsonl(proposal_root / "visual_frame_predictions.jsonl")
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
        FragmentGeometrySpec(
            label="fragment_center_median_reference",
            center_strategy="median_timestamp",
            support_strategy="none",
        ),
        FragmentGeometrySpec(
            label="fragment_center_highest_score",
            center_strategy="highest_score_frame",
            support_strategy="none",
        ),
        FragmentGeometrySpec(
            label="fragment_center_densest_cluster",
            center_strategy="densest_cluster",
            support_strategy="none",
        ),
        FragmentGeometrySpec(
            label="fragment_center_support_weighted",
            center_strategy="support_weighted",
            support_strategy="none",
        ),
        FragmentGeometrySpec(
            label="fragment_density_aware_support",
            center_strategy="support_weighted",
            support_strategy="density_aware",
            min_positive_frames=2,
            min_peak_score=0.90,
        ),
        FragmentGeometrySpec(
            label="fragment_density_margin_support",
            center_strategy="support_weighted",
            support_strategy="density_and_margin",
            min_positive_frames=2,
            min_peak_score=0.90,
            min_margin_sum=0.20,
        ),
    ]

    candidates: list[dict[str, Any]] = []
    for spec in specs:
        proposals = build_fragment_geometry_proposals(
            session_root=session_root,
            frame_rows=frame_rows,
            base_intervals=base_intervals,
            base_config=base_config,
            duration_seconds=float(ground_truth["duration_seconds"]),
            spec=spec,
        )
        result = evaluate_proposals(
            session_root=session_root,
            ground_truth=ground_truth,
            proposal_rows=proposals,
            label=spec.label,
        )
        result["center_strategy"] = spec.center_strategy
        result["support_strategy"] = spec.support_strategy
        result["support_min_positive_frames"] = spec.min_positive_frames
        result["support_min_peak_score"] = spec.min_peak_score
        result["support_min_margin_sum"] = spec.min_margin_sum
        result["beats_reference_union"] = result["union_recall"] > reference["union_recall"]
        result["beats_reference_recovered"] = result["recovered_anchor_count"] > reference["recovered_anchor_count"]
        result["beats_reference_burden"] = (
            result["unmatched_visual_count"] < reference["unmatched_visual_count"]
            or result["false_visual_proposals_per_minute"] < reference["false_visual_proposals_per_minute"]
        )
        candidates.append(result)

    best = max(
        candidates,
        key=lambda row: (
            row["union_recall"],
            row["recovered_anchor_count"],
            -row["unmatched_visual_count"],
            -row["false_visual_proposals_per_minute"],
            -row["visual_proposal_count"],
        ),
    )

    no_clear_gain = (
        best["union_recall"] <= reference["union_recall"]
        and best["recovered_anchor_count"] <= reference["recovered_anchor_count"]
        and best["unmatched_visual_count"] >= reference["unmatched_visual_count"]
        and best["false_visual_proposals_per_minute"] >= reference["false_visual_proposals_per_minute"]
    )

    summary = {
        "reference": reference,
        "fixed_control": {
            "session_root": str(session_root),
            "proposal_root": str(proposal_root),
            "mode": "audio-gated",
            "roi_mode": "full_frame",
            "fps": 1.0,
            "prompt_id": "diving_attempt",
            "decision_rule": "yes_no_first_token_margin",
            "internal_gap_seconds": INTERNAL_GAP_SECONDS,
        },
        "candidates": candidates,
        "best_candidate": best,
        "meaningful_gain_vs_reference": not no_clear_gain,
        "interval_geometry_remains_primary": True,
        "prefilter_now_justified": False,
    }

    outputs = Path("outputs")
    write_json(outputs / "r45_fragment_geometry_ablation.json", summary)
    write_json(outputs / "r45_fragment_geometry_candidate_comparison.json", {"candidates": candidates})
    summary_md, comparison_md = render_md(reference, candidates, best)
    write_md(outputs / "r45_fragment_geometry_ablation.md", summary_md)
    write_md(outputs / "r45_fragment_geometry_candidate_comparison.md", comparison_md)

    decision = "R45_FRAGMENT_GEOMETRY_GAIN" if not no_clear_gain else "R45_FRAGMENT_GEOMETRY_NO_CLEAR_GAIN"
    write_md(
        Path("docs/research/R45_FRAGMENT_SCORING_AND_EVENT_CENTER_GEOMETRY.md"),
        "\n".join(
            [
                "# R45 Fragment Scoring And Event-Center Geometry",
                "",
                f"- reference: `split_internal_gap_3s` (union recall `{reference['union_recall']}`, unmatched visual `{reference['unmatched_visual_count']}`).",
                f"- best candidate: `{best['label']}`",
                f"- best center strategy: `{best['center_strategy']}`",
                f"- best support strategy: `{best['support_strategy']}`",
                f"- best union recall: `{best['union_recall']}`",
                f"- best recovered anchors: `{best['recovered_anchor_count']}`",
                f"- best unmatched visual: `{best['unmatched_visual_count']}`",
                f"- best false visual/min: `{best['false_visual_proposals_per_minute']}`",
                f"- meaningful gain vs reference: `{not no_clear_gain}`",
                "- interpretation: fragment/event-center geometry was tested with bounded support/centering variants under fixed full-frame controls.",
                "- interpretation: interval geometry remains the primary next lever.",
                "- interpretation: prefilter is still premature.",
                "",
                "## Decisions",
                "",
                f"- `{decision}`",
                "- `R45_INTERVAL_GEOMETRY_REMAINS_PRIMARY`",
            ]
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
