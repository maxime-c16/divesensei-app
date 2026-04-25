from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass
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
PROMPT_ID = "diving_attempt"
DECISION_RULE = "yes_no_first_token_margin"


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
class SubsegSpec:
    label: str
    strategy: str
    peak_min_score: float = 0.88
    peak_min_margin: float = 0.03
    peak_separation_seconds: float = 3.0
    island_gap_seconds: float = 4.0
    max_subevents: int = 2
    long_fragment_min_seconds: float = 8.5


def split_internal_gap(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: float(row["timestamp_seconds"]))
    chunks = [[rows[0]]]
    for previous, current in zip(rows, rows[1:]):
        if float(current["timestamp_seconds"]) - float(previous["timestamp_seconds"]) > INTERNAL_GAP_SECONDS:
            chunks.append([current])
        else:
            chunks[-1].append(current)
    return chunks


def local_peaks(chunk: list[dict[str, Any]], spec: SubsegSpec) -> list[dict[str, Any]]:
    if not chunk:
        return []
    peaks: list[dict[str, Any]] = []
    for idx, row in enumerate(chunk):
        score = float(row.get("score", 0.0) or 0.0)
        margin = float(row.get("yes_no_first_token_margin", 0.0) or 0.0)
        if score < spec.peak_min_score or margin < spec.peak_min_margin:
            continue
        prev_score = float(chunk[idx - 1].get("score", 0.0) or 0.0) if idx > 0 else -1.0
        next_score = float(chunk[idx + 1].get("score", 0.0) or 0.0) if idx + 1 < len(chunk) else -1.0
        if score >= prev_score and score >= next_score:
            peaks.append(row)
    peaks.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    for row in peaks:
        ts = float(row["timestamp_seconds"])
        if any(abs(ts - float(chosen["timestamp_seconds"])) < spec.peak_separation_seconds for chosen in selected):
            continue
        selected.append(row)
        if len(selected) >= spec.max_subevents:
            break
    selected.sort(key=lambda row: float(row["timestamp_seconds"]))
    return selected


def island_centers(chunk: list[dict[str, Any]], spec: SubsegSpec) -> list[dict[str, Any]]:
    if not chunk:
        return []
    chunk = sorted(chunk, key=lambda row: float(row["timestamp_seconds"]))
    islands: list[list[dict[str, Any]]] = [[chunk[0]]]
    for previous, current in zip(chunk, chunk[1:]):
        if float(current["timestamp_seconds"]) - float(previous["timestamp_seconds"]) > spec.island_gap_seconds:
            islands.append([current])
        else:
            islands[-1].append(current)
    centers = [max(island, key=lambda row: float(row.get("score", 0.0) or 0.0)) for island in islands]
    centers = sorted(centers, key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)[: spec.max_subevents]
    centers.sort(key=lambda row: float(row["timestamp_seconds"]))
    return centers


def long_fragment_second_peak(chunk: list[dict[str, Any]], spec: SubsegSpec) -> list[dict[str, Any]]:
    chunk = sorted(chunk, key=lambda row: float(row["timestamp_seconds"]))
    if not chunk:
        return []
    start_ts = float(chunk[0]["timestamp_seconds"])
    end_ts = float(chunk[-1]["timestamp_seconds"])
    span = end_ts - start_ts
    primary = max(chunk, key=lambda row: float(row.get("score", 0.0) or 0.0))
    if span < spec.long_fragment_min_seconds:
        return [primary]
    peaks = local_peaks(chunk, spec)
    if not peaks:
        return [primary]
    secondary = None
    primary_ts = float(primary["timestamp_seconds"])
    for candidate in peaks:
        if abs(float(candidate["timestamp_seconds"]) - primary_ts) >= spec.peak_separation_seconds:
            secondary = candidate
            break
    if secondary is None:
        return [primary]
    centers = [primary, secondary]
    centers.sort(key=lambda row: float(row["timestamp_seconds"]))
    return centers[: spec.max_subevents]


def centers_for_chunk(chunk: list[dict[str, Any]], spec: SubsegSpec) -> list[dict[str, Any]]:
    if spec.strategy == "single_reference":
        return [chunk[int(len(chunk) / 2)]]
    if spec.strategy == "peak_separation":
        peaks = local_peaks(chunk, spec)
        return peaks or [max(chunk, key=lambda row: float(row.get("score", 0.0) or 0.0))]
    if spec.strategy == "island_subevents":
        centers = island_centers(chunk, spec)
        return centers or [max(chunk, key=lambda row: float(row.get("score", 0.0) or 0.0))]
    if spec.strategy == "long_fragment_second_peak":
        return long_fragment_second_peak(chunk, spec)
    raise ValueError(f"Unsupported strategy: {spec.strategy}")


def build_reference_intervals(
    *,
    frame_rows: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    base_config = VisualProposalConfig(
        mode="audio-gated",
        roi_mode="full_frame",
        confidence_threshold=0.845,
        grouping_threshold_seconds=2.5,
        buffer_start_seconds=1.5,
        buffer_end_seconds=3.0,
        merge_gap_seconds=3.5,
        prompt_ids=(PROMPT_ID,),
        decision_rules=(DECISION_RULE,),
    )
    zero_merge_config = VisualProposalConfig(
        mode=base_config.mode,
        roi_mode=base_config.roi_mode,
        confidence_threshold=base_config.confidence_threshold,
        grouping_threshold_seconds=base_config.grouping_threshold_seconds,
        buffer_start_seconds=base_config.buffer_start_seconds,
        buffer_end_seconds=base_config.buffer_end_seconds,
        merge_gap_seconds=0.0,
        prompt_ids=base_config.prompt_ids,
        decision_rules=base_config.decision_rules,
    )
    base_intervals = frame_predictions_to_intervals(
        frame_rows,
        config=base_config,
        decision_rule=DECISION_RULE,
        prompt_id=PROMPT_ID,
        duration_seconds=duration_seconds,
    )
    out: list[dict[str, Any]] = []
    for interval in base_intervals:
        start = float(interval["start_seconds"])
        end = float(interval["end_seconds"])
        positives = [
            row
            for row in frame_rows
            if row.get("prompt_id") == PROMPT_ID
            and row.get("decision_rule") == DECISION_RULE
            and bool(row.get("is_positive"))
            and start <= float(row["timestamp_seconds"]) <= end
        ]
        if not positives:
            out.append(dict(interval))
            continue
        for chunk in split_internal_gap(positives):
            out.extend(
                frame_predictions_to_intervals(
                    chunk,
                    config=zero_merge_config,
                    decision_rule=DECISION_RULE,
                    prompt_id=PROMPT_ID,
                    duration_seconds=duration_seconds,
                )
            )
    out.sort(key=lambda row: float(row["start_seconds"]))
    for idx, row in enumerate(out, start=1):
        row["visual_interval_id"] = f"vis-int-{idx:04d}"
    return out


def proposals_from_center_timestamps(
    *,
    session_root: Path,
    center_rows: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    proposals = []
    for idx, (ts, score) in enumerate(sorted(center_rows, key=lambda row: row[0]), start=1):
        proposals.append(
            {
                "proposal_id": f"vis-prop-{idx:04d}",
                "session_id": session_root.name,
                "source_video_path": str(session_root / "web/session_source_review.mp4"),
                "timestamp": round(float(ts), 3),
                "start_seconds": round(max(0.0, float(ts) - 1.5), 3),
                "end_seconds": round(float(ts) + 3.0, 3),
                "proposal_frontend": "visual_vlm_paligemma2",
                "proposal_provenance": "visual_vlm_paligemma2",
                "raw_proposal_score": float(score),
                "prompt_id": PROMPT_ID,
                "decision_rule": DECISION_RULE,
                "roi_mode": "full_frame",
                "mode": "audio-gated",
                "positive_frame_count": 1,
                "pipeline_selected": False,
                "pipeline_stage": "visual_proposal_only",
            }
        )
    return proposals


def render_md(reference: dict[str, Any], candidates: list[dict[str, Any]], best: dict[str, Any]) -> tuple[str, str]:
    summary = [
        "# R46 Multi-Event Fragment Sub-Segmentation",
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
        f"- strategy: `{best['strategy']}`",
        f"- visual proposals: `{best['visual_proposal_count']}`",
        f"- visual recall: `{best['visual_recall']}`",
        f"- union recall: `{best['union_recall']}`",
        f"- recovered anchors: `{best['recovered_anchor_count']}`",
        f"- unmatched visual proposals: `{best['unmatched_visual_count']}`",
        f"- false visual/min: `{best['false_visual_proposals_per_minute']}`",
    ]
    comparison = [
        "# R46 Fragment Sub-Segmentation Candidate Comparison",
        "",
        "| Candidate | Strategy | Visual proposals | Visual recall | Union recall | Recovered anchors | Unmatched visual | False visual/min | Interval min / med / max | Merged proposals |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in candidates:
        comparison.append(
            f"| {row['label']} | `{row['strategy']}` | {row['visual_proposal_count']} | {row['visual_recall']:.4f} | {row['union_recall']:.4f} | {row['recovered_anchor_count']} | {row['unmatched_visual_count']} | {row['false_visual_proposals_per_minute']:.3f} | {row['interval_length_min_seconds']:.1f} / {row['interval_length_median_seconds']:.1f} / {row['interval_length_max_seconds']:.1f} | {row['merged_proposal_count']} |"
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
    reference_intervals = build_reference_intervals(
        frame_rows=frame_rows,
        duration_seconds=float(ground_truth["duration_seconds"]),
    )

    specs = [
        SubsegSpec(label="reference_interval_anchor_passthrough", strategy="single_reference", max_subevents=1),
        SubsegSpec(
            label="peak_subseg_sep_3p0s_max2",
            strategy="peak_separation",
            peak_min_score=0.88,
            peak_min_margin=0.03,
            peak_separation_seconds=3.0,
            max_subevents=2,
        ),
        SubsegSpec(
            label="peak_subseg_sep_4p0s_max2",
            strategy="peak_separation",
            peak_min_score=0.90,
            peak_min_margin=0.04,
            peak_separation_seconds=4.0,
            max_subevents=2,
        ),
        SubsegSpec(
            label="island_subseg_gap_4p0s_max2",
            strategy="island_subevents",
            island_gap_seconds=4.0,
            max_subevents=2,
        ),
        SubsegSpec(
            label="long_fragment_second_peak_max2",
            strategy="long_fragment_second_peak",
            peak_min_score=0.88,
            peak_min_margin=0.03,
            peak_separation_seconds=3.5,
            max_subevents=2,
            long_fragment_min_seconds=8.5,
        ),
    ]

    candidates: list[dict[str, Any]] = []
    for spec in specs:
        center_rows: list[tuple[float, float]] = []
        for interval in reference_intervals:
            start = float(interval["start_seconds"])
            end = float(interval["end_seconds"])
            positives = [
                row
                for row in frame_rows
                if row.get("prompt_id") == PROMPT_ID
                and row.get("decision_rule") == DECISION_RULE
                and bool(row.get("is_positive"))
                and start <= float(row["timestamp_seconds"]) <= end
            ]
            if not positives:
                continue
            if spec.strategy == "single_reference":
                center_rows.append(
                    (
                        float(interval["anchor_timestamp_seconds"]),
                        float(interval.get("max_score", 0.0) or 0.0),
                    )
                )
                continue
            selected = centers_for_chunk(positives, spec)
            for row in selected[: spec.max_subevents]:
                center_rows.append((float(row["timestamp_seconds"]), float(row.get("score", 0.0) or 0.0)))
        proposals = proposals_from_center_timestamps(session_root=session_root, center_rows=center_rows)
        result = evaluate_proposals(
            session_root=session_root,
            ground_truth=ground_truth,
            proposal_rows=proposals,
            label=spec.label,
        )
        result["strategy"] = spec.strategy
        result["peak_min_score"] = spec.peak_min_score
        result["peak_min_margin"] = spec.peak_min_margin
        result["peak_separation_seconds"] = spec.peak_separation_seconds
        result["island_gap_seconds"] = spec.island_gap_seconds
        result["max_subevents"] = spec.max_subevents
        result["long_fragment_min_seconds"] = spec.long_fragment_min_seconds
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
            "prompt_id": PROMPT_ID,
            "decision_rule": DECISION_RULE,
            "internal_gap_seconds": INTERNAL_GAP_SECONDS,
        },
        "reference_interval_count": len(reference_intervals),
        "candidates": candidates,
        "best_candidate": best,
        "meaningful_gain_vs_reference": not no_clear_gain,
        "interval_geometry_remains_primary": True,
        "prefilter_now_justified": False,
    }

    outputs = Path("outputs")
    write_json(outputs / "r46_multi_event_fragment_subsegmentation.json", summary)
    write_json(outputs / "r46_fragment_subsegmentation_candidate_comparison.json", {"candidates": candidates})
    summary_md, comparison_md = render_md(reference, candidates, best)
    write_md(outputs / "r46_multi_event_fragment_subsegmentation.md", summary_md)
    write_md(outputs / "r46_fragment_subsegmentation_candidate_comparison.md", comparison_md)

    decision = "R46_SUBSEGMENTATION_GAIN" if not no_clear_gain else "R46_SUBSEGMENTATION_NO_CLEAR_GAIN"
    write_md(
        Path("docs/research/R46_MULTI_EVENT_FRAGMENT_SUBSEGMENTATION.md"),
        "\n".join(
            [
                "# R46 Multi-Event Fragment Sub-Segmentation",
                "",
                f"- reference: `split_internal_gap_3s` (union recall `{reference['union_recall']}`, recovered `{reference['recovered_anchor_count']}`, unmatched `{reference['unmatched_visual_count']}`).",
                f"- best candidate: `{best['label']}`",
                f"- strategy: `{best['strategy']}`",
                f"- best union recall: `{best['union_recall']}`",
                f"- best recovered anchors: `{best['recovered_anchor_count']}`",
                f"- best unmatched visual: `{best['unmatched_visual_count']}`",
                f"- best false visual/min: `{best['false_visual_proposals_per_minute']}`",
                f"- meaningful gain vs reference: `{not no_clear_gain}`",
                "- interpretation: bounded multi-event sub-segmentation was tested under fixed full-frame controls.",
                "- interpretation: interval geometry remains the primary next lever.",
                "- interpretation: prefilter is still premature.",
                "",
                "## Decisions",
                "",
                f"- `{decision}`",
                "- `R46_INTERVAL_GEOMETRY_REMAINS_PRIMARY`",
            ]
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
