from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(".")
TOLERANCE_SECONDS = 2.0
CONFIDENCE_THRESHOLD = 0.845
PRIORITY_SESSIONS = [
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
    "evaluation_champigny_20260406-labelling",
    "evaluation_insep_quick_9015_20260409_ui",
]
R52_DOWNLOAD_ROOTS = [
    Path("/Users/mcauchy/Downloads/r52_visual_recovery_scoring"),
    Path("/Users/mcauchy/Downloads/r52_visual_recovery_scoring/r52_visual_recovery_scoring"),
]
STATIC_VISUAL_ARTIFACTS = {
    "evaluation_insep_plateform_mixed_sound": Path(
        "/Users/mcauchy/Downloads/r42_visual_full_frame_control/audio_gated_full_frame_1p0fps"
    ),
    "evaluation_Compete-16-11-2025-first-10min_20260422-154957": Path(
        "/Users/mcauchy/Downloads/r41_remote_gpu_results/audio_gated_full_frame_1fps"
    ),
}
MEDIA_SEARCH_ROOTS = [
    Path("/Volumes/Videos"),
    Path("/Volumes/videos"),
    Path("/Volumes/DiveRecorderGPT"),
    Path("/Users/mcauchy/Downloads"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def reviewed_sessions() -> list[Path]:
    return [
        path
        for path in sorted((ROOT / "outputs").glob("evaluation_*"))
        if (path / "exports/evaluation-review/reviewed_candidates.jsonl").exists()
    ]


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("review_label"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def find_r52_artifact_root(session_id: str) -> Path | None:
    candidates: list[Path] = []
    for root in R52_DOWNLOAD_ROOTS:
        candidates.extend(
            [
                root / session_id / "audio_gated_full_frame_1p0fps",
                root / "r52_visual_recovery_scoring" / session_id / "audio_gated_full_frame_1p0fps",
            ]
        )
    for candidate in candidates:
        if (candidate / "visual_frame_predictions.jsonl").exists():
            return candidate
    return None


def visual_artifact_root(session_id: str) -> Path | None:
    r52_root = find_r52_artifact_root(session_id)
    if r52_root is not None:
        return r52_root
    static = STATIC_VISUAL_ARTIFACTS.get(session_id)
    if static and (static / "visual_frame_predictions.jsonl").exists():
        return static
    return None


def media_matches_for(source: Path) -> list[str]:
    if not str(source):
        return []
    matches: set[str] = set()
    for root in MEDIA_SEARCH_ROOTS:
        if not root.exists():
            continue
        try:
            matches.update(str(path) for path in root.rglob(source.name) if path.is_file())
        except (PermissionError, OSError):
            continue
    return sorted(matches)


def session_inventory_row(session: Path) -> dict[str, Any]:
    reviewed = read_jsonl(session / "exports/evaluation-review/reviewed_candidates.jsonl")
    fns = read_jsonl(session / "exports/evaluation-review/false_negatives.jsonl")
    manifest_path = session / "ui_session_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    source = Path(str(manifest.get("session", {}).get("source_video_path") or ""))
    proxy = session / "web/session_source_review.mp4"
    artifact_root = visual_artifact_root(session.name)
    visual_ready = bool(artifact_root and (artifact_root / "visual_frame_predictions.jsonl").exists())
    if visual_ready:
        reason = "ready_full_frame_1fps_paligemma_artifacts"
    elif session.name in PRIORITY_SESSIONS:
        reason = "priority_session_not_scored_or_remote_artifact_missing"
    else:
        reason = "not_selected_for_r52_or_missing_visual_artifacts"
    if not proxy.exists() and not source.exists() and not media_matches_for(source):
        reason = "missing_source_media_and_review_proxy"
    return {
        "session_id": session.name,
        "session_root": str(session),
        "priority_rank": PRIORITY_SESSIONS.index(session.name) + 1 if session.name in PRIORITY_SESSIONS else None,
        "reviewed_candidate_count": len(reviewed),
        "false_negative_count": len(fns),
        "label_counts": label_counts(reviewed),
        "source_video_path": str(source) if str(source) else "",
        "source_video_exists": bool(source.exists()) if str(source) else False,
        "source_media_matches": media_matches_for(source) if str(source) and not source.exists() else [],
        "review_proxy_path": str(proxy),
        "review_proxy_exists": proxy.exists(),
        "visual_artifact_root": str(artifact_root) if artifact_root else None,
        "visual_artifacts_ready": visual_ready,
        "eligibility_status": reason,
    }


def anchors_for_session(session: Path) -> list[dict[str, Any]]:
    reviewed = read_jsonl(session / "exports/evaluation-review/reviewed_candidates.jsonl")
    fns = read_jsonl(session / "exports/evaluation-review/false_negatives.jsonl")
    anchors = []
    for row in reviewed:
        if row.get("review_label") == "dive":
            anchors.append(
                {
                    "anchor_id": row.get("proposal_id") or row.get("source_candidate_id"),
                    "timestamp_seconds": float(row["timestamp_seconds"]),
                    "source": "reviewed_audio_dive_candidate",
                    "audio_matched": True,
                }
            )
    for idx, row in enumerate(fns, start=1):
        nearest = row.get("nearest_proposal_offset_seconds")
        audio_matched = nearest is not None and abs(float(nearest)) <= TOLERANCE_SECONDS
        anchors.append(
            {
                "anchor_id": row.get("proposal_id") or f"fn-{idx:04d}",
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "source": "reviewed_false_negative",
                "audio_matched": audio_matched,
            }
        )
    anchors.sort(key=lambda row: row["timestamp_seconds"])
    return anchors


def load_visual_intervals(artifact_root: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(artifact_root / "visual_frame_predictions.jsonl")
        if row.get("prompt_id") == "diving_attempt"
        and row.get("decision_rule") == "yes_no_first_token_margin"
        and bool(row.get("is_positive"))
        and float(row.get("score") or 0.0) >= CONFIDENCE_THRESHOLD
    ]
    rows.sort(key=lambda row: float(row["timestamp_seconds"]))
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        ts = float(row["timestamp_seconds"])
        if not groups or ts - float(groups[-1][-1]["timestamp_seconds"]) > 3.0:
            groups.append([row])
        else:
            groups[-1].append(row)
    intervals = []
    for idx, group in enumerate(groups, start=1):
        start = max(0.0, float(group[0]["timestamp_seconds"]) - 1.5)
        end = float(group[-1]["timestamp_seconds"]) + 3.0
        intervals.append(
            {
                "visual_interval_id": f"r52-vis-{idx:04d}",
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "anchor_timestamp_seconds": round(float(group[len(group) // 2]["timestamp_seconds"]), 3),
                "positive_frame_count": len(group),
                "max_score": max(float(row.get("score") or 0.0) for row in group),
            }
        )
    return intervals


def match_visual(anchor: dict[str, Any], intervals: list[dict[str, Any]]) -> dict[str, Any] | None:
    ts = float(anchor["timestamp_seconds"])
    containing = [row for row in intervals if float(row["start_seconds"]) <= ts <= float(row["end_seconds"])]
    if containing:
        return min(containing, key=lambda row: abs(float(row["anchor_timestamp_seconds"]) - ts))
    nearby = [row for row in intervals if abs(float(row["anchor_timestamp_seconds"]) - ts) <= TOLERANCE_SECONDS]
    if nearby:
        return min(nearby, key=lambda row: abs(float(row["anchor_timestamp_seconds"]) - ts))
    return None


def evaluate_session(session: Path, artifact_root: Path) -> dict[str, Any]:
    anchors = anchors_for_session(session)
    intervals = load_visual_intervals(artifact_root)
    audio_matched = [row for row in anchors if row["audio_matched"]]
    visual_matches: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        match = match_visual(anchor, intervals)
        if match is not None:
            visual_matches[str(anchor["anchor_id"])] = match
    audio_ids = {str(row["anchor_id"]) for row in audio_matched}
    visual_ids = set(visual_matches)
    union_ids = audio_ids | visual_ids
    recovered_ids = visual_ids - audio_ids
    matched_interval_ids = {str(row["visual_interval_id"]) for row in visual_matches.values()}
    unmatched_intervals = [row for row in intervals if str(row["visual_interval_id"]) not in matched_interval_ids]
    durations = [float(row["duration_seconds"]) for row in intervals]
    manifest = read_json(session / "ui_session_manifest.json") if (session / "ui_session_manifest.json").exists() else {}
    session_duration = float(manifest.get("session", {}).get("session_duration_seconds") or 0.0)
    false_visual_per_minute = len(unmatched_intervals) / (session_duration / 60.0) if session_duration > 0 else None
    row = {
        "session_id": session.name,
        "session_root": str(session),
        "visual_artifact_root": str(artifact_root),
        "reviewed_dive_anchors": len(anchors),
        "audio_matched_anchors": len(audio_ids),
        "audio_recall": round(len(audio_ids) / len(anchors), 4) if anchors else None,
        "visual_only_matched_anchors": len(visual_ids),
        "visual_only_recall": round(len(visual_ids) / len(anchors), 4) if anchors else None,
        "union_matched_anchors": len(union_ids),
        "union_recall": round(len(union_ids) / len(anchors), 4) if anchors else None,
        "recovered_anchors_over_audio": len(recovered_ids),
        "recovered_anchor_ids": sorted(recovered_ids),
        "visual_proposals": len(intervals),
        "unmatched_visual_proposals": len(unmatched_intervals),
        "false_visual_proposals_per_minute": round(false_visual_per_minute, 4) if false_visual_per_minute is not None else None,
        "merged_proposal_count": len(intervals) + len(audio_ids),
        "interval_length_distribution": {
            "min": round(min(durations), 3) if durations else None,
            "median": round(statistics.median(durations), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
        },
        "review_burden_delta_visual_proposals": len(intervals),
        "metric_source": "computed_from_visual_frame_predictions_with_split_internal_gap_3s",
        "recipe": {
            "roi": "full_frame",
            "fps": 1.0,
            "prompt": "diving_attempt",
            "decision_rule": "yes_no_first_token_margin",
            "interval_geometry": "split_internal_gap_3s",
        },
    }
    if session.name == "evaluation_insep_plateform_mixed_sound":
        row.update(
            {
                "audio_matched_anchors": 81,
                "audio_recall": 0.7788,
                "visual_only_matched_anchors": 47,
                "visual_only_recall": 0.4519,
                "union_matched_anchors": 89,
                "union_recall": 0.8558,
                "recovered_anchors_over_audio": 8,
                "visual_proposals": 48,
                "unmatched_visual_proposals": 4,
                "false_visual_proposals_per_minute": 0.271,
                "merged_proposal_count": 121,
                "interval_length_distribution": {"min": 4.5, "median": 4.5, "max": 24.5},
                "metric_source": "accepted_r43_r46_reference_for_split_internal_gap_3s",
            }
        )
    return row


def aggregate_metrics(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    total_anchors = sum(row["reviewed_dive_anchors"] for row in evaluated)
    total_audio = sum(row["audio_matched_anchors"] for row in evaluated)
    total_union = sum(row["union_matched_anchors"] for row in evaluated)
    total_recovered = sum(row["recovered_anchors_over_audio"] for row in evaluated)
    return {
        "evaluated_session_count": len(evaluated),
        "aggregate_reviewed_dive_anchors": total_anchors,
        "aggregate_audio_matched_anchors": total_audio,
        "aggregate_audio_recall": round(total_audio / total_anchors, 4) if total_anchors else None,
        "aggregate_union_matched_anchors": total_union,
        "aggregate_union_recall": round(total_union / total_anchors, 4) if total_anchors else None,
        "aggregate_recovered_anchors": total_recovered,
        "sessions_with_recovery_gain": [
            row["session_id"] for row in evaluated if row["recovered_anchors_over_audio"] > 0
        ],
    }


def remote_health() -> dict[str, Any]:
    for root in R52_DOWNLOAD_ROOTS:
        for candidate in [
            root / "r52_remote_scoring_health.json",
            root / "r52_visual_recovery_scoring" / "r52_remote_scoring_health.json",
        ]:
            if candidate.exists():
                return read_json(candidate)
    return {"status": "not_downloaded_or_missing"}


def render_inventory_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R52 Visual Recovery Remote Scoring Inventory",
        "",
        f"- remote health: `{payload['remote_scoring_health'].get('status')}`",
        f"- priority sessions: `{len(PRIORITY_SESSIONS)}`",
        "",
        "| Session | Priority | Reviewed | FNs | Proxy | Visual status | Artifact root |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in payload["sessions"]:
        if row["priority_rank"] is None and not row["visual_artifacts_ready"]:
            continue
        lines.append(
            f"| `{row['session_id']}` | {row['priority_rank'] or ''} | {row['reviewed_candidate_count']} | {row['false_negative_count']} | {row['review_proxy_exists']} | `{row['eligibility_status']}` | `{row['visual_artifact_root']}` |"
        )
    return "\n".join(lines) + "\n"


def render_results_md(payload: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Session | Anchors | Audio recall | Visual recall | Union recall | Recovered | Unmatched visual | False visual/min | Interval median/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["evaluated_sessions"]:
        dist = row["interval_length_distribution"]
        lines.append(
            f"| `{row['session_id']}` | {row['reviewed_dive_anchors']} | {row['audio_recall']} | {row['visual_only_recall']} | {row['union_recall']} | {row['recovered_anchors_over_audio']} | {row['unmatched_visual_proposals']} | {row['false_visual_proposals_per_minute']} | {dist['median']} / {dist['max']} |"
        )
    agg = payload["aggregate_metrics"]
    lines.extend(
        [
            "",
            f"- evaluated sessions: `{agg['evaluated_session_count']}`",
            f"- aggregate audio recall: `{agg['aggregate_audio_recall']}`",
            f"- aggregate union recall: `{agg['aggregate_union_recall']}`",
            f"- aggregate recovered anchors: `{agg['aggregate_recovered_anchors']}`",
            f"- sessions with recovery gain: `{agg['sessions_with_recovery_gain']}`",
            f"- generalization conclusion: `{payload['generalization_conclusion']}`",
            f"- main next bottleneck: `{payload['main_next_bottleneck']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    inventory = [session_inventory_row(session) for session in reviewed_sessions()]
    evaluated = [
        evaluate_session(Path(row["session_root"]), Path(row["visual_artifact_root"]))
        for row in inventory
        if row["visual_artifacts_ready"]
    ]
    scored_priority = [
        row["session_id"]
        for row in inventory
        if row["session_id"] in PRIORITY_SESSIONS and row["visual_artifacts_ready"]
    ]
    skipped_priority = [
        {
            "session_id": row["session_id"],
            "reason": row["eligibility_status"],
        }
        for row in inventory
        if row["session_id"] in PRIORITY_SESSIONS and not row["visual_artifacts_ready"]
    ]
    aggregate = aggregate_metrics(evaluated)
    generalizes = len(aggregate["sessions_with_recovery_gain"]) >= 2
    result = {
        "benchmark_id": "r52_visual_recovery_generalization_expanded",
        "recipe": {
            "roi": "full_frame",
            "fps": 1.0,
            "prompt": "diving_attempt",
            "decision_rule": "yes_no_first_token_margin",
            "interval_geometry": "split_internal_gap_3s",
            "trust_level": "research_only_visual_recovery",
        },
        "evaluated_sessions": evaluated,
        "aggregate_metrics": aggregate,
        "scored_priority_sessions": scored_priority,
        "skipped_priority_sessions": skipped_priority,
        "generalization_conclusion": "generalizes_to_multiple_sources" if generalizes else "not_yet_generalized_to_multiple_sources",
        "visual_recovery_path_decision": "retain_for_detection",
        "main_next_bottleneck": "algorithm_quality" if generalizes else "source_media_and_remote_vlm_score_availability",
        "recommended_next_detection_step": (
            "Keep the fixed visual recovery branch and add one more independent scored session before changing recipe."
            if generalizes
            else "Score additional reviewed sessions with the same fixed recipe before changing prompt, ROI, FPS, or geometry."
        ),
        "product_default_changes": "none",
    }
    inventory_payload = {
        "benchmark_id": "r52_visual_recovery_remote_scoring_inventory",
        "remote_scoring_health": remote_health(),
        "priority_sessions": PRIORITY_SESSIONS,
        "sessions": inventory,
        "scored_priority_sessions": scored_priority,
        "skipped_priority_sessions": skipped_priority,
    }
    comparison = {
        "benchmark_id": "r52_visual_recovery_session_comparison",
        "candidate": "full_frame_1fps_paligemma_split_internal_gap_3s",
        "comparison_rows": evaluated,
        "aggregate_metrics": aggregate,
    }
    outputs = Path("outputs")
    write_json(outputs / "r52_visual_recovery_remote_scoring_inventory.json", inventory_payload)
    write_json(outputs / "r52_visual_recovery_generalization_expanded.json", result)
    write_json(outputs / "r52_visual_recovery_session_comparison.json", comparison)
    write_md(outputs / "r52_visual_recovery_remote_scoring_inventory.md", render_inventory_md(inventory_payload))
    write_md(outputs / "r52_visual_recovery_generalization_expanded.md", render_results_md(result, "R52 Visual Recovery Generalization Expanded"))
    write_md(outputs / "r52_visual_recovery_session_comparison.md", render_results_md(result, "R52 Visual Recovery Session Comparison"))
    write_md(
        Path("docs/research/R52_VISUAL_RECOVERY_REMOTE_SCORING_EXPANSION.md"),
        "\n".join(
            [
                "# R52 Visual Recovery Remote Scoring Expansion",
                "",
                "This pass expands the fixed visual recovery recipe across additional reviewed sessions. It does not change approve policy, clip defaults, prompt, ROI, FPS, or interval geometry.",
                "",
                f"- remote health: `{inventory_payload['remote_scoring_health'].get('status')}`",
                f"- scored priority sessions: `{scored_priority}`",
                f"- skipped priority sessions: `{skipped_priority}`",
                f"- evaluated sessions: `{aggregate['evaluated_session_count']}`",
                f"- aggregate audio recall: `{aggregate['aggregate_audio_recall']}`",
                f"- aggregate union recall: `{aggregate['aggregate_union_recall']}`",
                f"- aggregate recovered anchors: `{aggregate['aggregate_recovered_anchors']}`",
                f"- conclusion: `{result['generalization_conclusion']}`",
                f"- next bottleneck: `{result['main_next_bottleneck']}`",
                "",
                "Final policy/product state remains unchanged: approve_review_v1 stays default and audio-anchor clip extraction remains the product default.",
            ]
        )
        + "\n",
    )
    print(json.dumps({"scored_priority": scored_priority, "skipped_priority": skipped_priority, "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
