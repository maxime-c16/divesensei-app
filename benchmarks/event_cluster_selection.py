from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from event_level_research import EventLevelResearch, ReviewEvent, to_serializable, zscore


PRACTICAL_TOLERANCE = 1.0
FN_NEIGHBORHOOD_SECONDS = 1.5
DENSE_CLUSTER_MIN_SIZE = 8
MIN_VIABLE_CANDIDATES = 2


class EventClusterSelectionResearch:
    def __init__(self, session_dir: Path) -> None:
        self.research = EventLevelResearch(session_dir)
        self.session_dir = session_dir
        self.summary = self.research.summary
        self.review = self.research.review
        self.false_negative_rows = self.research.false_negative_rows
        self.actual_selected = self._actual_selected_timestamps()
        self.reviewed_dives, self.reviewed_rebounds, self.reviewed_false_negatives = self.research.reviewed_events()

    def _suppression_events(self) -> int:
        proposal_failure = self.summary.get("proposal_failure_attribution", {})
        return int(proposal_failure.get("suppressed_or_merged_proposal_candidate", 0))

    def _actual_selected_timestamps(self) -> list[float]:
        return sorted(
            float(row["timestamp"])
            for row in self.research.diagnostic_rows
            if row.get("pipeline_selected")
        )

    def _nearest_actual_selected(self, timestamp: float, tolerance: float = PRACTICAL_TOLERANCE) -> float | None:
        best = None
        best_delta = None
        for selected in self.actual_selected:
            delta = abs(selected - timestamp)
            if delta <= tolerance and (best_delta is None or delta < best_delta):
                best = selected
                best_delta = delta
        return best

    def _cluster_for_timestamp(self, timestamp: float) -> list[dict]:
        return self.research._cluster_rows_for_timestamp(timestamp)

    def _viable_cluster_candidates(self, cluster: list[dict]) -> list[dict]:
        viable = [row for row in cluster if bool(row.get("threshold_passed"))]
        viable.sort(key=lambda row: row["_anchor_timestamp"])
        return viable

    def _is_dense_cluster(self, cluster: list[dict], viable: list[dict]) -> bool:
        if len(cluster) < DENSE_CLUSTER_MIN_SIZE:
            return False
        if len(viable) < MIN_VIABLE_CANDIDATES:
            return False
        span = cluster[-1]["_anchor_timestamp"] - cluster[0]["_anchor_timestamp"] if len(cluster) >= 2 else 0.0
        return span >= 1.0

    def _pattern_score(self, row: dict) -> float:
        return float(row.get("audio_pattern_score", 0.0))

    def _event_score(self, row: dict) -> float:
        return float(self.research.anchor_features[row["_row_index"]]["event_selection_score"])

    def _winner(self, viable: list[dict], rule: str) -> dict | None:
        if not viable:
            return None
        if rule == "baseline_peak":
            return max(viable, key=lambda row: (row["_raw_score"], self._pattern_score(row)))
        if rule == "event":
            return max(viable, key=lambda row: (self._event_score(row), row["_raw_score"]))
        if rule == "hybrid":
            raw = np.asarray([row["_raw_score"] for row in viable], dtype=np.float32)
            pattern = np.asarray([self._pattern_score(row) for row in viable], dtype=np.float32)
            event = np.asarray([self._event_score(row) for row in viable], dtype=np.float32)
            score = 0.4 * np.clip(zscore(raw), -2.0, 2.0) + 0.2 * np.clip(zscore(pattern), -2.0, 2.0) + 0.4 * np.clip(zscore(event), -2.0, 2.0)
            return viable[int(np.argmax(score))]
        raise ValueError(f"Unknown rule: {rule}")

    def _winner_payload(self, row: dict | None, event_timestamp: float) -> dict | None:
        if row is None:
            return None
        return {
            "timestamp": float(row["_anchor_timestamp"]),
            "frontend": row.get("proposal_frontend"),
            "offset_seconds": float(row["_anchor_timestamp"] - event_timestamp),
            "raw_score": float(row["_raw_score"]),
            "pattern_score": self._pattern_score(row),
            "event_score": self._event_score(row),
            "threshold_passed": bool(row.get("threshold_passed")),
            "rejection_stage": row.get("rejection_stage"),
        }

    def _event_case(self, event: ReviewEvent) -> dict:
        cluster = self._cluster_for_timestamp(event.timestamp)
        viable = self._viable_cluster_candidates(cluster)
        dense = self._is_dense_cluster(cluster, viable)
        baseline = self._winner(viable, "baseline_peak") if dense else None
        event_winner = self._winner(viable, "event") if dense else None
        hybrid = self._winner(viable, "hybrid") if dense else None
        candidate_nearby = any(abs(row["_anchor_timestamp"] - event.timestamp) <= PRACTICAL_TOLERANCE for row in viable)
        actual_detection = self._nearest_actual_selected(event.timestamp)
        return {
            "timestamp": float(event.timestamp),
            "label": event.label,
            "subtype": event.subtype,
            "cluster_size": len(cluster),
            "viable_candidate_count": len(viable),
            "dense_cluster": dense,
            "cluster_span_seconds": 0.0 if len(cluster) < 2 else float(cluster[-1]["_anchor_timestamp"] - cluster[0]["_anchor_timestamp"]),
            "candidate_nearby": candidate_nearby,
            "actual_selected_detection": None
            if actual_detection is None
            else {
                "timestamp": actual_detection,
                "offset_seconds": actual_detection - event.timestamp,
            },
            "baseline_peak_winner": self._winner_payload(baseline, event.timestamp),
            "event_winner": self._winner_payload(event_winner, event.timestamp),
            "hybrid_winner": self._winner_payload(hybrid, event.timestamp),
        }

    def _all_cases(self) -> tuple[list[dict], list[dict], list[dict]]:
        dive_cases = [self._event_case(event) for event in self.reviewed_dives]
        rebound_cases = [self._event_case(event) for event in self.reviewed_rebounds]
        fn_cases = [self._event_case(event) for event in self.reviewed_false_negatives]
        return dive_cases, rebound_cases, fn_cases

    def _variant_metrics(self, fn_cases: list[dict], rebound_cases: list[dict], rule: str) -> dict:
        accepted_detection_count = 0
        accepted_proposal_only_count = 0
        unresolved_count = 0
        nearby_frontend_candidates = 0
        nearby_final_proposals = 0
        winner_changes = 0
        rebound_replaced = 0
        fn_breakdown: list[dict] = []

        for case in fn_cases:
            baseline = case["baseline_peak_winner"]
            chosen = case[f"{rule}_winner"] if rule != "baseline_peak" else case["baseline_peak_winner"]
            event_winner = case["event_winner"]
            hybrid_winner = case["hybrid_winner"]
            actual = case["actual_selected_detection"]

            if case["candidate_nearby"]:
                nearby_frontend_candidates += 1

            if actual is not None:
                accepted_detection_count += 1
                nearby_final_proposals += 1
            elif chosen is not None and abs(float(chosen["offset_seconds"])) <= PRACTICAL_TOLERANCE:
                accepted_proposal_only_count += 1
                nearby_final_proposals += 1
            else:
                unresolved_count += 1

            if rule != "baseline_peak" and baseline is not None and chosen is not None:
                changed = (
                    abs(float(chosen["timestamp"]) - float(baseline["timestamp"])) > 1e-6
                    or str(chosen["frontend"]) != str(baseline["frontend"])
                )
                if changed:
                    winner_changes += 1
                if (
                    changed
                    and abs(float(chosen["offset_seconds"])) < abs(float(baseline["offset_seconds"]))
                    and float(chosen["event_score"]) > float(baseline["event_score"])
                ):
                    rebound_replaced += 1

            fn_breakdown.append(
                {
                    "timestamp": case["timestamp"],
                    "proposal_failure_category": next(
                        (
                            row.get("proposal_failure_category")
                            for row in self.false_negative_rows
                            if abs(float(row["timestamp_seconds"]) - float(case["timestamp"])) <= 1e-6
                        ),
                        None,
                    ),
                    "cluster_size": case["cluster_size"],
                    "viable_candidate_count": case["viable_candidate_count"],
                    "dense_cluster": case["dense_cluster"],
                    "baseline_peak_winner": baseline,
                    "event_winner": event_winner,
                    "hybrid_winner": hybrid_winner,
                    "selected_under_rule": chosen if actual is None else {
                        "timestamp": actual["timestamp"],
                        "frontend": "actual_selected",
                        "offset_seconds": actual["offset_seconds"],
                    },
                    "improves_over_baseline": False
                    if baseline is None or chosen is None
                    else abs(float(chosen["offset_seconds"])) < abs(float(baseline["offset_seconds"])),
                    "winner_changed": False
                    if rule == "baseline_peak" or baseline is None or chosen is None
                    else (
                        abs(float(chosen["timestamp"]) - float(baseline["timestamp"])) > 1e-6
                        or str(chosen["frontend"]) != str(baseline["frontend"])
                    ),
                }
            )

        rebound_hits = 0
        for case in rebound_cases:
            chosen = case[f"{rule}_winner"] if rule != "baseline_peak" else case["baseline_peak_winner"]
            if chosen is not None and abs(float(chosen["offset_seconds"])) <= PRACTICAL_TOLERANCE:
                rebound_hits += 1
        duration_minutes = float(self.summary["per_session_metrics"][0]["duration_seconds"]) / 60.0
        fp_per_min_proxy = rebound_hits / max(duration_minutes, 1e-6)

        return {
            "false_negative_count": len(fn_cases),
            "practical_1p0": {
                "accepted_detections": accepted_detection_count,
                "accepted_proposals_only": accepted_proposal_only_count,
                "unresolved": unresolved_count,
            },
            "nearby_frontend_candidates": nearby_frontend_candidates,
            "nearby_final_proposals": nearby_final_proposals,
            "winner_changes_in_fn_neighborhoods": winner_changes,
            "baseline_rebound_replaced_by_event_like": rebound_replaced,
            "candidate_count": int(self.summary["per_session_metrics"][0]["candidate_count"]),
            "replay_coverage_proxy": float(self.summary["replay_mapping_quality"]["mapping_coverage"]),
            "median_delta_seconds_proxy": float(self.summary["replay_mapping_quality"]["median_delta_seconds"]),
            "reviewed_fp_per_min_proxy": fp_per_min_proxy,
            "threshold_promotions": 0,
            "suppression_events": self._suppression_events(),
            "fn_neighborhoods": fn_breakdown,
        }

    def run(self) -> dict:
        dive_cases, rebound_cases, fn_cases = self._all_cases()
        actual_metrics = {
            "false_negative_count": int(self.summary["false_negative_count"]),
            "practical_1p0": {
                "accepted_detections": int(self.summary["practical_false_negative_resolution"]["1.0s"]["accepted_detection_count"]),
                "accepted_proposals_only": int(self.summary["practical_false_negative_resolution"]["1.0s"]["accepted_proposal_only_count"]),
                "unresolved": int(self.summary["practical_false_negative_resolution"]["1.0s"]["unresolved_count"]),
            },
            "nearby_frontend_candidates": int(self.summary["proposal_recall_summary"]["false_negative_nearby_frontend_candidate_count"]),
            "nearby_final_proposals": int(self.summary["proposal_recall_summary"]["false_negative_nearby_final_proposal_count"]),
            "winner_changes_in_fn_neighborhoods": 0,
            "baseline_rebound_replaced_by_event_like": 0,
            "candidate_count": int(self.summary["per_session_metrics"][0]["candidate_count"]),
            "replay_coverage_proxy": float(self.summary["replay_mapping_quality"]["mapping_coverage"]),
            "median_delta_seconds_proxy": float(self.summary["replay_mapping_quality"]["median_delta_seconds"]),
            "reviewed_fp_per_min_proxy": float(self.summary["per_session_metrics"][0]["reviewed_false_positives_per_minute"]),
            "threshold_promotions": 0,
            "suppression_events": self._suppression_events(),
        }
        baseline_metrics = self._variant_metrics(fn_cases, rebound_cases, "baseline_peak")
        event_metrics = self._variant_metrics(fn_cases, rebound_cases, "event")
        hybrid_metrics = self._variant_metrics(fn_cases, rebound_cases, "hybrid")

        best_variant = "event"
        best_metrics = event_metrics
        if (
            hybrid_metrics["practical_1p0"]["accepted_detections"],
            hybrid_metrics["practical_1p0"]["accepted_proposals_only"],
            hybrid_metrics["winner_changes_in_fn_neighborhoods"],
        ) > (
            event_metrics["practical_1p0"]["accepted_detections"],
            event_metrics["practical_1p0"]["accepted_proposals_only"],
            event_metrics["winner_changes_in_fn_neighborhoods"],
        ):
            best_variant = "hybrid"
            best_metrics = hybrid_metrics

        classification = "C"
        if (
            best_metrics["reviewed_fp_per_min_proxy"] > actual_metrics["reviewed_fp_per_min_proxy"] + 1e-6
            or best_metrics["candidate_count"] > actual_metrics["candidate_count"]
            or best_metrics["replay_coverage_proxy"] + 1e-6 < actual_metrics["replay_coverage_proxy"]
        ):
            classification = "D"
        elif (
            best_metrics["practical_1p0"]["accepted_detections"] > actual_metrics["practical_1p0"]["accepted_detections"]
            or best_metrics["practical_1p0"]["unresolved"] < actual_metrics["practical_1p0"]["unresolved"]
        ):
            classification = "A"
        elif (
            best_metrics["winner_changes_in_fn_neighborhoods"] > 0
            or best_metrics["baseline_rebound_replaced_by_event_like"] > 0
            or best_metrics["practical_1p0"]["accepted_proposals_only"] > actual_metrics["practical_1p0"]["accepted_proposals_only"]
        ):
            classification = "B"

        return {
            "session_dir": str(self.session_dir),
            "cluster_rules_tested": {
                "baseline_peak": "max raw proposal score among threshold-passed raw-peak hypotheses inside dense cluster",
                "event": "max event-level selection score among threshold-passed raw-peak hypotheses inside dense cluster",
                "hybrid": "0.4 raw zscore + 0.2 pattern zscore + 0.4 event zscore inside dense cluster",
            },
            "actual_validated_metrics": actual_metrics,
            "cluster_rule_variants": {
                "baseline_peak": baseline_metrics,
                "event": event_metrics,
                "hybrid": hybrid_metrics,
            },
            "fn_neighborhood_comparison": best_metrics["fn_neighborhoods"],
            "decision": {
                "classification": classification,
                "best_variant": best_variant,
                "conclusion": (
                    "Event-level cluster selection beats peak-first collapse structurally."
                    if classification in {"A", "B"}
                    else "Event-level cluster selection does not beat peak-first collapse enough to continue this line."
                ),
            },
        }


def render_markdown(result: dict) -> str:
    actual = result["actual_validated_metrics"]
    variants = result["cluster_rule_variants"]
    decision = result["decision"]
    lines = [
        "# Event Cluster Selection",
        "",
        f"Session: `{result['session_dir']}`",
        "",
        "## Cluster Winner Rules Tested",
        "",
        f"- baseline peak: {result['cluster_rules_tested']['baseline_peak']}",
        f"- event winner: {result['cluster_rules_tested']['event']}",
        f"- hybrid winner: {result['cluster_rules_tested']['hybrid']}",
        "",
        "## Metrics Comparison",
        "",
        f"- actual validated practical `±1.0s` accepted detections: `{actual['practical_1p0']['accepted_detections']}`",
        f"- actual validated practical `±1.0s` accepted proposals only: `{actual['practical_1p0']['accepted_proposals_only']}`",
        f"- actual validated practical `±1.0s` unresolved: `{actual['practical_1p0']['unresolved']}`",
        f"- baseline peak accepted proposals only: `{variants['baseline_peak']['practical_1p0']['accepted_proposals_only']}`",
        f"- event winner accepted proposals only: `{variants['event']['practical_1p0']['accepted_proposals_only']}`",
        f"- hybrid winner accepted proposals only: `{variants['hybrid']['practical_1p0']['accepted_proposals_only']}`",
        f"- baseline peak unresolved: `{variants['baseline_peak']['practical_1p0']['unresolved']}`",
        f"- event winner unresolved: `{variants['event']['practical_1p0']['unresolved']}`",
        f"- hybrid winner unresolved: `{variants['hybrid']['practical_1p0']['unresolved']}`",
        f"- event winner changes in FN neighborhoods: `{variants['event']['winner_changes_in_fn_neighborhoods']}`",
        f"- hybrid winner changes in FN neighborhoods: `{variants['hybrid']['winner_changes_in_fn_neighborhoods']}`",
        f"- event rebound-like winner replacements: `{variants['event']['baseline_rebound_replaced_by_event_like']}`",
        f"- hybrid rebound-like winner replacements: `{variants['hybrid']['baseline_rebound_replaced_by_event_like']}`",
        f"- actual reviewed FP/min: `{actual['reviewed_fp_per_min_proxy']:.4f}`",
        f"- baseline peak rebound FP/min proxy: `{variants['baseline_peak']['reviewed_fp_per_min_proxy']:.4f}`",
        f"- event winner rebound FP/min proxy: `{variants['event']['reviewed_fp_per_min_proxy']:.4f}`",
        f"- hybrid winner rebound FP/min proxy: `{variants['hybrid']['reviewed_fp_per_min_proxy']:.4f}`",
        "",
        "## Decision",
        "",
        f"- classification: `{decision['classification']}`",
        f"- best variant: `{decision['best_variant']}`",
        f"- conclusion: {decision['conclusion']}",
        "",
        "## FN Neighborhood Winner Changes",
        "",
    ]
    for row in result["fn_neighborhood_comparison"]:
        lines.append(f"- FN `{row['timestamp']:.3f}s`, failure `{row['proposal_failure_category']}`")
        lines.append(f"  cluster size `{row['cluster_size']}`, viable `{row['viable_candidate_count']}`, dense `{row['dense_cluster']}`")
        for key in ("baseline_peak_winner", "event_winner", "hybrid_winner"):
            winner = row[key]
            label = key.replace("_winner", "")
            if winner is None:
                lines.append(f"  {label}: none")
            else:
                lines.append(
                    f"  {label}: {winner['frontend']} @ {winner['timestamp']:.3f}s "
                    f"(offset {winner['offset_seconds']:.3f}s, raw {winner['raw_score']:.3f}, "
                    f"pattern {winner['pattern_score']:.3f}, event {winner['event_score']:.3f})"
                )
        selected = row["selected_under_rule"]
        if selected is None:
            lines.append("  selected under best variant: none within ±1.0s")
        else:
            lines.append(
                f"  selected under best variant: {selected['frontend']} @ {selected['timestamp']:.3f}s "
                f"(offset {selected['offset_seconds']:.3f}s)"
            )
        lines.append(f"  improves over baseline: `{row['improves_over_baseline']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline cluster-level event winner analysis on validated springboard session artifacts.")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("outputs/evaluation_insep_15min_validated"),
        help="Validated session root containing raw peaks, replayed review, and export artifacts.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs/analysis_event_cluster_selection.json"),
        help="Where to write the JSON analysis summary.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("outputs/analysis_event_cluster_selection.md"),
        help="Where to write the Markdown analysis summary.",
    )
    args = parser.parse_args()

    research = EventClusterSelectionResearch(args.session_dir)
    result = research.run()
    serializable = to_serializable(result)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(serializable, indent=2))
    args.md_out.write_text(render_markdown(serializable))
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": str(args.md_out),
                "classification": result["decision"]["classification"],
                "best_variant": result["decision"]["best_variant"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
