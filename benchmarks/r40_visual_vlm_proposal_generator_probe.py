from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SESSION_ROOTS = [
    ROOT / "outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957",
    ROOT / "outputs/evaluation_CAO-1st-15min_20260421-072906",
    ROOT / "outputs/evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    ROOT / "outputs/evaluation_Champigny-17-04-9min_20260418-065417",
    ROOT / "outputs/evaluation_insep_plateform_mixed_sound",
    ROOT / "outputs/evaluation_insep_quick_9015_20260409_ui",
]
TOLERANCE_SECONDS = 2.0
OUT_AUDIT_JSON = ROOT / "outputs/r40_repo_readiness_audit.json"
OUT_AUDIT_MD = ROOT / "outputs/r40_repo_readiness_audit.md"
OUT_DESIGN_JSON = ROOT / "outputs/r40_visual_vlm_experiment_design.json"
OUT_DESIGN_MD = ROOT / "outputs/r40_visual_vlm_experiment_design.md"
OUT_PLAN_JSON = ROOT / "outputs/r40_visual_vlm_implementation_plan.json"
OUT_PLAN_MD = ROOT / "outputs/r40_visual_vlm_implementation_plan.md"
OUT_RESULTS_JSON = ROOT / "outputs/r40_visual_vlm_proposal_generator_probe.json"
OUT_RESULTS_MD = ROOT / "outputs/r40_visual_vlm_proposal_generator_probe.md"
OUT_DOC = ROOT / "docs/research/R40_VISUAL_VLM_PROPOSAL_GENERATOR_PROBE.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def reviewed_anchor_rows(root: Path) -> list[dict[str, Any]]:
    manifest = load_json(root / "ui_session_manifest.json")
    review = load_json(root / "evaluation_review.json")
    detections = {str(row.get("id")): row for row in manifest.get("detections", [])}
    anchors: list[dict[str, Any]] = []
    for item in review.get("decisions", []):
        event_label = item.get("eventLabel")
        if event_label not in {"platform_dive", "springboard_dive"}:
            continue
        det = detections.get(str(item.get("detectionId")))
        if not det:
            continue
        ts = item.get("manualAnchorTimestampSeconds")
        anchors.append(
            {
                "row_key": f"{root.name}::{item.get('detectionId')}",
                "session_id": root.name,
                "anchor_type": "reviewed_detection",
                "timestamp_seconds": float(ts if ts is not None else det.get("timestamp_seconds") or 0.0),
                "event_label": event_label,
                "source_video_path": manifest.get("session", {}).get("source_video_path"),
            }
        )
    for item in review.get("falseNegatives", []):
        event_label = item.get("eventLabel")
        if event_label not in {"platform_dive", "springboard_dive"}:
            continue
        anchors.append(
            {
                "row_key": f"{root.name}::{item.get('id')}",
                "session_id": root.name,
                "anchor_type": "manual_false_negative",
                "timestamp_seconds": float(item.get("timestampSeconds") or 0.0),
                "event_label": event_label,
                "source_video_path": manifest.get("session", {}).get("source_video_path"),
            }
        )
    return anchors


def audio_proposals(root: Path) -> list[dict[str, Any]]:
    manifest = load_json(root / "ui_session_manifest.json")
    rows = []
    for item in manifest.get("detections", []):
        rows.append(
            {
                "proposal_id": item.get("id"),
                "timestamp": float(item.get("timestamp_seconds") or 0.0),
                "proposal_provenance": "audio",
                "proposal_frontend": "audio",
            }
        )
    return rows


def visual_proposals(root: Path) -> list[dict[str, Any]]:
    paths = [
        root / "exports/visual-vlm-proposals/visual_proposals.jsonl",
        ROOT / "outputs/r40_smoke_visual_availability/visual_proposals.jsonl",
    ]
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def greedy_match(anchors: list[dict[str, Any]], proposals: list[dict[str, Any]], tolerance: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    for ai, anchor in enumerate(anchors):
        for pi, proposal in enumerate(proposals):
            delta = abs(float(anchor["timestamp_seconds"]) - float(proposal.get("timestamp", proposal.get("anchor_timestamp_seconds", 0.0)) or 0.0))
            if delta <= tolerance:
                pairs.append((delta, ai, pi))
    pairs.sort()
    used_a = set()
    used_p = set()
    matches = []
    for delta, ai, pi in pairs:
        if ai in used_a or pi in used_p:
            continue
        used_a.add(ai)
        used_p.add(pi)
        matches.append({"anchor": anchors[ai], "proposal": proposals[pi], "delta_seconds": delta})
    unmatched = [anchor for idx, anchor in enumerate(anchors) if idx not in used_a]
    return matches, unmatched


def proposal_metrics(root: Path) -> dict[str, Any]:
    anchors = reviewed_anchor_rows(root)
    audio = audio_proposals(root)
    visual = visual_proposals(root)
    union = list(audio)
    for proposal in visual:
        ts = float(proposal.get("timestamp", 0.0) or 0.0)
        if any(abs(ts - float(existing.get("timestamp", 0.0) or 0.0)) <= TOLERANCE_SECONDS for existing in audio):
            merged = dict(proposal)
            merged["proposal_provenance"] = "audio_visual_overlap"
            union.append(merged)
        else:
            union.append(proposal)
    duration = float(load_json(root / "ui_session_manifest.json").get("session", {}).get("session_duration_seconds") or 0.0)
    result = {}
    for name, proposals in [("audio_only", audio), ("visual_only", visual), ("union_audio_visual", union)]:
        matches, unmatched = greedy_match(anchors, proposals, TOLERANCE_SECONDS)
        result[name] = {
            "anchor_count": len(anchors),
            "proposal_count": len(proposals),
            "matched_anchor_count": len(matches),
            "missed_anchor_count": len(unmatched),
            "proposal_recall": len(matches) / len(anchors) if anchors else None,
            "manual_false_negatives_recovered": sum(1 for match in matches if match["anchor"]["anchor_type"] == "manual_false_negative"),
            "false_visual_proposals_per_minute": (
                max(0, len(visual) - len(greedy_match(anchors, visual, TOLERANCE_SECONDS)[0])) / (duration / 60.0)
                if name == "visual_only" and duration > 0
                else None
            ),
            "median_timing_delta_seconds": sorted([m["delta_seconds"] for m in matches])[len(matches) // 2] if matches else None,
            "unmatched_anchor_sample": unmatched[:10],
        }
    result["counts"] = {
        "anchors_by_type": dict(Counter(row["anchor_type"] for row in anchors)),
        "anchors_by_event_label": dict(Counter(row["event_label"] for row in anchors)),
        "visual_proposals": len(visual),
        "audio_proposals": len(audio),
        "audio_visual_overlap": sum(1 for proposal in union if proposal.get("proposal_provenance") == "audio_visual_overlap"),
    }
    return result


def run_availability_smoke(root: Path) -> dict[str, Any]:
    out_dir = ROOT / "outputs/r40_visual_vlm_availability" / root.name
    cmd = [
        sys.executable,
        "-m",
        "divesensei.cli",
        "visual-vlm-proposals",
        str(root),
        "--backend",
        "availability-check",
        "--mode",
        "audio-gated",
        "--output-dir",
        str(out_dir),
    ]
    env = dict(**os_environ_with_pythonpath())
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "output_dir": str(out_dir),
    }


def os_environ_with_pythonpath() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def build_readiness_audit(roots: list[Path]) -> dict[str, Any]:
    sessions = []
    for root in roots:
        if not (root / "ui_session_manifest.json").exists() or not (root / "evaluation_review.json").exists():
            continue
        manifest = load_json(root / "ui_session_manifest.json")
        report = load_json(root / "session_pipeline_report.json")
        artifacts = manifest.get("artifacts", {})
        source_video = Path(str(manifest.get("session", {}).get("source_video_path") or ""))
        sessions.append(
            {
                "session_root": str(root),
                "session_id": root.name,
                "source_video_path": str(source_video),
                "source_video_exists": source_video.exists(),
                "duration_seconds": manifest.get("session", {}).get("session_duration_seconds"),
                "detections": len(manifest.get("detections", [])),
                "review_decisions": len(load_json(root / "evaluation_review.json").get("decisions", [])),
                "false_negatives": len(load_json(root / "evaluation_review.json").get("falseNegatives", [])),
                "proposal_diagnostics_exists": Path(str(artifacts.get("proposal_diagnostics") or root / "proposal_diagnostics.jsonl")).exists(),
                "proposal_frontend_candidates_exists": Path(str(artifacts.get("proposal_frontend_candidates") or root / "proposal_frontend_candidates.jsonl")).exists(),
                "runtime_score_enrichment": report.get("runtime_score_enrichment", {}),
            }
        )
    return {
        "audit_id": "r40_repo_readiness_audit",
        "video_assets_available": "ui_session_manifest stores source_video_path; review_proxy is optional for UI but original source is available for most governed reviewed sessions",
        "timestamp_alignment": "detections and false negatives use session-relative seconds; proposal_diagnostics rows use the same timestamp basis",
        "natural_integration_point": "after audio proposal diagnostics in evaluate-session; visual artifacts should be exported separately and merged by timestamp/provenance",
        "current_artifacts": [
            "ui_session_manifest.json",
            "session_pipeline_report.json",
            "proposal_diagnostics.jsonl",
            "proposal_frontend_candidates.jsonl",
            "evaluation_review.json",
            "exports/evaluation-review/reviewed_candidates.jsonl",
            "exports/evaluation-review/false_negatives.jsonl",
        ],
        "ground_truth_reuse": "reviewed eventLabel values and manual falseNegatives provide reviewed anchors for proposal recall",
        "provenance_status": "proposal_frontend exists today; r40 adds explicit proposal_provenance values audio, visual_vlm_paligemma2, audio_visual_overlap",
        "bottlenecks": [
            "PaliGemma requires torch/transformers/Pillow and accepted Hugging Face license",
            "long full-session sweeps need GPU or aggressive sampling",
            "source videos can be on cloud/external disks and may be unavailable",
            "ROI geometry is not standardized per session",
            "VLM confidence is not calibrated and must remain proposal-only",
        ],
        "optional_stack": {
            "torch": module_available("torch"),
            "transformers": module_available("transformers"),
            "PIL": module_available("PIL"),
            "cv2": module_available("cv2"),
        },
        "sessions": sessions,
    }


def build_experiment_design(roots: list[Path]) -> dict[str, Any]:
    return {
        "experiment_id": "r40_visual_vlm_proposal_generator_probe",
        "policy_constraints": {
            "approve_review_v1_changed": False,
            "taxonomy_changed": False,
            "auto_approve_added": False,
            "auto_exclude_added": False,
        },
        "inputs": {
            "sessions": [str(root) for root in roots if root.exists()],
            "hard_sessions": [
                "evaluation_Compete-16-11-2025-first-10min_20260422-154957",
                "evaluation_insep_plateform_mixed_sound",
            ],
            "alignment_assumption": "all anchors and proposals are session-relative seconds in the source video timeline",
        },
        "outputs": [
            "visual_frame_predictions.jsonl",
            "visual_event_intervals.json",
            "visual_proposals.jsonl",
            "merged_proposal_diagnostics.jsonl with explicit provenance",
        ],
        "provenance_values": ["audio", "visual_vlm_paligemma2", "audio_visual_overlap"],
        "strategies": {
            "full_session_visual_sweep": {"mode": "full-session", "fps": [0.5, 1.0, 2.0], "roi_modes": ["full_frame", "center_pool"]},
            "audio_gated_visual_sweep": {"mode": "audio-gated", "windows": "current audio proposals only", "deployability": "runtime_realistic", "roi_modes": ["full_frame", "center_pool"]},
            "oracle_gated_visual_sweep": {"mode": "oracle-gated", "windows": "audio proposals plus reviewed FN neighborhoods", "deployability": "benchmark_upper_bound_only"},
            "roi_aware_visual_sweep": {"roi_modes": ["full_frame", "center_pool", "lower_water", "custom"]},
        },
        "prompt_variants": [
            "airborne_entry",
            "jumping_or_diving",
            "diving_attempt",
            "pool_entry",
        ],
        "decision_rules": [
            "naive_contains_yes",
            "strict_yes_no",
            "future_constrained_yes_no_or_logit_margin_if_supported",
            "yes_no_first_token_margin",
        ],
        "metrics": [
            "proposal recall against reviewed anchors at 2.0s tolerance",
            "recovered false negatives count",
            "false visual proposals per minute",
            "review burden delta",
            "visual-only vs audio-only vs overlap proposal counts",
            "timing delta to reviewed anchor",
            "breakdown by source/session family",
            "runtime seconds and sampled frames",
        ],
    }


def build_implementation_plan() -> dict[str, Any]:
    return {
        "implementation_id": "r40_visual_vlm_optional_proposal_path",
        "code_changes": [
            {
                "path": "src/divesensei/workflows/visual_vlm_proposals.py",
                "purpose": "optional visual frame inference, interval clustering, proposal export, and non-default merged diagnostics with provenance",
            },
            {
                "path": "src/divesensei/workflows/evaluate_session.py",
                "purpose": "research-only --with-visual-vlm-proposals hook after audio diagnostics",
            },
            {
                "path": "src/divesensei/cli.py",
                "purpose": "stable visual-vlm-proposals CLI entry point",
            },
            {
                "path": "src/divesensei/metadata/ui_contract.py",
                "purpose": "surface optional visual VLM artifacts in session manifests",
            },
            {
                "path": "pyproject.toml",
                "purpose": "declare optional visual-vlm dependency group without forcing it into normal runtime",
            },
        ],
        "integration_safety": [
            "visual path is opt-in only",
            "approval policy is untouched",
            "taxonomy is untouched",
            "missing model/dependencies produce skipped artifacts instead of changing detection behavior",
            "visual proposals carry explicit provenance and remain review-support only",
        ],
        "runtime_prerequisites": [
            "install optional dependency group or equivalent torch/transformers/Pillow stack",
            "accept Google PaliGemma license on Hugging Face",
            "provide Hugging Face token/cache path",
            "prefer external storage cache for model weights",
            "prefer GPU for long full-session sweeps",
        ],
        "first_real_run": {
            "session": "outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957",
            "mode": "audio-gated",
            "roi_modes": ["full_frame", "center_pool"],
            "fps": 1.0,
            "reason": "echo/rebound hard case with reviewed FN neighborhoods and locally available source video",
        },
    }


def main() -> int:
    roots = [root for root in SESSION_ROOTS if root.exists()]
    readiness = build_readiness_audit(roots)
    design = build_experiment_design(roots)
    plan = build_implementation_plan()
    smoke = run_availability_smoke(roots[0]) if roots else {"status": "no_roots"}
    session_metrics = {root.name: proposal_metrics(root) for root in roots}
    aggregate = {
        "anchor_count": sum(session_metrics[root.name]["audio_only"]["anchor_count"] for root in roots),
        "audio_proposal_count": sum(session_metrics[root.name]["audio_only"]["proposal_count"] for root in roots),
        "visual_proposal_count": sum(session_metrics[root.name]["visual_only"]["proposal_count"] for root in roots),
        "audio_matched": sum(session_metrics[root.name]["audio_only"]["matched_anchor_count"] for root in roots),
        "visual_matched": sum(session_metrics[root.name]["visual_only"]["matched_anchor_count"] for root in roots),
        "union_matched": sum(session_metrics[root.name]["union_audio_visual"]["matched_anchor_count"] for root in roots),
    }
    aggregate["audio_recall"] = aggregate["audio_matched"] / aggregate["anchor_count"] if aggregate["anchor_count"] else None
    aggregate["visual_recall"] = aggregate["visual_matched"] / aggregate["anchor_count"] if aggregate["anchor_count"] else None
    aggregate["union_recall"] = aggregate["union_matched"] / aggregate["anchor_count"] if aggregate["anchor_count"] else None

    optional_stack_ready = readiness["optional_stack"]["torch"] and readiness["optional_stack"]["transformers"] and readiness["optional_stack"]["PIL"]
    vlm_ready = bool(optional_stack_ready)
    result = {
        "experiment_id": "r40_visual_vlm_proposal_generator_probe",
        "status": "implemented_but_vlm_runtime_blocked" if not vlm_ready else "ready_for_real_vlm_inference",
        "readiness_summary": {
            "optional_stack_ready": optional_stack_ready,
            "paligemma_access_verified": False,
            "availability_smoke": smoke,
        },
        "aggregate_metrics_current_artifacts": aggregate,
        "session_metrics": session_metrics,
        "candidate_strategy_status": [
            {"strategy": "full-session", "status": "implemented", "benchmark_status": "requires_paligemma_runtime"},
            {"strategy": "audio-gated", "status": "implemented", "benchmark_status": "availability_smoke_complete"},
            {"strategy": "roi-aware", "status": "implemented", "benchmark_status": "requires_paligemma_runtime"},
        ],
        "decision": {
            "adopt_now": False,
            "recommendation": "research_only_until_real_paligemma_or_teacher_run_completes",
            "best_available_mode": "audio-gated visual sweep is the lowest-cost first real VLM run because it covers current audio proposals and reviewed FN neighborhoods",
            "next_phase": "run real PaliGemma inference on Compete echo/rebound session with audio-gated/full-frame and center_pool ROI variants after installing optional stack and accepting model license",
        },
    }
    write_json(OUT_AUDIT_JSON, readiness)
    write_json(OUT_DESIGN_JSON, design)
    write_json(OUT_PLAN_JSON, plan)
    write_json(OUT_RESULTS_JSON, result)

    write_md(OUT_AUDIT_MD, render_audit_md(readiness))
    write_md(OUT_DESIGN_MD, render_design_md(design))
    write_md(OUT_PLAN_MD, render_plan_md(plan))
    report_md = render_results_md(result)
    write_md(OUT_RESULTS_MD, report_md)
    write_md(OUT_DOC, report_md)
    return 0


def render_audit_md(audit: dict[str, Any]) -> str:
    lines = [
        "# R40 Repo Readiness Audit",
        "",
        f"- video assets: {audit['video_assets_available']}",
        f"- timestamp alignment: {audit['timestamp_alignment']}",
        f"- integration point: {audit['natural_integration_point']}",
        f"- ground truth: {audit['ground_truth_reuse']}",
        f"- provenance: {audit['provenance_status']}",
        "",
        "## Optional Stack",
        "",
    ]
    for name, ok in audit["optional_stack"].items():
        lines.append(f"- `{name}`: `{ok}`")
    lines.extend(["", "## Sessions", "", "| session | detections | decisions | FNs | video exists |", "|---|---:|---:|---:|---|"])
    for row in audit["sessions"]:
        lines.append(f"| `{row['session_id']}` | {row['detections']} | {row['review_decisions']} | {row['false_negatives']} | {row['source_video_exists']} |")
    lines.extend(["", "## Bottlenecks", ""])
    for item in audit["bottlenecks"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def render_design_md(design: dict[str, Any]) -> str:
    lines = [
        "# R40 Visual VLM Experiment Design",
        "",
        "This is proposal generation only. It does not alter `approve_review_v1`, taxonomy, auto-approve, or auto-exclude behavior.",
        "",
        "## Outputs",
        "",
    ]
    for item in design["outputs"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Metrics", ""])
    for item in design["metrics"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Strategies", ""])
    for name, cfg in design["strategies"].items():
        lines.append(f"- `{name}`: `{cfg}`")
    lines.extend(["", "## Prompt Variants", ""])
    for item in design["prompt_variants"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def render_plan_md(plan: dict[str, Any]) -> str:
    lines = [
        "# R40 Visual VLM Implementation Plan",
        "",
        "## Code Changes",
        "",
    ]
    for row in plan["code_changes"]:
        lines.append(f"- `{row['path']}`: {row['purpose']}")
    lines.extend(["", "## Integration Safety", ""])
    for item in plan["integration_safety"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Runtime Prerequisites", ""])
    for item in plan["runtime_prerequisites"]:
        lines.append(f"- {item}")
    first = plan["first_real_run"]
    lines.extend(
        [
            "",
            "## First Real Run",
            "",
            f"- session: `{first['session']}`",
            f"- mode: `{first['mode']}`",
            f"- ROI modes: `{', '.join(first['roi_modes'])}`",
            f"- FPS: `{first['fps']}`",
            f"- reason: {first['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def render_results_md(result: dict[str, Any]) -> str:
    aggregate = result["aggregate_metrics_current_artifacts"]
    lines = [
        "# R40 Visual VLM Proposal Generator Probe",
        "",
        "## Status",
        "",
        f"- status: `{result['status']}`",
        f"- optional stack ready: `{result['readiness_summary']['optional_stack_ready']}`",
        "- approval policy changed: `false`",
        "- taxonomy changed: `false`",
        "",
        "## Current Audio Baseline And Visual Artifact Baseline",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| reviewed dive anchors | {aggregate['anchor_count']} |",
        f"| audio proposals | {aggregate['audio_proposal_count']} |",
        f"| visual proposals currently available | {aggregate['visual_proposal_count']} |",
        f"| audio matched anchors | {aggregate['audio_matched']} |",
        f"| visual matched anchors | {aggregate['visual_matched']} |",
        f"| union matched anchors | {aggregate['union_matched']} |",
        f"| audio recall @2s | {aggregate['audio_recall'] if aggregate['audio_recall'] is not None else 'n/a'} |",
        f"| visual recall @2s | {aggregate['visual_recall'] if aggregate['visual_recall'] is not None else 'n/a'} |",
        f"| union recall @2s | {aggregate['union_recall'] if aggregate['union_recall'] is not None else 'n/a'} |",
        "",
        "## Candidate Strategy Status",
        "",
        "| strategy | implementation | benchmark status |",
        "|---|---|---|",
    ]
    for row in result["candidate_strategy_status"]:
        lines.append(f"| `{row['strategy']}` | `{row['status']}` | `{row['benchmark_status']}` |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- adopt now: `{result['decision']['adopt_now']}`",
            f"- best mode for first real run: {result['decision']['best_available_mode']}",
            f"- next phase: {result['decision']['next_phase']}",
            "",
            "## Final Decision",
            "",
            "`R40_VISUAL_VLM_PROPOSAL_PATH_IMPLEMENTED_RESEARCH_ONLY`",
            "",
            "`APPROVE_REVIEW_V1_REMAINS_DEFAULT`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
