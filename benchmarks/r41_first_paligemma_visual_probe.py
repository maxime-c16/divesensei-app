from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SESSION_ROOT = ROOT / "outputs/evaluation_Compete-16-11-2025-first-10min_20260422-154957"
VENV = ROOT / ".venv-vlm/bin/python"
MODEL_ID = "google/paligemma2-3b-mix-224"
CACHE_DIR = Path(os.environ.get("DIVESENSEI_VLM_CACHE_DIR") or "/Users/mcauchy/.cache/huggingface")

OUT_PREFLIGHT_JSON = ROOT / "outputs/r41_visual_vlm_preflight.json"
OUT_PREFLIGHT_MD = ROOT / "outputs/r41_visual_vlm_preflight.md"
OUT_PLAN_JSON = ROOT / "outputs/r41_paligemma_benchmark_run_plan.json"
OUT_PLAN_MD = ROOT / "outputs/r41_paligemma_benchmark_run_plan.md"
OUT_RESULTS_JSON = ROOT / "outputs/r41_visual_vlm_real_probe_results.json"
OUT_RESULTS_MD = ROOT / "outputs/r41_visual_vlm_real_probe_results.md"
OUT_FAILURE_JSON = ROOT / "outputs/r41_visual_vlm_failure_analysis.json"
OUT_FAILURE_MD = ROOT / "outputs/r41_visual_vlm_failure_analysis.md"
OUT_DOC = ROOT / "docs/research/R41_FIRST_PALIGEMMA_VISUAL_PROBE.md"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    return {
        "command": " ".join(args),
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(result.get("stdout") or "{}")
    except Exception:
        return {}


def build_plan() -> dict[str, Any]:
    modes = [
        {
            "run_id": "full_session_full_frame_1fps",
            "mode": "full-session",
            "roi_mode": "full_frame",
            "fps": 1.0,
            "runtime_class": "honest_visual_only_baseline",
            "deployable": True,
            "uses_reviewed_fn_oracle": False,
        },
        {
            "run_id": "full_session_full_frame_2fps",
            "mode": "full-session",
            "roi_mode": "full_frame",
            "fps": 2.0,
            "runtime_class": "honest_visual_only_baseline",
            "deployable": True,
            "uses_reviewed_fn_oracle": False,
        },
        {
            "run_id": "audio_gated_full_frame_1fps",
            "mode": "audio-gated",
            "roi_mode": "full_frame",
            "fps": 1.0,
            "runtime_class": "runtime_realistic_hybrid",
            "deployable": True,
            "uses_reviewed_fn_oracle": False,
        },
        {
            "run_id": "audio_gated_center_pool_1fps",
            "mode": "audio-gated",
            "roi_mode": "center_pool",
            "fps": 1.0,
            "runtime_class": "runtime_realistic_hybrid",
            "deployable": True,
            "uses_reviewed_fn_oracle": False,
        },
        {
            "run_id": "oracle_gated_full_frame_1fps",
            "mode": "oracle-gated",
            "roi_mode": "full_frame",
            "fps": 1.0,
            "runtime_class": "benchmark_upper_bound_only",
            "deployable": False,
            "uses_reviewed_fn_oracle": True,
        },
    ]
    return {
        "benchmark_id": "r41_first_paligemma_visual_probe",
        "session_root": str(SESSION_ROOT),
        "model_id": MODEL_ID,
        "cache_dir": str(CACHE_DIR),
        "required_modes": modes[:4],
        "optional_oracle_mode": modes[4],
        "metrics": [
            "reviewed anchors",
            "visual proposals",
            "visual matched anchors @2s",
            "audio matched anchors @2s",
            "union matched anchors @2s",
            "visual/audio/union recall",
            "recovered false negatives",
            "false visual proposals per minute",
            "visual-only proposal count",
            "overlap proposal count",
            "review burden delta",
            "timing delta distribution",
            "wall clock and throughput",
        ],
    }


def run_probe() -> int:
    plan = build_plan()
    preflight_cmd = [
        str(VENV),
        "-m",
        "divesensei.cli",
        "visual-vlm-preflight",
        "--model-id",
        MODEL_ID,
        "--model-cache-dir",
        str(CACHE_DIR),
        "--check-processor",
    ]
    preflight_run = run_cmd(preflight_cmd)
    preflight = parse_json_stdout(preflight_run)
    write_json(OUT_PREFLIGHT_JSON, {"run": preflight_run, "preflight": preflight})
    write_json(OUT_PLAN_JSON, plan)

    runs: list[dict[str, Any]] = []
    if preflight_run["returncode"] != 0:
        blocked = {
            "benchmark_id": "r41_first_paligemma_visual_probe",
            "status": "blocked_before_real_inference",
            "blocker": "paligemma_model_access_not_available",
            "preflight": preflight,
            "executed_real_inference": False,
            "mode_results": [],
            "decision": {
                "visual_proposals_showed_incremental_recall": None,
                "review_burden_added": None,
                "recommendation": "blocked_until_huggingface_token_with_accepted_paligemma_license_is_available",
            },
        }
        write_json(OUT_RESULTS_JSON, blocked)
        failure = build_failure_analysis(blocked)
        write_json(OUT_FAILURE_JSON, failure)
        render_all(preflight, plan, blocked, failure)
        return 0

    for mode in [*plan["required_modes"], plan["optional_oracle_mode"]]:
        output_dir = ROOT / "outputs" / f"r41_paligemma_probe_compete_{mode['run_id']}"
        args = [
            str(VENV),
            "-m",
            "divesensei.cli",
            "visual-vlm-proposals",
            str(SESSION_ROOT),
            "--backend",
            "paligemma",
            "--mode",
            mode["mode"],
            "--roi-mode",
            mode["roi_mode"],
            "--fps",
            str(mode["fps"]),
            "--model-cache-dir",
            str(CACHE_DIR),
            "--output-dir",
            str(output_dir),
        ]
        run = run_cmd(args, timeout=60 * 60 * 6)
        summary = parse_json_stdout(run)
        runs.append({"mode": mode, "run": run, "summary": summary})
        if run["returncode"] != 0:
            break

    result = {
        "benchmark_id": "r41_first_paligemma_visual_probe",
        "status": "complete" if all(item["run"]["returncode"] == 0 for item in runs) else "partial_or_failed",
        "preflight": preflight,
        "executed_real_inference": bool(runs and any((item["summary"].get("status") == "complete") for item in runs)),
        "mode_results": runs,
        "decision": {
            "visual_proposals_showed_incremental_recall": "requires_metric_postprocess",
            "review_burden_added": "requires_metric_postprocess",
            "recommendation": "evaluate generated visual_proposals against reviewed anchors before any adoption decision",
        },
    }
    write_json(OUT_RESULTS_JSON, result)
    failure = build_failure_analysis(result)
    write_json(OUT_FAILURE_JSON, failure)
    render_all(preflight, plan, result, failure)
    return 0


def build_failure_analysis(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("executed_real_inference"):
        return {
            "status": "model_access_blocked_no_visual_error_analysis_possible",
            "qualitative_error_analysis": [],
            "observed_failure_mode": "Hugging Face gated PaliGemma access is unavailable; no decoded frame predictions were produced.",
            "not_evaluated_yet": [
                "diver too small in frame",
                "diver off-camera",
                "occlusion by board/platform/rails",
                "reflections/poolside clutter",
                "non-dive person near pool",
                "rebound/setup mistaken as dive",
                "ROI cropping out signal",
                "low-FPS sampling miss",
                "timestamp alignment issue",
            ],
        }
    return {
        "status": "needs_manual_error_review_from_visual_predictions",
        "qualitative_error_analysis": [],
    }


def render_all(preflight: dict[str, Any], plan: dict[str, Any], result: dict[str, Any], failure: dict[str, Any]) -> None:
    write_md(OUT_PREFLIGHT_MD, render_preflight(preflight))
    write_md(OUT_PLAN_MD, render_plan(plan))
    results_md = render_results(result)
    write_md(OUT_RESULTS_MD, results_md)
    write_md(OUT_FAILURE_MD, render_failure(failure))
    write_md(OUT_DOC, results_md + "\n\n" + render_failure(failure))


def render_preflight(preflight: dict[str, Any]) -> str:
    lines = ["# R41 Visual VLM Preflight", ""]
    lines.append(f"- can proceed: `{preflight.get('can_proceed')}`")
    lines.append(f"- backend: `{preflight.get('backend')}`")
    lines.append(f"- model: `{preflight.get('model_id')}`")
    lines.append(f"- device: `{preflight.get('device')}`")
    lines.append(f"- cache: `{preflight.get('cache_dir')}`")
    lines.append(f"- HF token present: `{preflight.get('hf_token_present')}`")
    lines.append(f"- model gated: `{preflight.get('model_gated')}`")
    lines.append(f"- processor load: `{preflight.get('processor_load_status')}`")
    if preflight.get("errors"):
        lines.extend(["", "## Errors", ""])
        for item in preflight["errors"]:
            lines.append(f"- `{item}`")
    if preflight.get("processor_load_error"):
        lines.extend(["", "## Processor Error", "", "```text", str(preflight["processor_load_error"]), "```"])
    lines.append("")
    return "\n".join(lines)


def render_plan(plan: dict[str, Any]) -> str:
    lines = [
        "# R41 PaliGemma Benchmark Run Plan",
        "",
        f"- session: `{plan['session_root']}`",
        f"- model: `{plan['model_id']}`",
        f"- cache: `{plan['cache_dir']}`",
        "",
        "| run | mode | ROI | FPS | runtime class | reviewed-FN oracle |",
        "|---|---|---|---:|---|---|",
    ]
    for row in [*plan["required_modes"], plan["optional_oracle_mode"]]:
        lines.append(f"| `{row['run_id']}` | `{row['mode']}` | `{row['roi_mode']}` | {row['fps']} | `{row['runtime_class']}` | `{row['uses_reviewed_fn_oracle']}` |")
    lines.append("")
    return "\n".join(lines)


def render_results(result: dict[str, Any]) -> str:
    lines = [
        "# R41 First PaliGemma Visual Probe Results",
        "",
        f"- status: `{result.get('status')}`",
        f"- executed real inference: `{result.get('executed_real_inference')}`",
        "",
    ]
    if result.get("blocker"):
        lines.append(f"- blocker: `{result['blocker']}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- recommendation: {result.get('decision', {}).get('recommendation')}",
            "- `approve_review_v1` remains default.",
            "- no taxonomy, auto-approve, or auto-exclude changes were made.",
            "",
        ]
    )
    return "\n".join(lines)


def render_failure(failure: dict[str, Any]) -> str:
    lines = ["# R41 Failure Analysis", "", f"- status: `{failure.get('status')}`", ""]
    if failure.get("observed_failure_mode"):
        lines.append(f"- observed failure mode: {failure['observed_failure_mode']}")
    if failure.get("not_evaluated_yet"):
        lines.extend(["", "## Visual Failure Modes Not Yet Evaluated", ""])
        for item in failure["not_evaluated_yet"]:
            lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(run_probe())
