#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
IMG_8237_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "img_8237_compare.json"
REVIEWED_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "reviewed_compare.json"
LONG_SESSION_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "long_session_compare.json"


def run_command(cmd: list[str]) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_validation(manifest: Path) -> dict:
    result = run_command([sys.executable, "-m", "divesensei.app.validation", str(manifest)])
    if result["returncode"] != 0:
        raise RuntimeError(json.dumps({"stage": "validate", "manifest": str(manifest), **result}, indent=2))
    return json.loads(manifest.with_suffix(".results.json").read_text())


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        img_8237 = run_validation(IMG_8237_MANIFEST)
        reviewed = run_validation(REVIEWED_MANIFEST)
        long_session = run_validation(LONG_SESSION_MANIFEST)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    img_summary = img_8237["summaries"]
    reviewed_summary = reviewed["summaries"]
    long_summary = long_session["summaries"]

    baseline_img = img_summary["audio_v1_heuristic"]
    advanced_img = img_summary["audio_v2_pcen_classifier"]
    hybrid_img = img_summary["audio_v2_hybrid_video"]
    baseline_reviewed = reviewed_summary["audio_v1_heuristic"]
    advanced_reviewed = reviewed_summary["audio_v2_pcen_classifier"]
    baseline_long = long_summary["audio_v1_heuristic"]
    advanced_long = long_summary["audio_v2_pcen_classifier"]

    checks = [
        (baseline_img["pass_rate"] == 0.0, "baseline should fail IMG_8237 hard case"),
        (advanced_img["pass_rate"] == 1.0, "advanced should pass IMG_8237 hard case"),
        (hybrid_img["pass_rate"] == 1.0, "hybrid should pass IMG_8237 hard case"),
        (advanced_reviewed["pass_rate"] >= baseline_reviewed["pass_rate"], "advanced reviewed pass rate regressed"),
        (advanced_reviewed["mean_detected_events"] <= baseline_reviewed["mean_detected_events"], "advanced reviewed false positives regressed"),
        (baseline_long["pass_rate"] == 1.0, "baseline long-session gate failed"),
        (advanced_long["pass_rate"] == 1.0, "advanced long-session gate failed"),
    ]
    failed = [message for ok, message in checks if not ok]
    if failed:
        print(
            json.dumps(
                {
                    "ok": False,
                    "errors": failed,
                    "img_8237": img_summary,
                    "reviewed": reviewed_summary,
                    "long_session": long_summary,
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "img_8237": img_summary,
                "reviewed": reviewed_summary,
                "long_session": long_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
