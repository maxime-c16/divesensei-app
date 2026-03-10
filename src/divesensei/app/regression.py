#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "reviewed_audio.json"
DEFAULT_SESSION = Path("/srv/nas/videos/Eindhoven 2026/IMG_8281.MOV")


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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    manifest = Path(argv[0]).resolve() if argv else DEFAULT_MANIFEST
    session_video = Path(argv[1]).resolve() if len(argv) > 1 else DEFAULT_SESSION
    output_dir = REPO_ROOT / ".tmp_regression_session"

    validate = run_command(
        [
            sys.executable,
            "-m",
            "divesensei.app.validation",
            str(manifest),
        ]
    )
    if validate["returncode"] != 0:
        print(json.dumps({"stage": "validate", **validate}, indent=2))
        return 1

    results_path = manifest.with_suffix(".results.json")
    summary = json.loads(results_path.read_text())["summary"]
    if summary["pass_rate"] < 0.7619047619047619 or summary["positive_recall_proxy"] < 0.9:
        print(json.dumps({"stage": "validate", "summary": summary, "error": "benchmark regression"}, indent=2))
        return 1

    detect = run_command(
        [
            sys.executable,
            "-m",
            "divesensei.app.session_pipeline",
            str(session_video),
            "--output-dir",
            str(output_dir),
            "--profile",
            "long-session",
            "--detect-only",
            "--json",
        ]
    )
    if detect["returncode"] != 0:
        print(json.dumps({"stage": "session", **detect}, indent=2))
        return 1

    session_report = json.loads((output_dir / "session_pipeline_report.json").read_text())
    session_summary = {
        "video_path": session_report["video_path"],
        "candidate_count": session_report["candidate_count"],
        "detector_seconds": session_report["detector_seconds"],
        "report_path": str(output_dir / "session_pipeline_report.json"),
        "extracted_count": len(session_report.get("extracted_paths", [])),
    }
    candidate_count = int(session_summary["candidate_count"])
    if not 40 <= candidate_count <= 55:
        print(
            json.dumps(
                {
                    "stage": "session",
                    "summary": session_summary,
                    "error": "session detection count out of band",
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "reviewed_summary": summary,
                "session_summary": session_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
