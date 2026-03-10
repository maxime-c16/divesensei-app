#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("Usage: create_review_template.py <session_pipeline_report.json> <output.csv>")
        return 1

    report_path = Path(argv[0]).resolve()
    output_path = Path(argv[1]).resolve()
    report = json.loads(report_path.read_text())
    candidates = report.get("candidates", [])

    fieldnames = [
        "index",
        "session_file",
        "clip_filename",
        "timestamp",
        "start_time",
        "end_time",
        "confidence",
        "audio_score",
        "review_label",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, candidate in enumerate(candidates, start=1):
            confidence_suffix = f"_{candidate['confidence']}" if candidate["confidence"] != "high" else ""
            clip_filename = f"dive_splash_{idx}_t{candidate['timestamp']:.1f}s{confidence_suffix}.mp4"
            writer.writerow(
                {
                    "index": idx,
                    "session_file": Path(report.get("video_path", "")).name,
                    "clip_filename": clip_filename,
                    "timestamp": candidate["timestamp"],
                    "start_time": candidate["start_time"],
                    "end_time": candidate["end_time"],
                    "confidence": candidate["confidence"],
                    "audio_score": candidate["audio_score"],
                    "review_label": "",
                    "notes": "",
                }
            )

    print(json.dumps({"rows": len(candidates), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
