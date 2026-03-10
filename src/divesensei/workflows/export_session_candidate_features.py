#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("Usage: export_session_candidate_features.py <session_pipeline_report.json> <output.jsonl>")
        return 1

    report_path = Path(argv[0]).resolve()
    output_path = Path(argv[1]).resolve()
    report = json.loads(report_path.read_text())
    session_file = Path(report.get("video_path", "")).name

    with output_path.open("w", encoding="utf-8") as handle:
        for candidate in report.get("candidates", []):
            details = dict(candidate.get("details", {}))
            row = {
                "file": session_file,
                "timestamp": candidate["timestamp"],
                "audio_score": candidate["audio_score"],
                "spectral_flux": details.get("spectral_flux"),
                "rms": details.get("rms"),
                "hf_ratio": details.get("hf_ratio"),
                "spectral_centroid_hz": details.get("spectral_centroid_hz"),
                "spectral_flatness": details.get("spectral_flatness"),
                "post_flux_ratio": details.get("post_flux_ratio"),
                "post_rms_ratio": details.get("post_rms_ratio"),
                "local_prominence": details.get("local_prominence"),
                "nearby_peaks_8s": details.get("nearby_peaks_8s"),
            }
            handle.write(json.dumps(row) + "\n")

    print(json.dumps({"rows": len(report.get("candidates", [])), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
