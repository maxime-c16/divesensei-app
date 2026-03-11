from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei",
        description="Production-ready audio-first dive detection and metadata pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("detect", help="Detect dives in a session video and export clips")
    subparsers.add_parser("validate", help="Run a benchmark manifest")

    inspect = subparsers.add_parser("inspect", help="Print a summary for a session report or UI manifest")
    inspect.add_argument("report_path")

    review_template = subparsers.add_parser("review-template", help="Create a review CSV from a session report")
    review_template.add_argument("report_path")
    review_template.add_argument("output_csv")

    subparsers.add_parser("label-audio", help="Save a labeled audio clip for future classifier training")
    train_audio_clip = subparsers.add_parser("train-audio-clip-model", help="Train the short-window audio clip classifier")
    train_audio_clip.add_argument("labels_path", nargs="?")
    train_audio_clip.add_argument("output_model", nargs="?")

    regress = subparsers.add_parser("regress", help="Run the non-regression suite")
    regress.add_argument("args", nargs=argparse.REMAINDER)

    library_index = subparsers.add_parser("library-index", help="Build a UI-ready library index from session manifests")
    library_index.add_argument("root", nargs="?")
    library_index.add_argument("output", nargs="?")
    return parser


def _inspect(path: Path) -> int:
    if not path.exists():
        print("Report not found.")
        return 1
    data = json.loads(path.read_text())
    if data.get("kind") == "divesensei.ui-session":
        session = data.get("session", {})
        telemetry = session.get("telemetry", {})
        print(f"Video: {session.get('source_video_path')}")
        print(f"Detected dives: {session.get('candidate_count')}")
        print(f"Clips written: {session.get('extracted_count')}")
        print(f"Detection time: {float(telemetry.get('detector_seconds') or 0.0):.2f}s")
        print(f"Extraction time: {float(telemetry.get('extract_seconds') or 0.0):.2f}s")
        print(f"Peak RSS: {telemetry.get('peak_rss_kb', 'n/a')} KB")
        print(f"Detections CSV: {data.get('artifacts', {}).get('detections_csv', 'n/a')}")
        print(f"First timestamp: {session.get('timestamp_range', {}).get('first')}")
        print(f"Last timestamp: {session.get('timestamp_range', {}).get('last')}")
        return 0
    debug = data.get("debug_summary", {})
    print(f"Video: {data.get('video_path')}")
    print(f"Detected dives: {data.get('candidate_count')}")
    print(f"Clips written: {len(data.get('extracted_paths', []))}")
    print(f"Detection time: {data.get('detector_seconds', 0.0):.2f}s")
    print(f"Extraction time: {data.get('extract_seconds', 0.0):.2f}s")
    print(f"Peak RSS: {data.get('peak_rss_kb', 'n/a')} KB")
    print(f"Detections CSV: {data.get('detections_csv', 'n/a')}")
    print(f"First timestamp: {debug.get('timestamp_range', {}).get('first')}")
    print(f"Last timestamp: {debug.get('timestamp_range', {}).get('last')}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0

    if argv[0] == "detect":
        if any(flag in argv[1:] for flag in ("-h", "--help")):
            from divesensei.app.session_pipeline import build_parser as detect_build_parser

            detect_build_parser().print_help()
            return 0

        from divesensei.app.session_pipeline import main as session_main
        from divesensei.preflight import format_missing_dependencies_message, missing_runtime_dependencies

        missing = missing_runtime_dependencies()
        if missing:
            print(format_missing_dependencies_message(missing))
            return 2
        return session_main(argv[1:])

    if argv[0] == "validate":
        from divesensei.app.validation import main as validation_main

        return validation_main(argv[1:])

    if argv[0] == "inspect":
        if len(argv) < 2:
            print("Report path required.")
            return 1
        return _inspect(Path(argv[1]).resolve())

    if argv[0] == "review-template":
        from divesensei.workflows.create_review_template import main as review_main

        return review_main(argv[1:])

    if argv[0] == "label-audio":
        from divesensei.workflows.save_audio_label import main as label_audio_main

        return label_audio_main(argv[1:])

    if argv[0] == "train-audio-clip-model":
        from divesensei.workflows.train_audio_clip_model import main as train_audio_clip_main

        forwarded = argv[1:]
        if not forwarded:
            forwarded = [".divesensei-runtime/audio-labels/labels.jsonl", ".divesensei-runtime/models/audio_clip_model.json"]
        return train_audio_clip_main(forwarded)

    if argv[0] == "regress":
        from divesensei.app.regression import main as regression_main

        return regression_main(argv[1:])

    if argv[0] == "library-index":
        from divesensei.metadata.build_library_index import main as library_index_main

        return library_index_main(argv[1:])

    from divesensei.app.session_pipeline import main as session_main
    from divesensei.preflight import format_missing_dependencies_message, missing_runtime_dependencies

    missing = missing_runtime_dependencies()
    if missing:
        print(format_missing_dependencies_message(missing))
        return 2
    return session_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
