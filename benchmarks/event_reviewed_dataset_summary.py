from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_OUTPUT_JSON = Path("outputs/event_reviewed_dataset_summary.json")
DEFAULT_OUTPUT_MD = Path("outputs/event_reviewed_dataset_summary.md")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(manifest_paths: list[Path]) -> dict:
    rows: list[dict] = []
    for path in manifest_paths:
        rows.extend(load_rows(path))

    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get("source_session_id") or "unknown")].append(row)

    final_label_counts = Counter()
    by_session_counts = {}
    by_provenance = Counter()
    unlabeled = 0
    uncertain = 0
    agree = 0
    disagree = 0
    for row in rows:
        final = row.get("final_human_event_label")
        if final:
            final_label_counts[str(final)] += 1
            by_provenance[str(row.get("final_human_event_label_provenance") or "unknown")] += 1
            if str(final) == str(row.get("suggested_event_label")):
                agree += 1
            else:
                disagree += 1
        else:
            unlabeled += 1
        if row.get("suggested_event_label") == "uncertain" or row.get("uncertainty_flag"):
            uncertain += 1

    for session_id, session_rows in by_session.items():
        by_session_counts[session_id] = {
            "row_count": len(session_rows),
            "final_label_counts": dict(Counter(str(r.get("final_human_event_label")) for r in session_rows if r.get("final_human_event_label"))),
            "reviewed_count": sum(1 for r in session_rows if r.get("human_reviewed_at_event_level")),
            "missing_count": sum(1 for r in session_rows if not r.get("final_human_event_label")),
            "agreement_count": sum(1 for r in session_rows if r.get("final_human_event_label") and str(r.get("final_human_event_label")) == str(r.get("suggested_event_label"))),
            "disagreement_count": sum(1 for r in session_rows if r.get("final_human_event_label") and str(r.get("final_human_event_label")) != str(r.get("suggested_event_label"))),
        }

    return {
        "total_rows": len(rows),
        "counts_by_final_human_event_label": dict(final_label_counts),
        "counts_by_session": by_session_counts,
        "counts_by_final_human_event_label_provenance": dict(by_provenance),
        "remaining_unlabeled_rows": unlabeled,
        "remaining_uncertain_rows": uncertain,
        "suggestion_vs_final_agreement": {"agree": agree, "disagree": disagree},
        "manifest_paths": [str(path) for path in manifest_paths],
    }


def write_md(path: Path, summary: dict) -> None:
    lines = [
        "# Event Reviewed Dataset Summary",
        "",
        f"- total rows: `{summary['total_rows']}`",
        f"- unlabeled rows: `{summary['remaining_unlabeled_rows']}`",
        f"- uncertain rows: `{summary['remaining_uncertain_rows']}`",
        f"- agreement: `{json.dumps(summary['suggestion_vs_final_agreement'], sort_keys=True)}`",
        "",
        "## Final Labels",
        "",
        json.dumps(summary["counts_by_final_human_event_label"], sort_keys=True),
        "",
        "## Provenance",
        "",
        json.dumps(summary["counts_by_final_human_event_label_provenance"], sort_keys=True),
        "",
        "## By Session",
        "",
    ]
    for session_id, session_summary in summary["counts_by_session"].items():
        lines.append(f"- `{session_id}`: `{json.dumps(session_summary, sort_keys=True)}`")
    lines.append("")
    lines.append("## Manifests")
    for manifest in summary["manifest_paths"]:
        lines.append(f"- `{manifest}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize reviewed event manifests.")
    parser.add_argument("manifests", nargs="+", help="Reviewed manifest JSONL paths")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args(argv)
    manifest_paths = [Path(path) for path in args.manifests]
    summary = summarize(manifest_paths)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(Path(args.output_md), summary)
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "total_rows": summary["total_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
