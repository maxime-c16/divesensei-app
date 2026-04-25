from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path("/Users/mcauchy/Library/Mobile Documents/com~apple~CloudDocs/Diving/Training")
EXTRA_POOLS = [
    Path("/Volumes/Videos"),
]

OUT_INVENTORY_JSON = ROOT / "outputs/r38_hard_negative_source_inventory.json"
OUT_INVENTORY_MD = ROOT / "outputs/r38_hard_negative_source_inventory.md"
OUT_CANDIDATES_JSON = ROOT / "outputs/r38_next_review_candidates.json"
OUT_CANDIDATES_MD = ROOT / "outputs/r38_next_review_candidates.md"


def known_reviewed_sources() -> set[str]:
    sources: set[str] = set()
    for manifest_path in (ROOT / "outputs").glob("evaluation_*/ui_session_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        source = manifest.get("session", {}).get("source_video_path")
        if source:
            sources.add(str(Path(source).name).lower())
    return sources


def find_videos() -> list[Path]:
    videos: list[Path] = []
    roots = [TRAINING_DIR, *EXTRA_POOLS]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".mov", ".mp4", ".m4v"}:
                videos.append(path)
    return sorted(set(videos), key=lambda path: str(path).lower())


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=25, check=False)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip()[:500]}
    try:
        data = json.loads(proc.stdout)
    except Exception as exc:
        return {"available": False, "error": f"ffprobe JSON parse failed: {exc}"}
    video_stream = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"), {})
    duration = float(data.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
    return {
        "available": True,
        "duration_seconds": duration,
        "duration_minutes": duration / 60.0 if duration else None,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "format_name": data.get("format", {}).get("format_name"),
    }


def source_family(path: Path) -> str:
    name = path.name.lower()
    if "cao" in name:
        return "CAO"
    if "insep" in name:
        return "INSEP"
    if "snmt" in name:
        return "SNMT"
    if "champigny" in name:
        return "Champigny"
    if name.startswith("img_"):
        return "unknown_IMG"
    if "compete" in name or "compet" in name:
        return "competition_unknown"
    return "unknown"


def infer_traits(path: Path, metadata: dict[str, Any], reviewed_names: set[str]) -> dict[str, Any]:
    name = path.name.lower()
    family = source_family(path)
    duration = float(metadata.get("duration_seconds") or 0.0)
    local = bool(metadata.get("available"))
    already_reviewed_name = path.name.lower() in reviewed_names
    likely_nuisance: list[str] = []
    if "compete" in name or "compet" in name:
        likely_nuisance += ["close-mic voice/whistle", "pool-deck clutter", "non-dive splash during meet flow"]
    if "cao" in name:
        likely_nuisance += ["independent pool/location", "platform-context clutter"]
    if "insep" in name:
        likely_nuisance += ["platform context", "handling noise possible"]
    if "snmt" in name:
        likely_nuisance += ["same-family SNMT context", "voice/handling possible"]
    if "grp" in name or "debutant" in name or "débutant" in name:
        likely_nuisance += ["beginner group clutter", "coach voice", "non-dive splash"]
    if re.match(r"img_\\d+", name):
        likely_nuisance += ["unknown context", "phone handling/camera movement possible"]

    if family == "CAO":
        independence = 4
    elif family == "competition_unknown":
        independence = 5
    elif family == "unknown_IMG":
        independence = 4
    elif family == "INSEP":
        independence = 2
    elif family == "SNMT":
        independence = 1
    else:
        independence = 3

    nuisance_value = 2 + min(3, len(set(likely_nuisance)))
    if "compete" in name:
        nuisance_value += 2
    if "grp" in name or "debutant" in name or "débutant" in name:
        nuisance_value += 1
    if family == "SNMT":
        nuisance_value -= 1

    platform_context = 3
    if family in {"CAO", "INSEP", "competition_unknown"}:
        platform_context += 1
    if family == "SNMT":
        platform_context -= 1

    prep_cost = 2
    if not local:
        prep_cost += 5
    if duration > 25 * 60:
        prep_cost += 2
    elif duration > 18 * 60:
        prep_cost += 1
    if already_reviewed_name:
        prep_cost += 2

    value_score = independence * 2 + nuisance_value * 2 + platform_context - prep_cost
    if already_reviewed_name:
        value_score -= 3

    return {
        "source_family": family,
        "already_reviewed_filename": already_reviewed_name,
        "likely_platform_context": platform_context >= 3,
        "likely_session_type": "platform_or_mixed" if platform_context >= 3 else "mixed_or_springboard",
        "likely_nuisance_signals": sorted(set(likely_nuisance)),
        "independence_score": independence,
        "nuisance_value_score": nuisance_value,
        "platform_context_score": platform_context,
        "prep_cost_score": prep_cost,
        "hard_negative_value_score": value_score,
    }


def main() -> None:
    reviewed_names = known_reviewed_sources()
    rows: list[dict[str, Any]] = []
    for path in find_videos():
        metadata = ffprobe(path)
        traits = infer_traits(path, metadata, reviewed_names)
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "pool": "icloud_training" if TRAINING_DIR in path.parents else "extra_local_pool",
                "metadata": metadata,
                **traits,
            }
        )

    ranked = sorted(
        rows,
        key=lambda item: (
            item["hard_negative_value_score"],
            item["nuisance_value_score"],
            item["independence_score"],
            -item["prep_cost_score"],
        ),
        reverse=True,
    )
    top = ranked[:10]
    recommendations = {
        "experiment_name": "r38_next_review_candidates",
        "selection_goal": "Find independent nuisance-heavy sessions likely to expose new approve_review_v1 dangerous approvals.",
        "single_best_next_source": top[0] if top else None,
        "backup_second_choice": top[1] if len(top) > 1 else None,
        "backup_third_choice": top[2] if len(top) > 2 else None,
        "ranked_top_10": top,
        "decision_rule": [
            "prefer independent source families over more SNMT-family volume",
            "prefer nuisance-rich competition/CAO/unknown phone footage",
            "avoid spending review time on same-family sessions unless top candidates are unavailable",
            "prepare only one source at a time and run make approve-safety-monitor after review/export",
        ],
        "final_decisions": [
            "R38_HARD_NEGATIVE_SOURCE_TRIAGE_COMPLETE",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }
    inventory = {
        "experiment_name": "r38_hard_negative_source_inventory",
        "source_pools": [str(TRAINING_DIR), *[str(pool) for pool in EXTRA_POOLS]],
        "video_count": len(rows),
        "available_count": sum(1 for row in rows if row["metadata"].get("available")),
        "reviewed_filename_overlap_count": sum(1 for row in rows if row["already_reviewed_filename"]),
        "family_counts": dict(sorted({family: sum(1 for row in rows if row["source_family"] == family) for family in {row["source_family"] for row in rows}}.items())),
        "rows": ranked,
    }

    OUT_INVENTORY_JSON.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    OUT_CANDIDATES_JSON.write_text(json.dumps(recommendations, indent=2), encoding="utf-8")

    lines = [
        "# R38 Hard-Negative Source Inventory",
        "",
        f"- source pools: `{json.dumps(inventory['source_pools'])}`",
        f"- videos found: `{inventory['video_count']}`",
        f"- ffprobe-available: `{inventory['available_count']}`",
        f"- reviewed filename overlaps: `{inventory['reviewed_filename_overlap_count']}`",
        f"- family counts: `{json.dumps(inventory['family_counts'], sort_keys=True)}`",
        "",
        "| rank | filename | family | duration min | value | nuisance | independence | prep cost | signals |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(ranked, start=1):
        duration = row["metadata"].get("duration_minutes")
        duration_text = "n/a" if duration is None else f"{duration:.1f}"
        lines.append(
            f"| {idx} | `{row['filename']}` | {row['source_family']} | {duration_text} | {row['hard_negative_value_score']} | {row['nuisance_value_score']} | {row['independence_score']} | {row['prep_cost_score']} | {', '.join(row['likely_nuisance_signals']) or 'none'} |"
        )
    OUT_INVENTORY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    candidate_lines = [
        "# R38 Next Review Candidates",
        "",
        "The shortlist prioritizes independent, nuisance-heavy, platform-context sources likely to reveal new `approve_review_v1` hard negatives.",
        "",
        "| priority | filename | path | why |",
        "|---:|---|---|---|",
    ]
    labels = ["single best", "backup second", "backup third"]
    for idx, row in enumerate(top[:3], start=1):
        why = (
            f"family={row['source_family']}; value={row['hard_negative_value_score']}; "
            f"signals={', '.join(row['likely_nuisance_signals']) or 'unknown'}; "
            f"prep_cost={row['prep_cost_score']}"
        )
        candidate_lines.append(f"| {idx} ({labels[idx-1]}) | `{row['filename']}` | `{row['path']}` | {why} |")
    candidate_lines += [
        "",
        "## Operating Rule",
        "",
        "Prepare and review one source at a time. After review/export, run `make approve-safety-monitor`. If dangerous approvals remain `0`, do not run policy search.",
        "",
        "## Decisions",
        "",
        "- `R38_HARD_NEGATIVE_SOURCE_TRIAGE_COMPLETE`",
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`",
    ]
    OUT_CANDIDATES_MD.write_text("\n".join(candidate_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "video_count": len(rows),
        "available_count": inventory["available_count"],
        "top_3": [
            {
                "filename": row["filename"],
                "path": row["path"],
                "family": row["source_family"],
                "value": row["hard_negative_value_score"],
                "duration_minutes": row["metadata"].get("duration_minutes"),
            }
            for row in top[:3]
        ],
        "decisions": recommendations["final_decisions"],
    }, indent=2))


if __name__ == "__main__":
    main()
