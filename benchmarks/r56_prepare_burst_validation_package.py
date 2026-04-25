from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(".")
PACKAGE_ROOT = Path("kaggle/r56_burst_validation_sessions")
DATASET_ROOT = PACKAGE_ROOT / "r56_burst_validation_sessions"
SESSIONS = [
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
    "evaluation_insep_plateform_mixed_sound",
]
NUISANCE_PRIORITY = ["handling_noise", "voice_whistle", "non_dive_splash", "unknown_transient", "board_rebound", "board_slap", None]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def selected_nuisance_rows(rows: list[dict[str, Any]], max_rows: int = 12) -> list[dict[str, Any]]:
    non_dive = [row for row in rows if row.get("review_label") != "dive"]
    selected: list[dict[str, Any]] = []
    for subtype in NUISANCE_PRIORITY:
        matches = [row for row in non_dive if row.get("subtype") == subtype and row not in selected]
        cap = 4 if subtype in {"voice_whistle", "board_rebound", None} else 6
        selected.extend(matches[:cap])
        if len(selected) >= max_rows:
            return selected[:max_rows]
    return selected[:max_rows]


def random_negative_windows(duration: float, dive_ts: list[float], candidate_ts: list[float], count: int = 5) -> list[float]:
    if duration <= 30:
        return []
    out: list[float] = []
    step = duration / (count + 1)
    for idx in range(1, count + 1):
        ts = step * idx
        for shift in [0, 7, -7, 13, -13, 21]:
            candidate = min(max(3.0, ts + shift), duration - 3.0)
            if all(abs(candidate - d) > 8.0 for d in dive_ts) and all(abs(candidate - c) > 4.0 for c in candidate_ts):
                out.append(round(candidate, 3))
                break
    return out


def main() -> int:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    inventory = []
    for session_id in SESSIONS:
        src = ROOT / "outputs" / session_id
        dst = DATASET_ROOT / "outputs" / session_id
        for rel in [
            "session_pipeline_report.json",
            "proposal_diagnostics.jsonl",
            "proposal_diagnostics_summary.json",
            "evaluation_review.json",
            "ui_session_manifest.json",
            "web/session_source_review.mp4",
            "exports/evaluation-review/reviewed_candidates.jsonl",
            "exports/evaluation-review/false_negatives.jsonl",
        ]:
            copy_file(src / rel, dst / rel)
        manifest = read_json(src / "ui_session_manifest.json")
        rows = read_jsonl(src / "exports/evaluation-review/reviewed_candidates.jsonl")
        selected = selected_nuisance_rows(rows)
        dive_ts = [float(row["timestamp_seconds"]) for row in rows if row.get("review_label") == "dive"]
        candidate_ts = [float(row["timestamp_seconds"]) for row in rows]
        duration = float(manifest.get("session", {}).get("session_duration_seconds") or 0.0)
        targets: list[dict[str, Any]] = []
        for row in selected:
            targets.append(
                {
                    "target_id": f"nuis-{len(targets)+1:04d}",
                    "target_type": "reviewed_nuisance_audio_candidate",
                    "timestamp_seconds": float(row["timestamp_seconds"]),
                    "source_candidate_id": row.get("source_candidate_id"),
                    "proposal_id": row.get("proposal_id"),
                    "review_label": row.get("review_label"),
                    "subtype": row.get("subtype") or "unknown",
                    "confidence": row.get("confidence"),
                    "expected_positive": False,
                }
            )
        for ts in random_negative_windows(duration, dive_ts, candidate_ts):
            targets.append(
                {
                    "target_id": f"rand-{len(targets)+1:04d}",
                    "target_type": "random_negative_away_from_reviewed_dives",
                    "timestamp_seconds": ts,
                    "source_candidate_id": None,
                    "proposal_id": None,
                    "review_label": "random_negative",
                    "subtype": "random_negative",
                    "confidence": None,
                    "expected_positive": False,
                }
            )
        manifest["detections"] = [
            {
                "id": target["target_id"],
                "timestamp_seconds": target["timestamp_seconds"],
                "pipeline_selected": False,
                "proposal_provenance": "r56_false_control_validation_target",
                "r56_target_type": target["target_type"],
                "r56_subtype": target["subtype"],
            }
            for target in targets
        ]
        write_json(dst / "ui_session_manifest.json", manifest)
        write_json(dst / "r56_validation_targets.json", {"session_id": session_id, "targets": targets})
        subtype_counts: dict[str, int] = {}
        for target in targets:
            subtype_counts[target["subtype"]] = subtype_counts.get(target["subtype"], 0) + 1
        inventory.append(
            {
                "session_id": session_id,
                "target_count": len(targets),
                "reviewed_nuisance_count": sum(1 for row in targets if row["target_type"] == "reviewed_nuisance_audio_candidate"),
                "random_negative_count": sum(1 for row in targets if row["target_type"] == "random_negative_away_from_reviewed_dives"),
                "subtype_counts": subtype_counts,
                "review_proxy_bytes": (src / "web/session_source_review.mp4").stat().st_size,
            }
        )
    write_json(
        PACKAGE_ROOT / "dataset-metadata.json",
        {
            "title": "divesensei-r56-burst-validation-sessions",
            "id": "maximecauchy/divesensei-r56-burst-validation-sessions",
            "licenses": [{"name": "unknown"}],
        },
    )
    payload = {
        "benchmark_id": "r56_audio_local_burst_recipe_validation_inventory",
        "remote_target_type": "false_controls_only_existing_r54_used_for_hard_targets",
        "window_definition": {"pre_seconds": 3.0, "post_seconds": 3.0, "fps_values": [4.0, 8.0]},
        "sessions": inventory,
        "total_false_control_windows": sum(row["target_count"] for row in inventory),
    }
    write_json(Path("outputs/r56_burst_false_control_audit.json"), payload)
    lines = [
        "# R56 Burst False-Control Audit",
        "",
        "| Session | False controls | Reviewed nuisance | Random negative | Subtypes | Proxy MB |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in inventory:
        lines.append(
            f"| `{row['session_id']}` | {row['target_count']} | {row['reviewed_nuisance_count']} | {row['random_negative_count']} | `{row['subtype_counts']}` | {row['review_proxy_bytes']/1_000_000:.1f} |"
        )
    lines.append(f"\n- total false-control windows: `{payload['total_false_control_windows']}`")
    Path("outputs/r56_burst_false_control_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
