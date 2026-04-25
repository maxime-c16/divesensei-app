from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(".")
PACKAGE_ROOT = Path("kaggle/r54_visual_burst_sessions")
DATASET_ROOT = PACKAGE_ROOT / "r54_visual_burst_sessions"
SESSION_IDS = [
    "evaluation_CAO-1st-15min_20260421-072906",
    "evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927",
    "evaluation_Compete-16-11-2025-first-10min_20260422-154957",
    "evaluation_SNMT-WED-8:4:26_20260419-142758",
    "evaluation_insep_plateform_mixed_sound",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    audit = read_json(Path("outputs/r53_visual_recovery_anchor_audit.json"))
    diagnosis = read_json(Path("outputs/r53_visual_recovery_failure_mode_diagnosis.json"))
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    targets_by_session: dict[str, list[dict[str, Any]]] = {}
    for session in audit["sessions"]:
        session_id = session["session_id"]
        targets: list[dict[str, Any]] = []
        for anchor in session["anchors"]:
            if anchor["category"] == "audio_missed_visual_also_missed":
                targets.append(
                    {
                        "target_id": f"avm-{len(targets)+1:04d}",
                        "target_type": "audio_missed_visual_also_missed",
                        "anchor_id": anchor["anchor_id"],
                        "timestamp_seconds": anchor["timestamp_seconds"],
                        "expected_positive": True,
                        "source_category": anchor["category"],
                    }
                )
            elif anchor["category"] == "audio_missed_visual_recovered":
                targets.append(
                    {
                        "target_id": f"avr-{len(targets)+1:04d}",
                        "target_type": "audio_missed_visual_recovered_control",
                        "anchor_id": anchor["anchor_id"],
                        "timestamp_seconds": anchor["timestamp_seconds"],
                        "expected_positive": True,
                        "source_category": anchor["category"],
                    }
                )
        diag_row = next(row for row in diagnosis["sessions"] if row["session_id"] == session_id)
        for interval in diag_row["session_diagnostics"]["unmatched_visual_intervals"]:
            targets.append(
                {
                    "target_id": f"uvp-{len(targets)+1:04d}",
                    "target_type": "unmatched_visual_proposal_control",
                    "anchor_id": None,
                    "timestamp_seconds": float(interval["anchor_timestamp_seconds"]),
                    "expected_positive": False,
                    "source_category": "unmatched_visual_proposal",
                    "source_interval": interval,
                }
            )
        targets_by_session[session_id] = sorted(targets, key=lambda row: float(row["timestamp_seconds"]))
    inventory_sessions = []
    for session_id in SESSION_IDS:
        src = ROOT / "outputs" / session_id
        dst = DATASET_ROOT / "outputs" / session_id
        copy_file(src / "session_pipeline_report.json", dst / "session_pipeline_report.json")
        copy_file(src / "proposal_diagnostics.jsonl", dst / "proposal_diagnostics.jsonl")
        copy_file(src / "proposal_diagnostics_summary.json", dst / "proposal_diagnostics_summary.json")
        copy_file(src / "evaluation_review.json", dst / "evaluation_review.json")
        copy_file(src / "web/session_source_review.mp4", dst / "web/session_source_review.mp4")
        copy_file(
            src / "exports/evaluation-review/reviewed_candidates.jsonl",
            dst / "exports/evaluation-review/reviewed_candidates.jsonl",
        )
        copy_file(src / "exports/evaluation-review/false_negatives.jsonl", dst / "exports/evaluation-review/false_negatives.jsonl")
        manifest = read_json(src / "ui_session_manifest.json")
        targets = targets_by_session.get(session_id, [])
        manifest["detections"] = [
            {
                "id": target["target_id"],
                "timestamp_seconds": float(target["timestamp_seconds"]),
                "pipeline_selected": False,
                "proposal_provenance": "r54_diagnostic_burst_target",
                "r54_target_type": target["target_type"],
                "r54_anchor_id": target["anchor_id"],
            }
            for target in targets
        ]
        write_json(dst / "ui_session_manifest.json", manifest)
        write_json(dst / "r54_burst_targets.json", {"session_id": session_id, "targets": targets})
        inventory_sessions.append(
            {
                "session_id": session_id,
                "source_session_root": str(src),
                "packaged_session_root": str(dst),
                "review_proxy_bytes": (src / "web/session_source_review.mp4").stat().st_size
                if (src / "web/session_source_review.mp4").exists()
                else 0,
                "target_count": len(targets),
                "primary_av_minus_v_minus_count": sum(1 for row in targets if row["target_type"] == "audio_missed_visual_also_missed"),
                "positive_control_av_minus_v_plus_count": sum(
                    1 for row in targets if row["target_type"] == "audio_missed_visual_recovered_control"
                ),
                "unmatched_visual_control_count": sum(1 for row in targets if row["target_type"] == "unmatched_visual_proposal_control"),
            }
        )
    metadata = {
        "title": "divesensei-r54-visual-burst-sessions",
        "id": "maximecauchy/divesensei-r54-visual-burst-sessions",
        "licenses": [{"name": "unknown"}],
    }
    write_json(PACKAGE_ROOT / "dataset-metadata.json", metadata)
    payload = {
        "benchmark_id": "r54_audio_local_visual_burst_sampling_inventory",
        "window_definition": {"pre_seconds": 3.0, "post_seconds": 3.0, "fps_values": [4.0, 8.0]},
        "sessions": inventory_sessions,
        "total_targets": sum(row["target_count"] for row in inventory_sessions),
        "total_primary_av_minus_v_minus": sum(row["primary_av_minus_v_minus_count"] for row in inventory_sessions),
        "total_positive_controls": sum(row["positive_control_av_minus_v_plus_count"] for row in inventory_sessions),
        "total_unmatched_visual_controls": sum(row["unmatched_visual_control_count"] for row in inventory_sessions),
    }
    write_json(Path("outputs/r54_audio_local_visual_burst_sampling_inventory.json"), payload)
    lines = [
        "# R54 Audio-Local Visual Burst Sampling Inventory",
        "",
        "| Session | Targets | A-V- primary | A-V+ controls | Unmatched controls | Proxy MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in inventory_sessions:
        lines.append(
            f"| `{row['session_id']}` | {row['target_count']} | {row['primary_av_minus_v_minus_count']} | {row['positive_control_av_minus_v_plus_count']} | {row['unmatched_visual_control_count']} | {row['review_proxy_bytes']/1_000_000:.1f} |"
        )
    lines.append("")
    lines.append(f"- total targets: `{payload['total_targets']}`")
    lines.append(f"- primary A-V- hard anchors: `{payload['total_primary_av_minus_v_minus']}`")
    Path("outputs/r54_audio_local_visual_burst_sampling_inventory.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
