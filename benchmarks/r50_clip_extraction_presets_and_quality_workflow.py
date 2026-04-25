from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SESSION_ROOT = Path("outputs/evaluation_insep_plateform_mixed_sound")
REVIEWED_PATH = SESSION_ROOT / "exports/evaluation-review/reviewed_candidates.jsonl"
UI_MANIFEST_PATH = SESSION_ROOT / "ui_session_manifest.json"
PRESETS = {
    "short": {"pre_seconds": 4.0, "post_seconds": 3.0},
    "medium": {"pre_seconds": 6.0, "post_seconds": 4.0},
    "long": {"pre_seconds": 8.0, "post_seconds": 5.0},
}
DEFAULT_PRESET = "medium"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clip_window(anchor: float, pre: float, post: float, duration: float) -> dict[str, Any]:
    raw_start = anchor - pre
    raw_end = anchor + post
    start = max(0.0, raw_start)
    end = max(start + 0.25, raw_end)
    if duration > 0:
        end = min(duration, end)
        start = min(start, max(0.0, end - 0.25))
    return {
        "clip_start_seconds": round(start, 3),
        "clip_end_seconds": round(end, 3),
        "duration_seconds": round(end - start, 3),
        "clamped_start": bool(start != raw_start),
        "clamped_end": bool(duration > 0 and end != raw_end),
    }


def build_manifest(reviewed: list[dict[str, Any]], ui_manifest: dict[str, Any]) -> dict[str, Any]:
    session = ui_manifest["session"]
    duration = float(session.get("session_duration_seconds") or 0.0)
    source_media_path = str(session.get("source_video_path") or "")
    rows = []
    for row in reviewed:
        anchor = float(row["timestamp_seconds"])
        for preset, config in PRESETS.items():
            window = clip_window(anchor, float(config["pre_seconds"]), float(config["post_seconds"]), duration)
            rows.append(
                {
                    "source_session_id": row.get("source_session_id") or session.get("id"),
                    "session_id": row.get("session_id") or session.get("id"),
                    "candidate_id": row.get("source_candidate_id") or row.get("proposal_id"),
                    "proposal_id": row.get("proposal_id"),
                    "anchor_timestamp_seconds": anchor,
                    "review_label": row.get("review_label"),
                    "subtype": row.get("subtype"),
                    "preset": preset,
                    "pre_seconds": config["pre_seconds"],
                    "post_seconds": config["post_seconds"],
                    **window,
                    "source_media_path": source_media_path,
                    "intended_output_clip_path": str(
                        SESSION_ROOT
                        / "clips"
                        / "presets"
                        / preset
                        / f"{row.get('proposal_id', 'proposal')}_{preset}.mp4"
                    ),
                    "virtual_clip_path": f"divesensei://session/{session.get('id')}/candidate/{row.get('proposal_id')}/clip/{preset}",
                    "bounds_clamped_by_media_duration": bool(window["clamped_start"] or window["clamped_end"]),
                    "provenance": "audio_anchor_clip_preset",
                    "exact_visual_contact_required": False,
                }
            )
    summary: dict[str, Any] = {
        "benchmark_id": "r50_clip_manifest_presets",
        "session_root": str(SESSION_ROOT),
        "source_media_path": source_media_path,
        "session_duration_seconds": duration,
        "candidate_count": len(reviewed),
        "preset_count": len(PRESETS),
        "manifest_row_count": len(rows),
        "default_preset": DEFAULT_PRESET,
        "presets": PRESETS,
        "rows": rows,
    }
    return summary


def review_burden(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest["rows"]
    out = []
    for preset in PRESETS:
        preset_rows = [row for row in rows if row["preset"] == preset]
        dive_rows = [row for row in preset_rows if row.get("review_label") == "dive"]
        nuisance_rows = [row for row in preset_rows if row.get("review_label") == "non_dive"]
        out.append(
            {
                "preset": preset,
                "clip_count": len(preset_rows),
                "dive_clip_count": len(dive_rows),
                "nuisance_clip_count": len(nuisance_rows),
                "total_review_minutes_all_candidates": round(sum(float(row["duration_seconds"]) for row in preset_rows) / 60.0, 3),
                "total_review_minutes_dive_candidates": round(sum(float(row["duration_seconds"]) for row in dive_rows) / 60.0, 3),
                "nuisance_review_minutes": round(sum(float(row["duration_seconds"]) for row in nuisance_rows) / 60.0, 3),
                "clamped_clip_count": sum(1 for row in preset_rows if row["bounds_clamped_by_media_duration"]),
            }
        )
    return out


def quality_schema() -> dict[str, Any]:
    return {
        "schema_id": "clip_quality_review_v1",
        "scope": "audio_anchor_clip_preset_review",
        "fields": {
            "clip_quality": ["good", "too_early", "too_late", "too_short", "too_long", "wrong_event", "unusable"],
            "contains_takeoff": ["yes", "no", "unknown"],
            "contains_entry": ["yes", "no", "unknown"],
            "contains_full_dive": ["yes", "no", "unknown"],
            "preferred_preset": ["short", "medium", "long", "unknown"],
            "notes": "free_text_optional",
        },
        "required_fields": ["clip_quality", "contains_takeoff", "contains_entry", "contains_full_dive", "preferred_preset"],
        "export_record_keys": [
            "session_id",
            "proposal_id",
            "preset",
            "clip_start_seconds",
            "clip_end_seconds",
            "clip_quality",
            "contains_takeoff",
            "contains_entry",
            "contains_full_dive",
            "preferred_preset",
            "notes",
            "reviewed_at",
        ],
        "policy_note": "Clip quality review is product UX evidence only. It does not change approve_review_v1 or candidate labels.",
    }


def workflow_payload(manifest: dict[str, Any], burden: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "benchmark_id": "r50_clip_review_workflow",
        "default_preset": DEFAULT_PRESET,
        "ui_integration": {
            "implemented": True,
            "changed_files": [
                "src/divesensei/metadata/ui_contract.py",
                "apps/mobile/src/native/types.ts",
                "apps/mobile/src/review/model.ts",
                "apps/mobile/src/review/ReviewWorkspace.ts",
                "apps/mobile/src/styles.css",
            ],
            "behavior": "Review card exposes short/medium/long local clip-window selector and clip-quality marker buttons. Quality markers are UI-only until native persistence is added.",
            "approve_policy_impact": "none",
        },
        "evaluation_summary": {
            "candidate_count": manifest["candidate_count"],
            "manifest_row_count": manifest["manifest_row_count"],
            "review_burden_by_preset": burden,
        },
        "next_persistence_step": "Add native persistence/export for clip_quality_review_v1 records after the UI control shape is accepted.",
    }


def visual_recovery_lane_contract() -> dict[str, Any]:
    return {
        "benchmark_id": "r50_visual_recovery_lane_contract",
        "lane_name": "Visual recovery suggestions",
        "trust_label": "Research visual suggestion; requires human review",
        "source_provenance": "visual_vlm_paligemma2",
        "accept_reject_behavior": {
            "keep": "creates or confirms a review candidate only after human action",
            "reject": "dismisses the visual suggestion",
            "unsure": "keeps it in review support",
        },
        "clip_generation": {
            "uses_same_presets": True,
            "default_preset": DEFAULT_PRESET,
            "anchor_source": "visual proposal timestamp, not approve policy",
        },
        "auto_approve_exclusion": "Visual recovery suggestions are excluded from approve_review_v1 and cannot be auto-approved.",
        "rationale": "r48 rejected audio-window hard verification; r42-r46 showed visual-only remains useful for recovering missed audio anchors.",
    }


def render_manifest_md(manifest: dict[str, Any], burden: list[dict[str, Any]]) -> str:
    lines = [
        "# R50 Clip Manifest Presets",
        "",
        f"- session: `{manifest['session_root']}`",
        f"- candidates: `{manifest['candidate_count']}`",
        f"- presets: `{list(PRESETS)}`",
        f"- manifest rows: `{manifest['manifest_row_count']}`",
        f"- default preset: `{manifest['default_preset']}`",
        "",
        "| Preset | Clips | Dive clips | Nuisance clips | Total minutes | Nuisance minutes | Clamped clips |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in burden:
        lines.append(
            f"| {row['preset']} | {row['clip_count']} | {row['dive_clip_count']} | {row['nuisance_clip_count']} | {row['total_review_minutes_all_candidates']} | {row['nuisance_review_minutes']} | {row['clamped_clip_count']} |"
        )
    return "\n".join(lines) + "\n"


def render_schema_md(schema: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R50 Clip Quality Schema",
            "",
            f"- schema: `{schema['schema_id']}`",
            f"- clip_quality: `{schema['fields']['clip_quality']}`",
            f"- contains_takeoff: `{schema['fields']['contains_takeoff']}`",
            f"- contains_entry: `{schema['fields']['contains_entry']}`",
            f"- contains_full_dive: `{schema['fields']['contains_full_dive']}`",
            f"- preferred_preset: `{schema['fields']['preferred_preset']}`",
            "",
            schema["policy_note"],
        ]
    ) + "\n"


def render_workflow_md(workflow: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R50 Clip Review Workflow",
            "",
            f"- default preset: `{workflow['default_preset']}`",
            f"- UI implemented: `{workflow['ui_integration']['implemented']}`",
            f"- behavior: {workflow['ui_integration']['behavior']}",
            f"- approve policy impact: `{workflow['ui_integration']['approve_policy_impact']}`",
            f"- next persistence step: {workflow['next_persistence_step']}",
        ]
    ) + "\n"


def render_visual_md(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R50 Visual Recovery Lane Contract",
            "",
            f"- lane name: `{contract['lane_name']}`",
            f"- trust label: {contract['trust_label']}",
            f"- provenance: `{contract['source_provenance']}`",
            f"- uses same clip presets: `{contract['clip_generation']['uses_same_presets']}`",
            f"- default preset: `{contract['clip_generation']['default_preset']}`",
            f"- auto-approve exclusion: {contract['auto_approve_exclusion']}",
        ]
    ) + "\n"


def main() -> int:
    reviewed = read_jsonl(REVIEWED_PATH)
    ui_manifest = read_json(UI_MANIFEST_PATH)
    manifest = build_manifest(reviewed, ui_manifest)
    burden = review_burden(manifest)
    schema = quality_schema()
    workflow = workflow_payload(manifest, burden)
    visual = visual_recovery_lane_contract()
    doc = "\n".join(
        [
            "# R50 Clip Extraction Presets And Quality Workflow",
            "",
            "R50 converts the r49 product architecture into a concrete audio-anchor clip preset workflow.",
            "",
            "- Medium remains the default preset.",
            "- Clip quality review is the next missing product signal.",
            "- Visual recovery stays in a separate lane and is excluded from auto-approve.",
            "- Exact visual splash/contact localization is not required.",
        ]
    ) + "\n"

    outputs = Path("outputs")
    write_json(outputs / "r50_clip_manifest_presets.json", manifest)
    write_json(outputs / "r50_clip_quality_schema.json", schema)
    write_json(outputs / "r50_clip_review_workflow.json", workflow)
    write_json(outputs / "r50_visual_recovery_lane_contract.json", visual)
    write_md(outputs / "r50_clip_manifest_presets.md", render_manifest_md(manifest, burden))
    write_md(outputs / "r50_clip_quality_schema.md", render_schema_md(schema))
    write_md(outputs / "r50_clip_review_workflow.md", render_workflow_md(workflow))
    write_md(outputs / "r50_visual_recovery_lane_contract.md", render_visual_md(visual))
    write_md(Path("docs/research/R50_CLIP_EXTRACTION_PRESETS_AND_QUALITY_WORKFLOW.md"), doc)
    print(json.dumps({"manifest_rows": manifest["manifest_row_count"], "burden": burden}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
