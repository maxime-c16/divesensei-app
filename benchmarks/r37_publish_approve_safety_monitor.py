from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
R36_MONITOR_PATH = ROOT / "outputs/r36_approve_v1_safety_monitor.json"
R36_GAP_PATH = ROOT / "outputs/r36_hard_negative_acquisition_gap.json"
LATEST_JSON = ROOT / "outputs/latest_approve_safety_monitor.json"
LATEST_MD = ROOT / "outputs/latest_approve_safety_monitor.md"
OUT_DOC = ROOT / "docs/research/R37_APPROVE_SAFETY_MONITOR_WORKFLOW.md"
TRAINING_DIR = Path("/Users/mcauchy/Library/Mobile Documents/com~apple~CloudDocs/Diving/Training")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def training_video_inventory(limit: int = 80) -> list[dict[str, Any]]:
    if not TRAINING_DIR.exists():
        return []
    videos = sorted(
        [
            path
            for path in TRAINING_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mov", ".mp4", ".m4v"}
        ]
    )
    inventory: list[dict[str, Any]] = []
    for path in videos[:limit]:
        try:
            size_bytes = path.stat().st_size
            available = size_bytes > 0
        except OSError:
            size_bytes = None
            available = False
        inventory.append(
            {
                "path": str(path),
                "name": path.name,
                "size_bytes": size_bytes,
                "available_locally": available,
            }
        )
    return inventory


def main() -> None:
    monitor = load_json(R36_MONITOR_PATH)
    gap = load_json(R36_GAP_PATH)
    independent = monitor.get("current_reviewed_bank", monitor["independent_bank_r35"])
    policy = monitor["current_default_policy"]
    known_dangerous = gap["known_v1_dangerous_rows"]
    independent_dangerous = known_dangerous.get("independent_r35", [])
    calibration_dangerous = known_dangerous.get("calibration_r34", [])
    dangerous_count = int(independent["v1_dangerous"])
    trigger_hard_negative_diagnosis = dangerous_count > 0
    training_inventory = training_video_inventory()

    latest = {
        "artifact": "latest_approve_safety_monitor",
        "source_reports": {
            "monitor": str(R36_MONITOR_PATH),
            "gap": str(R36_GAP_PATH),
        },
        "active_default_policy": policy,
        "reviewed_bank": {
            "rows_considered": independent["rows"],
            "label_counts": independent["label_counts"],
            "subtype_counts": independent["subtype_counts"],
            "session_counts": independent["session_counts"],
            "source_family_counts": independent["source_family_counts"],
        },
        "approve_review_v1_status": {
            "approvals": independent["v1_approve_count"],
            "platform_approvals": independent["v1_platform_approvals"],
            "precision": independent["v1_precision"],
            "dangerous_approvals": dangerous_count,
            "status": "clean" if dangerous_count == 0 else "trigger_required",
        },
        "known_dangerous_rows": {
            "calibration_only": calibration_dangerous,
            "independent": independent_dangerous,
        },
        "trigger": {
            "hard_negative_visual_diagnosis_should_run": trigger_hard_negative_diagnosis,
            "reason": (
                "dangerous approvals present in the independent monitored bank"
                if trigger_hard_negative_diagnosis
                else "no independent dangerous approvals; do not run policy/model search"
            ),
            "rule": [
                "after each reviewed session export, run make approve-safety-monitor",
                "if dangerous approvals == 0, keep approve_review_v1 and do not promote visual veto",
                "if dangerous approvals > 0, run focused hard-negative diagnosis on the new source family",
            ],
        },
        "visual_veto_status": monitor["policy_interpretation"]["visual_veto_status"],
        "future_acquisition_sources": {
            "training_dir": str(TRAINING_DIR),
            "video_count_sampled": len(training_inventory),
            "videos": training_inventory,
            "priority_session_traits": [
                "shammy/towel throw in platform context",
                "handling noise near platform",
                "close-mic whistle or voice",
                "non-dive splash in platform context",
                "phone/camera movement near impact",
                "pool-deck impact-like transients",
            ],
        },
        "final_decisions": [
            "R37_APPROVE_SAFETY_MONITOR_OPERATIONALIZED",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }
    LATEST_JSON.write_text(json.dumps(latest, indent=2), encoding="utf-8")

    status = latest["approve_review_v1_status"]
    LATEST_MD.write_text(
        "# Latest Approve Safety Monitor\n\n"
        "This is the stable operational status artifact for the approve/review product mode.\n\n"
        "## Active Default Policy\n\n"
        f"- policy: `{policy['policy_id']}`\n"
        f"- model: `{policy['model_ref']}`\n"
        f"- approve threshold: `{policy['approve_min_score']}`\n"
        f"- status: `{policy['status']}`\n\n"
        "## Current Reviewed Bank\n\n"
        f"- rows considered: `{independent['rows']}`\n"
        f"- newly included extension rows: `{independent.get('extension_rows', 0)}`\n"
        f"- v1 approvals: `{status['approvals']}`\n"
        f"- v1 platform approvals: `{status['platform_approvals']}`\n"
        f"- v1 precision: `{status['precision']:.4f}`\n"
        f"- dangerous approvals: `{status['dangerous_approvals']}`\n"
        f"- status: `{status['status']}`\n\n"
        "## Trigger Decision\n\n"
        f"- run hard-negative visual diagnosis: `{trigger_hard_negative_diagnosis}`\n"
        f"- reason: {latest['trigger']['reason']}\n\n"
        "## Known Dangerous Rows\n\n"
        f"- calibration-only dangerous rows: `{len(calibration_dangerous)}`\n"
        f"- independent dangerous rows: `{len(independent_dangerous)}`\n\n"
        "## Acquisition Note\n\n"
        f"- training video folder: `{TRAINING_DIR}`\n"
        f"- sampled local video entries: `{len(training_inventory)}`\n"
        "- prioritize sessions with shammy/towel throws, handling noise, close-mic voice/whistle, non-dive splash, camera movement, or deck impacts.\n\n"
        "## Decisions\n\n"
        "- `R37_APPROVE_SAFETY_MONITOR_OPERATIONALIZED`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )

    OUT_DOC.write_text(
        "# R37 Approve Safety Monitor Workflow\n\n"
        "This workflow makes the approve/review safety monitor part of the normal review/export loop.\n\n"
        "## Command\n\n"
        "Run after reviewing and exporting a session:\n\n"
        "```bash\n"
        "make approve-safety-monitor\n"
        "```\n\n"
        "The command refreshes the research monitor and publishes stable latest-status artifacts:\n\n"
        "- `outputs/latest_approve_safety_monitor.json`\n"
        "- `outputs/latest_approve_safety_monitor.md`\n\n"
        "## Operating Rule\n\n"
        "1. Review the session in the desktop UI.\n"
        "2. Export reviewed labels/artifacts.\n"
        "3. Run `make approve-safety-monitor`.\n"
        "4. If dangerous approvals are `0`, do not run policy search and do not promote the visual veto.\n"
        "5. If dangerous approvals are greater than `0`, open a focused hard-negative diagnosis pass for the new source family.\n\n"
        "## What Not To Do When Clean\n\n"
        "- Do not lower approval thresholds.\n"
        "- Do not promote the visual veto from calibration-only evidence.\n"
        "- Do not run broad approve-policy sweeps.\n"
        "- Do not work on auto-exclude.\n\n"
        "## Acquisition Targets\n\n"
        f"New videos are available under `{TRAINING_DIR}`. Use them as future reviewed sources when the goal is hard-negative acquisition. Prioritize sessions with:\n\n"
        "- shammy/towel throws in platform context\n"
        "- handling noise near platform\n"
        "- close-mic whistle or voice\n"
        "- non-dive splash in platform context\n"
        "- phone/camera movement near impact\n"
        "- pool-deck impact-like transients\n\n"
        "## Current Status\n\n"
        f"- rows considered: `{independent['rows']}`\n"
        f"- newly included extension rows: `{independent.get('extension_rows', 0)}`\n"
        f"- v1 approvals: `{status['approvals']}`\n"
        f"- dangerous approvals: `{status['dangerous_approvals']}`\n"
        f"- hard-negative diagnosis trigger: `{trigger_hard_negative_diagnosis}`\n\n"
        "## Decisions\n\n"
        "- `R37_APPROVE_SAFETY_MONITOR_OPERATIONALIZED`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "latest_json": str(LATEST_JSON),
        "latest_md": str(LATEST_MD),
        "rows_considered": independent["rows"],
        "extension_rows": independent.get("extension_rows", 0),
        "v1_approvals": status["approvals"],
        "dangerous_approvals": status["dangerous_approvals"],
        "trigger_hard_negative_visual_diagnosis": trigger_hard_negative_diagnosis,
        "training_videos_sampled": len(training_inventory),
        "decisions": latest["final_decisions"],
    }, indent=2))


if __name__ == "__main__":
    main()
