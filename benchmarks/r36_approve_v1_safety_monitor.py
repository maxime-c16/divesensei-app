from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
R34_PATH = ROOT / "outputs/r34_legacy_splash_morphology_expanded_probe.json"
R35_PATH = ROOT / "outputs/r35_independent_visual_veto_validation.json"
R35_FAILURE_PATH = ROOT / "outputs/r35_independent_visual_veto_failure_analysis.json"

OUT_MONITOR_JSON = ROOT / "outputs/r36_approve_v1_safety_monitor.json"
OUT_MONITOR_MD = ROOT / "outputs/r36_approve_v1_safety_monitor.md"
OUT_GAP_JSON = ROOT / "outputs/r36_hard_negative_acquisition_gap.json"
OUT_GAP_MD = ROOT / "outputs/r36_hard_negative_acquisition_gap.md"
OUT_DOC = ROOT / "docs/research/R36_APPROVE_V1_SAFETY_MONITOR_AND_HARD_NEGATIVE_GAP.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def reviewed_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in sorted((ROOT / "outputs").glob("evaluation_*")):
        manifest_path = root / "ui_session_manifest.json"
        review_path = root / "evaluation_review.json"
        if not manifest_path.exists() or not review_path.exists():
            continue
        try:
            manifest = load_json(manifest_path)
            review = load_json(review_path)
        except Exception as exc:
            rows.append({"session_id": root.name, "status": "unreadable", "error": str(exc)})
            continue
        decisions = [
            item
            for item in review.get("decisions", [])
            if item.get("eventLabel") in {"platform_dive", "noise_or_other"}
        ]
        detections = manifest.get("detections", [])
        governed_present = sum(1 for det in detections if (det.get("scores") or {}).get("governed_r9_score") is not None)
        visual_present = sum(1 for det in detections if (det.get("features") or {}).get("visual_late_fusion_logreg_c0.5") is not None)
        source = manifest.get("session", {}).get("source_video_path")
        source_exists = bool(source and Path(source).exists())
        proxy_exists = (root / "web/session_source_review.mp4").exists()
        rows.append(
            {
                "session_id": root.name,
                "reviewed_platform_noise_rows": len(decisions),
                "label_counts": dict(sorted(Counter(item.get("eventLabel") for item in decisions).items())),
                "subtype_counts": dict(sorted(Counter(item.get("subtype") or "none" for item in decisions).items())),
                "detections": len(detections),
                "governed_score_rows": governed_present,
                "visual_score_rows": visual_present,
                "source_video_path": source,
                "source_exists": source_exists,
                "review_proxy_exists": proxy_exists,
                "eligible_for_future_monitoring": bool(len(decisions) and (source_exists or proxy_exists)),
            }
        )
    return rows


def detection_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for det in manifest.get("detections", []):
        candidate_id = det.get("id") or det.get("candidate_id") or det.get("detection_id")
        if not candidate_id:
            continue
        by_id[str(candidate_id)] = det
        by_id[str(candidate_id).split(":")[-1]] = det
    return by_id


def governed_r9_score(det: dict[str, Any]) -> float | None:
    for container_name in ("scores", "features"):
        value = (det.get(container_name) or {}).get("governed_r9_score")
        if value is not None:
            return float(value)
    for key in ("governed_r9_score", "audio_model_probability"):
        if det.get(key) is not None:
            return float(det[key])
    return None


def collect_extension_rows(session_ids: set[str], approve_threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded_sessions = {
        "evaluation_r30_exact_scorepath_insep_quick",
        "evaluation_r30_exact_scorepath_champigny_proxy",
        "evaluation_r27_scorepath_insep_quick",
        "evaluation_r27_scorepath_insep_quick_v2",
        "evaluation_r27_scorepath_champigny_proxy",
    }
    for root in sorted((ROOT / "outputs").glob("evaluation_*")):
        if root.name in session_ids or root.name in excluded_sessions:
            continue
        manifest_path = root / "ui_session_manifest.json"
        review_path = root / "evaluation_review.json"
        if not manifest_path.exists() or not review_path.exists():
            continue
        try:
            manifest = load_json(manifest_path)
            review = load_json(review_path)
        except Exception:
            continue
        by_id = detection_by_id(manifest)
        for item in review.get("decisions", []):
            event_label = item.get("eventLabel")
            if event_label not in {"platform_dive", "noise_or_other"}:
                continue
            detection_id = item.get("detectionId") or item.get("detection_id") or str(item.get("id", "")).split(":")[-1]
            det = by_id.get(str(detection_id), {})
            score = governed_r9_score(det)
            if score is None:
                continue
            rows.append(
                {
                    "session_id": root.name,
                    "candidate_id": detection_id,
                    "event_label": event_label,
                    "subtype": item.get("subtype") or "none",
                    "governed_r9_score": score,
                    "approved": score >= approve_threshold,
                    "dangerous": event_label == "noise_or_other" and score >= approve_threshold,
                    "source_video_path": manifest.get("session", {}).get("source_video_path"),
                }
            )
    return rows


def source_family(session_id: str) -> str:
    lowered = session_id.lower()
    if "snmt" in lowered:
        return "SNMT"
    if "cao" in lowered:
        return "CAO"
    if "champigny" in lowered:
        return "Champigny"
    if "insep" in lowered:
        return "INSEP"
    if "img_8852" in lowered or "priority123" in lowered:
        return "IMG_8852"
    return "other"


def main() -> None:
    r34 = load_json(R34_PATH)
    r35 = load_json(R35_PATH)
    r35_failure = load_json(R35_FAILURE_PATH)
    inventory = reviewed_inventory()

    r34_v1 = r34["baseline_v1"]
    r35_v1 = next(item for item in r35["candidate_comparison"] if item["candidate"] == "approve_review_v1_exact_r9")
    r35_best_veto = r35["best_safe_candidate"]
    monitored_sessions = sorted(r35["session_counts"].keys())
    monitored_session_set = set(monitored_sessions)
    family_counts = dict(sorted(Counter(source_family(session) for session in monitored_sessions).items()))
    approve_threshold = 0.92158
    extension_rows = collect_extension_rows(monitored_session_set, approve_threshold)
    extension_session_counts = dict(sorted(Counter(row["session_id"] for row in extension_rows).items()))
    extension_label_counts = dict(sorted(Counter(row["event_label"] for row in extension_rows).items()))
    extension_subtype_counts = dict(sorted(Counter(row["subtype"] for row in extension_rows).items()))
    extension_approvals = [row for row in extension_rows if row["approved"]]
    extension_dangerous = [row for row in extension_rows if row["dangerous"]]
    current_session_counts = dict(sorted((Counter(r35["session_counts"]) + Counter(extension_session_counts)).items()))
    current_label_counts = dict(sorted((Counter(r35["label_counts"]) + Counter(extension_label_counts)).items()))
    current_subtype_counts = dict(sorted((Counter(r35["subtype_counts"]) + Counter(extension_subtype_counts)).items()))
    current_family_counts = dict(sorted(Counter(source_family(session) for session in current_session_counts).items()))
    current_rows = r35["row_count"] + len(extension_rows)
    current_approvals = r35_v1["approve_count"] + len(extension_approvals)
    current_platform_approvals = r35_v1["platform_approvals"] + sum(
        1 for row in extension_approvals if row["event_label"] == "platform_dive"
    )
    current_dangerous = r35_v1["dangerous_count"] + len(extension_dangerous)
    current_precision = (current_platform_approvals / current_approvals) if current_approvals else None
    eligible_not_in_r35 = [
        item
        for item in inventory
        if item.get("eligible_for_future_monitoring")
        and item["session_id"] not in set(monitored_sessions)
        and item["session_id"] not in {
            "evaluation_r30_exact_scorepath_insep_quick",
            "evaluation_r30_exact_scorepath_champigny_proxy",
            "evaluation_r27_scorepath_insep_quick",
            "evaluation_r27_scorepath_insep_quick_v2",
            "evaluation_r27_scorepath_champigny_proxy",
        }
    ]

    monitor = {
        "experiment_name": "r36_approve_v1_safety_monitor",
        "purpose": "Codify post-r35 approve_review_v1 safety state and prevent visual-veto promotion from source-local evidence.",
        "current_default_policy": {
            "policy_id": "approve_review_v1",
            "model_ref": "r9_compact_nuisance_generalization_weighted_exact_runtime",
            "approve_min_score": 0.92158,
            "status": "active_default",
        },
        "calibration_bank_r34": {
            "rows": r34["row_count"] if "row_count" in r34 else None,
            "v1_approve_count": r34_v1["approve_count"],
            "v1_precision": r34_v1["approve_precision"],
            "v1_dangerous": r34_v1["dangerous_count"],
            "hard_negative": r34_v1.get("dangerous_rows", []),
            "best_visual_veto": r34["best_safe_candidate"],
        },
        "independent_bank_r35": {
            "rows": r35["row_count"],
            "label_counts": r35["label_counts"],
            "subtype_counts": r35["subtype_counts"],
            "session_counts": r35["session_counts"],
            "source_family_counts": family_counts,
            "v1_approve_count": r35_v1["approve_count"],
            "v1_platform_approvals": r35_v1["platform_approvals"],
            "v1_precision": r35_v1["approve_precision"],
            "v1_dangerous": r35_v1["dangerous_count"],
            "best_visual_veto": r35_best_veto,
        },
        "current_reviewed_bank": {
            "rows": current_rows,
            "label_counts": current_label_counts,
            "subtype_counts": current_subtype_counts,
            "session_counts": current_session_counts,
            "source_family_counts": current_family_counts,
            "v1_approve_count": current_approvals,
            "v1_platform_approvals": current_platform_approvals,
            "v1_precision": current_precision,
            "v1_dangerous": current_dangerous,
            "extension_rows": len(extension_rows),
            "extension_session_counts": extension_session_counts,
            "extension_v1_approve_count": len(extension_approvals),
            "extension_v1_dangerous": len(extension_dangerous),
            "extension_dangerous_rows": extension_dangerous,
        },
        "policy_interpretation": {
            "visual_veto_status": "diagnostic_shadow_only",
            "reason": "The visual veto removed the r34 shammy hard negative, but on the independent r35 bank approve_review_v1 already has zero dangerous approvals and higher coverage than vetoed variants.",
            "do_not_promote_visual_veto": True,
            "approve_review_v1_remains_default": True,
        },
        "monitoring_rule": {
            "trigger": "Any future reviewed session where approve_review_v1 produces one or more noise_or_other approvals.",
            "required_action": "Extract r33/r34 legacy-v1-plus-morphology features and compare whether the veto rejects the new dangerous row without killing true platform approvals in the same source.",
            "promotion_bar": [
                "at least two independent source families with v1 dangerous approvals",
                "visual veto removes all dangerous approvals in those families",
                "visual veto preserves materially useful true-platform approval coverage",
                "no reviewed subtype dependency",
                "works from source/proxy video available before human review",
            ],
        },
        "final_decisions": [
            "R36_APPROVE_V1_SAFETY_MONITOR_READY",
            "VISUAL_VETO_REMAINS_DIAGNOSTIC_ONLY",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }

    gap = {
        "experiment_name": "r36_hard_negative_acquisition_gap",
        "purpose": "Identify why the next bottleneck is hard-negative evidence, not another policy sweep.",
        "reviewed_inventory_count": len(inventory),
        "eligible_reviewed_inventory_count": sum(1 for item in inventory if item.get("eligible_for_future_monitoring")),
        "r35_monitored_sessions": monitored_sessions,
        "r35_source_family_counts": family_counts,
        "known_v1_dangerous_rows": {
            "calibration_r34": r34_v1.get("dangerous_rows", []),
            "independent_r35": r35_failure.get("v1_dangerous_rows", []),
        },
        "eligible_not_in_r35": eligible_not_in_r35,
        "gap_assessment": {
            "main_gap": "Not enough independent v1-dangerous nuisance approvals to validate or reject a visual veto as product logic.",
            "r35_result": "Current independent bank supports approve_review_v1 safety, not visual-veto promotion.",
            "needed_next_data": "Fresh or existing reviewed sessions that actually produce v1 dangerous approvals, ideally shammy/handling/noise/non_dive_splash in platform context.",
        },
        "recommended_next_action": {
            "action": "Run this safety monitor after each new reviewed source; only start a new veto benchmark when a new independent v1-dangerous row appears.",
            "avoid": [
                "Do not lower approve thresholds.",
                "Do not promote visual veto from one calibration hard negative.",
                "Do not run broad policy searches while independent v1 has zero dangerous approvals.",
            ],
        },
        "final_decisions": [
            "R36_HARD_NEGATIVE_EVIDENCE_GAP_CONFIRMED",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }

    OUT_MONITOR_JSON.write_text(json.dumps(monitor, indent=2), encoding="utf-8")
    OUT_GAP_JSON.write_text(json.dumps(gap, indent=2), encoding="utf-8")

    OUT_MONITOR_MD.write_text(
        "# R36 Approve V1 Safety Monitor\n\n"
        "R36 codifies the post-r35 state so we do not promote a visual veto from source-local evidence.\n\n"
        "## Current Default\n\n"
        "- policy: `approve_review_v1`\n"
        "- model: `r9_compact_nuisance_generalization_weighted_exact_runtime`\n"
        "- approve threshold: `0.92158`\n"
        "- status: `active_default`\n\n"
        "## Evidence Summary\n\n"
        f"- r34 calibration v1 approvals: `{r34_v1['approve_count']}`\n"
        f"- r34 calibration dangerous approvals: `{r34_v1['dangerous_count']}`\n"
        f"- r35 independent rows: `{r35['row_count']}`\n"
        f"- r35 independent v1 approvals: `{r35_v1['approve_count']}`\n"
        f"- r35 independent v1 precision: `{r35_v1['approve_precision']:.4f}`\n"
        f"- r35 independent dangerous approvals: `{r35_v1['dangerous_count']}`\n"
        f"- r35 source families: `{json.dumps(family_counts, sort_keys=True)}`\n\n"
        "## Current Reviewed Bank Extension\n\n"
        f"- newly included rows: `{len(extension_rows)}`\n"
        f"- current rows: `{current_rows}`\n"
        f"- current v1 approvals: `{current_approvals}`\n"
        f"- current v1 precision: `{current_precision:.4f}`\n"
        f"- current dangerous approvals: `{current_dangerous}`\n"
        f"- extension sessions: `{json.dumps(extension_session_counts, sort_keys=True)}`\n\n"
        "## Interpretation\n\n"
        "The visual veto is useful diagnostically because it removes the r34 shammy hard negative. "
        "It is not justified as a default policy change because the independent r35 bank has zero v1 dangerous approvals, and the veto only reduces coverage there.\n\n"
        "## Monitoring Rule\n\n"
        "After each new reviewed source, rerun the safety monitor. Only start a new visual-veto benchmark if `approve_review_v1` produces a new independent `noise_or_other` approval.\n\n"
        "## Decisions\n\n"
        "- `R36_APPROVE_V1_SAFETY_MONITOR_READY`\n"
        "- `VISUAL_VETO_REMAINS_DIAGNOSTIC_ONLY`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )
    OUT_GAP_MD.write_text(
        "# R36 Hard-Negative Acquisition Gap\n\n"
        "R36 identifies the current evidence gap: there are not enough independent v1-dangerous nuisance approvals to validate a visual veto as product logic.\n\n"
        f"- reviewed inventories scanned: `{len(inventory)}`\n"
        f"- eligible inventories with source/proxy media: `{gap['eligible_reviewed_inventory_count']}`\n"
        f"- r35 monitored sessions: `{len(monitored_sessions)}`\n"
        f"- r35 source families: `{json.dumps(family_counts, sort_keys=True)}`\n\n"
        "## Known Dangerous Rows\n\n"
        f"- r34 calibration dangerous rows: `{len(r34_v1.get('dangerous_rows', []))}`\n"
        f"- r35 independent dangerous rows: `{len(r35_failure.get('v1_dangerous_rows', []))}`\n\n"
        "## Next Action\n\n"
        "Do not run another broad policy sweep. Keep `approve_review_v1` default, and mine/monitor for fresh independent v1-dangerous rows. "
        "When a new dangerous row appears, run the same r33/r34 visual morphology extraction against that source family.\n\n"
        "## Decisions\n\n"
        "- `R36_HARD_NEGATIVE_EVIDENCE_GAP_CONFIRMED`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(
        "# R36 Approve V1 Safety Monitor And Hard-Negative Gap\n\n"
        "The project should not promote the current visual veto. The exact-runtime `approve_review_v1` lane is clean on the independent r35 bank, while the veto only reduces approval coverage there. "
        "The r34 shammy case remains valuable as a calibration hard negative, but one calibration nuisance is not enough to create product logic.\n\n"
        "The next productive workflow is evidence accumulation: rerun the monitor after every newly reviewed source and only launch a new veto benchmark when a fresh independent v1-dangerous nuisance approval appears.\n\n"
        "Decisions:\n\n"
        "- `R36_APPROVE_V1_SAFETY_MONITOR_READY`\n"
        "- `R36_HARD_NEGATIVE_EVIDENCE_GAP_CONFIRMED`\n"
        "- `VISUAL_VETO_REMAINS_DIAGNOSTIC_ONLY`\n"
        "- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "r35_rows": r35["row_count"],
        "r35_v1_approvals": r35_v1["approve_count"],
        "r35_v1_dangerous": r35_v1["dangerous_count"],
        "known_calibration_dangerous": len(r34_v1.get("dangerous_rows", [])),
        "known_independent_dangerous": len(r35_failure.get("v1_dangerous_rows", [])),
        "decisions": monitor["final_decisions"] + gap["final_decisions"],
    }, indent=2))


if __name__ == "__main__":
    main()
