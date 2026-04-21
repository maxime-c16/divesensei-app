from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/r20_nuisance_hardening_bank.json"
MANIFEST_GLOB = "evaluation_*/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl"
SHADOW_BACKFILL_GLOB = "r23_*_shadow_score_backfill_rows.jsonl"
NEW_VIDEO = Path("~/Downloads/CAO-1st-15min.mov").expanduser()

OUT_ELIGIBILITY_JSON = ROOT / "outputs/r23_shadow_eligibility_expansion.json"
OUT_ELIGIBILITY_MD = ROOT / "outputs/r23_shadow_eligibility_expansion.md"
OUT_ACCUM_JSON = ROOT / "outputs/r23_shadow_bank_accumulation.json"
OUT_ACCUM_MD = ROOT / "outputs/r23_shadow_bank_accumulation.md"
OUT_DOC = ROOT / "docs/research/APPROVE_REVIEW_V2_ROLLOUT_EVIDENCE_UPDATE.md"
OUT_FRESH_JSON = ROOT / "outputs/r23_fresh_source_preparation.json"
OUT_FRESH_MD = ROOT / "outputs/r23_fresh_source_preparation.md"

V1_R9_MIN = 0.92158
V2_R9_FLOOR = 0.84
V2_VISUAL_MIN = 0.55
RISK_SUBTYPES = {"handling_noise", "voice_whistle", "non_dive_splash", "unknown_transient"}
RISK_SOURCE_MARKERS = {"img_8852"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flag_v1(row: dict[str, Any]) -> bool:
    return float(row.get("r9_score") or 0.0) >= V1_R9_MIN


def flag_v2(row: dict[str, Any]) -> bool:
    r9 = float(row.get("r9_score") or 0.0)
    visual = row.get("visual_score")
    if r9 >= V1_R9_MIN:
        return True
    if visual is None:
        return False
    return r9 >= V2_R9_FLOOR and float(visual) >= V2_VISUAL_MIN


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "source_session_id": row["source_session_id"],
        "label": row["label"],
        "legacy_subtype": row.get("legacy_subtype"),
        "r9_score": row.get("r9_score"),
        "visual_late_fusion_logreg_c0.5": row.get("visual_score"),
        "freshness_role": row.get("freshness_role"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    v1 = [row for row in rows if flag_v1(row)]
    v2 = [row for row in rows if flag_v2(row)]
    added = [row for row in rows if flag_v2(row) and not flag_v1(row)]
    suspicious = [row for row in added if row.get("label") != "platform_dive" or str(row.get("legacy_subtype") or "none") in RISK_SUBTYPES]
    source_risky = [row for row in added if any(marker in str(row.get("source_session_id", "")).lower() for marker in RISK_SOURCE_MARKERS)]
    near_boundary = [
        row for row in added
        if 0.84 <= float(row.get("r9_score") or 0.0) <= 0.88
        or V2_VISUAL_MIN <= float(row.get("visual_score") or 0.0) <= 0.65
    ]
    return {
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in rows).items())),
        "source_session_counts": dict(sorted(Counter(str(row.get("source_session_id")) for row in rows).items())),
        "v1_approved_count": len(v1),
        "v2_shadow_approved_count": len(v2),
        "v2_added_approve_count": len(added),
        "v1_coverage": len(v1) / len(rows) if rows else 0.0,
        "v2_shadow_coverage": len(v2) / len(rows) if rows else 0.0,
        "coverage_delta": (len(v2) - len(v1)) / len(rows) if rows else 0.0,
        "v2_added_label_counts": dict(sorted(Counter(str(row.get("label")) for row in added).items())),
        "v2_added_subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in added).items())),
        "v2_added_source_session_counts": dict(sorted(Counter(str(row.get("source_session_id")) for row in added).items())),
        "suspicious_added_approval_count": len(suspicious),
        "risky_source_added_approval_count": len(source_risky),
        "near_boundary_added_approval_count": len(near_boundary),
        "suspicious_added_approvals": [compact(row) for row in suspicious],
        "risky_source_added_approvals": [compact(row) for row in source_risky],
        "near_boundary_added_approvals": [compact(row) for row in near_boundary],
        "v2_added_approvals": [compact(row) for row in added],
    }


def manifest_row_key(row: dict[str, Any], index: int) -> str:
    sid = str(row.get("source_session_id") or row.get("session_id") or "")
    rid = str(row.get("legacy_candidate_id") or f"row-{index + 1:04d}")
    return f"{sid}::{rid}"


def main() -> None:
    bank = load_json(BANK)
    rows_by_key: dict[str, dict[str, Any]] = {}
    priority = {
        "internal_official_holdout": 3,
        "corrected_external_holdout": 3,
        "source_unit_snmt": 3,
        "source_unit_img_8852": 3,
        "source_unit_champigny_1704": 3,
    }
    for row in bank["rows"]:
        key = row["row_key"]
        current = rows_by_key.get(key)
        if current is None or priority.get(str(row.get("bank_split")), 1) > priority.get(str(current.get("bank_split")), 1):
            rows_by_key[key] = dict(row)
    backfilled_sources: list[dict[str, Any]] = []
    for path in sorted((ROOT / "outputs").glob(SHADOW_BACKFILL_GLOB)):
        backfill_rows = load_jsonl(path)
        for row in backfill_rows:
            row["freshness_role"] = "fresh_or_less_central"
            rows_by_key[str(row["row_key"])] = dict(row)
        backfilled_sources.append(
            {
                "path": str(path),
                "row_count": len(backfill_rows),
                "source_session_counts": dict(sorted(Counter(str(row.get("source_session_id")) for row in backfill_rows).items())),
            }
        )
    rows = list(rows_by_key.values())
    discovery_sessions = {"evaluation_insep_plateform_mixed_sound", "evaluation_insep_quick_9015_20260409_ui"}
    for row in rows:
        sid = str(row.get("source_session_id") or "")
        row["freshness_role"] = "fresh_or_less_central" if sid not in discovery_sessions else "central_benchmark"

    eligible_sources = sorted(set(str(row.get("source_session_id")) for row in rows))
    source_summaries = {source: summarize([row for row in rows if row.get("source_session_id") == source]) for source in eligible_sources}
    overall = summarize(rows)
    fresh_rows = [row for row in rows if row["freshness_role"] == "fresh_or_less_central"]
    fresh_summary = summarize(fresh_rows)

    reviewed_sources = []
    for path in sorted((ROOT / "outputs").glob(MANIFEST_GLOB)):
        manifest_rows = [row for row in load_jsonl(path) if row.get("final_human_event_label") in {"platform_dive", "noise_or_other"}]
        keys = [manifest_row_key(row, idx) for idx, row in enumerate(manifest_rows)]
        eligible_count = sum(1 for key in keys if key in rows_by_key)
        reviewed_sources.append({
            "source": path.parts[-4],
            "manifest_path": str(path),
            "platform_noise_row_count": len(manifest_rows),
            "eligible_shadow_scored_row_count": eligible_count,
            "eligible": eligible_count > 0,
            "exclusion_reason": None if eligible_count > 0 else "missing visual_late_fusion_logreg_c0.5 shadow score path",
        })
    prepared_dirs = sorted((ROOT / "outputs").glob("evaluation_CAO-1st-15min_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    prepared_dir = prepared_dirs[0] if prepared_dirs else None
    reviewed_manifest_exists = prepared_dir is not None and (prepared_dir / "exports/event-reviewed-manifest/event_reviewed_manifest.jsonl").exists()
    fresh_reason = "not prepared yet"
    if prepared_dir:
        if reviewed_manifest_exists:
            fresh_reason = "reviewed event manifest exists; excluded from shadow evidence because visual_late_fusion_logreg_c0.5 is not backfilled for this source"
        else:
            fresh_reason = "prepared for review but no human-reviewed event manifest yet"
    fresh_included = prepared_dir is not None and any(
        str(row.get("source_session_id")) == prepared_dir.name for row in rows_by_key.values()
    )
    if fresh_included:
        fresh_reason = "reviewed event manifest exists and visual_late_fusion_logreg_c0.5 was backfilled for shadow evidence"
    fresh_preparation = {
        "source_video_path": str(NEW_VIDEO),
        "source_video_exists": NEW_VIDEO.exists(),
        "prepared_session_dir": str(prepared_dir) if prepared_dir else None,
        "prepared_session_id": prepared_dir.name if prepared_dir else None,
        "review_url": f"http://127.0.0.1:5173/?session={prepared_dir.name}&tab=1" if prepared_dir else None,
        "review_ready": prepared_dir is not None and (prepared_dir / "ui_session_manifest.json").exists(),
        "reviewed_manifest_exists": reviewed_manifest_exists,
        "included_in_shadow_evidence": fresh_included,
        "reason": fresh_reason,
    }

    suspicious = overall["suspicious_added_approval_count"]
    shadow_clean = suspicious == 0 and overall["v2_added_label_counts"] == {"platform_dive": overall["v2_added_approve_count"]}
    enough_for_rollout = shadow_clean and len(eligible_sources) >= 7 and fresh_summary["v2_added_approve_count"] >= 10
    evidence_decision = "R23_SHADOW_EVIDENCE_STILL_CLEAN" if shadow_clean else "R23_SHADOW_EVIDENCE_REGRESSED"
    rollout_decision = "V2_READY_FOR_LIMITED_FLAGGED_ROLLOUT" if enough_for_rollout else "V2_REQUIRES_MORE_SHADOW_ACCUMULATION"

    eligibility = {
        "experiment_name": "r23_shadow_eligibility_expansion",
        "currently_eligible_sources": eligible_sources,
        "reviewed_sources": reviewed_sources,
        "newly_made_eligible_sources": backfilled_sources,
        "blocked_sources": [item for item in reviewed_sources if not item["eligible"]],
        "eligibility_decision": "R23_SHADOW_ELIGIBILITY_EXPANDED" if backfilled_sources else "R23_SHADOW_ELIGIBILITY_BLOCKED",
        "reason": "Backfilled visual late-fusion shadow scores for reviewed source rows." if backfilled_sources else "No additional reviewed sources could be made eligible from existing caches; missing visual late-fusion scores require clip/frame embedding backfill.",
    }
    accumulation = {
        "experiment_name": "r23_shadow_eligibility_expansion_and_fresh_source_evidence",
        "final_decision": evidence_decision,
        "rollout_decision": rollout_decision,
        "new_video": {
            "path": str(NEW_VIDEO),
            "exists": NEW_VIDEO.exists(),
            "included": fresh_included,
            "prepared_session_id": fresh_preparation["prepared_session_id"],
            "review_url": fresh_preparation["review_url"],
            "reason": fresh_reason,
        },
        "policy": {
            "default": "approve_review_v1",
            "shadow": "approve_review_v2_shadow",
            "shadow_logic": {
                "approve_if_any": [
                    {"r9_score_gte": V1_R9_MIN},
                    {"r9_score_gte": V2_R9_FLOOR, "visual_late_fusion_logreg_c0.5_gte": V2_VISUAL_MIN},
                ],
            },
        },
        "reviewed_sources": reviewed_sources,
        "backfilled_sources": backfilled_sources,
        "eligible_sources": eligible_sources,
        "overall": overall,
        "fresh_or_less_central": fresh_summary,
        "source_summaries": source_summaries,
        "promotion_readiness": {
            "shadow_evidence_clean": shadow_clean,
            "eligible_source_count": len(eligible_sources),
            "requires_more_shadow_accumulation": not enough_for_rollout,
            "reason": "clean after fresh CAO backfill, but rollout still needs continued shadow override evidence before replacing v1" if fresh_included else "clean but source coverage is still limited to existing shadow-scored bank; new CAO source needs review and shadow scoring before rollout decision",
        },
    }
    OUT_ELIGIBILITY_JSON.write_text(json.dumps(eligibility, indent=2), encoding="utf-8")
    OUT_ACCUM_JSON.write_text(json.dumps(accumulation, indent=2), encoding="utf-8")
    OUT_FRESH_JSON.write_text(json.dumps(fresh_preparation, indent=2), encoding="utf-8")

    source_table = [
        "| source | rows | v1 approved | shadow approved | added | suspicious |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, summary in source_summaries.items():
        source_table.append(
            f"| `{source}` | {summary['row_count']} | {summary['v1_approved_count']} | {summary['v2_shadow_approved_count']} | {summary['v2_added_approve_count']} | {summary['suspicious_added_approval_count']} |"
        )
    excluded = [item for item in reviewed_sources if not item["eligible"]]
    excluded_lines = [f"- `{item['source']}`: {item['exclusion_reason']} ({item['platform_noise_row_count']} platform/noise rows)" for item in excluded]
    OUT_ELIGIBILITY_MD.write_text(
        "\n".join([
            "# r23 Shadow Eligibility Expansion",
            "",
            f"- eligibility decision: `{'R23_SHADOW_ELIGIBILITY_EXPANDED' if backfilled_sources else 'R23_SHADOW_ELIGIBILITY_BLOCKED'}`",
            f"- currently eligible sources: `{len(eligible_sources)}`",
            f"- newly made eligible sources: `{len(backfilled_sources)}`",
            f"- shadow score backfills loaded: `{len(backfilled_sources)}`",
            "",
            "## Blocked Reviewed Sources",
            "",
            *(excluded_lines or ["- none"]),
        ]) + "\n",
        encoding="utf-8",
    )
    OUT_ACCUM_MD.write_text(
        "\n".join([
            "# r23 Shadow Bank Accumulation",
            "",
            f"- final decision: `{evidence_decision}`",
            f"- rollout decision: `{rollout_decision}`",
            f"- eligible sources: `{len(eligible_sources)}`",
            f"- unique eligible rows: `{overall['row_count']}`",
            f"- v1 approved: `{overall['v1_approved_count']}`",
            f"- shadow approved: `{overall['v2_shadow_approved_count']}`",
            f"- shadow-only added approvals: `{overall['v2_added_approve_count']}`",
            f"- suspicious added approvals: `{overall['suspicious_added_approval_count']}`",
            f"- new CAO video included: `{fresh_included}`",
            f"- new CAO review URL: `{fresh_preparation['review_url']}`",
            "",
            *source_table,
            "",
            "## Reviewed Sources Not Yet Eligible",
            "",
            *(excluded_lines or ["- none"]),
        ]) + "\n",
        encoding="utf-8",
    )
    audit_table = [
        "| row | source | label | subtype | r9 | visual |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in overall["v2_added_approvals"][:40]:
        audit_table.append(
            f"| `{row['row_key']}` | `{row['source_session_id']}` | `{row['label']}` | `{row.get('legacy_subtype') or 'none'}` | {float(row['r9_score']):.4f} | {float(row['visual_late_fusion_logreg_c0.5']):.4f} |"
        )
    # Keep the added-approval audit in the accumulation report for r23.
    OUT_ACCUM_MD.write_text(
        OUT_ACCUM_MD.read_text(encoding="utf-8") +
        "\n".join([
            "\n# Added Approvals Audit",
            "",
            f"- added approvals: `{overall['v2_added_approve_count']}`",
            f"- added label counts: `{json.dumps(overall['v2_added_label_counts'], sort_keys=True)}`",
            f"- added subtype counts: `{json.dumps(overall['v2_added_subtype_counts'], sort_keys=True)}`",
            f"- suspicious added approvals: `{overall['suspicious_added_approval_count']}`",
            f"- risky source added approvals: `{overall['risky_source_added_approval_count']}`",
            f"- near-boundary added approvals: `{overall['near_boundary_added_approval_count']}`",
            "",
            *audit_table,
        ]) + "\n",
        encoding="utf-8",
    )
    OUT_FRESH_MD.write_text(
        "\n".join([
            "# r23 Fresh Source Preparation",
            "",
            f"- source video: `{fresh_preparation['source_video_path']}`",
            f"- source exists: `{fresh_preparation['source_video_exists']}`",
            f"- prepared session: `{fresh_preparation['prepared_session_id']}`",
            f"- review ready: `{fresh_preparation['review_ready']}`",
            f"- reviewed manifest exists: `{fresh_preparation['reviewed_manifest_exists']}`",
            f"- included in shadow evidence: `{fresh_preparation['included_in_shadow_evidence']}`",
            f"- reason: {fresh_preparation['reason']}",
            f"- review URL: `{fresh_preparation['review_url']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(
        "\n".join([
            "# Approve Review v2 Rollout Evidence Update",
            "",
            f"- shadow evidence decision: `{evidence_decision}`",
            f"- rollout decision: `{rollout_decision}`",
            "",
            "The hardened v2 shadow policy is not ready to replace v1. The expanded bank now exposes suspicious v2-only approvals on a fresh CAO source.",
            "",
            "## Current Evidence",
            "",
            f"- eligible source count: `{len(eligible_sources)}`",
            f"- unique eligible rows: `{overall['row_count']}`",
            f"- shadow-only added approvals: `{overall['v2_added_approve_count']}`",
            f"- suspicious added approvals: `{overall['suspicious_added_approval_count']}`",
            f"- fresh/less-central shadow-only additions: `{fresh_summary['v2_added_approve_count']}`",
            "",
            "## Required Before Limited Rollout",
            "",
            "- Treat CAO-SUN voice_whistle rows as a hard blocker for the current v2 rule.",
            "- Run the next bounded policy hardening pass against source-aware voice_whistle false approvals before any rollout decision.",
            "- Keep suspicious v2-only approvals at zero in any replacement candidate.",
            "- Keep human override rate for v2-only approvals at zero or explicitly accepted.",
            "- Keep approve_review_v1 as default until these checks pass.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": [str(OUT_ELIGIBILITY_JSON), str(OUT_ELIGIBILITY_MD), str(OUT_ACCUM_JSON), str(OUT_ACCUM_MD), str(OUT_DOC), str(OUT_FRESH_JSON), str(OUT_FRESH_MD)], "eligibility_decision": eligibility["eligibility_decision"], "final_decision": evidence_decision, "rollout_decision": rollout_decision}, indent=2))


if __name__ == "__main__":
    main()
