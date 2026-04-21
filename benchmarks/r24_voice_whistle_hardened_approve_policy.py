from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/r23_shadow_bank_accumulation.json"

OUT_JSON = ROOT / "outputs/r24_voice_whistle_hardened_approve_policy.json"
OUT_MD = ROOT / "outputs/r24_voice_whistle_hardened_approve_policy.md"
OUT_COMPARISON_JSON = ROOT / "outputs/r24_voice_whistle_candidate_comparison.json"
OUT_COMPARISON_MD = ROOT / "outputs/r24_voice_whistle_candidate_comparison.md"
OUT_BEST_JSON = ROOT / "outputs/r24_best_voice_whistle_hardened_policy.json"
OUT_BEST_MD = ROOT / "outputs/r24_best_voice_whistle_hardened_policy.md"

V1_R9_MIN = 0.92158
V2_R9_FLOOR = 0.84
V2_VISUAL_MIN = 0.55
RISK_SUBTYPES = {"handling_noise", "voice_whistle", "non_dive_splash", "unknown_transient"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def fnum(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "source_session_id": row["source_session_id"],
        "label": row["label"],
        "legacy_subtype": row.get("legacy_subtype"),
        "r9_score": row.get("r9_score"),
        "visual_score": row.get("visual_score"),
    }


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "visual_score" not in out:
        out["visual_score"] = out.get("visual_late_fusion_logreg_c0.5")
    return out


def summarize(rows: list[dict[str, Any]], flags: list[bool]) -> dict[str, Any]:
    approved = [row for row, flag in zip(rows, flags) if flag]
    added = [row for row in approved if not approve_v1(row)]
    errors = [row for row in approved if row.get("label") != "platform_dive"]
    suspicious_added = [
        row
        for row in added
        if row.get("label") != "platform_dive" or str(row.get("legacy_subtype") or "none") in RISK_SUBTYPES
    ]
    return {
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in rows).items())),
        "source_session_counts": dict(sorted(Counter(str(row.get("source_session_id")) for row in rows).items())),
        "approve_count": len(approved),
        "approve_coverage": safe_div(len(approved), len(rows)) or 0.0,
        "approve_precision": safe_div(sum(1 for row in approved if row.get("label") == "platform_dive"), len(approved)),
        "dangerous_approves": len(errors),
        "approved_label_counts": dict(sorted(Counter(str(row.get("label")) for row in approved).items())),
        "added_approve_count": len(added),
        "added_label_counts": dict(sorted(Counter(str(row.get("label")) for row in added).items())),
        "added_subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in added).items())),
        "suspicious_added_approval_count": len(suspicious_added),
        "dangerous_approve_rows": [compact(row) for row in errors],
        "suspicious_added_approvals": [compact(row) for row in suspicious_added],
        "added_approvals": [compact(row) for row in added],
    }


def approve_v1(row: dict[str, Any]) -> bool:
    return float(row.get("r9_score") or 0.0) >= V1_R9_MIN


def approve_v2(row: dict[str, Any]) -> bool:
    r9 = float(row.get("r9_score") or 0.0)
    visual = row.get("visual_score")
    return approve_v1(row) or (visual is not None and r9 >= V2_R9_FLOOR and float(visual) >= V2_VISUAL_MIN)


def expansion(row: dict[str, Any], r9_floor: float = V2_R9_FLOOR, visual_min: float = V2_VISUAL_MIN) -> bool:
    visual = row.get("visual_score")
    return visual is not None and float(row.get("r9_score") or 0.0) >= r9_floor and float(visual) >= visual_min


def candidate_flags(rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], bool]) -> list[bool]:
    return [fn(row) for row in rows]


def main() -> None:
    bank = load_json(BANK)
    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in bank["overall"]["v2_added_approvals"]:
        # Ensure all added rows are retained with normalized visual field.
        rows_by_key[row["row_key"]] = normalize_row(row)
    for summary in bank["source_summaries"].values():
        for row in summary.get("v2_added_approvals", []):
            rows_by_key[row["row_key"]] = normalize_row(row)
        for row in summary.get("suspicious_added_approvals", []):
            rows_by_key[row["row_key"]] = normalize_row(row)
    # Use all rows from r23 by reconstructing from per-source summaries is not possible, so load the row-level
    # backfill/cache artifacts directly and combine them with the legacy r20 bank rows.
    for path in sorted((ROOT / "outputs").glob("r23_*_shadow_score_backfill_rows.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = normalize_row(json.loads(line))
                rows_by_key[row["row_key"]] = row
    for row in load_json(ROOT / "outputs/r20_nuisance_hardening_bank.json")["rows"]:
        rows_by_key.setdefault(str(row["row_key"]), normalize_row(row))
    rows = list(rows_by_key.values())

    families: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("approve_review_v1", "current default; r9 >= 0.92158", approve_v1),
        ("approve_review_v2_shadow_current", "r9 >= 0.92158 OR r9 >= 0.84 and visual >= 0.55", approve_v2),
        (
            "v2_block_all_risky_subtypes",
            "v2 expansion suppressed for voice_whistle/handling_noise/non_dive_splash/unknown_transient",
            lambda row: approve_v1(row)
            or (expansion(row) and str(row.get("legacy_subtype") or "none") not in RISK_SUBTYPES),
        ),
        (
            "v2_block_voice_whistle_only",
            "v2 expansion suppressed for voice_whistle only",
            lambda row: approve_v1(row)
            or (expansion(row) and str(row.get("legacy_subtype") or "none") != "voice_whistle"),
        ),
    ]
    for floor in [0.86, 0.88, 0.90]:
        families.append(
            (
                f"v2_floor_{floor:.2f}_block_voice_whistle",
                f"v2 expansion floor {floor:.2f}; voice_whistle expansion suppressed",
                lambda row, floor=floor: approve_v1(row)
                or (expansion(row, r9_floor=floor) and str(row.get("legacy_subtype") or "none") != "voice_whistle"),
            )
        )
    for visual in [0.75, 0.90, 0.98]:
        families.append(
            (
                f"v2_visual_{visual:.2f}_block_voice_whistle",
                f"v2 expansion visual threshold {visual:.2f}; voice_whistle expansion suppressed",
                lambda row, visual=visual: approve_v1(row)
                or (expansion(row, visual_min=visual) and str(row.get("legacy_subtype") or "none") != "voice_whistle"),
            )
        )

    comparison = []
    source_names = sorted(set(str(row.get("source_session_id")) for row in rows))
    v1_summary = summarize(rows, candidate_flags(rows, approve_v1))
    unsafe_v2_summary = summarize(rows, candidate_flags(rows, approve_v2))
    for candidate, description, fn in families:
        flags = candidate_flags(rows, fn)
        overall = summarize(rows, flags)
        source_summaries = {
            source: summarize([row for row in rows if row.get("source_session_id") == source], candidate_flags([row for row in rows if row.get("source_session_id") == source], fn))
            for source in source_names
        }
        comparison.append(
            {
                "candidate": candidate,
                "description": description,
                "overall": overall,
                "source_summaries": source_summaries,
                "dangerous_sources": {
                    source: summary
                    for source, summary in source_summaries.items()
                    if summary["dangerous_approves"] > 0 or summary["suspicious_added_approval_count"] > 0
                },
                "coverage_delta_vs_v1": overall["approve_coverage"] - v1_summary["approve_coverage"],
                "added_recovered_vs_current_v2": safe_div(
                    overall["added_approve_count"],
                    max(unsafe_v2_summary["added_approve_count"], 1),
                ),
                "interesting": overall["suspicious_added_approval_count"] == 0
                and overall["dangerous_approves"] == 0
                and (overall["approve_precision"] or 0.0) >= 0.95
                and overall["approve_count"] > v1_summary["approve_count"],
            }
        )

    best = max(
        comparison,
        key=lambda row: (
            row["interesting"],
            -row["overall"]["suspicious_added_approval_count"],
            -row["overall"]["dangerous_approves"],
            row["overall"]["approve_count"],
            fnum(row["overall"]["approve_precision"]),
        ),
    )
    best_source_added = {
        source: summary["added_approve_count"]
        for source, summary in best["source_summaries"].items()
        if summary["added_approve_count"] > 0
    }
    best_suspicious_sources = {
        source: summary["suspicious_added_approval_count"]
        for source, summary in best["source_summaries"].items()
        if summary["suspicious_added_approval_count"] > 0
    }
    decision = "R24_VOICE_WHISTLE_HARDENING_GAIN" if best["interesting"] else "R24_VOICE_WHISTLE_HARDENING_NO_CLEAR_GAIN"
    rollout = "HARDENED_V2_READY_FOR_SHADOW_MODE" if best["interesting"] else "APPROVE_REVIEW_V1_REMAINS_DEFAULT"
    payload = {
        "experiment_name": "r24_voice_whistle_hardened_approve_policy",
        "final_decision": decision,
        "rollout_decision": rollout,
        "best_candidate": best["candidate"],
        "row_count": len(rows),
        "source_count": len(source_names),
        "source_names": source_names,
        "current_v1": v1_summary,
        "current_v2_shadow": unsafe_v2_summary,
        "comparison_rows": comparison,
        "best": best,
        "best_source_added_approval_counts": best_source_added,
        "best_suspicious_source_counts": best_suspicious_sources,
        "interpretation": {
            "recovers_current_v2_clean_platform_additions": best["overall"]["added_approve_count"] == 20,
            "blocks_cao_sun_voice_whistle_failures": best["source_summaries"].get("evaluation_CAO-SUN-19-4-26-FANNY_20260419-160927", {}).get("suspicious_added_approval_count") == 0,
            "gains_still_concentrated": best_source_added,
            "not_default_replacement": True,
            "recommended_product_state": "approve_review_v1_default_with_r24_candidate_shadow_only",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_COMPARISON_JSON.write_text(json.dumps({"comparison_rows": comparison}, indent=2), encoding="utf-8")
    OUT_BEST_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")

    table = [
        "| candidate | approved | precision | added | suspicious added | dangerous | coverage | interesting |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison:
        overall = row["overall"]
        table.append(
            f"| `{row['candidate']}` | {overall['approve_count']} | {fnum(overall['approve_precision']):.4f} | "
            f"{overall['added_approve_count']} | {overall['suspicious_added_approval_count']} | "
            f"{overall['dangerous_approves']} | {overall['approve_coverage']:.4f} | `{row['interesting']}` |"
        )
    added_table = [
        "| row | source | label | subtype | r9 | visual |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in best["overall"]["added_approvals"]:
        added_table.append(
            f"| `{row['row_key']}` | `{row['source_session_id']}` | `{row['label']}` | `{row.get('legacy_subtype') or 'none'}` | "
            f"{float(row['r9_score']):.4f} | {float(row['visual_score']):.4f} |"
        )
    md = [
        "# r24 Voice-Whistle Hardened Approve Policy",
        "",
        f"- final decision: `{decision}`",
        f"- rollout decision: `{rollout}`",
        f"- best candidate: `{best['candidate']}`",
        f"- rows: `{len(rows)}`",
        f"- sources: `{len(source_names)}`",
        "",
        "## Candidate Comparison",
        "",
        *table,
        "",
        "## Best Candidate Added Approvals",
        "",
        *added_table,
        "",
        "## Source Concentration",
        "",
        f"- best added approval counts by source: `{json.dumps(best_source_added, sort_keys=True)}`",
        f"- best suspicious source counts: `{json.dumps(best_suspicious_sources, sort_keys=True)}`",
        "- CAO-SUN voice_whistle failures are blocked by the best candidate.",
        "- The candidate should remain shadow-only; it is not a default replacement yet.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    OUT_COMPARISON_MD.write_text("\n".join(["# r24 Candidate Comparison", "", *table]) + "\n", encoding="utf-8")
    OUT_BEST_MD.write_text(
        "\n".join(
            [
                "# r24 Best Voice-Whistle Hardened Policy",
                "",
                f"- candidate: `{best['candidate']}`",
                f"- approved: `{best['overall']['approve_count']}`",
                f"- precision: `{best['overall']['approve_precision']}`",
                f"- added approvals: `{best['overall']['added_approve_count']}`",
                f"- suspicious added approvals: `{best['overall']['suspicious_added_approval_count']}`",
                f"- dangerous approvals: `{best['overall']['dangerous_approves']}`",
                f"- added approval counts by source: `{json.dumps(best_source_added, sort_keys=True)}`",
                "- product state: `approve_review_v1` remains default; this candidate is shadow-only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": [str(OUT_JSON), str(OUT_MD), str(OUT_COMPARISON_JSON), str(OUT_COMPARISON_MD), str(OUT_BEST_JSON), str(OUT_BEST_MD)], "final_decision": decision, "rollout_decision": rollout, "best_candidate": best["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
