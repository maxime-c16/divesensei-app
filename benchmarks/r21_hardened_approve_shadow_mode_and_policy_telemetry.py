from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/r20_nuisance_hardening_bank.json"
BEST = ROOT / "outputs/r20_best_hardened_policy.json"

OUT_COMPARISON_JSON = ROOT / "outputs/r21_shadow_mode_policy_comparison.json"
OUT_COMPARISON_MD = ROOT / "outputs/r21_shadow_mode_policy_comparison.md"
OUT_TELEMETRY_JSON = ROOT / "outputs/r21_shadow_mode_telemetry.json"
OUT_TELEMETRY_MD = ROOT / "outputs/r21_shadow_mode_telemetry.md"
OUT_DOC = ROOT / "docs/research/APPROVE_REVIEW_V2_SHADOW_MODE.md"

V1_R9_MIN = 0.92158
V2_R9_FLOOR = 0.84
V2_VISUAL_MIN = 0.55


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def flag_v1(row: dict[str, Any]) -> bool:
    return float(row.get("r9_score") or 0.0) >= V1_R9_MIN


def flag_v2_shadow(row: dict[str, Any]) -> bool:
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
        "bank_split": row.get("bank_split"),
        "source_session_id": row.get("source_session_id"),
        "label": row.get("label"),
        "legacy_subtype": row.get("legacy_subtype"),
        "r9_score": row.get("r9_score"),
        "visual_late_fusion_logreg_c0.5": row.get("visual_score"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    v1_rows = [row for row in rows if flag_v1(row)]
    v2_rows = [row for row in rows if flag_v2_shadow(row)]
    added = [row for row in rows if flag_v2_shadow(row) and not flag_v1(row)]
    disagreements = [row for row in rows if flag_v1(row) != flag_v2_shadow(row)]
    suspicious = [
        row for row in added
        if row.get("label") != "platform_dive"
        or str(row.get("legacy_subtype") or "none") in {"handling_noise", "voice_whistle", "non_dive_splash", "unknown_transient"}
    ]
    return {
        "row_count": len(rows),
        "v1_approved_count": len(v1_rows),
        "v2_shadow_approved_count": len(v2_rows),
        "v2_added_approve_count": len(added),
        "policy_disagreement_count": len(disagreements),
        "v1_coverage": len(v1_rows) / len(rows) if rows else 0.0,
        "v2_shadow_coverage": len(v2_rows) / len(rows) if rows else 0.0,
        "coverage_delta": (len(v2_rows) - len(v1_rows)) / len(rows) if rows else 0.0,
        "v2_added_label_counts": dict(sorted(Counter(str(row.get("label")) for row in added).items())),
        "v2_added_subtype_counts": dict(sorted(Counter(str(row.get("legacy_subtype") or "none") for row in added).items())),
        "v2_added_source_session_counts": dict(sorted(Counter(str(row.get("source_session_id")) for row in added).items())),
        "suspicious_added_approve_count": len(suspicious),
        "suspicious_added_approvals": [compact(row) for row in suspicious],
        "v1_approved_rows": [compact(row) for row in v1_rows],
        "v2_added_approved_rows": [compact(row) for row in added],
        "policy_disagreement_rows": [compact(row) for row in disagreements],
    }


def main() -> None:
    bank = load_json(BANK)
    best = load_json(BEST)["best_candidate"]
    rows = list(bank["rows"])
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row.get("bank_split") or "unknown"), []).append(row)
    split_summaries = {split: summarize(items) for split, items in sorted(by_split.items())}
    overall = summarize(rows)
    suspicious_total = overall["suspicious_added_approve_count"]
    ready = (
        best["source_aware_dangerous_approves"] == 0
        and best["fixed_external"]["dangerous_approves"] == 0
        and best["fixed_internal"]["dangerous_approves"] == 0
        and suspicious_total == 0
    )
    final_decision = "R21_HARDENED_V2_SHADOW_MODE_READY" if ready else "R21_HARDENED_V2_SHADOW_MODE_NOT_READY"
    payload = {
        "experiment_name": "r21_hardened_approve_shadow_mode_and_policy_telemetry",
        "final_decision": final_decision,
        "default_policy": "approve_review_v1",
        "shadow_policy": {
            "policy_id": "approve_review_v2_shadow",
            "source_candidate": "hardened_floor_0.84_visual_0.55",
            "logic": {
                "approve_if_any": [
                    {"r9_score_gte": V1_R9_MIN},
                    {"r9_score_gte": V2_R9_FLOOR, "visual_late_fusion_logreg_c0.5_gte": V2_VISUAL_MIN},
                ],
                "missing_visual_score_behavior": "fall_back_to_approve_review_v1; no added shadow approval",
            },
            "status": "shadow_mode_only",
        },
        "score_path": {
            "r9_score": {
                "generated_by": "r9_compact_nuisance_generalization_weighted platform/noise scorer",
                "stored_in_app_manifest": "scores.audio_model_probability",
                "surface": "approve_review_v1 policy card and row policy score",
            },
            "visual_late_fusion_logreg_c0.5": {
                "generated_by": "offline r15/r20 visual late-fusion probe over clip embeddings plus morphology features",
                "required_for": "approve_review_v2_shadow expansion branch",
                "stored_in_shadow_outputs": "visual_score in r20/r21 telemetry rows",
                "stored_in_app_manifest": "features.visual_late_fusion_logreg_c0.5 when upstream scoring provides it",
                "missing_behavior": "do not add v2 shadow approval; fall back to v1",
            },
        },
        "overall": overall,
        "split_summaries": split_summaries,
        "promotion_replacement_criteria": [
            "visual_late_fusion_logreg_c0.5 generated serially in reviewed manifests for all eligible rows",
            "continued shadow telemetry shows zero suspicious added approvals",
            "source-aware dangerous approvals remain zero on new independent reviewed sources",
            "external approve precision remains near 1.0 and coverage remains above approve_review_v1",
            "human override rate for v2-added approvals remains zero or explicitly accepted",
        ],
    }
    OUT_COMPARISON_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_TELEMETRY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = [
        "| split | rows | v1 approvals | shadow approvals | added | delta | suspicious added |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, summary in split_summaries.items():
        table.append(
            f"| `{split}` | {summary['row_count']} | {summary['v1_approved_count']} | {summary['v2_shadow_approved_count']} | {summary['v2_added_approve_count']} | {summary['coverage_delta']:+.4f} | {summary['suspicious_added_approve_count']} |"
        )
    md = "\n".join([
        "# r21 Shadow Mode Policy Comparison",
        "",
        f"- final decision: `{final_decision}`",
        "- active default: `approve_review_v1`",
        "- shadow policy: `approve_review_v2_shadow`",
        f"- overall v1 approvals: `{overall['v1_approved_count']}`",
        f"- overall shadow approvals: `{overall['v2_shadow_approved_count']}`",
        f"- additional shadow approvals: `{overall['v2_added_approve_count']}`",
        f"- suspicious added approvals: `{overall['suspicious_added_approve_count']}`",
        "",
        *table,
    ]) + "\n"
    OUT_COMPARISON_MD.write_text(md, encoding="utf-8")
    OUT_TELEMETRY_MD.write_text(md.replace("# r21 Shadow Mode Policy Comparison", "# r21 Shadow Mode Telemetry"), encoding="utf-8")
    OUT_DOC.write_text(
        "\n".join([
            "# Approve Review v2 Shadow Mode",
            "",
            "`approve_review_v1` remains the active default.",
            "",
            "`approve_review_v2_shadow` is available for metadata-only evaluation of the r20 hardened approve expansion candidate.",
            "",
            "## Shadow Policy",
            "",
            "- approve if `r9_score >= 0.92158`",
            "- or approve if `r9_score >= 0.84` and `visual_late_fusion_logreg_c0.5 >= 0.55`",
            "- otherwise `needs_review`",
            "- if the visual score is missing, fall back to v1 and do not add a shadow approval",
            "",
            "## Operational Status",
            "",
            f"- decision: `{final_decision}`",
            "- rollout: shadow mode only",
            "- default replacement: not allowed yet",
            "",
            "## Replacement Criteria",
            "",
            *[f"- {item}" for item in payload["promotion_replacement_criteria"]],
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": [str(OUT_COMPARISON_JSON), str(OUT_COMPARISON_MD), str(OUT_TELEMETRY_JSON), str(OUT_TELEMETRY_MD), str(OUT_DOC)], "final_decision": final_decision}, indent=2))


if __name__ == "__main__":
    main()
