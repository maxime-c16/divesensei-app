from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
V1_THRESHOLD = 0.92158
SESSION_ROOTS = [
    ROOT / "outputs/evaluation_r30_exact_scorepath_insep_quick",
    ROOT / "outputs/evaluation_r30_exact_scorepath_champigny_proxy",
]
OUT_JSON = ROOT / "outputs/r31_runtime_only_approve_reevaluation_after_r30_parity.json"
OUT_MD = ROOT / "outputs/r31_runtime_only_approve_reevaluation_after_r30_parity.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in SESSION_ROOTS:
        manifest = load_json(root / "ui_session_manifest.json")
        review = load_json(root / "evaluation_review.json")
        decisions = {
            str(item.get("detectionId")): item
            for item in review.get("decisions", [])
            if item.get("eventLabel") in {"platform_dive", "noise_or_other"}
        }
        for detection in manifest.get("detections", []):
            decision = decisions.get(str(detection.get("id")))
            if decision is None:
                continue
            scores = dict(detection.get("scores", {}) or {})
            features = dict(detection.get("features", {}) or {})
            rows.append(
                {
                    "session_id": root.name,
                    "detection_id": detection.get("id"),
                    "label": decision.get("eventLabel"),
                    "subtype": decision.get("subtype"),
                    "r9": float(scores.get("governed_r9_score") or 0.0),
                    "visual": features.get("visual_late_fusion_logreg_c0.5"),
                    "timestamp_seconds": detection.get("timestamp_seconds"),
                }
            )
    return rows


def metrics(rows: list[dict[str, Any]], policy_id: str, approve: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    approved = [row for row in rows if approve(row)]
    dangerous = [row for row in approved if row["label"] != "platform_dive"]
    platform = sum(1 for row in approved if row["label"] == "platform_dive")
    return {
        "policy_id": policy_id,
        "approve_count": len(approved),
        "approve_coverage": len(approved) / len(rows) if rows else 0.0,
        "approve_precision": None if not approved else platform / len(approved),
        "dangerous_approvals": len(dangerous),
        "approved_label_counts": dict(sorted(Counter(str(row["label"]) for row in approved).items())),
        "approved_source_counts": dict(sorted(Counter(str(row["session_id"]) for row in approved).items())),
        "dangerous_rows": dangerous,
    }


def build_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("approve_review_v1", lambda row: row["r9"] >= V1_THRESHOLD)
    ]
    for threshold in [0.99, 0.97, 0.95, 0.94, 0.93, 0.92158, 0.92, 0.90, 0.88, 0.86, 0.84, 0.80]:
        policies.append((f"r9_score_gate::{threshold:.5g}", lambda row, threshold=threshold: row["r9"] >= threshold))
    for r9_min in [0.84, 0.86, 0.88, 0.90]:
        for visual_min in [0.55, 0.70, 0.85, 0.95, 0.99]:
            policies.append(
                (
                    f"runtime_or_visual_gate::r9_{r9_min:.2f}::visual_{visual_min:.2f}",
                    lambda row, r9_min=r9_min, visual_min=visual_min: row["r9"] >= V1_THRESHOLD
                    or (
                        row["r9"] >= r9_min
                        and row.get("visual") is not None
                        and float(row["visual"]) >= visual_min
                    ),
                )
            )
    results = [metrics(rows, policy_id, approve) for policy_id, approve in policies]
    v1 = results[0]
    for result in results:
        result["coverage_delta_vs_v1"] = result["approve_coverage"] - v1["approve_coverage"]
        result["approve_count_delta_vs_v1"] = result["approve_count"] - v1["approve_count"]
        result["safe"] = result["dangerous_approvals"] == 0
        result["safe_improver"] = result["safe"] and result["approve_count"] > v1["approve_count"]
    return results


def write_markdown(report: dict[str, Any]) -> None:
    rows = sorted(
        report["candidate_comparison"],
        key=lambda row: (row["dangerous_approvals"], -row["approve_count"]),
    )[:15]
    lines = [
        "# R31 Runtime-Only Approve Reevaluation After R30 Parity",
        "",
        "R31 reruns the runtime-only approve benchmark after exact governed runtime/offline parity and governed 3-second runtime window scoring.",
        "",
        f"- rows: `{report['row_count']}`",
        f"- labels: `{json.dumps(report['label_counts'], sort_keys=True)}`",
        f"- visual present: `{report['visual_present_count']}/{report['row_count']}`",
        "",
        "## Baseline v1",
        "",
        f"- approve count: `{report['v1']['approve_count']}`",
        f"- approve coverage: `{report['v1']['approve_coverage']:.4f}`",
        f"- approve precision: `{report['v1']['approve_precision']}`",
        f"- dangerous approvals: `{report['v1']['dangerous_approvals']}`",
        f"- dangerous rows: `{json.dumps(report['v1']['dangerous_rows'], sort_keys=True)}`",
        "",
        "## Candidate Comparison",
        "",
        "| policy | approvals | coverage | precision | dangerous | delta vs v1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        precision = "n/a" if row["approve_precision"] is None else f"{row['approve_precision']:.4f}"
        lines.append(
            f"| `{row['policy_id']}` | {row['approve_count']} | {row['approve_coverage']:.4f} | {precision} | {row['dangerous_approvals']} | {row['coverage_delta_vs_v1']:.4f} |"
        )
    best = report["best_candidate"]
    lines += [
        "",
        "## Best Candidate",
        "",
        f"- policy: `{best['policy_id']}`",
        f"- approve count: `{best['approve_count']}`",
        f"- approve coverage: `{best['approve_coverage']:.4f}`",
        f"- approve precision: `{best['approve_precision']}`",
        f"- dangerous approvals: `{best['dangerous_approvals']}`",
        "",
        "## Decisions",
        "",
        *[f"- `{decision}`" for decision in report["final_decisions"]],
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    results = build_results(rows)
    v1 = results[0]
    safe_improvers = [row for row in results if row["safe_improver"]]
    safe_nonzero = [row for row in results if row["safe"] and row["approve_count"] > 0]
    best = max(safe_improvers, key=lambda row: (row["approve_count"], row["approve_precision"] or 0.0), default=None)
    if best is None:
        best = max(safe_nonzero, key=lambda row: (row["approve_count"], row["approve_precision"] or 0.0), default=v1)
    report = {
        "experiment_name": "r31_runtime_only_approve_reevaluation_after_r30_parity",
        "sessions": [root.name for root in SESSION_ROOTS],
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "visual_present_count": sum(1 for row in rows if row.get("visual") is not None),
        "v1": v1,
        "best_candidate": best,
        "safe_improver_count": len(safe_improvers),
        "candidate_comparison": results,
        "final_decisions": [
            "R31_RUNTIME_ONLY_APPROVE_REEVALUATION_GAIN"
            if safe_improvers
            else "R31_RUNTIME_ONLY_APPROVE_REEVALUATION_NO_CLEAR_GAIN",
            "APPROVE_REVIEW_V1_REMAINS_DEFAULT",
        ],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print(json.dumps({
        "row_count": report["row_count"],
        "v1": {key: v1[key] for key in ["approve_count", "approve_coverage", "approve_precision", "dangerous_approvals"]},
        "best": {key: best[key] for key in ["policy_id", "approve_count", "approve_coverage", "approve_precision", "dangerous_approvals", "safe_improver"]},
        "safe_improver_count": len(safe_improvers),
        "final_decisions": report["final_decisions"],
    }, indent=2))


if __name__ == "__main__":
    main()
