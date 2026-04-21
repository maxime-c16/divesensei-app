import type { ApproveReviewPolicy, ApproveReviewSummary, Detection } from "@/types/ui";

export const APPROVE_REVIEW_V1_POLICY: ApproveReviewPolicy = {
  policy_id: "approve_review_v1",
  model_ref: "r9_compact_nuisance_generalization_weighted",
  mode: "approve_review",
  approve_min_score: 0.92158,
  score_field: "scores.governed_r9_score (fallback: scores.audio_model_probability)",
  active_default: true,
  shadow_only: false,
  rollout_assumption: "Narrow high-precision approve lane only; every non-approved row remains review-required.",
  source_experiment: "r17_high_precision_approve_coverage_benchmark",
  governed_external_approve_precision: 1.0,
  governed_external_approve_coverage: 0.1717171717171717,
  governed_dangerous_external_auto_approves: 0,
  governed_dangerous_internal_auto_approves: 0,
};

export const APPROVE_REVIEW_V2_SHADOW_POLICY: ApproveReviewPolicy = {
  policy_id: "approve_review_v2_shadow",
  model_ref: "r9_compact_nuisance_generalization_weighted",
  mode: "approve_review",
  approve_min_score: 0.92158,
  score_field: "scores.governed_r9_score (fallback: scores.audio_model_probability)",
  active_default: false,
  shadow_only: true,
  visual_score_field: "features.visual_late_fusion_logreg_c0.5",
  expansion_rule: {
    r9_score_min: 0.84,
    visual_score_min: 0.55,
    suppressed_subtypes: ["handling_noise", "voice_whistle", "non_dive_splash", "unknown_transient"],
  },
  missing_shadow_score_behavior: "fall_back_to_approve_review_v1; never add shadow approvals when the visual score is absent",
  rollout_assumption: "r24 source-aware shadow candidate; v1 remains default. Expansion is suppressed for nuisance-risk subtypes after CAO-SUN exposed voice_whistle leakage.",
  source_experiment: "r24_voice_whistle_hardened_approve_policy",
  shadow_replaces_policy_experiment: "r20_source_aware_nuisance_hardening_for_approve_expansion",
  governed_external_approve_precision: 1.0,
  governed_external_approve_coverage: 0.15404699738903394,
  governed_dangerous_external_auto_approves: 0,
  governed_dangerous_internal_auto_approves: 0,
  governed_source_aware_dangerous_auto_approves: 0,
  governed_source_count: 7,
  governed_row_count: 383,
  governed_shadow_added_approvals: 20,
  governed_suspicious_added_approvals: 0,
};

export function approveReviewScoreForDetection(detection: Detection): number | null {
  const governed = Number(detection.scores?.governed_r9_score);
  if (Number.isFinite(governed)) return governed;
  const fallback = Number(detection.scores?.audio_model_probability);
  return Number.isFinite(fallback) ? fallback : null;
}

export function visualLateFusionScoreForDetection(detection: Detection): number | null {
  const score = Number(detection.features?.["visual_late_fusion_logreg_c0.5"]);
  return Number.isFinite(score) ? score : null;
}

export function approveReviewSubtypeForDetection(detection: Detection): string | null {
  const subtype = String(detection.subtype ?? "").trim();
  return subtype.length > 0 ? subtype : null;
}

export function approveReviewLaneForDetection(detection: Detection, policy: ApproveReviewPolicy = APPROVE_REVIEW_V1_POLICY): "auto_approved" | "needs_review" {
  const score = approveReviewScoreForDetection(detection);
  if (score !== null && score >= policy.approve_min_score) return "auto_approved";
  if (policy.policy_id === "approve_review_v2_shadow" && policy.expansion_rule) {
    const visualScore = visualLateFusionScoreForDetection(detection);
    const suppressedSubtypes = policy.expansion_rule.suppressed_subtypes ?? [];
    const subtype = approveReviewSubtypeForDetection(detection);
    if (subtype && suppressedSubtypes.includes(subtype as typeof suppressedSubtypes[number])) return "needs_review";
    if (
      score !== null &&
      visualScore !== null &&
      score >= policy.expansion_rule.r9_score_min &&
      visualScore >= policy.expansion_rule.visual_score_min
    ) {
      return "auto_approved";
    }
  }
  return "needs_review";
}

export function summarizeApproveReviewDetections(
  detections: Detection[],
  policy: ApproveReviewPolicy = APPROVE_REVIEW_V1_POLICY,
): ApproveReviewSummary {
  const summary: ApproveReviewSummary = {
    policy_id: policy.policy_id,
    model_ref: policy.model_ref,
    approve_min_score: policy.approve_min_score,
    total_count: detections.length,
    scored_count: 0,
    auto_approved_count: 0,
    needs_review_count: 0,
  };
  for (const detection of detections) {
    const score = approveReviewScoreForDetection(detection);
    if (score !== null) summary.scored_count += 1;
    if (approveReviewLaneForDetection(detection, policy) === "auto_approved") {
      summary.auto_approved_count += 1;
    } else {
      summary.needs_review_count += 1;
    }
  }
  return summary;
}
