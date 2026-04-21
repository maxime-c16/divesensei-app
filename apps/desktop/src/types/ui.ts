export type ConfidenceLevel = "high" | "medium" | "low";
export type SessionMode = "standard" | "evaluation";
export type ReviewDecisionLabel = "keep" | "reject" | "unsure" | "dive" | "non_dive";
export type EvaluationReviewSubtype =
  | "board_rebound"
  | "board_slap"
  | "non_dive_splash"
  | "voice_whistle"
  | "handling_noise"
  | "unknown_transient";

export interface SessionSummary {
  id: string;
  title: string;
  session_name?: string;
  mode?: SessionMode;
  profile: string;
  detector_id?: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  session_duration_seconds?: number;
  source_availability?: "available" | "missing" | "relink-required";
  candidate_count: number;
  extracted_count: number;
  source_video_path: string;
  output_dir: string;
  manifest_path?: string;
  timestamp_range: {
    first: number;
    last: number;
  };
  telemetry: {
    detector_seconds: number;
    extract_seconds: number;
    total_runtime_seconds: number;
    peak_rss_kb: number;
  };
}

export interface Detection {
  id: string;
  index: number;
  timestamp_seconds: number;
  start_time_seconds: number;
  end_time_seconds: number;
  duration_seconds: number;
  review_start_seconds?: number;
  review_end_seconds?: number;
  review_duration_seconds?: number;
  confidence: ConfidenceLevel;
  scores: {
    audio: number;
    video: number;
    combined: number;
    governed_r9_score?: number | null;
    audio_model_probability: number;
    audio_clip_probability: number;
  };
  features: Record<string, number | null>;
  clip: {
    path: string | null;
    browser_path?: string | null;
    filename: string | null;
  };
  reviewLabel?: ReviewDecisionLabel | null;
  eventLabel?: "springboard_dive" | "springboard_rebound_only" | "platform_dive" | "noise_or_other" | "uncertain" | null;
  subtype?: EvaluationReviewSubtype | null;
}

export interface SessionManifest {
  schema_version: string;
  kind: string;
  generated_at: string;
  session: SessionSummary;
  artifacts: Record<string, string | null>;
  detections: Detection[];
}

export interface LibraryIndex {
  schema_version: string;
  kind: string;
  generated_at: string;
  session_count: number;
  sessions: SessionSummary[];
}

export interface DebugLogEntry {
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "DEBUG";
  message: string;
  stage: string;
}

export interface ApproveReviewPolicy {
  policy_id: "approve_review_v1" | "approve_review_v2_shadow";
  model_ref: "r9_compact_nuisance_generalization_weighted";
  mode: "approve_review";
  approve_min_score: number;
  score_field: string;
  active_default?: boolean;
  shadow_only?: boolean;
  visual_score_field?: string;
  expansion_rule?: {
    r9_score_min: number;
    visual_score_min: number;
    suppressed_subtypes?: EvaluationReviewSubtype[];
  };
  missing_shadow_score_behavior?: string;
  rollout_assumption: string;
  source_experiment: string;
  governed_external_approve_precision: number;
  governed_external_approve_coverage: number;
  governed_dangerous_external_auto_approves: number;
  governed_dangerous_internal_auto_approves: number;
  governed_source_aware_dangerous_auto_approves?: number;
  governed_source_count?: number;
  governed_row_count?: number;
  governed_shadow_added_approvals?: number;
  governed_suspicious_added_approvals?: number;
  shadow_replaces_policy_experiment?: string;
}

export interface ApproveReviewSummary {
  policy_id: string;
  model_ref: string;
  approve_min_score: number;
  total_count: number;
  scored_count: number;
  auto_approved_count: number;
  needs_review_count: number;
}

export interface UiDataBundle {
  library: LibraryIndex;
  manifest: SessionManifest;
  selectedSessionId: string;
  logs: DebugLogEntry[];
  artifactsPreview: Record<string, string>;
  eventReviewSupport: Array<Record<string, unknown>>;
  eventReviewSupportSummary: Record<string, unknown> | null;
  eventReviewQueueRows: Array<Record<string, unknown>>;
  eventReviewQueueTitle?: string;
  eventReviewQueueNote?: string;
  approveReviewPolicy: ApproveReviewPolicy;
  approveReviewSummary: ApproveReviewSummary;
  approveReviewShadowPolicies: ApproveReviewPolicy[];
  approveReviewShadowSummaries: ApproveReviewSummary[];
}

export interface ReviewDecision {
  id: string;
  analysisRunId: string;
  detectionId: string;
  label: ReviewDecisionLabel;
  eventLabel?: "springboard_dive" | "springboard_rebound_only" | "platform_dive" | "noise_or_other" | "uncertain" | null;
  subtype?: EvaluationReviewSubtype | null;
  manualAnchorTimestampSeconds?: number | null;
  manualWindowStartSeconds?: number | null;
  manualWindowEndSeconds?: number | null;
  manualCorrectionType?: "retime_earlier" | "retime_later" | "relabel" | "keep" | "other" | null;
  manualCorrectionRationale?: string | null;
  notes: string;
  createdAt: string;
  updatedAt: string;
}

export interface EvaluationFalseNegativeAnnotation {
  id: string;
  analysisRunId: string;
  timestampSeconds: number;
  reviewStartSeconds: number;
  reviewEndSeconds: number;
  label: "false_negative";
  eventLabel?: "springboard_dive" | "springboard_rebound_only" | "platform_dive" | "noise_or_other" | "uncertain" | null;
  subtype?: EvaluationReviewSubtype | null;
  notes: string;
  createdAt: string;
  updatedAt: string;
}
