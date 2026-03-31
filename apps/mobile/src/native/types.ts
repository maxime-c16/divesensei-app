import type { Plugin } from "@capacitor/core";

export type ISODateString = string;
export type URLString = string;

export type SourceRef = string; // Opaque native identifier, e.g. "src_01..."
export type SessionId = string; // Opaque native identifier
export type JobId = string; // Opaque native identifier
export type ExportId = string; // Opaque native identifier
export type DetectionId = string; // Stable manifest detection identifier
export type LibraryAssetId = string; // Native library identifier when relevant

export type SourceOrigin = "photos" | "files";
export type ReviewDecisionLabel = "keep" | "reject" | "unsure";
export type JobKind = "analysis" | "export";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type DetectorId =
  | "audio_v1_heuristic"
  | "audio_v2_pcen_classifier"
  | "audio_v2_hybrid_video";

export type SessionProfile = "long-session" | "reviewed";

export type SourceAvailability =
  | "available"
  | "needs_download"
  | "missing"
  | "permission_denied"
  | "unsupported";

export type SessionStatus =
  | "created"
  | "analyzing"
  | "review_pending"
  | "review_ready"
  | "exporting"
  | "complete_with_errors"
  | "failed"
  | "deleted";

export type JobPhase =
  | "source_resolving"
  | "source_downloading"
  | "audio_decode"
  | "detecting"
  | "manifest_writing"
  | "proxy_generating"
  | "review_ready"
  | "export_preparing"
  | "exporting"
  | "saving_to_library"
  | "completed"
  | "failed"
  | "cancelled";

export type ExportStatus =
  | "pending"
  | "running"
  | "saved_local"
  | "saved_to_library"
  | "failed";

export type ProxyStatus = "pending" | "ready" | "failed";
export type ConfidenceLevel = "high" | "medium" | "low";

export type PlayerBackend = "html_video" | "native_avplayer";

export interface SourceSummary {
  sourceRef: SourceRef;
  origin: SourceOrigin;
  displayName: string;
  availability: SourceAvailability;
  durationSeconds?: number;
  fileSizeBytes?: number;
  canPersist: boolean;
}

export interface PickSourceVideoRequest {}

export interface PickSourceVideoResponse {
  cancelled?: boolean;
  source?: SourceSummary;
}

export interface GetSourceAvailabilityRequest {
  sourceRef: SourceRef;
}

export interface GetSourceAvailabilityResponse {
  sourceRef: SourceRef;
  availability: SourceAvailability;
  origin?: SourceOrigin;
  displayName?: string;
  durationSeconds?: number;
  fileSizeBytes?: number;
  lastResolvedAt?: ISODateString;
}

export interface RepairSourceRequest {
  sessionId: SessionId;
}

export interface RepairSourceResponse {
  repaired: boolean;
  source?: SourceSummary;
  availability: SourceAvailability;
}

export interface CreateSessionRequest {
  sourceRef: SourceRef;
  sessionName?: string;
  profile: SessionProfile;
  detectorId: DetectorId;
}

export interface CreateSessionResponse {
  sessionId: SessionId;
  status: SessionStatus;
  createdAt: ISODateString;
}

export interface StartAnalysisRequest {
  sessionId: SessionId;
}

export interface StartAnalysisResponse {
  jobId: JobId;
  sessionId: SessionId;
  kind: "analysis";
  status: "running";
}

export interface GetJobRequest {
  jobId: JobId;
}

export interface JobRecord {
  jobId: JobId;
  sessionId: SessionId;
  kind: JobKind;
  phase: JobPhase;
  status: JobStatus;
  progress?: number;
  message?: string;
  errorCode?: string;
  errorMessage?: string;
  startedAt?: ISODateString;
  updatedAt: ISODateString;
  finishedAt?: ISODateString;
}

export interface GetJobResponse {
  job: JobRecord;
}

export interface ObserveJobRequest {
  jobId: JobId;
}

export interface CancelJobRequest {
  jobId: JobId;
}

export interface CancelJobResponse {
  cancelled: boolean;
}

export interface SessionLibraryItem {
  sessionId: SessionId;
  sessionName: string;
  sourceRef: SourceRef;
  sourceDisplayName?: string;
  sourceOrigin?: SourceOrigin;
  sourceAvailability: SourceAvailability;
  profile: SessionProfile;
  detectorId: DetectorId;
  status: SessionStatus;
  candidateCount?: number;
  keptCount?: number;
  rejectCount?: number;
  unsureCount?: number;
  exportCount?: number;
  createdAt: ISODateString;
  updatedAt: ISODateString;
  lastOpenedAt?: ISODateString | null;
}

export interface ListSessionsRequest {}

export interface ListSessionsResponse {
  sessions: SessionLibraryItem[];
}

export interface TimestampRange {
  first: number;
  last: number;
}

export interface SessionTelemetry {
  detector_seconds: number;
  extract_seconds: number;
  total_runtime_seconds: number;
  peak_rss_kb: number;
}

export interface SessionSummaryManifest {
  id: SessionId;
  title: string;
  session_name?: string;
  profile: SessionProfile;
  detector_id?: DetectorId;
  status: SessionStatus;
  created_at?: ISODateString;
  updated_at?: ISODateString;
  session_duration_seconds?: number;
  source_ref: SourceRef;
  source_origin?: SourceOrigin;
  source_display_name?: string;
  source_availability?: SourceAvailability;
  candidate_count: number;
  extracted_count: number;
  timestamp_range: TimestampRange;
  telemetry: SessionTelemetry;
}

export interface DetectionScores {
  audio: number;
  video: number;
  combined: number;
  audio_model_probability: number;
  audio_clip_probability: number;
}

export interface DetectionExportRef {
  exportId?: ExportId | null;
  localUrl?: URLString | null;
  filename?: string | null;
  status?: ExportStatus | null;
}

export interface Detection {
  id: DetectionId;
  index: number;
  timestamp_seconds: number;
  start_time_seconds: number;
  end_time_seconds: number;
  duration_seconds: number;
  review_start_seconds?: number;
  review_end_seconds?: number;
  review_duration_seconds?: number;
  confidence: ConfidenceLevel;
  scores: DetectionScores;
  features: Record<string, number | null>;
  export_ref?: DetectionExportRef | null;
}

export interface SessionManifest {
  schema_version: string;
  kind: string;
  generated_at: ISODateString;
  session: SessionSummaryManifest;
  artifacts: Record<string, string | number | boolean | null>;
  detections: Detection[];
}

export interface GetSessionManifestRequest {
  sessionId: SessionId;
}

export interface GetSessionManifestResponse {
  manifest: SessionManifest;
}

export interface DecisionRecord {
  id: string;
  sessionId: SessionId;
  detectionId: DetectionId;
  label: ReviewDecisionLabel;
  notes: string;
  createdAt: ISODateString;
  updatedAt: ISODateString;
}

export interface SaveDecisionRequest {
  sessionId: SessionId;
  detectionId: DetectionId;
  label: ReviewDecisionLabel;
  notes?: string;
}

export interface SaveDecisionResponse {
  decision: DecisionRecord;
}

export interface ListDecisionsRequest {
  sessionId: SessionId;
}

export interface ListDecisionsResponse {
  decisions: DecisionRecord[];
}

export interface ReviewProxyRecord {
  sessionId: SessionId;
  status: ProxyStatus;
  url?: URLString;
  durationSeconds?: number;
  updatedAt?: ISODateString;
  playerBackend?: PlayerBackend;
}

export interface GetReviewProxyRequest {
  sessionId: SessionId;
}

export interface GetReviewProxyResponse {
  proxy: ReviewProxyRecord;
}

export interface StartExportRequest {
  sessionId: SessionId;
  saveToLibrary: boolean;
}

export interface StartExportResponse {
  jobId: JobId;
  sessionId: SessionId;
  kind: "export";
  status: "running";
}

export interface ExportRecord {
  exportId: ExportId;
  sessionId: SessionId;
  detectionId: DetectionId;
  status: ExportStatus;
  fileName: string;
  localUrl?: URLString;
  libraryAssetId?: LibraryAssetId;
  createdAt: ISODateString;
  updatedAt: ISODateString;
  errorCode?: string;
  errorMessage?: string;
}

export interface ListExportsRequest {
  sessionId: SessionId;
}

export interface ListExportsResponse {
  exports: ExportRecord[];
}

export interface DeleteSessionRequest {
  sessionId: SessionId;
  deleteExports?: boolean;
}

export interface DeleteSessionResponse {
  deleted: boolean;
}

export interface JobProgressEvent {
  jobId: JobId;
  sessionId: SessionId;
  kind: JobKind;
  phase: JobPhase;
  status: JobStatus;
  progress?: number;
  message?: string;
  startedAt?: ISODateString;
  updatedAt: ISODateString;
  errorCode?: string;
  errorMessage?: string;
}

export interface SessionUpdatedEvent {
  sessionId: SessionId;
  status: SessionStatus;
  updatedAt: ISODateString;
}

export interface ReviewProxyUpdatedEvent {
  sessionId: SessionId;
  status: ProxyStatus;
  url?: URLString;
  updatedAt: ISODateString;
  playerBackend?: PlayerBackend;
}

export interface ExportsUpdatedEvent {
  sessionId: SessionId;
  exports: ExportRecord[];
  updatedAt: ISODateString;
}

/**
 * Main Capacitor plugin surface.
 *
 * Rules:
 * - No raw source video bytes cross the bridge.
 * - No exported clip bytes cross the bridge.
 * - No native filesystem paths are exposed to the web layer.
 * - JS interacts only through opaque IDs, manifests, lightweight metadata, and playable URLs.
 */
export interface DiveSenseiMediaPlugin extends Plugin {
  pickSourceVideo(input: PickSourceVideoRequest): Promise<PickSourceVideoResponse>;
  getSourceAvailability(input: GetSourceAvailabilityRequest): Promise<GetSourceAvailabilityResponse>;
  repairSource(input: RepairSourceRequest): Promise<RepairSourceResponse>;

  createSession(input: CreateSessionRequest): Promise<CreateSessionResponse>;
  listSessions(input: ListSessionsRequest): Promise<ListSessionsResponse>;

  startAnalysis(input: StartAnalysisRequest): Promise<StartAnalysisResponse>;
  getJob(input: GetJobRequest): Promise<GetJobResponse>;
  observeJob(input: ObserveJobRequest): Promise<void>;
  cancelJob(input: CancelJobRequest): Promise<CancelJobResponse>;

  getSessionManifest(input: GetSessionManifestRequest): Promise<GetSessionManifestResponse>;

  saveDecision(input: SaveDecisionRequest): Promise<SaveDecisionResponse>;
  listDecisions(input: ListDecisionsRequest): Promise<ListDecisionsResponse>;

  getReviewProxy(input: GetReviewProxyRequest): Promise<GetReviewProxyResponse>;

  startExport(input: StartExportRequest): Promise<StartExportResponse>;
  listExports(input: ListExportsRequest): Promise<ListExportsResponse>;

  deleteSession(input: DeleteSessionRequest): Promise<DeleteSessionResponse>;
}

export const DiveSenseiMediaEvents = {
  JobProgress: "DiveSenseiMedia.jobProgress",
  SessionUpdated: "DiveSenseiMedia.sessionUpdated",
  ReviewProxyUpdated: "DiveSenseiMedia.reviewProxyUpdated",
  ExportsUpdated: "DiveSenseiMedia.exportsUpdated",
} as const;

/**
 * Optional web-side player adapter abstraction.
 * Keep the review UI wired to this interface, not directly to HTMLVideoElement.
 * That makes it possible to switch from HTML video to native AVPlayer later.
 */
export interface ReviewPlayerAdapter {
  readonly backend: PlayerBackend;

  attach(target: unknown): Promise<void>;
  detach(): Promise<void>;

  load(url: URLString): Promise<void>;
  unload(): Promise<void>;

  play(): Promise<void>;
  pause(): Promise<void>;
  seek(seconds: number): Promise<void>;

  getCurrentTime(): Promise<number>;
  getDuration(): Promise<number | null>;
  isReady(): Promise<boolean>;

  onTimeUpdate(listener: (currentTimeSeconds: number) => void): () => void;
  onEnded(listener: () => void): () => void;
  onError(listener: (error: { code?: string; message: string }) => void): () => void;
}
