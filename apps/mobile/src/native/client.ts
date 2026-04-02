import { WebPlugin, registerPlugin } from "@capacitor/core";
import type {
  CancelJobRequest,
  CancelJobResponse,
  ClearDecisionRequest,
  ClearDecisionResponse,
  CreateSessionRequest,
  CreateSessionResponse,
  DecisionRecord,
  DeleteSessionRequest,
  DeleteSessionResponse,
  DiveSenseiMediaPlugin,
  ExportRecord,
  ExportsUpdatedEvent,
  GetJobRequest,
  GetJobResponse,
  GetReviewProxyRequest,
  GetReviewProxyResponse,
  GetSessionManifestRequest,
  GetSessionManifestResponse,
  GetSourceAvailabilityRequest,
  GetSourceAvailabilityResponse,
  JobId,
  JobProgressEvent,
  JobRecord,
  ListDecisionsRequest,
  ListDecisionsResponse,
  ListExportsRequest,
  ListExportsResponse,
  ListSessionsRequest,
  ListSessionsResponse,
  ObserveJobRequest,
  PickSourceVideoRequest,
  PickSourceVideoResponse,
  RepairSourceRequest,
  RepairSourceResponse,
  ReviewProxyUpdatedEvent,
  SaveDecisionRequest,
  SaveDecisionResponse,
  SessionId,
  SessionLibraryItem,
  SessionManifest,
  SessionUpdatedEvent,
  SourceRef,
  SourceSummary,
  StartAnalysisRequest,
  StartAnalysisResponse,
  StartExportRequest,
  StartExportResponse,
} from "@/native/types";
import { DiveSenseiMediaEvents } from "@/native/types";

type ListenerHandle = { remove: () => Promise<void> };
type EventName =
  | typeof DiveSenseiMediaEvents.JobProgress
  | typeof DiveSenseiMediaEvents.SessionUpdated
  | typeof DiveSenseiMediaEvents.ReviewProxyUpdated
  | typeof DiveSenseiMediaEvents.ExportsUpdated;

function nowIso(): string {
  return new Date().toISOString();
}

class DiveSenseiMediaWeb extends WebPlugin implements DiveSenseiMediaPlugin {
  private sourceCounter = 0;
  private sessionCounter = 0;
  private jobCounter = 0;
  private exportCounter = 0;

  private readonly sources = new Map<SourceRef, SourceSummary>();
  private readonly sessions = new Map<SessionId, SessionLibraryItem>();
  private readonly manifests = new Map<SessionId, SessionManifest>();
  private readonly decisions = new Map<SessionId, DecisionRecord[]>();
  private readonly exports = new Map<SessionId, ExportRecord[]>();
  private readonly jobs = new Map<JobId, JobRecord>();

  async pickSourceVideo(_input: PickSourceVideoRequest): Promise<PickSourceVideoResponse> {
    this.sourceCounter += 1;
    const sourceRef = `src_${String(this.sourceCounter).padStart(4, "0")}`;
    const source: SourceSummary = {
      sourceRef,
      origin: "photos",
      displayName: `Stub Session ${this.sourceCounter}.mov`,
      availability: "available",
      durationSeconds: 90,
      fileSizeBytes: 1_500_000_000,
      canPersist: true,
    };
    this.sources.set(sourceRef, source);
    return { cancelled: false, source };
  }

  async getSourceAvailability(input: GetSourceAvailabilityRequest): Promise<GetSourceAvailabilityResponse> {
    const source = this.sources.get(input.sourceRef);
    if (!source) {
      throw new Error("source_not_found");
    }
    return {
      sourceRef: source.sourceRef,
      availability: source.availability,
      origin: source.origin,
      displayName: source.displayName,
      durationSeconds: source.durationSeconds,
      fileSizeBytes: source.fileSizeBytes,
      lastResolvedAt: nowIso(),
    };
  }

  async repairSource(input: RepairSourceRequest): Promise<RepairSourceResponse> {
    const session = this.sessions.get(input.sessionId);
    if (!session) {
      throw new Error("session_not_found");
    }
    const source = this.sources.get(session.sourceRef);
    return {
      repaired: false,
      source,
      availability: source?.availability ?? "missing",
    };
  }

  async createSession(input: CreateSessionRequest): Promise<CreateSessionResponse> {
    const source = this.sources.get(input.sourceRef);
    if (!source) {
      throw new Error("source_not_found");
    }

    this.sessionCounter += 1;
    const sessionId = `sess_${String(this.sessionCounter).padStart(4, "0")}`;
    const createdAt = nowIso();
    const session: SessionLibraryItem = {
      sessionId,
      sessionName: input.sessionName?.trim() || source.displayName.replace(/\.[^.]+$/, ""),
      sourceRef: input.sourceRef,
      sourceDisplayName: source.displayName,
      sourceOrigin: source.origin,
      sourceAvailability: source.availability,
      profile: input.profile,
      detectorId: input.detectorId,
      status: "created",
      candidateCount: 2,
      keptCount: 0,
      rejectCount: 0,
      unsureCount: 0,
      exportCount: 0,
      createdAt,
      updatedAt: createdAt,
      lastOpenedAt: null,
    };
    this.sessions.set(sessionId, session);
    this.decisions.set(sessionId, []);
    this.exports.set(sessionId, []);
    this.manifests.set(sessionId, {
      schema_version: "1.0.0",
      kind: "divesensei.ui-session",
      generated_at: createdAt,
      session: {
        id: sessionId,
        title: session.sessionName,
        session_name: session.sessionName,
        profile: session.profile,
        detector_id: session.detectorId,
        status: "created",
        created_at: createdAt,
        updated_at: createdAt,
        session_duration_seconds: source.durationSeconds,
        source_ref: source.sourceRef,
        source_origin: source.origin,
        source_display_name: source.displayName,
        source_availability: source.availability,
        candidate_count: 2,
        extracted_count: 0,
        timestamp_range: { first: 12.4, last: 41.2 },
        telemetry: {
          detector_seconds: 0,
          extract_seconds: 0,
          total_runtime_seconds: 0,
          peak_rss_kb: 0,
        },
      },
      artifacts: {
        review_mode: "stub",
      },
      detections: [
        {
          id: "det-0001",
          index: 1,
          timestamp_seconds: 12.4,
          start_time_seconds: 6.4,
          end_time_seconds: 15.4,
          duration_seconds: 9,
          review_start_seconds: 10.4,
          review_end_seconds: 14.4,
          review_duration_seconds: 4,
          confidence: "high",
          scores: {
            audio: 8.6,
            video: 0,
            combined: 8.6,
            audio_model_probability: 0.91,
            audio_clip_probability: 0.88,
          },
          features: {
            spectral_flux: 1.2,
            rms: 0.7,
          },
          export_ref: null,
        },
        {
          id: "det-0002",
          index: 2,
          timestamp_seconds: 41.2,
          start_time_seconds: 35.2,
          end_time_seconds: 44.2,
          duration_seconds: 9,
          review_start_seconds: 39.2,
          review_end_seconds: 43.2,
          review_duration_seconds: 4,
          confidence: "medium",
          scores: {
            audio: 5.1,
            video: 0,
            combined: 5.1,
            audio_model_probability: 0.64,
            audio_clip_probability: 0.57,
          },
          features: {
            spectral_flux: 0.8,
            rms: 0.5,
          },
          export_ref: null,
        },
      ],
    });
    this.notifyListeners(DiveSenseiMediaEvents.SessionUpdated, {
      sessionId,
      status: "created",
      updatedAt: createdAt,
    } satisfies SessionUpdatedEvent);
    return { sessionId, status: "created", createdAt };
  }

  async listSessions(_input: ListSessionsRequest): Promise<ListSessionsResponse> {
    return { sessions: [...this.sessions.values()] };
  }

  async startAnalysis(input: StartAnalysisRequest): Promise<StartAnalysisResponse> {
    const session = this.sessions.get(input.sessionId);
    if (!session) {
      throw new Error("session_not_found");
    }
    this.jobCounter += 1;
    const jobId = `job_${String(this.jobCounter).padStart(4, "0")}`;
    const startedAt = nowIso();
    const job: JobRecord = {
      jobId,
      sessionId: input.sessionId,
      kind: "analysis",
      phase: "detecting",
      status: "running",
      progress: 0,
      message: "Stub analysis started.",
      startedAt,
      updatedAt: startedAt,
    };
    this.jobs.set(jobId, job);
    this.sessions.set(input.sessionId, {
      ...session,
      status: "analyzing",
      updatedAt: startedAt,
    });
    this.notifyListeners(DiveSenseiMediaEvents.SessionUpdated, {
      sessionId: input.sessionId,
      status: "analyzing",
      updatedAt: startedAt,
    } satisfies SessionUpdatedEvent);
    window.setTimeout(() => {
      const updatedAt = nowIso();
      this.jobs.set(jobId, {
        ...job,
        phase: "review_ready",
        status: "completed",
        progress: 1,
        message: "Stub analysis completed.",
        updatedAt,
        finishedAt: updatedAt,
      });
      const manifest = this.manifests.get(input.sessionId);
      if (manifest) {
        manifest.session.status = "review_ready";
        manifest.session.updated_at = updatedAt;
        manifest.generated_at = updatedAt;
        this.manifests.set(input.sessionId, manifest);
      }
      const nextSession = this.sessions.get(input.sessionId);
      if (nextSession) {
        this.sessions.set(input.sessionId, {
          ...nextSession,
          status: "review_ready",
          updatedAt,
        });
      }
      this.notifyListeners(DiveSenseiMediaEvents.JobProgress, {
        jobId,
        sessionId: input.sessionId,
        kind: "analysis",
        phase: "review_ready",
        status: "completed",
        progress: 1,
        message: "Stub analysis completed.",
        startedAt,
        updatedAt,
      } satisfies JobProgressEvent);
      this.notifyListeners(DiveSenseiMediaEvents.SessionUpdated, {
        sessionId: input.sessionId,
        status: "review_ready",
        updatedAt,
      } satisfies SessionUpdatedEvent);
      this.notifyListeners(DiveSenseiMediaEvents.ReviewProxyUpdated, {
        sessionId: input.sessionId,
        status: "ready",
        url: `https://example.invalid/review-proxy/${input.sessionId}.mp4`,
        updatedAt,
        playerBackend: "html_video",
      } satisfies ReviewProxyUpdatedEvent);
    }, 800);
    return { jobId, sessionId: input.sessionId, kind: "analysis", status: "running" };
  }

  async getJob(input: GetJobRequest): Promise<GetJobResponse> {
    const job = this.jobs.get(input.jobId);
    if (!job) {
      throw new Error("job_not_found");
    }
    return { job };
  }

  async observeJob(_input: ObserveJobRequest): Promise<void> {}

  async cancelJob(input: CancelJobRequest): Promise<CancelJobResponse> {
    const job = this.jobs.get(input.jobId);
    if (!job) return { cancelled: false };
    this.jobs.set(input.jobId, {
      ...job,
      status: "cancelled",
      phase: "cancelled",
      updatedAt: nowIso(),
      finishedAt: nowIso(),
    });
    return { cancelled: true };
  }

  async getSessionManifest(input: GetSessionManifestRequest): Promise<GetSessionManifestResponse> {
    const manifest = this.manifests.get(input.sessionId);
    if (!manifest) {
      throw new Error("manifest_not_found");
    }
    return { manifest };
  }

  async saveDecision(input: SaveDecisionRequest): Promise<SaveDecisionResponse> {
    const existing = this.decisions.get(input.sessionId) ?? [];
    const now = nowIso();
    const previous = existing.find((item) => item.detectionId === input.detectionId);
    const decision: DecisionRecord = {
      id: previous?.id ?? `${input.sessionId}:${input.detectionId}`,
      sessionId: input.sessionId,
      detectionId: input.detectionId,
      label: input.label,
      notes: input.notes ?? "",
      createdAt: previous?.createdAt ?? now,
      updatedAt: now,
    };
    const next = [...existing.filter((item) => item.detectionId !== input.detectionId), decision];
    this.decisions.set(input.sessionId, next);
    this.updateSessionDecisionCounts(input.sessionId, next, now);
    return { decision };
  }

  async clearDecision(input: ClearDecisionRequest): Promise<ClearDecisionResponse> {
    const existing = this.decisions.get(input.sessionId) ?? [];
    const next = existing.filter((item) => item.detectionId !== input.detectionId);
    const cleared = next.length !== existing.length;
    this.decisions.set(input.sessionId, next);
    if (cleared) {
      this.updateSessionDecisionCounts(input.sessionId, next, nowIso());
    }
    return { cleared };
  }

  async listDecisions(input: ListDecisionsRequest): Promise<ListDecisionsResponse> {
    return { decisions: this.decisions.get(input.sessionId) ?? [] };
  }

  async getReviewProxy(input: GetReviewProxyRequest): Promise<GetReviewProxyResponse> {
    return {
      proxy: {
        sessionId: input.sessionId,
        status: "ready",
        url: `https://example.invalid/review-proxy/${input.sessionId}.mp4`,
        durationSeconds: 90,
        updatedAt: nowIso(),
        playerBackend: "html_video",
      },
    };
  }

  async startExport(input: StartExportRequest): Promise<StartExportResponse> {
    this.jobCounter += 1;
    const jobId = `job_${String(this.jobCounter).padStart(4, "0")}`;
    const startedAt = nowIso();
    this.jobs.set(jobId, {
      jobId,
      sessionId: input.sessionId,
      kind: "export",
      phase: "exporting",
      status: "running",
      progress: 0,
      message: "Stub export started.",
      startedAt,
      updatedAt: startedAt,
    });
    window.setTimeout(() => {
      this.exportCounter += 1;
      const exportRecord: ExportRecord = {
        exportId: `exp_${String(this.exportCounter).padStart(4, "0")}`,
        sessionId: input.sessionId,
        detectionId: "det-0001",
        status: input.saveToLibrary ? "saved_to_library" : "saved_local",
        fileName: "clip_det-0001.mp4",
        localUrl: "https://example.invalid/export/clip_det-0001.mp4",
        libraryAssetId: input.saveToLibrary ? "lib_0001" : undefined,
        createdAt: startedAt,
        updatedAt: nowIso(),
      };
      this.exports.set(input.sessionId, [exportRecord]);
      this.notifyListeners(DiveSenseiMediaEvents.ExportsUpdated, {
        sessionId: input.sessionId,
        exports: [exportRecord],
        updatedAt: nowIso(),
      } satisfies ExportsUpdatedEvent);
    }, 600);
    return { jobId, sessionId: input.sessionId, kind: "export", status: "running" };
  }

  async listExports(input: ListExportsRequest): Promise<ListExportsResponse> {
    return { exports: this.exports.get(input.sessionId) ?? [] };
  }

  async deleteSession(input: DeleteSessionRequest): Promise<DeleteSessionResponse> {
    const deleted = this.sessions.delete(input.sessionId);
    this.manifests.delete(input.sessionId);
    this.decisions.delete(input.sessionId);
    if (input.deleteExports) {
      this.exports.delete(input.sessionId);
    }
    return { deleted };
  }

  private updateSessionDecisionCounts(sessionId: SessionId, decisions: DecisionRecord[], updatedAt: string): void {
    const session = this.sessions.get(sessionId);
    if (!session) return;
    this.sessions.set(sessionId, {
      ...session,
      keptCount: decisions.filter((item) => item.label === "keep").length,
      rejectCount: decisions.filter((item) => item.label === "reject").length,
      unsureCount: decisions.filter((item) => item.label === "unsure").length,
      updatedAt,
    });
    this.notifyListeners(DiveSenseiMediaEvents.SessionUpdated, {
      sessionId,
      status: session.status,
      updatedAt,
    } satisfies SessionUpdatedEvent);
  }
}

export const DiveSenseiMedia = registerPlugin<DiveSenseiMediaPlugin>("DiveSenseiMedia", {
  web: () => Promise.resolve(new DiveSenseiMediaWeb()),
});

export type DiveSenseiMediaListenerMap = {
  [DiveSenseiMediaEvents.JobProgress]: JobProgressEvent;
  [DiveSenseiMediaEvents.SessionUpdated]: SessionUpdatedEvent;
  [DiveSenseiMediaEvents.ReviewProxyUpdated]: ReviewProxyUpdatedEvent;
  [DiveSenseiMediaEvents.ExportsUpdated]: ExportsUpdatedEvent;
};

export async function addDiveSenseiListener<TEventName extends EventName>(
  eventName: TEventName,
  listener: (payload: DiveSenseiMediaListenerMap[TEventName]) => void,
): Promise<ListenerHandle> {
  return DiveSenseiMedia.addListener(eventName, listener as (payload: unknown) => void);
}
