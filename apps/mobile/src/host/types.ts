import type {
  CreateSessionRequest,
  CreateSessionResponse,
  DecisionRecord,
  GetReviewProxyResponse,
  JobRecord,
  ListSessionsResponse,
  RepairSourceResponse,
  ReviewDecisionLabel,
  SessionId,
  SessionManifest,
  SourceSummary,
  StartAnalysisResponse,
} from "@/native/types";

export interface ReviewHostService {
  pickSourceVideo(): Promise<SourceSummary | null>;
  repairSource(sessionId: SessionId): Promise<RepairSourceResponse>;
  createSession(input: CreateSessionRequest): Promise<CreateSessionResponse>;
  listSessions(): Promise<ListSessionsResponse>;
  startAnalysis(sessionId: SessionId): Promise<StartAnalysisResponse>;
  getJob(jobId: string): Promise<JobRecord>;
  getSessionManifest(sessionId: SessionId): Promise<SessionManifest>;
  listDecisions(sessionId: SessionId): Promise<DecisionRecord[]>;
  saveDecision(sessionId: SessionId, detectionId: string, label: ReviewDecisionLabel, notes?: string): Promise<DecisionRecord>;
  getReviewProxy(sessionId: SessionId): Promise<GetReviewProxyResponse>;
}
