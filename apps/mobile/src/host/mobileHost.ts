import { DiveSenseiMedia } from "@/native/client";
import type { ReviewHostService } from "@/host/types";
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

export class MobileReviewHostService implements ReviewHostService {
  async pickSourceVideo(): Promise<SourceSummary | null> {
    const response = await DiveSenseiMedia.pickSourceVideo({});
    return response.cancelled ? null : response.source ?? null;
  }

  repairSource(sessionId: SessionId): Promise<RepairSourceResponse> {
    return DiveSenseiMedia.repairSource({ sessionId });
  }

  createSession(input: CreateSessionRequest): Promise<CreateSessionResponse> {
    return DiveSenseiMedia.createSession(input);
  }

  listSessions(): Promise<ListSessionsResponse> {
    return DiveSenseiMedia.listSessions({});
  }

  startAnalysis(sessionId: SessionId): Promise<StartAnalysisResponse> {
    return DiveSenseiMedia.startAnalysis({ sessionId });
  }

  async getJob(jobId: string): Promise<JobRecord> {
    const response = await DiveSenseiMedia.getJob({ jobId });
    return response.job;
  }

  async getSessionManifest(sessionId: SessionId): Promise<SessionManifest> {
    const response = await DiveSenseiMedia.getSessionManifest({ sessionId });
    return response.manifest;
  }

  async listDecisions(sessionId: SessionId): Promise<DecisionRecord[]> {
    const response = await DiveSenseiMedia.listDecisions({ sessionId });
    return response.decisions;
  }

  async saveDecision(
    sessionId: SessionId,
    detectionId: string,
    label: ReviewDecisionLabel,
    notes = "",
  ): Promise<DecisionRecord> {
    const response = await DiveSenseiMedia.saveDecision({ sessionId, detectionId, label, notes });
    return response.decision;
  }

  getReviewProxy(sessionId: SessionId): Promise<GetReviewProxyResponse> {
    return DiveSenseiMedia.getReviewProxy({ sessionId });
  }
}
