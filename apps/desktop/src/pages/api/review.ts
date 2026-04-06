import type { APIRoute } from "astro";
import {
  addEvaluationFalseNegative,
  isEvaluationSession,
  listEvaluationFalseNegatives,
  listEvaluationReviewDecisions,
  saveEvaluationReviewDecision,
} from "@/lib/evaluation-review-store";
import { getManifestPathForAnalysisRun, listReviewDecisions, saveReviewDecision } from "@/lib/session-catalog";
import { readManifest } from "@/lib/session-catalog-core";

export const GET: APIRoute = async ({ url }) => {
  const analysisRunId = url.searchParams.get("analysisRunId");
  if (!analysisRunId) {
    return Response.json({ error: "analysisRunId is required." }, { status: 400 });
  }

  const manifestPath = getManifestPathForAnalysisRun(analysisRunId);
  const manifest = manifestPath ? readManifest(manifestPath) : null;
  if (isEvaluationSession(manifest)) {
    return Response.json({
      decisions: manifest ? listEvaluationReviewDecisions(manifest) : [],
      falseNegatives: manifest ? listEvaluationFalseNegatives(manifest) : [],
      mode: "evaluation",
    });
  }

  return Response.json({
    decisions: listReviewDecisions(analysisRunId),
    falseNegatives: [],
    mode: "standard",
  });
};

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json().catch(() => null) as
    | {
        analysisRunId?: string;
        detectionId?: string;
        label?: "keep" | "reject" | "unsure" | "dive" | "non_dive" | "false_negative";
        notes?: string;
        timestampSeconds?: number;
      }
    | null;

  if (!body?.analysisRunId || !body.label) {
    return Response.json({ error: "analysisRunId and label are required." }, { status: 400 });
  }

  try {
    const manifestPath = getManifestPathForAnalysisRun(body.analysisRunId);
    const manifest = manifestPath ? readManifest(manifestPath) : null;
    if (isEvaluationSession(manifest)) {
      if (!manifest) {
        return Response.json({ error: "Evaluation session manifest not found." }, { status: 404 });
      }
      if (body.label === "false_negative") {
        if (typeof body.timestampSeconds !== "number" || Number.isNaN(body.timestampSeconds)) {
          return Response.json({ error: "timestampSeconds is required for false_negative." }, { status: 400 });
        }
        const annotation = addEvaluationFalseNegative(manifest, body.analysisRunId, body.timestampSeconds, body.notes ?? "");
        return Response.json({ annotation, mode: "evaluation" });
      }
      if (!body.detectionId) {
        return Response.json({ error: "detectionId is required for evaluation candidate decisions." }, { status: 400 });
      }
      if (!["dive", "non_dive", "unsure"].includes(body.label)) {
        return Response.json({ error: "Invalid evaluation label." }, { status: 400 });
      }
      const decision = saveEvaluationReviewDecision(manifest, body.analysisRunId, body.detectionId, body.label, body.notes ?? "");
      return Response.json({ decision, mode: "evaluation" });
    }

    if (!body.detectionId || !["keep", "reject", "unsure"].includes(body.label)) {
      return Response.json({ error: "analysisRunId, detectionId, and a standard review label are required." }, { status: 400 });
    }
    const decision = saveReviewDecision(body.analysisRunId, body.detectionId, body.label, body.notes ?? "");
    return Response.json({ decision, mode: "standard" });
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "Failed to save review decision." }, { status: 400 });
  }
};
