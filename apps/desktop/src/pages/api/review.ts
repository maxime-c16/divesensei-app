import type { APIRoute } from "astro";
import { listReviewDecisions, saveReviewDecision } from "@/lib/session-catalog";

export const GET: APIRoute = async ({ url }) => {
  const analysisRunId = url.searchParams.get("analysisRunId");
  if (!analysisRunId) {
    return Response.json({ error: "analysisRunId is required." }, { status: 400 });
  }

  return Response.json({
    decisions: listReviewDecisions(analysisRunId),
  });
};

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json().catch(() => null) as
    | { analysisRunId?: string; detectionId?: string; label?: "keep" | "reject" | "unsure"; notes?: string }
    | null;

  if (!body?.analysisRunId || !body.detectionId || !body.label) {
    return Response.json({ error: "analysisRunId, detectionId, and label are required." }, { status: 400 });
  }

  try {
    const decision = saveReviewDecision(body.analysisRunId, body.detectionId, body.label, body.notes ?? "");
    return Response.json({ decision });
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "Failed to save review decision." }, { status: 400 });
  }
};
