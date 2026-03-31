import type { APIRoute } from "astro";
import { getSessionExportState } from "@/lib/export-state";
import { readManifest } from "@/lib/session-catalog-core";
import { getManifestPathForAnalysisRun, listReviewDecisions } from "@/lib/session-catalog";
import { getExportJob, startExportJob } from "@/lib/export-jobs";
import { exportJobsRoot } from "@/lib/runtime-config";

export const GET: APIRoute = async ({ url }) => {
  const jobId = url.searchParams.get("job");
  const analysisRunId = url.searchParams.get("analysisRunId");
  if (!jobId) {
    if (analysisRunId) {
      const manifestPath = getManifestPathForAnalysisRun(analysisRunId);
      if (!manifestPath) {
        return Response.json({ error: "Session manifest not found." }, { status: 404 });
      }
      const manifest = readManifest(manifestPath);
      if (!manifest) {
        return Response.json({ error: "Session manifest could not be read." }, { status: 404 });
      }
      const exportState = getSessionExportState(manifest.session.output_dir);
      return Response.json({
        analysisRunId,
        exportedPaths: exportState.exportedPaths,
        exportedDetectionIds: exportState.exportedDetectionIds,
        exportedCount: exportState.exportedCount,
      });
    }
    return Response.json({ jobsRoot: exportJobsRoot });
  }

  const job = getExportJob(jobId);
  if (!job) {
    return Response.json({ error: "Job not found." }, { status: 404 });
  }
  return Response.json(job);
};

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json().catch(() => null) as { analysisRunId?: string } | null;
  const analysisRunId = String(body?.analysisRunId ?? "");
  if (!analysisRunId) {
    return Response.json({ error: "analysisRunId is required." }, { status: 400 });
  }

  const manifestPath = getManifestPathForAnalysisRun(analysisRunId);
  if (!manifestPath) {
    return Response.json({ error: "Session manifest not found." }, { status: 404 });
  }

  const keptDetections = listReviewDecisions(analysisRunId)
    .filter((entry) => entry.label === "keep")
    .map((entry) => entry.detectionId);

  if (keptDetections.length === 0) {
    return Response.json({ error: "No kept detections are available to export." }, { status: 400 });
  }

  try {
    const job = startExportJob(analysisRunId, manifestPath, keptDetections);
    return Response.json(job, { status: 202 });
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "Failed to start export job." }, { status: 400 });
  }
};
