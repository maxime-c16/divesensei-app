import fs from "node:fs";
import type { APIRoute } from "astro";
import { getAnalysisJob, listAllowedRoots, startAnalysisJob } from "@/lib/analysis-jobs";
import { analysisJobsRoot } from "@/lib/runtime-config";

export const GET: APIRoute = async ({ url }) => {
  const jobId = url.searchParams.get("job");
  if (!jobId) {
    return Response.json({ allowedRoots: listAllowedRoots(), jobsRoot: analysisJobsRoot });
  }

  const job = getAnalysisJob(jobId);
  if (!job) {
    return Response.json({ error: "Job not found." }, { status: 404 });
  }

  return Response.json(job);
};

export const POST: APIRoute = async ({ request }) => {
  const contentType = request.headers.get("content-type") ?? "";
  let videoPath = "";
  let sessionName = "";
  let profile = "long-session";
  let detectorId = "audio_v2_pcen_classifier";

  if (contentType.includes("application/json")) {
    const body = await request.json();
    videoPath = String(body.videoPath ?? "");
    sessionName = String(body.sessionName ?? "");
    profile = String(body.profile ?? "long-session");
    detectorId = String(body.detectorId ?? "audio_v2_pcen_classifier");
  } else {
    const form = await request.formData();
    videoPath = String(form.get("videoPath") ?? "");
    sessionName = String(form.get("sessionName") ?? "");
    profile = String(form.get("profile") ?? "long-session");
    detectorId = String(form.get("detectorId") ?? "audio_v2_pcen_classifier");
  }

  if (!videoPath) {
    return Response.json({ error: "Video path is required." }, { status: 400 });
  }

  if (!fs.existsSync(videoPath)) {
    return Response.json({ error: "Video path does not exist." }, { status: 400 });
  }

  try {
    const job = startAnalysisJob(videoPath, profile, detectorId, sessionName);
    return Response.json(job, { status: 202 });
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "Failed to start analysis." }, { status: 400 });
  }
};
