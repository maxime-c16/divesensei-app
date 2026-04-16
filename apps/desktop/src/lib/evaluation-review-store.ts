import fs from "node:fs";
import path from "node:path";
import type {
  EvaluationFalseNegativeAnnotation,
  EvaluationReviewSubtype,
  ReviewDecision,
  ReviewDecisionLabel,
  SessionManifest,
} from "@/types/ui";

interface EvaluationReviewStore {
  schemaVersion: string;
  decisions: ReviewDecision[];
  falseNegatives: EvaluationFalseNegativeAnnotation[];
}

const EMPTY_STORE: EvaluationReviewStore = {
  schemaVersion: "1.0.0",
  decisions: [],
  falseNegatives: [],
};

function readJsonFile<T>(filePath: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

export function isEvaluationSession(manifest: SessionManifest | null | undefined): boolean {
  return (manifest?.session.mode ?? "standard") === "evaluation";
}

export function evaluationReviewStorePath(manifest: SessionManifest): string {
  const artifactPath = manifest.artifacts?.evaluation_review;
  return artifactPath && artifactPath.length > 0
    ? artifactPath
    : path.join(manifest.session.output_dir, "evaluation_review.json");
}

export function loadEvaluationReviewStore(manifest: SessionManifest): EvaluationReviewStore {
  const filePath = evaluationReviewStorePath(manifest);
  return readJsonFile<EvaluationReviewStore>(filePath) ?? { ...EMPTY_STORE };
}

function saveEvaluationReviewStore(manifest: SessionManifest, store: EvaluationReviewStore): void {
  const filePath = evaluationReviewStorePath(manifest);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(store, null, 2));
}

export function listEvaluationReviewDecisions(manifest: SessionManifest): ReviewDecision[] {
  return loadEvaluationReviewStore(manifest).decisions.slice().sort((a, b) => {
    if (a.updatedAt === b.updatedAt) return a.detectionId.localeCompare(b.detectionId);
    return b.updatedAt.localeCompare(a.updatedAt);
  });
}

export function saveEvaluationReviewDecision(
  manifest: SessionManifest,
  analysisRunId: string,
  detectionId: string,
  label: Extract<ReviewDecisionLabel, "dive" | "non_dive" | "unsure">,
  eventLabel: ReviewDecision["eventLabel"] = null,
  subtype: EvaluationReviewSubtype | null = null,
  notes = "",
): ReviewDecision {
  const store = loadEvaluationReviewStore(manifest);
  const now = new Date().toISOString();
  const existing = store.decisions.find((entry) => entry.analysisRunId === analysisRunId && entry.detectionId === detectionId)
    ?? store.decisions.find((entry) => entry.detectionId === detectionId);
  const decision: ReviewDecision = {
    id: `${analysisRunId}:${detectionId}`,
    analysisRunId,
    detectionId,
    label,
    eventLabel,
    subtype,
    notes,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  };
  if (existing) {
    Object.assign(existing, decision);
  } else {
    store.decisions.push(decision);
  }
  saveEvaluationReviewStore(manifest, store);
  return decision;
}

export function listEvaluationFalseNegatives(manifest: SessionManifest): EvaluationFalseNegativeAnnotation[] {
  return loadEvaluationReviewStore(manifest).falseNegatives.slice().sort((a, b) => {
    if (a.timestampSeconds === b.timestampSeconds) return a.id.localeCompare(b.id);
    return a.timestampSeconds - b.timestampSeconds;
  });
}

export function addEvaluationFalseNegative(
  manifest: SessionManifest,
  analysisRunId: string,
  timestampSeconds: number,
  eventLabel: EvaluationFalseNegativeAnnotation["eventLabel"] = null,
  subtype: EvaluationReviewSubtype | null = null,
  notes = "",
): EvaluationFalseNegativeAnnotation {
  const store = loadEvaluationReviewStore(manifest);
  const now = new Date().toISOString();
  const annotation: EvaluationFalseNegativeAnnotation = {
    id: `fn-${Math.round(timestampSeconds * 1000)}-${Date.now()}`,
    analysisRunId,
    timestampSeconds,
    reviewStartSeconds: Math.max(0, timestampSeconds - 2.0),
    reviewEndSeconds: Math.max(timestampSeconds + 2.0, timestampSeconds + 0.5),
    label: "false_negative",
    eventLabel,
    subtype,
    notes,
    createdAt: now,
    updatedAt: now,
  };
  store.falseNegatives.push(annotation);
  saveEvaluationReviewStore(manifest, store);
  return annotation;
}

export function saveEvaluationFalseNegativeAnnotation(
  manifest: SessionManifest,
  analysisRunId: string,
  annotationId: string,
  eventLabel: EvaluationFalseNegativeAnnotation["eventLabel"] = null,
  subtype: EvaluationReviewSubtype | null = null,
  notes = "",
): EvaluationFalseNegativeAnnotation {
  const store = loadEvaluationReviewStore(manifest);
  const now = new Date().toISOString();
  const existing = store.falseNegatives.find((entry) => entry.id === annotationId && entry.analysisRunId === analysisRunId);
  if (!existing) {
    throw new Error(`False negative annotation not found: ${annotationId}`);
  }
  existing.eventLabel = eventLabel;
  existing.subtype = subtype;
  existing.notes = notes;
  existing.updatedAt = now;
  saveEvaluationReviewStore(manifest, store);
  return existing;
}

export function removeEvaluationFalseNegativeAnnotation(
  manifest: SessionManifest,
  analysisRunId: string,
  annotationId: string,
): EvaluationFalseNegativeAnnotation {
  const store = loadEvaluationReviewStore(manifest);
  const index = store.falseNegatives.findIndex((entry) =>
    entry.id === annotationId && entry.analysisRunId === analysisRunId
  );
  if (index < 0) {
    throw new Error(`False negative annotation not found: ${annotationId}`);
  }
  const [removed] = store.falseNegatives.splice(index, 1);
  saveEvaluationReviewStore(manifest, store);
  return removed;
}
