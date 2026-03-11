import type { APIRoute } from "astro";
import { deleteSessionRun, listCatalogSessions, markSessionOpened, refreshCatalogAvailability, relinkCatalogSource, renameSessionRun } from "@/lib/session-catalog";

function jsonWithNoStore(body: unknown, init?: ResponseInit): Response {
  const response = Response.json(body, init);
  response.headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
  response.headers.set("Pragma", "no-cache");
  response.headers.set("Expires", "0");
  return response;
}

export const GET: APIRoute = async () => {
  return jsonWithNoStore({
    sessions: listCatalogSessions(),
  });
};

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json().catch(() => null) as
    | { action?: string; mediaSourceId?: string; nextPath?: string; analysisRunId?: string; sessionName?: string }
    | null;

  if (!body?.action) {
    return jsonWithNoStore({ error: "Missing action." }, { status: 400 });
  }

  try {
    if (body.action === "refresh") {
      return jsonWithNoStore({ sessions: refreshCatalogAvailability() });
    }

    if (body.action === "relink") {
      if (!body.mediaSourceId || !body.nextPath) {
        return jsonWithNoStore({ error: "Missing relink parameters." }, { status: 400 });
      }
      relinkCatalogSource(body.mediaSourceId, body.nextPath);
      return jsonWithNoStore({ sessions: listCatalogSessions() });
    }

    if (body.action === "open") {
      if (!body.analysisRunId) {
        return jsonWithNoStore({ error: "Missing analysisRunId." }, { status: 400 });
      }
      markSessionOpened(body.analysisRunId);
      return jsonWithNoStore({ sessions: listCatalogSessions() });
    }

    if (body.action === "delete") {
      if (!body.analysisRunId) {
        return jsonWithNoStore({ error: "Missing analysisRunId." }, { status: 400 });
      }
      const deleted = deleteSessionRun(body.analysisRunId);
      return jsonWithNoStore({ deleted, sessions: listCatalogSessions() });
    }

    if (body.action === "rename") {
      if (!body.analysisRunId || !body.sessionName) {
        return jsonWithNoStore({ error: "Missing rename parameters." }, { status: 400 });
      }
      renameSessionRun(body.analysisRunId, body.sessionName);
      return jsonWithNoStore({ sessions: listCatalogSessions() });
    }

    return jsonWithNoStore({ error: "Unsupported action." }, { status: 400 });
  } catch (error) {
    return jsonWithNoStore({ error: error instanceof Error ? error.message : "Catalog action failed." }, { status: 400 });
  }
};
