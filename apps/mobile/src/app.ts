import { addDiveSenseiListener, DiveSenseiMedia } from "@/native/client";
import { DiveSenseiMediaEvents } from "@/native/types";
import { MobileReviewHostService } from "@/host/mobileHost";
import type { SessionLibraryItem, SourceSummary } from "@/native/types";
import { ReviewWorkspace } from "@/review/ReviewWorkspace";

const host = new MobileReviewHostService();

type AppState = {
  source: SourceSummary | null;
  sessions: SessionLibraryItem[];
  selectedSessionId: string | null;
  eventLog: Array<{ type: string; payload: unknown }>;
  lastError: string | null;
};

const state: AppState = {
  source: null,
  sessions: [],
  selectedSessionId: null,
  eventLog: [],
  lastError: null,
};

function appendEvent(type: string, payload: unknown): void {
  state.eventLog = [{ type, payload }, ...state.eventLog].slice(0, 12);
  render();
}

async function refreshSessions(): Promise<void> {
  const response = await host.listSessions();
  state.sessions = response.sessions;
  if (!state.selectedSessionId && response.sessions[0]) {
    state.selectedSessionId = response.sessions[0].sessionId;
  }
}

function appShell(): string {
  return `
    <main class="shell">
      <section class="card">
        <h1>DiveSensei Mobile</h1>
        <p class="muted">Capacitor scaffold + typed bridge smoke path.</p>
        <div class="button-row">
          <button id="pick-source">Pick Source</button>
          <button id="list-sessions">List Sessions</button>
          <button id="create-session" ${state.source ? "" : "disabled"}>Create Session</button>
          <button id="start-analysis" ${state.selectedSessionId ? "" : "disabled"}>Start Analysis</button>
        </div>
        <pre class="json-block">${escapeHtml(JSON.stringify(state.source, null, 2) || "null")}</pre>
        ${state.lastError ? `<p class="error">${escapeHtml(state.lastError)}</p>` : ""}
      </section>
      <section class="card">
        <h2>Sessions</h2>
        <ul class="session-list">
          ${state.sessions.map((session) => `
            <li>
              <button class="session-button ${session.sessionId === state.selectedSessionId ? "is-active" : ""}" data-session-id="${session.sessionId}">
                ${escapeHtml(session.sessionName)} · ${escapeHtml(session.status)}
              </button>
            </li>
          `).join("") || "<li class='muted'>No sessions yet.</li>"}
        </ul>
      </section>
      <div id="review-root"></div>
      <section class="card">
        <h2>Event Log</h2>
        <div class="event-log">
          ${state.eventLog.map((entry) => `
            <article class="event-item">
              <strong>${escapeHtml(entry.type)}</strong>
              <pre class="json-block">${escapeHtml(JSON.stringify(entry.payload, null, 2))}</pre>
            </article>
          `).join("") || "<p class='muted'>No events yet.</p>"}
        </div>
      </section>
    </main>
  `;
}

function bindApp(root: HTMLElement): void {
  root.querySelector<HTMLButtonElement>("#pick-source")?.addEventListener("click", async () => {
    try {
      state.lastError = null;
      state.source = await host.pickSourceVideo();
      render();
    } catch (error) {
      state.lastError = formatError(error);
      render();
    }
  });

  root.querySelector<HTMLButtonElement>("#list-sessions")?.addEventListener("click", async () => {
    try {
      state.lastError = null;
      await refreshSessions();
      render();
    } catch (error) {
      state.lastError = formatError(error);
      render();
    }
  });

  root.querySelector<HTMLButtonElement>("#create-session")?.addEventListener("click", async () => {
    if (!state.source) return;
    try {
      state.lastError = null;
      const created = await host.createSession({
        sourceRef: state.source.sourceRef,
        sessionName: state.source.displayName.replace(/\.[^.]+$/, ""),
        profile: "long-session",
        detectorId: "audio_v2_pcen_classifier",
      });
      state.selectedSessionId = created.sessionId;
      await refreshSessions();
      render();
    } catch (error) {
      state.lastError = formatError(error);
      render();
    }
  });

  root.querySelector<HTMLButtonElement>("#start-analysis")?.addEventListener("click", async () => {
    if (!state.selectedSessionId) return;
    try {
      state.lastError = null;
      await host.startAnalysis(state.selectedSessionId);
      await refreshSessions();
      render();
    } catch (error) {
      state.lastError = formatError(error);
      render();
    }
  });

  root.querySelectorAll<HTMLButtonElement>("[data-session-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedSessionId = button.dataset.sessionId ?? null;
      render();
    });
  });
}

async function renderReview(root: HTMLElement): Promise<void> {
  const reviewRoot = root.querySelector<HTMLElement>("#review-root");
  if (!reviewRoot) return;
  if (!state.selectedSessionId) {
    reviewRoot.innerHTML = "";
    return;
  }
  const review = new ReviewWorkspace(host, reviewRoot);
  await review.render(state.selectedSessionId);
}

async function render(): Promise<void> {
  const root = document.querySelector<HTMLElement>("#app");
  if (!root) return;
  root.innerHTML = appShell();
  bindApp(root);
  await renderReview(root);
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function bootApp(): Promise<void> {
  await addDiveSenseiListener(DiveSenseiMediaEvents.JobProgress, (payload) => {
    appendEvent("jobProgress", payload);
    void refreshSessions().then(render);
  });
  await addDiveSenseiListener(DiveSenseiMediaEvents.SessionUpdated, (payload) => {
    appendEvent("sessionUpdated", payload);
    void refreshSessions().then(render);
  });
  await addDiveSenseiListener(DiveSenseiMediaEvents.ReviewProxyUpdated, (payload) => {
    appendEvent("reviewProxyUpdated", payload);
  });
  await addDiveSenseiListener(DiveSenseiMediaEvents.ExportsUpdated, (payload) => {
    appendEvent("exportsUpdated", payload);
  });

  const nativeVersion = typeof DiveSenseiMedia.listSessions === "function";
  appendEvent("bridgeReady", { nativeVersion });
  await refreshSessions();
  await render();
}
