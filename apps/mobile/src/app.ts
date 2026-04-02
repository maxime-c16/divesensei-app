import { addDiveSenseiListener, DiveSenseiMedia } from "@/native/client";
import { MobileReviewHostService } from "@/host/mobileHost";
import {
  type DetectorId,
  DiveSenseiMediaEvents,
  type SessionLibraryItem,
  type SessionProfile,
  type SourceSummary,
} from "@/native/types";
import { ReviewWorkspace } from "@/review/ReviewWorkspace";
import brandLockupDark from "../../desktop/src/assets/divesensei-lockup-dark.svg";
import brandMarkMono from "../../desktop/src/assets/divesensei-mark-mono.svg";

const host = new MobileReviewHostService();
let activeReviewWorkspace: ReviewWorkspace | null = null;

const tabs = [
  { id: "create", label: "Create" },
  { id: "review", label: "Review" },
  { id: "exports", label: "Exports" },
  { id: "library", label: "Library" },
] as const;

type MobileTab = (typeof tabs)[number]["id"];

type AppState = {
  activeTab: MobileTab;
  source: SourceSummary | null;
  sessions: SessionLibraryItem[];
  selectedSessionId: string | null;
  draftSessionName: string;
  draftProfile: SessionProfile;
  draftDetectorId: DetectorId;
  eventLog: Array<{ type: string; payload: unknown }>;
  lastError: string | null;
};

const state: AppState = {
  activeTab: "create",
  source: null,
  sessions: [],
  selectedSessionId: null,
  draftSessionName: "",
  draftProfile: "long-session",
  draftDetectorId: "audio_v2_pcen_classifier",
  eventLog: [],
  lastError: null,
};

function parseHashRoute(): { tab?: MobileTab; sessionId?: string } {
  const raw = window.location.hash.replace(/^#/, "").trim();
  if (!raw) return {};
  const [tabPart, sessionId] = raw.split("/");
  if (tabs.some((tab) => tab.id === tabPart)) {
    return { tab: tabPart as MobileTab, sessionId: sessionId || undefined };
  }
  return {};
}

function syncHashRoute(): void {
  const next = `#${state.activeTab}${state.selectedSessionId ? `/${state.selectedSessionId}` : ""}`;
  if (window.location.hash !== next) {
    history.replaceState(null, "", next);
  }
}

function applyHashRoute(): void {
  const route = parseHashRoute();
  if (route.tab) {
    state.activeTab = route.tab;
  }
  if (route.sessionId && state.sessions.some((session) => session.sessionId === route.sessionId)) {
    state.selectedSessionId = route.sessionId;
  }
}

function appendEvent(type: string, payload: unknown): void {
  state.eventLog = [{ type, payload }, ...state.eventLog].slice(0, 10);
  patchDiagnostics();
}

function diagnosticsMarkup(): string {
  return state.eventLog.map((entry) => `
    <article class="event-chip">
      <strong>${escapeHtml(entry.type)}</strong>
      <pre>${escapeHtml(JSON.stringify(entry.payload, null, 2))}</pre>
    </article>
  `).join("") || "<p class='supporting-copy'>No events yet.</p>";
}

function patchDiagnostics(): void {
  const eventLog = document.querySelector<HTMLElement>(".event-log");
  if (!eventLog) return;
  eventLog.innerHTML = diagnosticsMarkup();
}

function selectedSession(): SessionLibraryItem | null {
  return state.sessions.find((session) => session.sessionId === state.selectedSessionId) ?? null;
}

function sourceDisplayBaseName(source: SourceSummary): string {
  return source.displayName.replace(/\.[^.]+$/, "");
}

async function refreshSessions(): Promise<void> {
  const response = await host.listSessions();
  state.sessions = response.sessions;
  if (!state.selectedSessionId && response.sessions[0]) {
    state.selectedSessionId = response.sessions[0].sessionId;
  }
  if (state.selectedSessionId && !response.sessions.some((session) => session.sessionId === state.selectedSessionId)) {
    state.selectedSessionId = response.sessions[0]?.sessionId ?? null;
  }
}

function formatDate(value?: string | null): string {
  if (!value) return "Just now";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-GB", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

function statusLabel(status: SessionLibraryItem["status"]): string {
  switch (status) {
    case "created":
      return "Created";
    case "analyzing":
      return "Analyzing";
    case "review_pending":
      return "Review pending";
    case "review_ready":
      return "Review ready";
    case "exporting":
      return "Exporting";
    case "complete_with_errors":
      return "Complete with errors";
    case "failed":
      return "Failed";
    case "deleted":
      return "Deleted";
    default:
      return status;
  }
}

function availabilityLabel(availability: SessionLibraryItem["sourceAvailability"] | SourceSummary["availability"]): string {
  switch (availability) {
    case "available":
      return "Ready";
    case "missing":
      return "Missing";
    case "needs_download":
      return "Needs download";
    case "permission_denied":
      return "Permission denied";
    case "unsupported":
      return "Unsupported";
    default:
      return availability;
  }
}

function detectorLabel(detectorId: DetectorId): string {
  switch (detectorId) {
    case "audio_v1_heuristic":
      return "Legacy audio";
    case "audio_v2_hybrid_video":
      return "Audio + video";
    default:
      return "Standard";
  }
}

function profileLabel(profile: SessionProfile): string {
  return profile === "reviewed" ? "Reviewed" : "Training";
}

function emptyState(title: string, note: string, actionLabel?: string, actionTab?: MobileTab): string {
  return `
    <section class="empty-state">
      <img class="empty-state__mark" src="${brandMarkMono}" alt="" aria-hidden="true" />
      <h2>${escapeHtml(title)}</h2>
      <p>${escapeHtml(note)}</p>
      ${actionLabel && actionTab ? `<button class="secondary-action" type="button" data-set-tab="${actionTab}">${escapeHtml(actionLabel)}</button>` : ""}
    </section>
  `;
}

function renderScreenHeader(title: string, note: string, meta: string[] = []): string {
  return `
    <header class="screen-header">
      <div>
        <h1>${escapeHtml(title)}</h1>
        <p>${escapeHtml(note)}</p>
      </div>
      ${meta.length > 0 ? `
        <div class="screen-header__meta">
          ${meta.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
        </div>
      ` : ""}
    </header>
  `;
}

function renderCreatePanel(): string {
  const currentSession = selectedSession();
  const source = state.source;

  return `
    ${renderScreenHeader(
      "Create session",
      "Pick a source, set the detector, then prepare the review queue.",
      [
        `${state.sessions.length} sessions`,
        `${state.sessions.filter((session) => session.status === "review_ready").length} ready`,
      ],
    )}

    <section class="panel-card source-panel">
      <div class="panel-card__header">
        <div>
          <h2>${source ? escapeHtml(source.displayName) : "Pick a dive video"}</h2>
        </div>
        <button class="primary-action" id="pick-source" type="button">${source ? "Change source" : "Pick source"}</button>
      </div>
      ${source ? `
        <div class="source-summary">
          <div class="source-summary__meta">
            <span class="pill pill--soft">${escapeHtml(source.origin)}</span>
            <span class="pill">${escapeHtml(availabilityLabel(source.availability))}</span>
          </div>
          <dl class="key-grid">
            <div><dt>Duration</dt><dd>${source.durationSeconds ? `${Math.round(source.durationSeconds)}s` : "Unknown"}</dd></div>
            <div><dt>Size</dt><dd>${source.fileSizeBytes ? formatBytes(source.fileSizeBytes) : "Unknown"}</dd></div>
            <div><dt>Persistence</dt><dd>${source.canPersist ? "Native" : "Temporary"}</dd></div>
          </dl>
        </div>
      ` : `
        <p class="supporting-copy">Use the native Photos picker. Once a source is chosen, the session form unlocks below.</p>
      `}
      ${state.lastError ? `<p class="inline-error">${escapeHtml(state.lastError)}</p>` : ""}
    </section>

    <section class="panel-card">
      <div class="panel-card__header">
        <div>
          <h2>Analysis setup</h2>
        </div>
      </div>
      <label class="field">
        <span>Session name</span>
        <input id="session-name" type="text" value="${escapeHtmlAttribute(state.draftSessionName)}" placeholder="Morning training set" ${source ? "" : "disabled"} />
      </label>
      <div class="stack">
        <div>
          <span class="field-label">Session type</span>
          <div class="segmented">
            ${(["long-session", "reviewed"] as const).map((profile) => `
              <button
                type="button"
                class="segmented__item ${state.draftProfile === profile ? "is-active" : ""}"
                data-profile="${profile}"
                ${source ? "" : "disabled"}
              >
                ${profileLabel(profile)}
              </button>
            `).join("")}
          </div>
        </div>
        <div>
          <span class="field-label">Detector</span>
          <div class="segmented segmented--detectors">
            ${([
              "audio_v2_pcen_classifier",
              "audio_v1_heuristic",
              "audio_v2_hybrid_video",
            ] as const).map((detectorId) => `
              <button
                type="button"
                class="segmented__item ${state.draftDetectorId === detectorId ? "is-active" : ""}"
                data-detector="${detectorId}"
                ${source ? "" : "disabled"}
              >
                ${escapeHtml(detectorLabel(detectorId))}
              </button>
            `).join("")}
          </div>
        </div>
      </div>
      <div class="action-cluster">
        <button class="primary-action" id="create-session" type="button" ${source ? "" : "disabled"}>Create session</button>
        <button class="secondary-action" id="start-analysis" type="button" ${currentSession ? "" : "disabled"}>Start analysis</button>
      </div>
      <p class="supporting-copy">${currentSession
        ? `Selected session: ${escapeHtml(currentSession.sessionName)} · ${escapeHtml(statusLabel(currentSession.status))}`
        : "Create a session first, then analysis can begin."}</p>
    </section>

    <section class="panel-card checklist-card">
      <div class="panel-card__header">
        <div>
          <h2>What happens next</h2>
        </div>
        <button class="ghost-link" type="button" data-set-tab="library">Open library</button>
      </div>
      <div class="checklist-grid">
        <article><strong>1</strong><p>Pick the source video from Photos.</p></article>
        <article><strong>2</strong><p>Create the session with the right detector.</p></article>
        <article><strong>3</strong><p>Let analysis prepare the reviewable attempts.</p></article>
        <article><strong>4</strong><p>Open Review to keep, reject, and export.</p></article>
      </div>
    </section>
  `;
}

function renderReviewPanel(): string {
  const currentSession = selectedSession();
  if (!currentSession) {
    return emptyState("No session selected", "Create or choose a session before opening Review.", "Go to Create", "create");
  }

  return `
    <section class="review-panel">
      <div id="review-host" class="review-host"></div>
    </section>
  `;
}

function renderExportsPanel(): string {
  const currentSession = selectedSession();
  if (!currentSession) {
    return emptyState("Nothing to export yet", "Exports are attached to a selected session. Pick one from Library or create a new one first.", "Go to Library", "library");
  }

  return `
    ${renderScreenHeader(
      "Exports",
      "This tab will become the mobile export workspace once review flow is stable.",
      [
        `${currentSession.keptCount ?? 0} kept`,
        `${currentSession.exportCount ?? 0} exports`,
      ],
    )}
    <section class="panel-card">
      <div class="panel-card__header">
        <div>
          <h2>Export flow comes after review</h2>
        </div>
      </div>
      <p class="supporting-copy">Keep using Review to decide what stays. Once export is implemented here, this tab will become the mobile destination for clip output, destinations, and saved artifacts.</p>
      <div class="key-grid">
        <div><dt>Session status</dt><dd>${escapeHtml(statusLabel(currentSession.status))}</dd></div>
        <div><dt>Pending review</dt><dd>${Math.max((currentSession.candidateCount ?? 0) - ((currentSession.keptCount ?? 0) + (currentSession.rejectCount ?? 0) + (currentSession.unsureCount ?? 0)), 0)}</dd></div>
      </div>
    </section>
  `;
}

function renderLibraryPanel(): string {
  const currentSession = selectedSession();

  return `
    ${renderScreenHeader(
      "Library",
      "Reopen sessions, switch context fast, and jump back into review.",
      [
        `${state.sessions.length} saved`,
        `${state.sessions.filter((session) => session.status === "analyzing").length} running`,
      ],
    )}

    <section class="panel-card">
      <div class="panel-card__header">
        <div>
          <h2>${state.sessions.length > 0 ? "Your recent sessions" : "No sessions yet"}</h2>
        </div>
      </div>
      ${state.sessions.length > 0 ? `
        <div class="session-stack">
          ${state.sessions.map((session) => `
            <article class="session-card ${session.sessionId === state.selectedSessionId ? "is-current" : ""}" data-session-id="${session.sessionId}">
              <button class="session-card__main" type="button" data-session-id="${session.sessionId}">
                <div class="session-card__topline">
                  <strong>${escapeHtml(session.sessionName)}</strong>
                  <span class="pill ${session.status === "review_ready" ? "pill--success" : ""}">${escapeHtml(statusLabel(session.status))}</span>
                </div>
                <p>${escapeHtml(session.sourceDisplayName ?? "Unknown source")}</p>
                <div class="session-card__meta">
                  <span>${escapeHtml(detectorLabel(session.detectorId))}</span>
                  <span>${escapeHtml(profileLabel(session.profile))}</span>
                  <span>${escapeHtml(availabilityLabel(session.sourceAvailability))}</span>
                </div>
                <div class="session-card__stats">
                  <span>${session.candidateCount ?? 0} attempts</span>
                  <span>${session.keptCount ?? 0} kept</span>
                  <span>${session.exportCount ?? 0} exports</span>
                  <span>${escapeHtml(formatDate(session.updatedAt))}</span>
                </div>
              </button>
              <div class="session-card__actions">
                <button class="secondary-action" type="button" data-open-review="${session.sessionId}">Review</button>
                <button class="secondary-action" type="button" data-start-analysis-for="${session.sessionId}">Analyze</button>
              </div>
            </article>
          `).join("")}
        </div>
      ` : emptyState("Create your first session", "Pick a source in Create to start building the library.", "Go to Create", "create")}
    </section>

    ${currentSession ? `
      <section class="panel-card">
        <div class="panel-card__header">
          <div>
            <h2>${escapeHtml(currentSession.sessionName)}</h2>
          </div>
          <button class="ghost-link" type="button" data-open-review="${currentSession.sessionId}">Open review</button>
        </div>
        <div class="key-grid">
          <div><dt>Status</dt><dd>${escapeHtml(statusLabel(currentSession.status))}</dd></div>
          <div><dt>Source</dt><dd>${escapeHtml(currentSession.sourceDisplayName ?? "Unknown")}</dd></div>
          <div><dt>Updated</dt><dd>${escapeHtml(formatDate(currentSession.updatedAt))}</dd></div>
          <div><dt>Exports</dt><dd>${currentSession.exportCount ?? 0}</dd></div>
        </div>
      </section>
    ` : ""}
  `;
}

function renderDiagnostics(): string {
  return `
    <section class="diagnostics-card">
      <div class="panel-card__header">
        <div>
          <h2>Diagnostics</h2>
        </div>
      </div>
      ${state.lastError ? `<p class="inline-error">${escapeHtml(state.lastError)}</p>` : "<p class='supporting-copy'>Native bridge is connected. Recent events appear here while the mobile shell is still evolving.</p>"}
      <div class="event-log">
        ${diagnosticsMarkup()}
      </div>
    </section>
  `;
}

function activePanel(): string {
  switch (state.activeTab) {
    case "review":
      return renderReviewPanel();
    case "exports":
      return renderExportsPanel();
    case "library":
      return renderLibraryPanel();
    default:
      return renderCreatePanel();
  }
}

function appShell(): string {
  const currentSession = selectedSession();
  const selectedLabel = currentSession ? currentSession.sessionName : "No session";
  return `
    <main class="mobile-app-shell ${state.activeTab === "review" ? "mobile-app-shell--review" : ""}">
      <div class="app-topbar">
        <div class="app-topbar__brand">
          <img class="app-topbar__brand-image" src="${brandLockupDark}" alt="DiveSensei" />
          <strong>${escapeHtml(tabs.find((tab) => tab.id === state.activeTab)?.label ?? "Create")}</strong>
        </div>
        <div class="app-topbar__status">
          <span>${state.sessions.length} sessions</span>
          <span class="app-topbar__selected">${escapeHtml(selectedLabel)}</span>
        </div>
      </div>

      <section class="app-stage ${state.activeTab === "review" ? "app-stage--review" : ""}">
        ${activePanel()}
        ${state.activeTab === "create" || state.lastError ? renderDiagnostics() : ""}
      </section>

      <nav class="thumb-nav" aria-label="Primary">
        ${tabs.map((tab) => `
          <button
            class="thumb-nav__item ${state.activeTab === tab.id ? "is-active" : ""}"
            type="button"
            data-set-tab="${tab.id}"
          >
            <span class="thumb-nav__indicator" aria-hidden="true"></span>
            <span class="thumb-nav__label">${escapeHtml(tab.label)}</span>
          </button>
        `).join("")}
      </nav>
    </main>
  `;
}

function bindApp(root: HTMLElement): void {
  root.querySelectorAll<HTMLButtonElement>("[data-set-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTab = (button.dataset.setTab as MobileTab) ?? "create";
      void render();
    });
  });

  root.querySelector<HTMLInputElement>("#session-name")?.addEventListener("input", (event) => {
    state.draftSessionName = (event.currentTarget as HTMLInputElement).value;
  });

  root.querySelectorAll<HTMLButtonElement>("[data-profile]").forEach((button) => {
    button.addEventListener("click", () => {
      state.draftProfile = (button.dataset.profile as SessionProfile) ?? "long-session";
      void render();
    });
  });

  root.querySelectorAll<HTMLButtonElement>("[data-detector]").forEach((button) => {
    button.addEventListener("click", () => {
      state.draftDetectorId = (button.dataset.detector as DetectorId) ?? "audio_v2_pcen_classifier";
      void render();
    });
  });

  root.querySelector<HTMLButtonElement>("#pick-source")?.addEventListener("click", async () => {
    try {
      state.lastError = null;
      const source = await host.pickSourceVideo();
      state.source = source;
      if (source && !state.draftSessionName) {
        state.draftSessionName = sourceDisplayBaseName(source);
      }
      await render();
    } catch (error) {
      state.lastError = formatError(error);
      await render();
    }
  });

  root.querySelector<HTMLButtonElement>("#create-session")?.addEventListener("click", async () => {
    if (!state.source) return;
    try {
      state.lastError = null;
      const created = await host.createSession({
        sourceRef: state.source.sourceRef,
        sessionName: state.draftSessionName || sourceDisplayBaseName(state.source),
        profile: state.draftProfile,
        detectorId: state.draftDetectorId,
      });
      state.selectedSessionId = created.sessionId;
      state.activeTab = "library";
      await refreshSessions();
      await render();
    } catch (error) {
      state.lastError = formatError(error);
      await render();
    }
  });

  root.querySelector<HTMLButtonElement>("#start-analysis")?.addEventListener("click", async () => {
    const sessionId = state.selectedSessionId;
    if (!sessionId) return;
    await runAnalysis(sessionId);
  });

  root.querySelectorAll<HTMLButtonElement>("[data-session-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedSessionId = button.dataset.sessionId ?? null;
      void render();
    });
  });

  root.querySelectorAll<HTMLButtonElement>("[data-open-review]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedSessionId = button.dataset.openReview ?? null;
      state.activeTab = "review";
      void render();
    });
  });

  root.querySelectorAll<HTMLButtonElement>("[data-start-analysis-for]").forEach((button) => {
    button.addEventListener("click", async () => {
      const sessionId = button.dataset.startAnalysisFor;
      if (!sessionId) return;
      state.selectedSessionId = sessionId;
      await runAnalysis(sessionId);
    });
  });

  root.querySelectorAll<HTMLButtonElement>("[data-repair-source]").forEach((button) => {
    button.addEventListener("click", async () => {
      const sessionId = button.dataset.repairSource;
      if (!sessionId) return;

      try {
        state.lastError = null;
        const response = await host.repairSource(sessionId);
        state.selectedSessionId = sessionId;
        if (response.source) {
          state.source = response.source;
        }
        await refreshSessions();
        await render();
      } catch (error) {
        state.lastError = formatError(error);
        await render();
      }
    });
  });
}

async function runAnalysis(sessionId: string): Promise<void> {
  try {
    state.lastError = null;
    await host.startAnalysis(sessionId);
    state.activeTab = "review";
    await refreshSessions();
    await render();
  } catch (error) {
    state.lastError = formatError(error);
    await render();
  }
}

async function renderReview(root: HTMLElement): Promise<void> {
  const reviewRoot = root.querySelector<HTMLElement>("#review-host");
  if (!reviewRoot) return;
  if (state.activeTab !== "review" || !state.selectedSessionId) {
    activeReviewWorkspace?.destroy();
    activeReviewWorkspace = null;
    reviewRoot.innerHTML = "";
    return;
  }

  activeReviewWorkspace?.destroy();
  const review = new ReviewWorkspace(host, reviewRoot);
  activeReviewWorkspace = review;
  await review.render(state.selectedSessionId);
}

async function render(): Promise<void> {
  const root = document.querySelector<HTMLElement>("#app");
  if (!root) return;
  activeReviewWorkspace?.destroy();
  activeReviewWorkspace = null;
  syncHashRoute();
  root.innerHTML = appShell();
  bindApp(root);

  try {
    await renderReview(root);
  } catch (error) {
    state.lastError = formatError(error);
    const reviewRoot = root.querySelector<HTMLElement>("#review-host");
    if (reviewRoot) {
      reviewRoot.innerHTML = `
        <section class="empty-state">
          <h2>Review failed to load</h2>
          <p>${escapeHtml(state.lastError)}</p>
        </section>
      `;
    }
  }
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeHtmlAttribute(value: string): string {
  return escapeHtml(value).replace(/"/g, "&quot;");
}

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function bootApp(): Promise<void> {
  try {
    const listenerResults = await Promise.allSettled([
      addDiveSenseiListener(DiveSenseiMediaEvents.JobProgress, (payload) => {
        appendEvent("jobProgress", payload);
        void refreshSessions().then(render);
      }),
      addDiveSenseiListener(DiveSenseiMediaEvents.SessionUpdated, (payload) => {
        appendEvent("sessionUpdated", payload);
        void refreshSessions().then(render);
      }),
      addDiveSenseiListener(DiveSenseiMediaEvents.ReviewProxyUpdated, (payload) => {
        appendEvent("reviewProxyUpdated", payload);
      }),
      addDiveSenseiListener(DiveSenseiMediaEvents.ExportsUpdated, (payload) => {
        appendEvent("exportsUpdated", payload);
      }),
    ]);

    const listenerFailures = listenerResults
      .filter((result): result is PromiseRejectedResult => result.status === "rejected")
      .map((result) => formatError(result.reason));

    if (listenerFailures.length > 0) {
      appendEvent("bridgeListenerWarning", { listenerFailures });
    }

    const nativeVersion = typeof DiveSenseiMedia.listSessions === "function";
    appendEvent("bridgeReady", { nativeVersion });
    await refreshSessions();
    applyHashRoute();
  } catch (error) {
    state.lastError = formatError(error);
  }

  await render();
}
