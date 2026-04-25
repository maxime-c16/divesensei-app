import type { ReviewHostService } from "@/host/types";
import type {
  ClipPresetName,
  DecisionRecord,
  Detection,
  ReviewDecisionLabel,
  ReviewProxyRecord,
  SessionId,
  SessionManifest,
} from "@/native/types";
import { clipWindowFor, formatSeconds, reviewEndFor, reviewStartFor } from "@/review/model";

type DeckDecision = ReviewDecisionLabel | "pending";

type DeckItem = {
  detection: Detection;
  decision: DeckDecision;
  windowStart: number;
  windowEnd: number;
};

type PersistedDecisionSnapshot = {
  detectionId: string;
  previous: DecisionRecord | null;
  applied: ReviewDecisionLabel;
};

type GestureState = {
  pointerId: number;
  startX: number;
  startY: number;
  dx: number;
  dy: number;
  startedAt: number;
  dragging: boolean;
};

type SessionUiState = {
  pendingIndex: number;
  clipPreset: ClipPresetName;
  lastDecision?: PersistedDecisionSnapshot;
};

const sessionUiState = new Map<SessionId, SessionUiState>();

function uiStateFor(sessionId: SessionId): SessionUiState {
  const existing = sessionUiState.get(sessionId);
  if (existing) return existing;
  const created: SessionUiState = { pendingIndex: 0, clipPreset: "medium" };
  sessionUiState.set(sessionId, created);
  return created;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function decisionTone(label: DeckDecision): string {
  switch (label) {
    case "keep":
      return "pill--success";
    case "reject":
      return "pill--danger";
    case "unsure":
      return "pill--warn";
    default:
      return "pill--soft";
  }
}

function proxyStateNote(proxy: ReviewProxyRecord): string {
  switch (proxy.status) {
    case "pending":
      return "Preparing review video.";
    case "failed":
      return "Review video could not be prepared.";
    default:
      return "Review proxy ready.";
  }
}

function cardOutcomeLabel(dx: number): string {
  if (dx > 14) return "KEEP";
  if (dx < -14) return "REJECT";
  return "REVIEW";
}

export class ReviewWorkspace {
  private destroyed = false;
  private visibilityHandler: (() => void) | null = null;
  private pageHideHandler: (() => void) | null = null;
  private activeVideo: HTMLVideoElement | null = null;
  private loopBounds: { start: number; end: number } | null = null;
  private loopGuardHandler: (() => void) | null = null;
  private rerenderTimer: number | null = null;

  constructor(
    private readonly host: ReviewHostService,
    private readonly root: HTMLElement,
  ) {}

  destroy(): void {
    this.destroyed = true;
    if (this.rerenderTimer !== null) {
      window.clearTimeout(this.rerenderTimer);
      this.rerenderTimer = null;
    }
    this.teardownVideoBindings();
    if (this.visibilityHandler) {
      document.removeEventListener("visibilitychange", this.visibilityHandler);
      this.visibilityHandler = null;
    }
    if (this.pageHideHandler) {
      window.removeEventListener("pagehide", this.pageHideHandler);
      this.pageHideHandler = null;
    }
  }

  async render(sessionId: SessionId): Promise<void> {
    if (this.rerenderTimer !== null) {
      window.clearTimeout(this.rerenderTimer);
      this.rerenderTimer = null;
    }

    const [manifest, decisions, proxyResult] = await Promise.all([
      this.host.getSessionManifest(sessionId),
      this.host.listDecisions(sessionId),
      this.host.getReviewProxy(sessionId).then((response) => ({ ok: true as const, proxy: response.proxy })).catch((error) => ({
        ok: false as const,
        error: error instanceof Error ? error.message : String(error),
      })),
    ]);

    if (this.destroyed) return;

    const decisionMap = new Map(decisions.map((item) => [item.detectionId, item]));
    const ordered = manifest.detections.map((detection) => this.deckItemFromDetection(detection, decisionMap.get(detection.id) ?? null));
    const pending = ordered.filter((item) => item.decision === "pending");
    const decided = ordered.filter((item) => item.decision !== "pending");
    const uiState = uiStateFor(sessionId);
    uiState.pendingIndex = clamp(uiState.pendingIndex, 0, Math.max(pending.length - 1, 0));

    this.root.innerHTML = this.renderMarkup({
      sessionId,
      manifest,
      proxy: proxyResult.ok ? proxyResult.proxy : null,
      proxyError: proxyResult.ok ? null : proxyResult.error,
      pending,
      decided,
      pendingIndex: uiState.pendingIndex,
      decisions,
      canUndo: Boolean(uiState.lastDecision),
    });

    if (proxyResult.ok && proxyResult.proxy.status === "pending") {
      this.rerenderTimer = window.setTimeout(() => {
        void this.render(sessionId);
      }, 1600);
    }

    this.bind(sessionId, manifest, decisions, pending, proxyResult.ok ? proxyResult.proxy : null);
  }

  private deckItemFromDetection(detection: Detection, decision: DecisionRecord | null): DeckItem {
    return {
      detection,
      decision: decision?.label ?? "pending",
      windowStart: reviewStartFor(detection),
      windowEnd: reviewEndFor(detection),
    };
  }

  private renderMarkup(input: {
    sessionId: SessionId;
    manifest: SessionManifest;
    proxy: ReviewProxyRecord | null;
    proxyError: string | null;
    pending: DeckItem[];
    decided: DeckItem[];
    pendingIndex: number;
    decisions: DecisionRecord[];
    canUndo: boolean;
  }): string {
    const {
      manifest,
      proxy,
      proxyError,
      pending,
      pendingIndex,
    } = input;

    const activeItem = pending[pendingIndex] ?? null;
    const selectedPreset = uiStateFor(input.sessionId).clipPreset;
    const selectedWindow = activeItem ? clipWindowFor(activeItem.detection, selectedPreset) : null;

    const activeReadyMarkup = activeItem && proxy?.status === "ready" && proxy.url ? `
      <section class="review-deck review-deck--ready">
        <section class="review-deck__stage">
          <div class="review-stage__backdrop"></div>
          <article
            class="review-card review-card--active"
            data-review-card
            data-detection-id="${escapeHtmlAttribute(activeItem.detection.id)}"
            data-window-start="${selectedWindow?.start ?? activeItem.windowStart}"
            data-window-end="${selectedWindow?.end ?? activeItem.windowEnd}"
          >
            <div class="review-card__decision review-card__decision--reject">Reject</div>
            <div class="review-card__decision review-card__decision--keep">Keep</div>
            <div class="review-card__surface">
              <div class="review-card__video-shell">
                <video
                  class="review-card__video"
                  data-review-video
                  playsinline
                  webkit-playsinline="true"
                  muted
                  autoplay
                  preload="auto"
                  src="${escapeHtmlAttribute(proxy.url)}"
                ></video>
                <div class="review-card__video-overlay">
                  <div class="review-card__video-topline">
                    <span class="review-card__window-pill">${formatSeconds(selectedWindow?.start ?? activeItem.windowStart)} to ${formatSeconds(selectedWindow?.end ?? activeItem.windowEnd)}</span>
                  </div>
                  <span class="review-card__video-state" data-video-state-label>Loading</span>
                </div>
                <div class="review-card__video-caption">
                  <strong>${manifest.session.session_name ?? manifest.session.title}</strong>
                  <span>Audio-anchored ${selectedPreset} clip. Swipe left to reject, right to keep.</span>
                </div>
                <div class="review-card__preset-strip" aria-label="Clip preset">
                  ${(["short", "medium", "long"] as const).map((preset) => `
                    <button
                      class="preset-button ${preset === selectedPreset ? "preset-button--active" : ""}"
                      type="button"
                      data-clip-preset="${preset}"
                    >${preset}</button>
                  `).join("")}
                </div>
                <div class="review-card__quality-strip" aria-label="Clip quality marker">
                  ${["good", "too early", "too late", "too short"].map((quality) => `
                    <button class="quality-button" type="button" data-quality-marker="${escapeHtmlAttribute(quality)}">${escapeHtml(quality)}</button>
                  `).join("")}
                </div>
                <div class="review-card__action-rail">
                  <button class="decision-button decision-button--reject" type="button" data-decision-action="reject">Reject</button>
                  <button class="decision-button decision-button--unsure" type="button" data-decision-action="unsure">Unsure</button>
                  <button class="decision-button decision-button--keep" type="button" data-decision-action="keep">Keep</button>
                </div>
              </div>
            </div>
          </article>
        </section>

        <pre class="review-deck__debug-live review-deck__debug-live--hidden" data-review-debug-live>Awaiting player events.</pre>
      </section>
    ` : "";

    return `
      ${activeReadyMarkup}

      ${proxyError ? `
        <section class="review-deck">
          <section class="review-deck__message review-deck__message--error">
            <h4>Review video unavailable</h4>
            <p>${escapeHtml(proxyError)}</p>
            <div class="review-deck__message-actions">
              <button class="secondary-action" type="button" data-review-retry>Retry proxy</button>
              <button class="secondary-action" type="button" data-repair-source="${escapeHtmlAttribute(manifest.session.id)}">Repair source</button>
            </div>
          </section>
        </section>
      ` : ""}

      ${!activeReadyMarkup && !proxyError && proxy && proxy.status !== "ready" ? `
        <section class="review-deck review-deck--ready review-deck--pending">
          <section class="review-deck__stage">
            <div class="review-stage__backdrop"></div>
            <article class="review-card review-card--active review-card--loading">
              <div class="review-card__surface">
                <div class="review-card__video-shell review-card__video-shell--placeholder">
                  <div class="review-card__video-overlay">
                    <div class="review-card__video-topline">
                      <span class="pill pill--soft">${pending.length} pending</span>
                    </div>
                    <span class="review-card__video-state">${proxy.status === "pending" ? "Preparing review loop" : "Review loop unavailable"}</span>
                  </div>
                  <div class="review-card__video-caption">
                    <strong>${manifest.session.session_name ?? manifest.session.title}</strong>
                    <span>${escapeHtml(proxyStateNote(proxy))}</span>
                  </div>
                </div>
              </div>
            </article>
          </section>

          <div class="review-deck__controls">
            <button class="decision-button decision-button--reject" type="button" disabled>Reject</button>
            <button class="decision-button decision-button--unsure" type="button" disabled>Unsure</button>
            <button class="decision-button decision-button--keep" type="button" disabled>Keep</button>
          </div>

          <div class="review-deck__pending-actions">
            <button class="secondary-action" type="button" data-review-retry>Retry proxy</button>
            ${proxy.status === "failed" ? `<button class="secondary-action" type="button" data-repair-source="${escapeHtmlAttribute(manifest.session.id)}">Repair source</button>` : ""}
          </div>
        </section>
      ` : ""}

        ${!activeItem && !proxyError ? `
          <section class="review-deck__completion">
            <div>
              <h4>Queue complete</h4>
              <p>All attempts in this session have a decision. You can reopen the library, export kept clips, or undo the last swipe.</p>
            </div>
            <div class="review-deck__message-actions">
              <button class="secondary-action" type="button" data-review-retry>Reload</button>
            </div>
          </section>
        ` : ""}
      </section>
    `;
  }

  private bind(
    sessionId: SessionId,
    manifest: SessionManifest,
    decisions: DecisionRecord[],
    pending: DeckItem[],
    proxy: ReviewProxyRecord | null,
  ): void {
    this.root.querySelector<HTMLElement>("[data-review-retry]")?.addEventListener("click", () => {
      void this.render(sessionId);
    });

    this.root.querySelector<HTMLButtonElement>(`[data-repair-source="${sessionId}"]`)?.addEventListener("click", async () => {
      await this.host.repairSource(sessionId);
      await this.render(sessionId);
    });

    this.root.querySelectorAll<HTMLButtonElement>("[data-clip-preset]").forEach((button) => {
      button.addEventListener("click", async () => {
        const preset = button.dataset.clipPreset as ClipPresetName | undefined;
        if (!preset) return;
        uiStateFor(sessionId).clipPreset = preset;
        await this.render(sessionId);
      });
    });

    this.root.querySelectorAll<HTMLButtonElement>("[data-quality-marker]").forEach((button) => {
      button.addEventListener("click", () => {
        this.root.querySelectorAll<HTMLButtonElement>("[data-quality-marker]").forEach((item) => item.classList.remove("quality-button--active"));
        button.classList.add("quality-button--active");
      });
    });

    const undoButton = this.root.querySelector<HTMLButtonElement>("[data-undo-last]");
    if (undoButton) {
      undoButton.addEventListener("click", async () => {
        const uiState = uiStateFor(sessionId);
        const last = uiState.lastDecision;
        if (!last) return;

        undoButton.disabled = true;
        try {
          if (last.previous) {
            await this.host.saveDecision(sessionId, last.detectionId, last.previous.label, last.previous.notes);
          } else {
            await this.host.clearDecision(sessionId, last.detectionId);
          }
          uiState.pendingIndex = Math.max(uiState.pendingIndex - 1, 0);
          uiState.lastDecision = undefined;
          await this.render(sessionId);
        } catch {
          undoButton.disabled = false;
        }
      });
    }

    if (!proxy || proxy.status !== "ready" || !proxy.url || pending.length === 0) {
      return;
    }

    const activeItem = pending[uiStateFor(sessionId).pendingIndex] ?? null;
    const video = this.root.querySelector<HTMLVideoElement>("[data-review-video]");
    const card = this.root.querySelector<HTMLElement>("[data-review-card]");
    if (!video || !card || !activeItem) {
      return;
    }

    const clipWindow = clipWindowFor(activeItem.detection, uiStateFor(sessionId).clipPreset);
    this.bindVideo(video, clipWindow.start, clipWindow.end);

    this.root.querySelectorAll<HTMLButtonElement>("[data-decision-action]").forEach((button) => {
      button.addEventListener("click", () => {
        const label = button.dataset.decisionAction as ReviewDecisionLabel | undefined;
        if (!label) return;
        void this.commitDecision({
          sessionId,
          detectionId: activeItem.detection.id,
          label,
          manifest,
          decisions,
          trigger: button,
          animationDirection: label === "reject" ? -1 : label === "keep" ? 1 : 0,
        });
      });
    });

    this.bindSwipeGesture(card, {
      sessionId,
      detectionId: activeItem.detection.id,
      manifest,
      decisions,
    });

    this.visibilityHandler = () => {
      if (!this.activeVideo) return;
      if (document.hidden) {
        this.activeVideo.pause();
      } else {
        void this.activeVideo.play().catch(() => undefined);
      }
    };
    document.addEventListener("visibilitychange", this.visibilityHandler);

    this.pageHideHandler = () => {
      if (this.activeVideo) {
        this.activeVideo.pause();
      }
    };
    window.addEventListener("pagehide", this.pageHideHandler);
  }

  private bindVideo(video: HTMLVideoElement, start: number, end: number): void {
    this.teardownVideoBindings();
    this.activeVideo = video;
    this.loopBounds = { start, end };
    const stateLabel = this.root.querySelector<HTMLElement>("[data-video-state-label]");

    const setVideoState = (text: string) => {
      if (stateLabel) {
        stateLabel.textContent = text;
        stateLabel.hidden = text.length === 0;
      }
      this.updateReviewDebug(text);
    };

    const beginPlayback = async () => {
      if (!this.activeVideo || !this.loopBounds) return;
      try {
        if (Math.abs(this.activeVideo.currentTime - this.loopBounds.start) > 0.08) {
          this.activeVideo.currentTime = this.loopBounds.start;
        }
      } catch {
        setVideoState("Clip unavailable");
        return;
      }

      try {
        await this.activeVideo.play();
        setVideoState("");
      } catch {
        setVideoState("Tap video to play");
      }
    };

    const guardLoop = () => {
      if (!this.activeVideo || !this.loopBounds) return;
      if (this.activeVideo.currentTime >= this.loopBounds.end - 0.04) {
        this.activeVideo.currentTime = this.loopBounds.start;
        void this.activeVideo.play().catch(() => undefined);
      }
    };

    this.loopGuardHandler = guardLoop;
    video.addEventListener("loadedmetadata", () => {
      this.updateReviewDebug(`loadedmetadata duration=${video.duration.toFixed(3)} size=${video.videoWidth}x${video.videoHeight} readyState=${video.readyState}`);
      void beginPlayback();
    }, { once: true });
    video.addEventListener("loadeddata", () => {
      this.updateReviewDebug(`loadeddata currentTime=${video.currentTime.toFixed(3)} readyState=${video.readyState}`);
      void beginPlayback();
    }, { once: true });
    video.addEventListener("canplay", () => {
      this.updateReviewDebug(`canplay currentTime=${video.currentTime.toFixed(3)} readyState=${video.readyState}`);
      if (video.paused) {
        void beginPlayback();
      }
    });
    video.addEventListener("seeked", () => {
      this.updateReviewDebug(`seeked currentTime=${video.currentTime.toFixed(3)} readyState=${video.readyState}`);
    });
    video.addEventListener("timeupdate", guardLoop);
    video.addEventListener("error", () => {
      const mediaError = video.error ? `${video.error.code}` : "unknown";
      this.updateReviewDebug(`error code=${mediaError} networkState=${video.networkState} readyState=${video.readyState}`);
      setVideoState("Review clip failed to load");
    });
    video.addEventListener("click", () => {
      void beginPlayback();
    });

    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      void beginPlayback();
    } else {
      setVideoState("Loading review loop");
    }

    video.load();
  }

  private updateReviewDebug(text: string): void {
    const node = this.root.querySelector<HTMLElement>("[data-review-debug-live]");
    if (!node) return;
    node.textContent = text;
  }

  private teardownVideoBindings(): void {
    if (this.activeVideo && this.loopGuardHandler) {
      this.activeVideo.removeEventListener("timeupdate", this.loopGuardHandler);
    }
    if (this.activeVideo) {
      this.activeVideo.pause();
    }
    this.activeVideo = null;
    this.loopBounds = null;
    this.loopGuardHandler = null;
  }

  private bindSwipeGesture(
    card: HTMLElement,
    input: {
      sessionId: SessionId;
      detectionId: string;
      manifest: SessionManifest;
      decisions: DecisionRecord[];
    },
  ): void {
    const state: GestureState = {
      pointerId: -1,
      startX: 0,
      startY: 0,
      dx: 0,
      dy: 0,
      startedAt: 0,
      dragging: false,
    };

    const updateCard = () => {
      const rotate = state.dx / 20;
      card.style.setProperty("--review-card-offset-x", `${state.dx}px`);
      card.style.setProperty("--review-card-offset-y", `${Math.min(Math.max(state.dy * 0.1, -8), 8)}px`);
      card.style.setProperty("--review-card-rotate", `${rotate}deg`);
      card.dataset.dragState = state.dx > 18 ? "keep" : state.dx < -18 ? "reject" : "idle";
      const label = card.querySelector<HTMLElement>(".review-card__video-state");
      if (label) {
        label.textContent = cardOutcomeLabel(state.dx);
      }
    };

    const resetCard = () => {
      state.pointerId = -1;
      state.dragging = false;
      state.dx = 0;
      state.dy = 0;
      card.classList.remove("is-animating-out");
      card.dataset.dragState = "idle";
      card.style.removeProperty("--review-card-offset-x");
      card.style.removeProperty("--review-card-offset-y");
      card.style.removeProperty("--review-card-rotate");
      const label = card.querySelector<HTMLElement>(".review-card__video-state");
      if (label) {
        label.textContent = "REVIEW";
      }
    };

    card.addEventListener("pointerdown", (event) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("button")) return;
      state.pointerId = event.pointerId;
      state.startX = event.clientX;
      state.startY = event.clientY;
      state.dx = 0;
      state.dy = 0;
      state.startedAt = performance.now();
      state.dragging = true;
      card.setPointerCapture(event.pointerId);
    });

    card.addEventListener("pointermove", (event) => {
      if (!state.dragging || event.pointerId !== state.pointerId) return;
      state.dx = event.clientX - state.startX;
      state.dy = event.clientY - state.startY;
      updateCard();
    });

    const finish = async () => {
      if (!state.dragging) return;
      const elapsed = Math.max(performance.now() - state.startedAt, 1);
      const velocity = state.dx / elapsed;
      const label = state.dx > 88 || velocity > 0.65
        ? "keep"
        : state.dx < -88 || velocity < -0.65
          ? "reject"
          : null;

      if (!label) {
        resetCard();
        return;
      }

      card.classList.add("is-animating-out");
      card.style.setProperty("--review-card-offset-x", `${label === "keep" ? 440 : -440}px`);
      card.style.setProperty("--review-card-offset-y", `${Math.min(Math.max(state.dy * 0.12, -18), 18)}px`);
      card.style.setProperty("--review-card-rotate", `${label === "keep" ? 14 : -14}deg`);

      window.setTimeout(() => {
        void this.commitDecision({
          sessionId: input.sessionId,
          detectionId: input.detectionId,
          label,
          manifest: input.manifest,
          decisions: input.decisions,
          animationDirection: label === "keep" ? 1 : -1,
        });
      }, 150);
      state.dragging = false;
    };

    card.addEventListener("pointerup", (event) => {
      if (event.pointerId !== state.pointerId) return;
      void finish();
    });

    card.addEventListener("pointercancel", () => {
      resetCard();
    });
  }

  private async commitDecision(input: {
    sessionId: SessionId;
    detectionId: string;
    label: ReviewDecisionLabel;
    manifest: SessionManifest;
    decisions: DecisionRecord[];
    trigger?: HTMLButtonElement;
    animationDirection: number;
  }): Promise<void> {
    const { sessionId, detectionId, label, decisions } = input;
    const uiState = uiStateFor(sessionId);
    const previous = decisions.find((item) => item.detectionId === detectionId) ?? null;

    if (input.trigger) {
      input.trigger.disabled = true;
    }

    try {
      await this.host.saveDecision(sessionId, detectionId, label, "");
      uiState.lastDecision = { detectionId, previous, applied: label };
      await this.render(sessionId);
    } catch (error) {
      const stage = this.root.querySelector<HTMLElement>(".review-deck__toolbar p");
      if (stage) {
        stage.textContent = error instanceof Error ? error.message : String(error);
      }
      if (input.trigger) {
        input.trigger.disabled = false;
      }
      const card = this.root.querySelector<HTMLElement>("[data-review-card]");
      if (card) {
        card.classList.remove("is-animating-out");
        card.dataset.dragState = "idle";
        card.style.removeProperty("--review-card-offset-x");
        card.style.removeProperty("--review-card-offset-y");
        card.style.removeProperty("--review-card-rotate");
      }
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
