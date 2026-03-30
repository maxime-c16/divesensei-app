import type { ReviewHostService } from "@/host/types";
import type { DecisionRecord, SessionId, SessionManifest } from "@/native/types";
import { formatSeconds, reviewEndFor, reviewStartFor } from "@/review/model";

export class ReviewWorkspace {
  constructor(
    private readonly host: ReviewHostService,
    private readonly root: HTMLElement,
  ) {}

  async render(sessionId: SessionId): Promise<void> {
    const [manifest, decisions] = await Promise.all([
      this.host.getSessionManifest(sessionId),
      this.host.listDecisions(sessionId),
    ]);
    this.root.innerHTML = this.renderMarkup(manifest, decisions);
    this.bindDecisionButtons(sessionId, manifest);
  }

  private renderMarkup(manifest: SessionManifest, decisions: DecisionRecord[]): string {
    const decisionMap = new Map(decisions.map((item) => [item.detectionId, item]));
    return `
      <section class="card">
        <h2>Review</h2>
        <p class="muted">${manifest.session.session_name ?? manifest.session.title}</p>
        <div class="review-grid">
          <div>
            <strong>Status</strong>
            <div>${manifest.session.status}</div>
          </div>
          <div>
            <strong>Attempts</strong>
            <div>${manifest.detections.length}</div>
          </div>
        </div>
        <table class="review-table">
          <thead>
            <tr>
              <th>Attempt</th>
              <th>Splash</th>
              <th>Window</th>
              <th>Decision</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            ${manifest.detections.map((item) => {
              const decision = decisionMap.get(item.id);
              return `
                <tr>
                  <td>${item.id}</td>
                  <td>${formatSeconds(item.timestamp_seconds)}</td>
                  <td>${formatSeconds(reviewStartFor(item))} to ${formatSeconds(reviewEndFor(item))}</td>
                  <td>${decision?.label ?? "pending"}</td>
                  <td class="actions-cell">
                    <button data-decision="keep" data-detection-id="${item.id}">Keep</button>
                    <button data-decision="reject" data-detection-id="${item.id}">Reject</button>
                    <button data-decision="unsure" data-detection-id="${item.id}">Unsure</button>
                  </td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      </section>
    `;
  }

  private bindDecisionButtons(sessionId: SessionId, manifest: SessionManifest): void {
    this.root.querySelectorAll<HTMLButtonElement>("[data-decision]").forEach((button) => {
      button.addEventListener("click", async () => {
        const label = button.dataset.decision;
        const detectionId = button.dataset.detectionId;
        if (!label || !detectionId) return;
        await this.host.saveDecision(sessionId, detectionId, label as "keep" | "reject" | "unsure");
        await this.render(manifest.session.id);
      });
    });
  }
}
