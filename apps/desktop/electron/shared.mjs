import { setTimeout as delay } from "node:timers/promises";

export async function waitForUrl(url, timeoutMs = 30000) {
  const start = Date.now();
  let lastError = null;

  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(url, { redirect: "manual" });
      if (response.ok || response.status < 500) {
        return;
      }
      lastError = new Error(`Unexpected status ${response.status}`);
    } catch (error) {
      lastError = error;
    }

    await delay(400);
  }

  throw lastError ?? new Error(`Timed out waiting for ${url}`);
}

export async function requestJson(baseUrl, pathname, init) {
  const response = await fetch(new URL(pathname, baseUrl), init);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error ?? `Request failed: ${response.status}`);
  }
  return payload;
}
