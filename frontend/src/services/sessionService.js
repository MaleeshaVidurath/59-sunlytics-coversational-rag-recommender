import { request, requestJson } from "./http";

export function getSessions(userId) {
  return requestJson(`/api/sessions?user_id=${encodeURIComponent(userId)}`, { retries: 2 });
}

export function getSessionHistory(sessionId, userId) {
  return requestJson(
    `/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
    { retries: 2 },
  );
}

/** Not retried: a repeated DELETE would race the list reload that follows it. */
export function deleteSession(sessionId, userId) {
  return requestJson(
    `/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" },
  );
}

/**
 * Clears the Redis active-session pointer so the next message opens a fresh
 * session. Failure is non-fatal — the next send passes force_new anyway — so
 * this resolves either way and lets the caller decide whether to care.
 */
export async function startNewSession(userId) {
  await request(`/api/sessions/new?user_id=${encodeURIComponent(userId)}`, { method: "POST" });
}
