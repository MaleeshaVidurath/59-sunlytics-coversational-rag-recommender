import { request, requestJson } from "./http";

/**
 * Session calls.
 *
 * user_id is deliberately absent from every signature: the server derives it
 * from the caller's access token. Passing one used to be how any client could
 * read or delete another user's chats.
 */

export function getSessions() {
  return requestJson("/api/sessions", { retries: 2 });
}

export function getSessionHistory(sessionId) {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`, { retries: 2 });
}

/** Not retried: a repeated DELETE would race the list reload that follows it. */
export function deleteSession(sessionId) {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

/**
 * Clears the server's active-session pointer so the next message opens a fresh
 * session. Failure is non-fatal — the next send passes force_new anyway.
 */
export async function startNewSession() {
  await request("/api/sessions/new", { method: "POST" });
}
