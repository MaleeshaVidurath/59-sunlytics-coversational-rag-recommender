import { requestJson } from "./http";

/**
 * Sends one conversational turn.
 *
 * Deliberately never retried: the turn is not idempotent, and a retry after a
 * timeout would append a second message the user never typed.
 */
export function sendMessage({ userId, customerId, message, sessionId, forceNew, selectedModel }) {
  return requestJson("/api/chat", {
    method: "POST",
    body: {
      user_id:           userId,
      customer_id:       customerId,
      message,
      session_id:        sessionId || null,
      force_new_session: forceNew || false,
      selected_model:    selectedModel || "m3",
    },
  });
}
