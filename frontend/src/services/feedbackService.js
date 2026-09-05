import { request } from "./http";

/**
 * Records a thumbs up/down for the RL signal collector.
 *
 * Deliberately swallows failures: the rating is already reflected optimistically
 * in the UI, and a lost training signal must not surface as a chat error. This is
 * the one endpoint where silence is the correct behaviour — everywhere else an
 * error reaches the user.
 */
export async function submitFeedback({ sessionId, userId, recommendationId, turnId, rating, articleIds }) {
  try {
    await request("/api/rl/feedback", {
      method: "POST",
      body: {
        session_id:        sessionId,
        user_id:           userId,
        recommendation_id: recommendationId || "",
        turn_id:           turnId || "",
        rating,
        article_ids:       articleIds || [],
      },
    });
  } catch (e) {
    console.warn("RL feedback submission failed:", e.userMessage ?? e);
  }
}
