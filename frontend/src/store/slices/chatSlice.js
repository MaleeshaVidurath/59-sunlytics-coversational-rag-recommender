import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { getSessionHistory, startNewSession } from "../../services/sessionService";
import { sendMessage } from "../../services/chatService";
import { submitFeedback } from "../../services/feedbackService";
import { CONSENT_TRIGGER } from "../../utils/constants";
import { toApiError } from "../../services/ApiError";
import {
  activeSessionSet, activeSessionCleared, refreshSessionsUntilPresent,
} from "./sessionsSlice";
import { logout, sessionExpired } from "./authSlice";

const now = () => new Date().toISOString();

/** Rehydrates a stored transcript into the shape the UI renders. */
function mapHistoryMessages(data, sessionId) {
  return (data.messages || []).map(m => ({
    id: m.turn_id || Math.random(),
    role: m.role,
    content: m.content,
    timestamp: m.timestamp,
    label: m.label,
    // Restore the product cards and their "why this for you" reasons so a
    // reopened chat looks the same as when it was live.
    items: m.items_recommended || [],
    // Catalogue values that changed after this message quoted them.
    corrections: m.corrections || [],
    recommendation_id: m.recommendation_id || null,
    turn_id: m.turn_id || null,
    session_id: sessionId,
    isConsentQuestion: m.role === "assistant"
      && (m.content || "").includes(CONSENT_TRIGGER),
  }));
}

/** Builds an assistant turn from a /api/chat response. */
function buildBotMessage(res) {
  return {
    id: Date.now() + 1,
    role: "assistant",
    content: res.response_text,
    timestamp: now(),
    label: res.label,
    confidence: res.confidence,
    items: res.items_recommended || [],
    hallucination_flag: res.hallucination_flag,
    contradiction_found: res.contradiction_found,
    recommendation_id: res.recommendation_id || null,
    turn_id: res.turn_id || null,
    session_id: res.session_id || null,
    feedbackGiven: null,
    isConsentQuestion: (res.response_text || "").includes(CONSENT_TRIGGER),
  };
}

/**
 * Assistant turn describing a failed request.
 *
 * Uses ApiError.userMessage so the user learns whether the backend is down, the
 * request timed out, or the server errored — the previous single generic string
 * gave them nothing to act on.
 */
const errorMessage = (err) => ({
  id: Date.now() + 1,
  role: "assistant",
  content: toApiError(err).userMessage,
  isError: true,
  timestamp: now(),
});

/**
 * Re-reads the transcript from the server after a catalogue revision.
 *
 * A revision annotates messages sent EARLIER in this conversation, and only
 * the server knows which assistant turns quoted the old value. Re-reading is
 * cheaper to reason about than trying to locate those messages client-side,
 * and it happens only on the rare turn where a value actually changed.
 */
export const refreshTranscript = createAsyncThunk(
  "chat/refreshTranscript",
  async ({ sessionId }, { dispatch }) => {
    if (!sessionId) return;
    try {
      const data = await getSessionHistory(sessionId);
      dispatch(messagesSet(mapHistoryMessages(data, sessionId)));
    } catch (e) {
      // Non-fatal: the reply itself arrived, only the correction note is missing.
      dispatch(chatErrorSet(toApiError(e).userMessage));
    }
  },
);

/** Opens a chat from the sidebar and loads its transcript. */
export const openSession = createAsyncThunk(
  "chat/openSession",
  async ({ session }, { dispatch }) => {
    dispatch(consentCleared());
    dispatch(activeSessionSet(session.session_id));
    try {
      const data = await getSessionHistory(session.session_id);
      dispatch(messagesSet(mapHistoryMessages(data, session.session_id)));
    } catch (e) {
      // The user clicked a chat and got nothing — this must be visible.
      dispatch(chatErrorSet(toApiError(e).userMessage));
    }
  },
);

/**
 * Starts a fresh chat: drops the server's active-session pointer and empties
 * the open transcript. The caller then returns the user to model selection.
 *
 * The transcript reset is explicit here because chat state now outlives the
 * ChatPage component — unmounting no longer clears it the way local state did.
 */
export const startNewChat = createAsyncThunk(
  "chat/startNew",
  async (_, { dispatch }) => {
    // Non-fatal: the next send passes force_new anyway, so a failed pointer
    // clear cannot strand the user in the old session.
    try { await startNewSession(); }
    catch (e) { console.warn("could not clear session pointer:", toApiError(e).userMessage); }
    dispatch(activeSessionCleared());
  },
);

export const sendChatMessage = createAsyncThunk(
  "chat/send",
  async ({ text, model }, { getState, dispatch }) => {
    dispatch(messageAppended({ id: Date.now(), role: "user", content: text, timestamp: now() }));

    const activeSession = getState().sessions.activeId;

    try {
      const res = await sendMessage({
        message: text,
        sessionId: activeSession,
        // No session open in the UI means the model was just picked, so this
        // is the first message of a fresh chat. Without force_new the backend
        // resumes whatever session its Redis pointer still holds, and that
        // session's model lock overrides the model the user just selected.
        forceNew: !activeSession,
        selectedModel: model,
      });

      // Set the active session before appending the reply, so anything keyed
      // off the open session sees the new id.
      const newSessionId = res.session_id;
      if (!activeSession) dispatch(activeSessionSet(newSessionId));

      const botMsg = buildBotMessage(res);
      dispatch(messageAppended(botMsg));
      if (botMsg.isConsentQuestion) dispatch(consentAwaited());

      // A catalogue value changed mid-conversation: pull the transcript back
      // from the server so the correction note appears under the earlier
      // message that quoted the old value.
      if ((res.revisions || []).length > 0) {
        await dispatch(refreshTranscript({ sessionId: newSessionId || activeSession }));
      }

      // Deliberately not awaited — the sidebar catching up must not hold the
      // composer disabled.
      dispatch(refreshSessionsUntilPresent({ sessionId: newSessionId }));
    } catch (e) {
      dispatch(messageAppended(errorMessage(e)));
    }
  },
  {
    condition: ({ text }, { getState }) => {
      const { chat } = getState();
      return Boolean(text) && !chat.sending && !chat.awaitingConsent;
    },
  },
);

/** "Yes" to a consent question — re-runs the search and shows the reply. */
export const acceptConsent = createAsyncThunk(
  "chat/acceptConsent",
  async ({ model }, { getState, dispatch }) => {
    dispatch(consentCleared());
    dispatch(messageAppended({ id: Date.now(), role: "user", content: "Yes", timestamp: now() }));

    const activeSession = getState().sessions.activeId;
    try {
      const res = await sendMessage({
        message: "yes", sessionId: activeSession, selectedModel: model,
      });
      const botMsg = buildBotMessage(res);
      dispatch(messageAppended(botMsg));
      if (botMsg.isConsentQuestion) dispatch(consentAwaited());
      if ((res.revisions || []).length > 0) {
        await dispatch(refreshTranscript({ sessionId: res.session_id || activeSession }));
      }
    } catch (e) {
      dispatch(messageAppended(errorMessage(e)));
    }
  },
);

/**
 * "No" to a consent question.
 *
 * Still calls the backend so it clears its pending-consent flag, but shows no
 * assistant reply and never raises the typing indicator — the user declined,
 * and answering anyway would read as the assistant ignoring them.
 */
export const declineConsent = createAsyncThunk(
  "chat/declineConsent",
  async ({ model }, { getState, dispatch }) => {
    dispatch(consentCleared());
    dispatch(messageAppended({ id: Date.now(), role: "user", content: "No", timestamp: now() }));

    const activeSession = getState().sessions.activeId;
    try {
      await sendMessage({
        message: "no", sessionId: activeSession, selectedModel: model,
      });
    } catch (e) { /* nothing to show either way */ }
  },
);

/** Records a thumbs up/down, reflecting it in the UI before the request lands. */
export const rateMessage = createAsyncThunk(
  "chat/rate",
  async ({ msg, rating, userId }, { getState, dispatch }) => {
    dispatch(feedbackRecorded({ id: msg.id, rating }));
    // The RL endpoint still takes an explicit user_id; it is a research signal
    // collector, not a data-access route, so it was left out of phase 3.
    await submitFeedback({
      sessionId: msg.session_id || getState().sessions.activeId,
      userId,
      recommendationId: msg.recommendation_id,
      turnId: msg.turn_id,
      rating,
      articleIds: (msg.items || []).map(i => i.article_id).filter(Boolean),
    });
  },
);

/** Signing out, or losing the session, clears the transcript. */
const resetChat = state => {
  state.messages = []; state.sending = false;
  state.awaitingConsent = false; state.error = null;
};

const chatSlice = createSlice({
  name: "chat",
  initialState: { messages: [], sending: false, awaitingConsent: false, error: null },
  reducers: {
    messagesSet(state, action)     { state.messages = action.payload; },
    messageAppended(state, action) { state.messages.push(action.payload); },
    consentAwaited(state)          { state.awaitingConsent = true; },
    consentCleared(state)          { state.awaitingConsent = false; },
    chatErrorSet(state, action)    { state.error = action.payload; },
    chatErrorCleared(state)        { state.error = null; },
    feedbackRecorded(state, action) {
      const m = state.messages.find(m => m.id === action.payload.id);
      if (m) m.feedbackGiven = action.payload.rating;
    },
  },
  extraReducers: builder => {
    builder
      // Only the two turns that show a reply drive the typing indicator;
      // declining consent is silent and must not raise it.
      .addCase(sendChatMessage.pending,   state => { state.sending = true; state.error = null; })
      .addCase(sendChatMessage.fulfilled, state => { state.sending = false; })
      .addCase(sendChatMessage.rejected,  state => { state.sending = false; })
      .addCase(acceptConsent.pending,     state => { state.sending = true; })
      .addCase(acceptConsent.fulfilled,   state => { state.sending = false; })
      .addCase(acceptConsent.rejected,    state => { state.sending = false; })
      // The open session went away, so the transcript on screen no longer
      // belongs to anything.
      .addCase(activeSessionCleared, state => {
        state.messages = []; state.awaitingConsent = false; state.error = null;
      })
      .addCase(logout.fulfilled, resetChat)
      .addCase(sessionExpired, resetChat);
  },
});

export const {
  messagesSet, messageAppended, consentAwaited, consentCleared, feedbackRecorded,
  chatErrorSet, chatErrorCleared,
} = chatSlice.actions;

export const selectMessages        = state => state.chat.messages;
export const selectSending         = state => state.chat.sending;
export const selectAwaitingConsent = state => state.chat.awaitingConsent;
export const selectChatError       = state => state.chat.error;

export default chatSlice.reducer;
