import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { getSessions, deleteSession } from "../../services/sessionService";
import { toApiError } from "../../services/ApiError";
import { logout, sessionExpired } from "./authSlice";

export const fetchSessions = createAsyncThunk(
  "sessions/fetch",
  async (_, { rejectWithValue }) => {
    try {
      const data = await getSessions();
      return data.sessions || [];
    } catch (err) {
      return rejectWithValue(toApiError(err).toPayload());
    }
  },
);

/**
 * Reloads the session list until the just-created session appears in it.
 *
 * The session is written to MongoDB as the chat turn completes, so an immediate
 * read can miss it. Retries three times at 600ms, then accepts whatever the
 * server last returned rather than leaving the sidebar indefinitely stale.
 */
export const refreshSessionsUntilPresent = createAsyncThunk(
  "sessions/refreshUntilPresent",
  async ({ sessionId, retries = 3, delay = 600 }, { dispatch }) => {
    for (let i = 0; i < retries; i++) {
      await new Promise(r => setTimeout(r, delay));
      try {
        const data = await getSessions();
        const list = data.sessions || [];
        if (list.some(s => s.session_id === sessionId) || i === retries - 1) {
          dispatch(sessionsReceived(list));
          break;
        }
      } catch (err) {
        // Background catch-up: the turn itself already succeeded, so a stale
        // sidebar is not worth interrupting the user for.
        console.warn("session list refresh failed:", toApiError(err).userMessage);
        break;
      }
    }
  },
);

export const removeSession = createAsyncThunk(
  "sessions/remove",
  async ({ sessionId }, { getState, dispatch, rejectWithValue }) => {
    try {
      await deleteSession(sessionId);
    } catch (err) {
      return rejectWithValue(toApiError(err).toPayload());
    }
    // Clear before reloading the list, so the deleted chat is never briefly
    // shown as the open one.
    if (getState().sessions.activeId === sessionId) dispatch(activeSessionCleared());
    await dispatch(fetchSessions());
    return sessionId;
  },
);

/** Signing out, or losing the session, empties the sidebar. */
const resetSessions = state => {
  state.items = []; state.activeId = null; state.error = null;
};

const sessionsSlice = createSlice({
  name: "sessions",
  initialState: { items: [], activeId: null, error: null },
  reducers: {
    sessionsReceived(state, action)  { state.items = action.payload; },
    activeSessionSet(state, action)  { state.activeId = action.payload; },
    /** The open session went away — deleted, or the user started a new chat. */
    activeSessionCleared(state)      { state.activeId = null; },
    sessionsErrorCleared(state)      { state.error = null; },
  },
  extraReducers: builder => {
    builder
      .addCase(fetchSessions.pending,   state => { state.error = null; })
      .addCase(fetchSessions.fulfilled, (state, action) => {
        state.items = action.payload; state.error = null;
      })
      // A failed list read leaves the previous list in place: a stale sidebar is
      // less disruptive mid-conversation than an empty one. Unlike before, the
      // failure is surfaced to the user rather than silently becoming [].
      .addCase(fetchSessions.rejected, (state, action) => {
        state.error = action.payload?.message ?? "Could not load your chats.";
      })
      .addCase(removeSession.rejected, (state, action) => {
        state.error = action.payload?.message ?? "Could not delete that chat.";
      })
      .addCase(logout.fulfilled, resetSessions)
      .addCase(sessionExpired, resetSessions);
  },
});

export const {
  sessionsReceived, activeSessionSet, activeSessionCleared, sessionsErrorCleared,
} = sessionsSlice.actions;

export const selectSessions      = state => state.sessions.items;
export const selectActiveSession = state => state.sessions.activeId;
export const selectSessionsError = state => state.sessions.error;

export default sessionsSlice.reducer;
