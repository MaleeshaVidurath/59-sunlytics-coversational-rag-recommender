import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import * as authService from "../../services/authService";
import { toApiError } from "../../services/ApiError";

/**
 * Authentication state.
 *
 * The session is NOT stored here or in localStorage — it lives in httpOnly
 * cookies the page cannot read, so a cross-site script has nothing to steal.
 * What this slice holds is only the profile the server told us about, which is
 * re-established on every page load by calling /api/auth/me.
 *
 *   status: "bootstrapping" -> we do not yet know if anyone is signed in
 *           "anonymous"     -> confirmed signed out
 *           "authenticated" -> confirmed signed in, `user` is populated
 */

/** Restores the session from cookies on page load. */
export const bootstrapSession = createAsyncThunk(
  "auth/bootstrap",
  async (_, { rejectWithValue }) => {
    try {
      return await authService.fetchMe();
    } catch (err) {
      // A 401 here is the normal signed-out case, not an error worth showing.
      return rejectWithValue(toApiError(err).status === 401 ? null : toApiError(err).userMessage);
    }
  },
);

export const login = createAsyncThunk(
  "auth/login",
  async ({ username, password }, { rejectWithValue }) => {
    try {
      return await authService.login(username, password);
    } catch (err) {
      return rejectWithValue(toApiError(err).userMessage);
    }
  },
);

export const register = createAsyncThunk(
  "auth/register",
  async ({ username, password }, { rejectWithValue }) => {
    try {
      return await authService.register(username, password);
    } catch (err) {
      return rejectWithValue(toApiError(err).userMessage);
    }
  },
);

export const logout = createAsyncThunk(
  "auth/logout",
  async () => {
    // Never rejects: the local session must be cleared even if the server call
    // fails, otherwise a network blip leaves the user apparently signed in.
    try { await authService.logout(); }
    catch (e) { console.warn("logout request failed:", toApiError(e).userMessage); }
  },
);

const authSlice = createSlice({
  name: "auth",
  initialState: {
    user: null,
    status: "bootstrapping",
    error: null,
  },
  reducers: {
    /**
     * The access token could not be refreshed — the session is over.
     * Dispatched by the HTTP layer, so any 401 anywhere lands the user back on
     * the sign-in screen instead of leaving a half-dead UI.
     */
    sessionExpired(state) {
      state.user = null;
      state.status = "anonymous";
      state.error = "Your session expired. Please sign in again.";
    },
    authErrorCleared(state) { state.error = null; },
  },
  extraReducers: builder => {
    const authenticated = (state, action) => {
      state.user = action.payload;
      state.status = "authenticated";
      state.error = null;
    };

    builder
      .addCase(bootstrapSession.fulfilled, authenticated)
      .addCase(bootstrapSession.rejected, (state, action) => {
        state.user = null;
        state.status = "anonymous";
        // null payload = an ordinary signed-out 401; anything else is a real
        // failure (server down) worth telling the user about.
        state.error = action.payload ?? null;
      })
      .addCase(login.pending, state => { state.error = null; })
      .addCase(login.fulfilled, authenticated)
      .addCase(login.rejected, (state, action) => {
        state.user = null;
        state.status = "anonymous";
        state.error = action.payload ?? "Sign in failed.";
      })
      .addCase(register.pending, state => { state.error = null; })
      .addCase(register.fulfilled, authenticated)
      .addCase(register.rejected, (state, action) => {
        state.error = action.payload ?? "Could not create that account.";
      })
      .addCase(logout.fulfilled, state => {
        state.user = null;
        state.status = "anonymous";
        state.error = null;
      });
  },
});

export const { sessionExpired, authErrorCleared } = authSlice.actions;

export const selectUser        = state => state.auth.user;
export const selectAuthStatus  = state => state.auth.status;
export const selectAuthError   = state => state.auth.error;

export default authSlice.reducer;
