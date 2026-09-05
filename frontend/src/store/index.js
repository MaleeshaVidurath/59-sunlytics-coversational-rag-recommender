import { configureStore } from "@reduxjs/toolkit";
import authReducer, { sessionExpired } from "./slices/authSlice";
import modelReducer from "./slices/modelSlice";
import sessionsReducer from "./slices/sessionsSlice";
import chatReducer from "./slices/chatSlice";
import { setSessionExpiredHandler } from "../services/http";

/**
 * There is no persistence middleware any more.
 *
 * The session used to be mirrored into localStorage, which meant any injected
 * script could read it. It now lives entirely in httpOnly cookies: the page
 * cannot read them, and the profile in this store is rebuilt on load from
 * /api/auth/me.
 */
export const store = configureStore({
  reducer: {
    auth:     authReducer,
    model:    modelReducer,
    sessions: sessionsReducer,
    chat:     chatReducer,
  },
});

/**
 * Lets the HTTP layer report a dead session without importing the store —
 * which would be a cycle, since the store's thunks import the HTTP layer.
 */
setSessionExpiredHandler(() => store.dispatch(sessionExpired()));

export default store;
