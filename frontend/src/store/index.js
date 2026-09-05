import { configureStore, createListenerMiddleware } from "@reduxjs/toolkit";
import authReducer, { loggedIn, loggedOut, STORAGE_KEY } from "./slices/authSlice";
import modelReducer from "./slices/modelSlice";
import sessionsReducer from "./slices/sessionsSlice";
import chatReducer from "./slices/chatSlice";

/**
 * Keeps localStorage in step with the auth slice.
 *
 * Done here rather than inside the reducers so those stay pure — a reducer that
 * writes to storage cannot be replayed or tested in isolation.
 */
const persistence = createListenerMiddleware();

persistence.startListening({
  actionCreator: loggedIn,
  effect: action => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(action.payload)); }
    catch (e) { console.warn("Could not persist session", e); }
  },
});

persistence.startListening({
  actionCreator: loggedOut,
  effect: () => {
    try { localStorage.removeItem(STORAGE_KEY); }
    catch (e) { console.warn("Could not clear persisted session", e); }
  },
});

export const store = configureStore({
  reducer: {
    auth:     authReducer,
    model:    modelReducer,
    sessions: sessionsReducer,
    chat:     chatReducer,
  },
  middleware: getDefault => getDefault().prepend(persistence.middleware),
});

export default store;
