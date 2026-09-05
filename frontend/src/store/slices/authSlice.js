import { createSlice } from "@reduxjs/toolkit";

export const STORAGE_KEY = "sunlytics_user";

/**
 * The signed-in user is restored from localStorage so a refresh does not sign
 * them out. A malformed or absent entry simply reads as signed-out.
 */
function readPersistedUser() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)); }
  catch { return null; }
}

const authSlice = createSlice({
  name: "auth",
  initialState: { user: readPersistedUser() },
  reducers: {
    // Reducers stay pure — writing localStorage is handled by the persistence
    // listener registered in store/index.js.
    loggedIn(state, action) { state.user = action.payload; },
    loggedOut(state)        { state.user = null; },
  },
});

export const { loggedIn, loggedOut } = authSlice.actions;

export const selectUser = state => state.auth.user;

export default authSlice.reducer;
