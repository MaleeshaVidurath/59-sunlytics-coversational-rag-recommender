import { createSlice } from "@reduxjs/toolkit";
import { loggedOut } from "./authSlice";

/**
 * Which recommendation model the current chat is bound to.
 *
 * Deliberately not persisted: one model per session, per login, so every fresh
 * visit makes the choice explicit rather than silently reusing the last one.
 */
const modelSlice = createSlice({
  name: "model",
  initialState: { selected: null },
  reducers: {
    modelSelected(state, action) { state.selected = action.payload; },
    modelReset(state)            { state.selected = null; },
  },
  extraReducers: builder => {
    // Signing out must also drop the model choice, otherwise the next user to
    // sign in on this browser would land straight in someone else's model.
    builder.addCase(loggedOut, state => { state.selected = null; });
  },
});

export const { modelSelected, modelReset } = modelSlice.actions;

export const selectModel = state => state.model.selected;

export default modelSlice.reducer;
