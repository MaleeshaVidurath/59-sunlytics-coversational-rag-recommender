// Shared colour palette. Single source of truth for every inline style in the
// app — imported as `C` so existing call sites read unchanged.
export const C = {
  bg: "#0f0f0f", sidebar: "#161616", card: "#1c1c1c",
  border: "#2a2a2a", accent: "#c9a96e", accentDim: "#8a6f3e",
  user: "#1e3a2f", bot: "#1c1c1c", text: "#f0ebe3",
  textDim: "#8a8078", textMuted: "#555",
  flag: "#7c2d2d", flagText: "#fca5a5",
  contra: "#2d4a1e", contraText: "#86efac", tag: "#222",
  // Catalogue revisions are not errors — the message was true when sent — so
  // they read as an informational amber note, distinct from the red
  // hallucination flag and the green contradiction badge.
  revise: "#3a2f16", reviseText: "#e0c079", reviseBorder: "#5c4a22",
};

export default C;
