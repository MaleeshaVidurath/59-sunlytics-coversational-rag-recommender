// ── Consent flow ──────────────────────────────────────────────────────────────
// The backend signals "I can re-run the search for you" by ending its reply
// with this exact sentence. The UI matches on it to swap the composer for
// Yes/No buttons, so the string must stay byte-identical to the server's.
export const CONSENT_TRIGGER = "Would you like to see new recommendations for this?";

// ── Model selection ───────────────────────────────────────────────────────────
export const MODEL_OPTIONS = [
  {
    id: "m3",
    title: "Member 3 · Conversational RAG",
    desc: "Text-based RAG with session memory, hallucination checking and contradiction detection.",
    tag: "M3",
  },
  {
    id: "m2",
    title: "Member 2 · Multimodal RAG",
    desc: "Multimodal retrieval combining text and visual features for richer recommendations.",
    tag: "M2",
  },
  {
    id: "m1",
    title: "Member 1 · Graph RAG",
    desc: "Knowledge-graph-based retrieval leveraging product relationship networks.",
    tag: "M1",
  },
];

export const MODEL_META = {
  m1: { label: "M1 · Graph RAG",          tag: "M1", color: "#7ec87e", bg: "#1e3a1e" },
  m2: { label: "M2 · Multimodal RAG",     tag: "M2", color: "#6e9bcf", bg: "#1e2a3a" },
  m3: { label: "M3 · Conversational RAG", tag: "M3", color: "#c9a96e", bg: "#2a200e" },
};

// Heading for the justification block. The two models justify a pick on
// different grounds, so one shared label would misdescribe one of them:
// M3's lines come from this user's own purchase statistics, while M2's come
// from the item's visual/style reasoning and are user-independent.
export const WHY_HEADING = {
  m3: "Why this for you",
  m2: "Why this item",
  m1: "Why this item",
};
