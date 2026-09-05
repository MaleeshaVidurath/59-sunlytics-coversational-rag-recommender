/**
 * Colour for an intent label badge. The three groups read as one colour each:
 * gold for product requests, blue for questions about shown items, green for
 * conversational turns.
 */
export function labelColor(label) {
  const m = {
    INITIAL_REQUEST: "#c9a96e", REFINEMENT: "#c9a96e",
    ATTRIBUTE_QUESTION: "#6e9bcf", COMPARISON: "#6e9bcf",
    EXPLANATION_WHY: "#6e9bcf", SELECTION_REFERENCE: "#6e9bcf",
    FEEDBACK: "#7ec87e", CHITCHAT: "#7ec87e",
  };
  return m[label] || "#666";
}
