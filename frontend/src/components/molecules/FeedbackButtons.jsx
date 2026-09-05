/**
 * Thumbs up/down on a recommendation, feeding the RL signal collector.
 *
 * Only rendered for turns that actually carry a recommendation_id, and locked
 * once a rating is given — the backend treats the first signal per turn as the
 * real one, so the UI must not offer a second.
 */
export default function FeedbackButtons({ msg, onFeedback }) {
  if (!msg.recommendation_id) return null;

  const given = msg.feedbackGiven;

  const rating = (kind, activeBg, activeBorder, activeColor) => ({
    background: given === kind ? activeBg : "#1a1a1a",
    border: `1px solid ${given === kind ? activeBorder : "#333"}`,
    borderRadius: 8,
    padding: "3px 10px",
    cursor: given ? "default" : "pointer",
    color: given === kind ? activeColor : "#555",
    fontSize: 14,
    transition: "all 0.15s",
    opacity: given && given !== kind ? 0.35 : 1,
  });

  return (
    <div style={{ display:"flex", alignItems:"center", gap:6, marginTop:8 }}>
      <span style={{ fontSize:10, color:"#555", marginRight:2 }}>Was this helpful?</span>
      <button
        onClick={() => !given && onFeedback(msg, "up")}
        title="Good recommendation"
        style={rating("up", "#1e3a2f", "#2d5a3d", "#7ec87e")}>
        👍
      </button>
      <button
        onClick={() => !given && onFeedback(msg, "down")}
        title="Could be better"
        style={rating("down", "#3a1e1e", "#5a2d2d", "#f87171")}>
        👎
      </button>
      {given && (
        <span style={{ fontSize:10, color: given === "up" ? "#7ec87e" : "#f87171" }}>
          {given === "up" ? "Thanks for your feedback!" : "We'll improve!"}
        </span>
      )}
    </div>
  );
}
