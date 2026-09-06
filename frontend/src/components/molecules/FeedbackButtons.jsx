import styles from "./FeedbackButtons.module.css";

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

  const classesFor = kind => [
    styles.rating,
    styles[kind],
    given === kind ? styles.chosen : "",
    given ? styles.locked : "",
    given && given !== kind ? styles.dimmed : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={styles.row}>
      <span className={styles.prompt}>Was this helpful?</span>
      <button
        className={classesFor("up")}
        onClick={() => !given && onFeedback(msg, "up")}
        title="Good recommendation"
        aria-pressed={given === "up"}>
        👍
      </button>
      <button
        className={classesFor("down")}
        onClick={() => !given && onFeedback(msg, "down")}
        title="Could be better"
        aria-pressed={given === "down"}>
        👎
      </button>
      {given && (
        <span className={`${styles.thanks} ${given === "up" ? styles.thanksUp : styles.thanksDown}`}>
          {given === "up" ? "Thanks for your feedback!" : "We'll improve!"}
        </span>
      )}
    </div>
  );
}
