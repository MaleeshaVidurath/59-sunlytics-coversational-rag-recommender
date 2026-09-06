import Avatar from "./Avatar";
import styles from "./TypingIndicator.module.css";

/** Three bouncing dots shown in the assistant's bubble while a reply is in flight. */
export default function TypingIndicator() {
  return (
    <div className={styles.wrapper}>
      <Avatar className={styles.avatar}>S</Avatar>
      <div className={styles.bubble} role="status" aria-label="Assistant is typing">
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>
    </div>
  );
}
