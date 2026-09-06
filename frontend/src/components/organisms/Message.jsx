import Avatar from "../atoms/Avatar";
import ProductCard from "../molecules/ProductCard";
import MetaBadge from "../molecules/MetaBadge";
import CorrectionNotes from "../molecules/CorrectionNotes";
import ConsentButtons from "../molecules/ConsentButtons";
import FeedbackButtons from "../molecules/FeedbackButtons";
import styles from "./Message.module.css";

/** One turn in the transcript: bubble, product cards, guard badges and controls. */
export default function Message({ msg, onFeedback, model, awaitingConsent, onConsentYes, onConsentNo }) {
  const isUser = msg.role === "user";
  const showConsent = !isUser && awaitingConsent && msg.isConsentQuestion;

  const bubbleClasses = [
    styles.bubble,
    isUser ? styles.bubbleUser : styles.bubbleBot,
    msg.isError ? styles.bubbleError : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={`${styles.row} ${isUser ? styles.fromUser : styles.fromBot}`}>
      {!isUser && <Avatar className={styles.avatarBot}>S</Avatar>}

      <div className={styles.column}>
        <div className={bubbleClasses}>{msg.content}</div>

        {msg.items && msg.items.length > 0 && (
          <div className={styles.items}>
            {msg.items.map((item, i) => (
              <ProductCard key={i} item={item} model={model} />
            ))}
          </div>
        )}

        {!isUser && <CorrectionNotes corrections={msg.corrections} />}
        {!isUser && msg.label && (
          <MetaBadge label={msg.label} confidence={msg.confidence || 0}
            hallucination={msg.hallucination_flag}
            contradiction={msg.contradiction_found} />
        )}

        {showConsent
          ? <ConsentButtons onYes={onConsentYes} onNo={onConsentNo} />
          : !isUser && <FeedbackButtons msg={msg} onFeedback={onFeedback} />}

        <div className={`${styles.timestamp} ${isUser ? styles.timestampUser : ""}`.trim()}>
          {msg.timestamp
            ? new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
            : ""}
        </div>
      </div>

      {isUser && (
        <Avatar variant="user" fontSize={13} className={styles.avatarUser}>U</Avatar>
      )}
    </div>
  );
}
