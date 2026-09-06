import { MAX_MESSAGE_LENGTH, MESSAGE_WARN_AT, validateMessage } from "../../utils/validation";
import styles from "./MessageInput.module.css";

/**
 * The chat composer: auto-growing textarea plus send button.
 *
 * Both are disabled while a consent question is outstanding — the backend is
 * waiting on a Yes/No for that turn, and a free-text message sent in between
 * would be answered against the wrong pending state.
 *
 * Validation is advisory rather than blocking at the keystroke level: the
 * textarea accepts overlong text so a paste is never silently truncated, and
 * the send button explains why it is disabled instead.
 */
export default function MessageInput({
  value, onChange, onKeyDown, onSend, sending, awaitingConsent, inputRef,
}) {
  const { valid, error } = validateMessage(value);
  const canSend = valid && !sending && !awaitingConsent;
  const trimmedLength = value.trim().length;
  const showCount = value.length >= MESSAGE_WARN_AT;
  const overLimit = trimmedLength > MAX_MESSAGE_LENGTH;

  return (
    <div className={styles.wrapper}>
      {/* Only complain about length — an empty box is not an error worth saying. */}
      {overLimit && <div className={styles.error} role="alert">{error}</div>}

      <div className={`${styles.field} ${overLimit ? styles.invalid : ""}`.trim()}>
        <textarea
          ref={inputRef}
          className={styles.textarea}
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={awaitingConsent ? "Please select Yes or No above…" : "Message Sunlytics..."}
          disabled={awaitingConsent}
          aria-label="Message"
          aria-invalid={overLimit || undefined}
          rows={1}
          onInput={e => {
            e.target.style.height = "auto";
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
          }}
        />

        {showCount && (
          <span className={`${styles.counter} ${overLimit ? styles.counterOver : ""}`.trim()}>
            {trimmedLength}/{MAX_MESSAGE_LENGTH}
          </span>
        )}

        <button
          className={styles.send}
          onClick={onSend}
          disabled={!canSend}
          title={overLimit ? "Message is too long to send" : "Send"}
          aria-label="Send message">
          {sending ? "…" : "↑"}
        </button>
      </div>

      <div className={styles.footnote}>
        Sunlytics CRS · Fashion Recommendation Research System
      </div>
    </div>
  );
}
