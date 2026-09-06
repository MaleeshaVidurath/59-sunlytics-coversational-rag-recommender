import Message from "./Message";
import Wordmark from "../atoms/Wordmark";
import Button from "../atoms/Button";
import TypingIndicator from "../atoms/TypingIndicator";
import ErrorBanner from "../molecules/ErrorBanner";
import styles from "./MessageList.module.css";

const SUGGESTIONS = [
  "I want a black dress under £50",
  "Show me casual summer tops",
  "Need 3 shirts in different colours",
];

/** Prompt shown before the first message of a chat. */
function EmptyState({ onSuggestion }) {
  return (
    <div className={styles.empty}>
      <Wordmark size={42} letterSpacing={6} color="var(--accent-dim)" marginBottom={12} />
      <div className={styles.tagline}>
        Your personalised fashion assistant.<br />Tell me what you are looking for today.
      </div>
      <div className={styles.suggestions}>
        {SUGGESTIONS.map(s => (
          <Button key={s} variant="chip" onClick={() => onSuggestion(s)}>{s}</Button>
        ))}
      </div>
    </div>
  );
}

/**
 * Scrollable transcript. `endRef` marks the bottom of the list so the page can
 * scroll new turns into view.
 */
export default function MessageList({
  messages, sending, model, awaitingConsent,
  onFeedback, onConsentYes, onConsentNo, onSuggestion, endRef,
  error, onDismissError,
}) {
  const empty = messages.length === 0 && !sending && !error;
  return (
    <div className={styles.list}>
      <ErrorBanner message={error} onDismiss={onDismissError} />
      {empty ? (
        <EmptyState onSuggestion={onSuggestion} />
      ) : (
        <>
          {messages.map(msg => (
            <Message key={msg.id} msg={msg} onFeedback={onFeedback}
              model={model}
              awaitingConsent={awaitingConsent}
              onConsentYes={onConsentYes}
              onConsentNo={onConsentNo} />
          ))}
          {sending && <TypingIndicator />}
          <div ref={endRef} />
        </>
      )}
    </div>
  );
}
