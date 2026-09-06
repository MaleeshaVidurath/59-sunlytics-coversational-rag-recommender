import styles from "./ErrorBanner.module.css";

/**
 * Inline, dismissible error notice.
 *
 * Errors are shown where the failed action lives rather than in a floating
 * toast, so it is always clear which thing failed. `onRetry` is only rendered
 * when the caller has something meaningful to retry.
 */
export default function ErrorBanner({ message, onRetry, onDismiss }) {
  if (!message) return null;
  return (
    <div className={styles.banner} role="alert">
      <span className={styles.icon} aria-hidden="true">⚠</span>
      <span className={styles.message}>{message}</span>
      {onRetry && <button className={styles.retry} onClick={onRetry}>Retry</button>}
      {onDismiss && (
        <button className={styles.dismiss} onClick={onDismiss} title="Dismiss"
          aria-label="Dismiss error">×</button>
      )}
    </div>
  );
}
