import styles from "./Wordmark.module.css";

/**
 * The SUNLYTICS wordmark. Appears at four sizes — login, model select, sidebar
 * header and the empty-chat state — differing only in scale and colour.
 */
export default function Wordmark({
  size = 18,
  letterSpacing = 3,
  color,
  weight = 700,
  marginBottom = 0,
  className = "",
}) {
  return (
    <div
      className={`${styles.wordmark} ${className}`.trim()}
      style={{
        "--wordmark-size": `${size}px`,
        "--wordmark-tracking": `${letterSpacing}px`,
        "--wordmark-weight": weight,
        "--wordmark-gap": `${marginBottom}px`,
        ...(color && { "--wordmark-color": color }),
      }}
    >
      SUNLYTICS
    </div>
  );
}
