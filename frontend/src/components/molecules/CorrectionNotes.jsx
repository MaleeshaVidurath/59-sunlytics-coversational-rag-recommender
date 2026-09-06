import styles from "./CorrectionNotes.module.css";

/**
 * In-chat notice that a value quoted in THIS message has since changed in the
 * catalogue. The message itself is never rewritten: it was accurate when sent,
 * and silently editing history would be the dishonest fix. The note sits under
 * the bubble it applies to so the user can see exactly which statement aged.
 */
export default function CorrectionNotes({ corrections }) {
  if (!corrections || corrections.length === 0) return null;
  return (
    <div className={styles.list}>
      {corrections.map((c, i) => (
        <div key={i} className={styles.note}>
          <strong>Update:</strong>{" "}
          {c.product_name ? `the ${c.label} of ${c.product_name}` : `the ${c.label}`}
          {" "}changed from{" "}
          <span className={styles.oldValue}>{c.old_value}</span>{" "}
          to <strong>{c.new_value}</strong> after this message.
        </div>
      ))}
    </div>
  );
}
