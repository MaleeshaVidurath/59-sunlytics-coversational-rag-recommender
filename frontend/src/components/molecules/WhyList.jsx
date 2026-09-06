import { WHY_HEADING } from "../../utils/constants";
import styles from "./WhyList.module.css";

/**
 * Why this item was picked. M3 generates these from real purchase statistics;
 * M2 from its hallucination-guard-verified explanation. Both are safe to render
 * verbatim — see WHY_HEADING for why the label differs between them.
 */
export default function WhyList({ reasons = [], model }) {
  if (reasons.length === 0) return null;
  return (
    <div className={styles.block}>
      <div className={styles.heading}>{WHY_HEADING[model] || "Why this for you"}</div>
      {reasons.map((reason, i) => (
        <div key={i} className={styles.reason}>
          <span className={styles.tick}>✓</span>
          <span>{reason}</span>
        </div>
      ))}
    </div>
  );
}
