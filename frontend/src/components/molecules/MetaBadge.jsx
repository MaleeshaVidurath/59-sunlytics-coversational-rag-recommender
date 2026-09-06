import Badge from "../atoms/Badge";
import { labelColor } from "../../utils/labels";
import styles from "./MetaBadge.module.css";

/** Row of pills under an assistant reply: intent, confidence, and guard verdicts. */
export default function MetaBadge({ label, confidence, hallucination, contradiction }) {
  return (
    <div className={styles.row}>
      {label && <Badge variant="intent" color={labelColor(label)}>{label}</Badge>}
      {confidence > 0 && <Badge variant="mono">{(confidence * 100).toFixed(1)}%</Badge>}
      {hallucination && <Badge variant="flag">⚠ hallucination flagged</Badge>}
      {contradiction && <Badge variant="contra">✓ contradiction corrected</Badge>}
    </div>
  );
}
