import { MODEL_META } from "../../utils/constants";
import styles from "./ModelBadge.module.css";

/** Pill identifying which recommendation model a chat is bound to. */
export default function ModelBadge({ model, size = "normal" }) {
  const meta = MODEL_META[model] || MODEL_META.m3;
  const classes = [styles.badge, size === "small" ? styles.small : ""]
    .filter(Boolean).join(" ");

  return (
    <span
      className={classes}
      style={{ "--model-color": meta.color, "--model-bg": meta.bg }}
    >
      <span className={styles.tag}>{meta.tag}</span>
      <span className={styles.label}>{meta.label}</span>
    </span>
  );
}
