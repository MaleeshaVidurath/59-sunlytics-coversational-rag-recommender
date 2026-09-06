import Avatar from "../atoms/Avatar";
import styles from "./ModelOptionCard.module.css";

/**
 * One selectable model on the model-select screen.
 *
 * A <button> rather than a clickable <div>: it is genuinely an action, and this
 * way it is reachable by keyboard and announced correctly.
 */
export default function ModelOptionCard({ option, onSelect }) {
  return (
    <button type="button" className={styles.card} onClick={() => onSelect(option.id)}>
      <Avatar variant="tag" size={44} radius={10} fontSize={13}>{option.tag}</Avatar>
      <div className={styles.body}>
        <div className={styles.title}>{option.title}</div>
        <div className={styles.description}>{option.desc}</div>
      </div>
      <div className={styles.arrow} aria-hidden="true">→</div>
    </button>
  );
}
