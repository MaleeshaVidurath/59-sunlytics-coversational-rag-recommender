import styles from "./CenteredTemplate.module.css";

/**
 * Full-height centred shell shared by the login, register and model-select
 * screens — the views that appear before a chat exists.
 */
export default function CenteredTemplate({ children, font = "sans" }) {
  return (
    <div className={`${styles.centered} ${styles[font]}`}>
      {children}
    </div>
  );
}
