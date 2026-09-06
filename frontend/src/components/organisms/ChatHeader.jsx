import IconButton from "../atoms/IconButton";
import ModelBadge from "../atoms/ModelBadge";
import styles from "./ChatHeader.module.css";

/** Top bar: sidebar toggle, the chat's model, and which session is open. */
export default function ChatHeader({ model, activeSession, onToggleSidebar }) {
  return (
    <div className={styles.header}>
      <IconButton tone="muted" onClick={onToggleSidebar}
        title="Toggle sidebar" aria-label="Toggle sidebar">☰</IconButton>
      <div className={styles.meta}>
        <ModelBadge model={model} size="normal" />
        <span className={styles.session}>
          {activeSession ? `Session · ${activeSession.slice(-8)}` : "New conversation"}
        </span>
      </div>
    </div>
  );
}
