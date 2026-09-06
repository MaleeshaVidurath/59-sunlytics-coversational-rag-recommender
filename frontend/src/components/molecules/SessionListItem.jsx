import IconButton from "../atoms/IconButton";
import { timeAgo } from "../../utils/time";
import styles from "./SessionListItem.module.css";

/** One chat in the sidebar list. Reveals its delete control on hover or focus. */
export default function SessionListItem({ session, active, onSelect, onDelete }) {
  return (
    <div
      className={`${styles.item} ${active ? styles.active : ""}`.trim()}
      onClick={() => onSelect(session)}
    >
      <div className={styles.title}>{session.title || "New conversation"}</div>
      <div className={styles.meta}>
        {timeAgo(session.last_activity_at)}
        {session.message_count > 0 && ` · ${session.message_count} messages`}
      </div>
      <IconButton
        small
        floating
        className={styles.delete}
        title="Delete chat"
        onClick={e => { e.stopPropagation(); onDelete(session.session_id); }}
      >🗑</IconButton>
    </div>
  );
}
