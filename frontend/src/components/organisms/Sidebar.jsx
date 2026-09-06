import Wordmark from "../atoms/Wordmark";
import Avatar from "../atoms/Avatar";
import Button from "../atoms/Button";
import IconButton from "../atoms/IconButton";
import ModelBadge from "../atoms/ModelBadge";
import SessionListItem from "../molecules/SessionListItem";
import ErrorBanner from "../molecules/ErrorBanner";
import styles from "./Sidebar.module.css";

/**
 * Chat sidebar: new-chat action, the session list, and the signed-in user.
 *
 * Sessions are filtered to the model this chat is bound to. A session is locked
 * to the model it was opened with, so listing chats from another model would
 * offer the user conversations they cannot resume from here.
 */
export default function Sidebar({
  user, sessions, activeSession, model,
  onNewChat, onSelectSession, onDeleteSession, onLogout,
  error, onRetry,
}) {
  const visible = sessions.filter(s => (s.selected_model || "m3") === model);

  return (
    <>
      <div className={styles.header}>
        <Wordmark size={18} letterSpacing={3} marginBottom={12} />
        <Button variant="ghost" fullWidth onClick={onNewChat}>
          <span>+</span> New Chat
        </Button>
      </div>

      <div className={styles.list}>
        {/* A failed list read keeps whatever was already loaded, so the error and
            the stale list can legitimately appear together. */}
        <ErrorBanner message={error} onRetry={onRetry} />
        {visible.length === 0
          ? <div className={styles.emptyList}>No previous chats yet.</div>
          : visible.map(s => (
              <SessionListItem key={s.session_id} session={s}
                active={s.session_id === activeSession}
                onSelect={onSelectSession}
                onDelete={onDeleteSession} />
            ))}
      </div>

      <div className={styles.modelRow}>
        <ModelBadge model={model} size="small" />
      </div>

      <div className={styles.userRow}>
        <div className={styles.user}>
          <Avatar variant="user" fontSize={13}>
            {user.age ? user.age.toString()[0] : "U"}
          </Avatar>
          <div className={styles.userDetails}>
            <div className={styles.username}>{user.username}</div>
            <div className={styles.userMeta}>
              {user.purchase_summary?.budget_tier || ""}
            </div>
          </div>
          <IconButton onClick={onLogout} title="Sign out" aria-label="Sign out">↩</IconButton>
        </div>
      </div>
    </>
  );
}
