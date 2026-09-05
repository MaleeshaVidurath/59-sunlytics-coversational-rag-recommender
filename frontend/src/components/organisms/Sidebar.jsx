import Wordmark from "../atoms/Wordmark";
import Avatar from "../atoms/Avatar";
import Button from "../atoms/Button";
import IconButton from "../atoms/IconButton";
import ModelBadge from "../atoms/ModelBadge";
import SessionListItem from "../molecules/SessionListItem";
import ErrorBanner from "../molecules/ErrorBanner";
import { C } from "../../styles/theme";

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
      <div style={{ padding:"18px 14px 12px", borderBottom:`1px solid ${C.border}` }}>
        <Wordmark size={18} letterSpacing={3} color={C.accent} weight={700} marginBottom={12} />
        <Button onClick={onNewChat}
          style={{ width:"100%", background:"transparent", border:`1px solid ${C.border}`,
            borderRadius:8, color:C.textDim, padding:"8px 12px", fontSize:13,
            cursor:"pointer", display:"flex", alignItems:"center", gap:8, transition:"all 0.15s" }}
          hoverStyle={{ border:`1px solid ${C.accent}`, color:C.accent }}>
          <span style={{fontSize:16}}>+</span> New Chat
        </Button>
      </div>

      <div style={{ flex:1, overflowY:"auto", padding:"10px" }}>
        {/* A failed list read keeps whatever was already loaded, so the error and
            the stale list can legitimately appear together. */}
        <ErrorBanner message={error} onRetry={onRetry} />
        {visible.length === 0
          ? <div style={{color:C.textMuted,fontSize:12,padding:"16px 4px"}}>No previous chats yet.</div>
          : visible.map(s => (
              <SessionListItem key={s.session_id} session={s}
                active={s.session_id === activeSession}
                onSelect={onSelectSession}
                onDelete={onDeleteSession} />
            ))}
      </div>

      <div style={{ padding:"8px 14px", borderTop:`1px solid ${C.border}` }}>
        <ModelBadge model={model} size="small" />
      </div>

      <div style={{ padding:"10px 14px", borderTop:`1px solid ${C.border}` }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <Avatar background={C.user} fontSize={13}
            style={{ border:"1px solid #2d5a3d", color:C.accent, fontWeight:700 }}>
            {user.age?user.age.toString()[0]:"U"}
          </Avatar>
          <div style={{ flex:1, minWidth:0 }}>
            <div style={{ color:C.text, fontSize:12, fontWeight:500,
              whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
              {user.customer_id?.slice(0,18)}...
            </div>
            <div style={{ color:C.textDim, fontSize:11 }}>
              {user.purchase_summary?.budget_tier || ""}
            </div>
          </div>
          <IconButton onClick={onLogout} title="Sign out" hoverColor={C.flagText}>↩</IconButton>
        </div>
      </div>
    </>
  );
}
