import { useState } from "react";
import IconButton from "../atoms/IconButton";
import { C } from "../../styles/theme";
import { timeAgo } from "../../utils/time";

/** One chat in the sidebar list. Reveals its delete control on hover. */
export default function SessionListItem({ session, active, onSelect, onDelete }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ position:"relative", padding:"11px 14px", cursor:"pointer",
        borderRadius:8, marginBottom:4,
        background:active?"#252525":hovered?"#1e1e1e":"transparent",
        border:active?`1px solid ${C.border}`:"1px solid transparent",
        transition:"all 0.15s" }}
      onClick={() => onSelect(session)}
    >
      <div style={{ color:active?C.text:C.textDim, fontSize:13,
        fontWeight:active?500:400, paddingRight:hovered?24:0,
        whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
        {session.title || "New conversation"}
      </div>
      <div style={{ color:C.textMuted, fontSize:11, marginTop:3 }}>
        {timeAgo(session.last_activity_at)}
        {session.message_count > 0 && ` · ${session.message_count} messages`}
      </div>
      {hovered && (
        <IconButton
          title="Delete chat"
          onClick={e => { e.stopPropagation(); onDelete(session.session_id); }}
          hoverColor={C.flagText}
          fontSize={14}
          borderRadius={4}
          style={{ position:"absolute", right:8, top:"50%", transform:"translateY(-50%)" }}
        >🗑</IconButton>
      )}
    </div>
  );
}
