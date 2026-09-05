import IconButton from "../atoms/IconButton";
import ModelBadge from "../atoms/ModelBadge";
import { C } from "../../styles/theme";

/** Top bar: sidebar toggle, the chat's model, and which session is open. */
export default function ChatHeader({ model, activeSession, onToggleSidebar }) {
  return (
    <div style={{ padding:"0 20px", height:56, borderBottom:`1px solid ${C.border}`,
      display:"flex", alignItems:"center", gap:12, background:C.bg, flexShrink:0 }}>
      <IconButton onClick={onToggleSidebar} color={C.textDim} padding="4px 8px">☰</IconButton>
      <div style={{ display:"flex", alignItems:"center", gap:10 }}>
        <ModelBadge model={model} size="normal" />
        <span style={{ color:C.textMuted, fontSize:12 }}>
          {activeSession ? `Session · ${activeSession.slice(-8)}` : "New conversation"}
        </span>
      </div>
    </div>
  );
}
