import Avatar from "./Avatar";
import { C } from "../../styles/theme";

/** Three bouncing dots shown in the assistant's bubble while a reply is in flight. */
export default function TypingIndicator() {
  return (
    <div style={{ display:"flex", padding:"0 16px", marginBottom:16, alignItems:"center" }}>
      <Avatar style={{ marginRight:10 }}>S</Avatar>
      <div style={{ background:C.bot, border:`1px solid ${C.border}`,
        borderRadius:"18px 18px 18px 4px", padding:"12px 18px",
        display:"flex", gap:5, alignItems:"center" }}>
        {[0,1,2].map(i => (
          <div key={i} style={{ width:7, height:7, borderRadius:"50%",
            background:C.accentDim,
            animation:`bounce 1.2s ease-in-out ${i*0.2}s infinite` }} />
        ))}
      </div>
    </div>
  );
}
