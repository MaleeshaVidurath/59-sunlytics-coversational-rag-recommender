import { useState } from "react";
import Avatar from "../atoms/Avatar";
import { C } from "../../styles/theme";

/** One selectable model on the model-select screen. */
export default function ModelOptionCard({ option, onSelect }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={() => onSelect(option.id)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? "#222" : C.card,
        border: `1px solid ${hovered ? C.accent : C.border}`,
        borderRadius:12, padding:"18px 20px", marginBottom:12,
        cursor:"pointer", transition:"all 0.15s", display:"flex",
        alignItems:"center", gap:16,
      }}>
      <Avatar size={44} radius={10} fontSize={13}
        style={{ fontWeight:700, color:"#0f0f0f", fontFamily:"monospace" }}>
        {option.tag}
      </Avatar>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ color:C.text, fontWeight:600, fontSize:14, marginBottom:4 }}>
          {option.title}
        </div>
        <div style={{ color:C.textDim, fontSize:12, lineHeight:1.5 }}>
          {option.desc}
        </div>
      </div>
      <div style={{ color:hovered ? C.accent : C.textMuted, fontSize:18, flexShrink:0 }}>
        →
      </div>
    </div>
  );
}
