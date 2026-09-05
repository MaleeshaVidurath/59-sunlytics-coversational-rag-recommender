import { MODEL_META } from "../../utils/constants";

/** Pill identifying which recommendation model a chat is bound to. */
export default function ModelBadge({ model, size = "normal" }) {
  const meta = MODEL_META[model] || MODEL_META.m3;
  const isSmall = size === "small";
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:isSmall?4:6,
      background:meta.bg, border:`1px solid ${meta.color}55`,
      borderRadius:20, padding:isSmall?"2px 8px":"4px 12px",
      flexShrink:0,
    }}>
      <span style={{
        width:isSmall?16:20, height:isSmall?16:20, borderRadius:"50%", flexShrink:0,
        background:meta.color, display:"flex", alignItems:"center", justifyContent:"center",
        fontSize:isSmall?9:11, fontWeight:700, color:"#0f0f0f", fontFamily:"monospace",
      }}>{meta.tag}</span>
      <span style={{ color:meta.color, fontSize:isSmall?10:12,
        fontFamily:"monospace", whiteSpace:"nowrap" }}>{meta.label}</span>
    </span>
  );
}
