import Badge from "../atoms/Badge";
import { C } from "../../styles/theme";
import { labelColor } from "../../utils/labels";

/** Row of pills under an assistant reply: intent, confidence, and guard verdicts. */
export default function MetaBadge({ label, confidence, hallucination, contradiction }) {
  return (
    <div style={{ display:"flex", flexWrap:"wrap", gap:4, marginTop:8 }}>
      {label && (
        <Badge style={{ background:C.tag, border:`1px solid ${labelColor(label)}33`,
          color:labelColor(label), fontFamily:"monospace" }}>{label}</Badge>
      )}
      {confidence > 0 && (
        <Badge style={{ background:C.tag, border:"1px solid #333",
          color:C.textDim, fontFamily:"monospace" }}>
          {(confidence*100).toFixed(1)}%
        </Badge>
      )}
      {hallucination && (
        <Badge style={{ background:C.flag, color:C.flagText,
          padding:"2px 8px", fontWeight:600 }}>
          ⚠ hallucination flagged
        </Badge>
      )}
      {contradiction && (
        <Badge style={{ background:C.contra, color:C.contraText,
          padding:"2px 8px", fontWeight:600 }}>
          ✓ contradiction corrected
        </Badge>
      )}
    </div>
  );
}
