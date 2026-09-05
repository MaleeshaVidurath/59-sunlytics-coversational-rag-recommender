import { C } from "../../styles/theme";
import { WHY_HEADING } from "../../utils/constants";

/**
 * Why this item was picked. M3 generates these from real purchase statistics;
 * M2 from its hallucination-guard-verified explanation. Both are safe to render
 * verbatim — see WHY_HEADING for why the label differs between them.
 */
export default function WhyList({ reasons = [], model }) {
  if (reasons.length === 0) return null;
  const heading = WHY_HEADING[model] || "Why this for you";
  return (
    <div style={{ marginTop:6, paddingTop:6, borderTop:`1px solid ${C.border}` }}>
      <div style={{ color:C.textMuted, fontSize:9, letterSpacing:0.5,
        textTransform:"uppercase", marginBottom:3 }}>
        {heading}
      </div>
      {reasons.map((reason, i) => (
        <div key={i} style={{ color:C.textDim, fontSize:10, marginTop:2,
          display:"flex", gap:5, lineHeight:1.35 }}>
          <span style={{ color:C.accent, flexShrink:0 }}>✓</span>
          <span>{reason}</span>
        </div>
      ))}
    </div>
  );
}
