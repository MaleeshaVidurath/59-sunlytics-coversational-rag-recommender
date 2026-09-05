import { C } from "../../styles/theme";

/**
 * In-chat notice that a value quoted in THIS message has since changed in the
 * catalogue. The message itself is never rewritten: it was accurate when sent,
 * and silently editing history would be the dishonest fix. The note sits under
 * the bubble it applies to so the user can see exactly which statement aged.
 */
export default function CorrectionNotes({ corrections }) {
  if (!corrections || corrections.length === 0) return null;
  return (
    <div style={{ marginTop:6, display:"flex", flexDirection:"column", gap:4 }}>
      {corrections.map((c, i) => (
        <div key={i} style={{ background:C.revise,
          border:`1px solid ${C.reviseBorder}`, borderRadius:8,
          padding:"6px 10px", fontSize:11.5, color:C.reviseText,
          lineHeight:1.5 }}>
          <strong>Update:</strong>{" "}
          {c.product_name ? `the ${c.label} of ${c.product_name}` : `the ${c.label}`}
          {" "}changed from{" "}
          <span style={{ textDecoration:"line-through", opacity:0.75 }}>
            {c.old_value}
          </span>{" "}
          to <strong>{c.new_value}</strong> after this message.
        </div>
      ))}
    </div>
  );
}
