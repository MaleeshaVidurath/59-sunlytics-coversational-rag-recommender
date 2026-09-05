import { C } from "../../styles/theme";

/** Customer-profile dropdown and sign-in action on the login screen. */
export default function CustomerPicker({
  customers, selected, onSelect, loading, logging, error, onSubmit,
}) {
  if (loading) return <div style={{color:C.textMuted,fontSize:13}}>Loading...</div>;

  return (
    <>
      <select value={selected} onChange={e => onSelect(e.target.value)}
        style={{ width:"100%", background:C.card, border:`1px solid ${C.border}`,
          borderRadius:8, color:selected?C.text:C.textMuted, padding:"12px 14px",
          fontSize:13, fontFamily:"monospace", cursor:"pointer",
          outline:"none", marginBottom:16, appearance:"none" }}>
        <option value="">Choose a customer ID...</option>
        {customers.map(c => (
          <option key={c.customer_id} value={c.customer_id}>
            {c.customer_id.slice(0,20)}...{c.age?` · Age ${c.age}`:""}
            {c.club_member_status?` · ${c.club_member_status}`:""}
          </option>
        ))}
      </select>

      {error && <div style={{ color:C.flagText, background:C.flag,
        borderRadius:8, padding:"8px 14px", fontSize:12, marginBottom:16 }}>{error}</div>}

      <button onClick={onSubmit} disabled={!selected||logging}
        style={{ width:"100%", background:selected?C.accent:C.textMuted,
          border:"none", borderRadius:8, color:"#0f0f0f", padding:"13px",
          fontSize:14, fontWeight:700, cursor:selected?"pointer":"not-allowed",
          fontFamily:"system-ui,sans-serif", letterSpacing:1,
          textTransform:"uppercase", transition:"background 0.2s" }}>
        {logging?"Signing in...":"Enter"}
      </button>
    </>
  );
}
