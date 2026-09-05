import { C } from "../../styles/theme";
import { MAX_MESSAGE_LENGTH, MESSAGE_WARN_AT, validateMessage } from "../../utils/validation";

/**
 * The chat composer: auto-growing textarea plus send button.
 *
 * Both are disabled while a consent question is outstanding — the backend is
 * waiting on a Yes/No for that turn, and a free-text message sent in between
 * would be answered against the wrong pending state.
 *
 * Validation is advisory rather than blocking at the keystroke level: the
 * textarea accepts overlong text so a paste is never silently truncated, and the
 * send button explains why it is disabled instead.
 */
export default function MessageInput({
  value, onChange, onKeyDown, onSend, sending, awaitingConsent, inputRef,
}) {
  const { valid, error } = validateMessage(value);
  const canSend = valid && !sending && !awaitingConsent;
  const showCount = value.length >= MESSAGE_WARN_AT;
  const overLimit = value.trim().length > MAX_MESSAGE_LENGTH;

  return (
    <div style={{ padding:"12px 20px 20px", borderTop:`1px solid ${C.border}`,
      background:C.bg, flexShrink:0 }}>

      {/* Only complain about length — an empty box is not an error worth saying. */}
      {overLimit && (
        <div role="alert" style={{ color:C.flagText, fontSize:11, marginBottom:6, paddingLeft:4 }}>
          {error}
        </div>
      )}

      <div style={{ display:"flex", gap:10, alignItems:"flex-end",
        background:C.card,
        border:`1px solid ${overLimit ? C.flagText : C.border}`,
        borderRadius:14, padding:"10px 14px" }}>
        <textarea ref={inputRef} value={value}
          onChange={e => onChange(e.target.value)} onKeyDown={onKeyDown}
          placeholder={awaitingConsent ? "Please select Yes or No above…" : "Message Sunlytics..."}
          disabled={awaitingConsent}
          aria-label="Message"
          aria-invalid={overLimit || undefined}
          rows={1}
          style={{ flex:1, background:"transparent", border:"none",
            color: awaitingConsent ? C.textMuted : C.text,
            fontSize:14, resize:"none", outline:"none",
            lineHeight:1.6, maxHeight:120, overflowY:"auto",
            fontFamily:"system-ui,sans-serif",
            cursor: awaitingConsent ? "not-allowed" : "text" }}
          onInput={e => {
            e.target.style.height = "auto";
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
          }} />

        {showCount && (
          <span style={{ fontSize:10, color: overLimit ? C.flagText : C.textMuted,
            fontFamily:"monospace", flexShrink:0, alignSelf:"center" }}>
            {value.trim().length}/{MAX_MESSAGE_LENGTH}
          </span>
        )}

        <button onClick={onSend} disabled={!canSend}
          title={overLimit ? "Message is too long to send" : "Send"}
          aria-label="Send message"
          style={{ background: canSend ? C.accent : C.textMuted,
            border:"none", borderRadius:9, width:36, height:36,
            display:"flex", alignItems:"center", justifyContent:"center",
            cursor: canSend ? "pointer" : "not-allowed",
            fontSize:16, flexShrink:0, transition:"background 0.2s" }}>
          {sending ? "…" : "↑"}
        </button>
      </div>
      <div style={{ textAlign:"center", fontSize:10, color:C.textMuted, marginTop:8 }}>
        Sunlytics CRS · Fashion Recommendation Research System
      </div>
    </div>
  );
}
