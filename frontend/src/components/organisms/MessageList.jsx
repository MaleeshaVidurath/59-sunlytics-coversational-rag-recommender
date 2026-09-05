import Message from "./Message";
import Wordmark from "../atoms/Wordmark";
import Button from "../atoms/Button";
import TypingIndicator from "../atoms/TypingIndicator";
import ErrorBanner from "../molecules/ErrorBanner";
import { C } from "../../styles/theme";

const SUGGESTIONS = [
  "I want a black dress under £50",
  "Show me casual summer tops",
  "Need 3 shirts in different colours",
];

/** Prompt shown before the first message of a chat. */
function EmptyState({ onSuggestion }) {
  return (
    <div style={{ height:"100%", display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center",
      color:C.textMuted, padding:40 }}>
      <Wordmark size={42} letterSpacing={6} color={C.accentDim} marginBottom={12} />
      <div style={{ fontSize:14, maxWidth:340, textAlign:"center", lineHeight:1.7 }}>
        Your personalised fashion assistant.<br/>Tell me what you are looking for today.
      </div>
      <div style={{ marginTop:28, display:"flex", gap:10, flexWrap:"wrap", justifyContent:"center" }}>
        {SUGGESTIONS.map(s => (
          <Button key={s} onClick={() => onSuggestion(s)}
            style={{ background:C.card, border:`1px solid ${C.border}`,
              borderRadius:20, color:C.textDim, padding:"8px 16px",
              fontSize:12, cursor:"pointer", transition:"all 0.15s" }}
            hoverStyle={{ border:`1px solid ${C.accent}`, color:C.accent }}>
            {s}
          </Button>
        ))}
      </div>
    </div>
  );
}

/**
 * Scrollable transcript. `endRef` marks the bottom of the list so the page can
 * scroll new turns into view.
 */
export default function MessageList({
  messages, sending, model, awaitingConsent,
  onFeedback, onConsentYes, onConsentNo, onSuggestion, endRef,
  error, onDismissError,
}) {
  const empty = messages.length === 0 && !sending && !error;
  return (
    <div style={{ flex:1, overflowY:"auto", padding:"24px 0 8px" }}>
      <ErrorBanner message={error} onDismiss={onDismissError} />
      {empty ? (
        <EmptyState onSuggestion={onSuggestion} />
      ) : (
        <>
          {messages.map(msg => (
            <Message key={msg.id} msg={msg} onFeedback={onFeedback}
              model={model}
              awaitingConsent={awaitingConsent}
              onConsentYes={onConsentYes}
              onConsentNo={onConsentNo} />
          ))}
          {sending && <TypingIndicator />}
          <div ref={endRef} />
        </>
      )}
    </div>
  );
}
