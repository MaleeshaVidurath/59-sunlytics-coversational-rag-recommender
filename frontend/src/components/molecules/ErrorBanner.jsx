import { C } from "../../styles/theme";

/**
 * Inline, dismissible error notice.
 *
 * Errors are shown where the failed action lives rather than in a floating toast,
 * so it is always clear which thing failed. `onRetry` is only rendered when the
 * caller has something meaningful to retry.
 */
export default function ErrorBanner({ message, onRetry, onDismiss }) {
  if (!message) return null;
  return (
    <div role="alert" style={{
      display:"flex", alignItems:"center", gap:10,
      background:C.flag, color:C.flagText,
      border:"1px solid #a33", borderRadius:8,
      padding:"8px 12px", fontSize:12, lineHeight:1.5,
      margin:"0 16px 10px",
    }}>
      <span style={{ flexShrink:0 }}>⚠</span>
      <span style={{ flex:1, minWidth:0 }}>{message}</span>
      {onRetry && (
        <button onClick={onRetry}
          style={{ background:"transparent", border:`1px solid ${C.flagText}`,
            borderRadius:6, color:C.flagText, fontSize:11, padding:"3px 10px",
            cursor:"pointer", flexShrink:0 }}>
          Retry
        </button>
      )}
      {onDismiss && (
        <button onClick={onDismiss} title="Dismiss"
          style={{ background:"transparent", border:"none", color:C.flagText,
            fontSize:15, cursor:"pointer", padding:"0 2px", flexShrink:0 }}>
          ×
        </button>
      )}
    </div>
  );
}
