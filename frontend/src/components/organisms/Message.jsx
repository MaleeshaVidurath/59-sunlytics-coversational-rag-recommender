import Avatar from "../atoms/Avatar";
import ProductCard from "../molecules/ProductCard";
import MetaBadge from "../molecules/MetaBadge";
import CorrectionNotes from "../molecules/CorrectionNotes";
import ConsentButtons from "../molecules/ConsentButtons";
import FeedbackButtons from "../molecules/FeedbackButtons";
import { C } from "../../styles/theme";

/** One turn in the transcript: bubble, product cards, guard badges and controls. */
export default function Message({ msg, onFeedback, model, awaitingConsent, onConsentYes, onConsentNo }) {
  const isUser = msg.role === "user";
  const showConsent = !isUser && awaitingConsent && msg.isConsentQuestion;
  return (
    <div style={{ display:"flex", justifyContent:isUser?"flex-end":"flex-start",
      marginBottom:16, padding:"0 16px" }}>
      {!isUser && (
        <Avatar style={{ marginRight:10, marginTop:2 }}>S</Avatar>
      )}
      <div style={{ maxWidth:"70%", minWidth:60 }}>
        <div style={{ background:isUser?C.user:C.bot,
          border:`1px solid ${isUser?"#2d5a3d":C.border}`,
          borderRadius:isUser?"18px 18px 4px 18px":"18px 18px 18px 4px",
          padding:"10px 15px", color:C.text, fontSize:14,
          lineHeight:1.6, wordBreak:"break-word", whiteSpace:"pre-wrap" }}>
          {msg.content}
        </div>
        {msg.items && msg.items.length > 0 && (
          <div style={{ marginTop:6 }}>
            {msg.items.map((item, i) => (
              <ProductCard key={i} item={item} model={model} />
            ))}
          </div>
        )}
        {!isUser && <CorrectionNotes corrections={msg.corrections} />}
        {!isUser && msg.label && (
          <MetaBadge label={msg.label} confidence={msg.confidence||0}
            hallucination={msg.hallucination_flag}
            contradiction={msg.contradiction_found} />
        )}
        {showConsent
          ? <ConsentButtons onYes={onConsentYes} onNo={onConsentNo} />
          : !isUser && <FeedbackButtons msg={msg} onFeedback={onFeedback} />
        }
        <div style={{ fontSize:10, color:C.textMuted, marginTop:4,
          textAlign:isUser?"right":"left" }}>
          {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([],
            {hour:"2-digit",minute:"2-digit"}) : ""}
        </div>
      </div>
      {isUser && (
        <Avatar background="#1e3a2f" fontSize={13}
          style={{ border:"1px solid #2d5a3d", marginLeft:10, marginTop:2, color:C.accent }}>U</Avatar>
      )}
    </div>
  );
}
