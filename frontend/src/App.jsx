import { useState, useEffect, useRef } from "react";

const BASE = "http://localhost:8000";
const M2_IMAGE_BASE = "http://localhost:8001";   // M2 serves product photos at /api/images/{article_id}

async function apiGetCustomers() {
  const r = await fetch(`${BASE}/api/auth/customers`);
  return r.json();
}
async function apiLogin(customerId) {
  const r = await fetch(`${BASE}/api/auth/login`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id: customerId }),
  });
  if (!r.ok) throw new Error("Login failed");
  return r.json();
}
async function apiGetSessions(userId) {
  const r = await fetch(`${BASE}/api/sessions?user_id=${userId}`);
  return r.json();
}
async function apiGetHistory(sessionId, userId) {
  const r = await fetch(`${BASE}/api/sessions/${sessionId}?user_id=${userId}`);
  return r.json();
}
async function apiSendMessage({ userId, customerId, message, sessionId, forceNew, selectedModel }) {
  const r = await fetch(`${BASE}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      customer_id: customerId,
      message,
      session_id: sessionId || null,
      force_new_session: forceNew || false,
      selected_model: selectedModel || "m3",
    }),
  });
  if (!r.ok) throw new Error("Chat failed");
  return r.json();
}
async function apiDeleteSession(sessionId, userId) {
  const r = await fetch(`${BASE}/api/sessions/${sessionId}?user_id=${userId}`,
    { method: "DELETE" });
  if (!r.ok) throw new Error("Delete failed");
  return r.json();
}

async function apiNewSession(userId) {
  // Clears the Redis active session pointer so next message creates fresh session
  const r = await fetch(`${BASE}/api/sessions/new?user_id=${userId}`,
    { method: "POST" });
  if (!r.ok) console.warn("Could not clear session pointer");
}

async function apiSubmitFeedback({ sessionId, userId, recommendationId, turnId, rating, articleIds }) {
  try {
    await fetch(`${BASE}/api/rl/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id:        sessionId,
        user_id:           userId,
        recommendation_id: recommendationId || "",
        turn_id:           turnId || "",
        rating,
        article_ids:       articleIds || [],
      }),
    });
  } catch (e) {
    console.warn("RL feedback submission failed:", e);
  }
}

const C = {
  bg: "#0f0f0f", sidebar: "#161616", card: "#1c1c1c",
  border: "#2a2a2a", accent: "#c9a96e", accentDim: "#8a6f3e",
  user: "#1e3a2f", bot: "#1c1c1c", text: "#f0ebe3",
  textDim: "#8a8078", textMuted: "#555",
  flag: "#7c2d2d", flagText: "#fca5a5",
  contra: "#2d4a1e", contraText: "#86efac", tag: "#222",
  // Catalogue revisions are not errors — the message was true when sent — so
  // they read as an informational amber note, distinct from the red
  // hallucination flag and the green contradiction badge.
  revise: "#3a2f16", reviseText: "#e0c079", reviseBorder: "#5c4a22",
};

function timeAgo(ts) {
  if (!ts) return "";
  const s = Math.floor((new Date() - new Date(ts)) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  if (s < 86400) return `${Math.floor(s/3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

function labelColor(label) {
  const m = { INITIAL_REQUEST:"#c9a96e", REFINEMENT:"#c9a96e",
    ATTRIBUTE_QUESTION:"#6e9bcf", COMPARISON:"#6e9bcf",
    EXPLANATION_WHY:"#6e9bcf", SELECTION_REFERENCE:"#6e9bcf",
    FEEDBACK:"#7ec87e", CHITCHAT:"#7ec87e" };
  return m[label] || "#666";
}

function ProductCard({ item }) {
  const why = item.why || [];
  const [imgFailed, setImgFailed] = useState(false);
  const [preview, setPreview]     = useState(false);
  const imgSrc = `${M2_IMAGE_BASE}/api/images/${item.article_id}`;
  return (
    <div style={{ background:"#1a1a1a", border:`1px solid ${C.border}`,
      borderRadius:10, padding:"10px 14px", marginTop:8,
      display:"flex", alignItems:"flex-start", gap:14 }}>
      {item.article_id && !imgFailed ? (
        <img
          src={imgSrc}
          alt={item.name || "product"}
          title="Click to enlarge"
          onError={() => setImgFailed(true)}
          onClick={() => setPreview(true)}
          style={{ width:80, height:80, borderRadius:10, flexShrink:0,
            objectFit:"cover", background:"#242424", cursor:"zoom-in",
            border:`1px solid ${C.border}` }}
        />
      ) : (
        <div style={{ width:80, height:80, borderRadius:10, flexShrink:0,
          background:`linear-gradient(135deg,${C.accentDim},${C.accent})`,
          display:"flex", alignItems:"center", justifyContent:"center", fontSize:26 }}>
          👗
        </div>
      )}
      {preview && (
        <div
          onClick={() => setPreview(false)}
          style={{ position:"fixed", inset:0, zIndex:1000,
            background:"rgba(0,0,0,0.88)", cursor:"pointer",
            display:"flex", flexDirection:"column",
            alignItems:"center", justifyContent:"center", gap:14 }}>
          <img src={imgSrc} alt={item.name || "product"}
            style={{ maxWidth:"82vw", maxHeight:"72vh", borderRadius:12,
              boxShadow:"0 24px 80px rgba(0,0,0,0.7)" }} />
          <div style={{ color:"#fff", fontWeight:600, fontSize:16 }}>{item.name}</div>
          <div style={{ color:"#aaa", fontSize:13 }}>
            {item.colour} · {item.type}{item.price ? ` · ${item.price}` : ""}
          </div>
          <div style={{ color:"#fff", fontSize:14, marginTop:6, opacity:0.85,
            fontWeight:600 }}>Close</div>
        </div>
      )}
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ color:C.text, fontWeight:600, fontSize:13,
          whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
          {item.name}
        </div>
        <div style={{ color:C.textDim, fontSize:11, marginTop:2 }}>
          {item.colour} · {item.type} · <span style={{color:C.accent}}>{item.price}</span>
        </div>
        {item.description && (
          <div style={{ color:C.textMuted, fontSize:10, marginTop:3,
            display:"-webkit-box", WebkitLineClamp:2,
            WebkitBoxOrient:"vertical", overflow:"hidden" }}>
            {item.description}
          </div>
        )}
        {/* Why this item was picked for this user. Generated from real
            statistics by the ranker, so it is safe to render verbatim. */}
        {why.length > 0 && (
          <div style={{ marginTop:6, paddingTop:6,
            borderTop:`1px solid ${C.border}` }}>
            <div style={{ color:C.textMuted, fontSize:9, letterSpacing:0.5,
              textTransform:"uppercase", marginBottom:3 }}>
              Why this for you
            </div>
            {why.map((reason, i) => (
              <div key={i} style={{ color:C.textDim, fontSize:10, marginTop:2,
                display:"flex", gap:5, lineHeight:1.35 }}>
                <span style={{ color:C.accent, flexShrink:0 }}>✓</span>
                <span>{reason}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end",
        gap:4, flexShrink:0 }}>
        <div style={{ fontSize:10, color:C.textMuted, fontFamily:"monospace" }}>
          #{item.article_id?.slice(-6)}
        </div>
        {/* Already a clamped 0-100 figure from the ranker; absent when nothing
            personalised matched, in which case no badge is shown at all. */}
        {typeof item.match_percent === "number" && (
          <div title="Personalised match score"
            style={{ fontSize:9, color:C.accent, fontFamily:"monospace",
              border:`1px solid ${C.border}`, borderRadius:5, padding:"1px 5px" }}>
            {item.match_percent}%
          </div>
        )}
      </div>
    </div>
  );
}

function MetaBadge({ label, confidence, hallucination, contradiction }) {
  return (
    <div style={{ display:"flex", flexWrap:"wrap", gap:4, marginTop:8 }}>
      {label && (
        <span style={{ background:C.tag, border:`1px solid ${labelColor(label)}33`,
          color:labelColor(label), fontSize:10, padding:"2px 7px",
          borderRadius:12, fontFamily:"monospace" }}>{label}</span>
      )}
      {confidence > 0 && (
        <span style={{ background:C.tag, border:`1px solid #333`,
          color:C.textDim, fontSize:10, padding:"2px 7px",
          borderRadius:12, fontFamily:"monospace" }}>
          {(confidence*100).toFixed(1)}%
        </span>
      )}
      {hallucination && (
        <span style={{ background:C.flag, color:C.flagText,
          fontSize:10, padding:"2px 8px", borderRadius:12, fontWeight:600 }}>
          ⚠ hallucination flagged
        </span>
      )}
      {contradiction && (
        <span style={{ background:C.contra, color:C.contraText,
          fontSize:10, padding:"2px 8px", borderRadius:12, fontWeight:600 }}>
          ✓ contradiction corrected
        </span>
      )}
    </div>
  );
}

/**
 * In-chat notice that a value quoted in THIS message has since changed in the
 * catalogue. The message itself is never rewritten: it was accurate when sent,
 * and silently editing history would be the dishonest fix. The note sits under
 * the bubble it applies to so the user can see exactly which statement aged.
 */
function CorrectionNotes({ corrections }) {
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

function ConsentButtons({ onYes, onNo }) {
  return (
    <div style={{ display:"flex", gap:8, marginTop:10 }}>
      <button onClick={onYes}
        style={{ background:"#1e3a2f", border:"1px solid #2d5a3d",
          borderRadius:8, color:"#7ec87e", padding:"6px 18px",
          fontSize:13, cursor:"pointer", fontWeight:600,
          transition:"all 0.15s" }}
        onMouseEnter={e=>{e.currentTarget.style.background="#253f35";}}
        onMouseLeave={e=>{e.currentTarget.style.background="#1e3a2f";}}>
        Yes
      </button>
      <button onClick={onNo}
        style={{ background:"#1c1c1c", border:`1px solid ${C.border}`,
          borderRadius:8, color:C.textDim, padding:"6px 18px",
          fontSize:13, cursor:"pointer", fontWeight:600,
          transition:"all 0.15s" }}
        onMouseEnter={e=>{e.currentTarget.style.borderColor="#555";}}
        onMouseLeave={e=>{e.currentTarget.style.borderColor=C.border;}}>
        No
      </button>
    </div>
  );
}

function FeedbackButtons({ msg, onFeedback }) {
  if (!msg.recommendation_id) return null;

  const given = msg.feedbackGiven;

  return (
    <div style={{ display:"flex", alignItems:"center", gap:6, marginTop:8 }}>
      <span style={{ fontSize:10, color:"#555", marginRight:2 }}>Was this helpful?</span>
      <button
        onClick={() => !given && onFeedback(msg, "up")}
        title="Good recommendation"
        style={{
          background: given === "up" ? "#1e3a2f" : "#1a1a1a",
          border: `1px solid ${given === "up" ? "#2d5a3d" : "#333"}`,
          borderRadius: 8,
          padding: "3px 10px",
          cursor: given ? "default" : "pointer",
          color: given === "up" ? "#7ec87e" : "#555",
          fontSize: 14,
          transition: "all 0.15s",
          opacity: given && given !== "up" ? 0.35 : 1,
        }}>
        👍
      </button>
      <button
        onClick={() => !given && onFeedback(msg, "down")}
        title="Could be better"
        style={{
          background: given === "down" ? "#3a1e1e" : "#1a1a1a",
          border: `1px solid ${given === "down" ? "#5a2d2d" : "#333"}`,
          borderRadius: 8,
          padding: "3px 10px",
          cursor: given ? "default" : "pointer",
          color: given === "down" ? "#f87171" : "#555",
          fontSize: 14,
          transition: "all 0.15s",
          opacity: given && given !== "down" ? 0.35 : 1,
        }}>
        👎
      </button>
      {given && (
        <span style={{ fontSize:10, color: given === "up" ? "#7ec87e" : "#f87171" }}>
          {given === "up" ? "Thanks for your feedback!" : "We'll improve!"}
        </span>
      )}
    </div>
  );
}

const CONSENT_TRIGGER = "Would you like to see new recommendations for this?";

function Message({ msg, onFeedback, awaitingConsent, onConsentYes, onConsentNo }) {
  const isUser = msg.role === "user";
  const showConsent = !isUser && awaitingConsent && msg.isConsentQuestion;
  return (
    <div style={{ display:"flex", justifyContent:isUser?"flex-end":"flex-start",
      marginBottom:16, padding:"0 16px" }}>
      {!isUser && (
        <div style={{ width:32, height:32, borderRadius:"50%", flexShrink:0,
          background:`linear-gradient(135deg,${C.accentDim},${C.accent})`,
          display:"flex", alignItems:"center", justifyContent:"center",
          fontSize:14, marginRight:10, marginTop:2 }}>S</div>
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
            {msg.items.map((item, i) => <ProductCard key={i} item={item} />)}
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
        <div style={{ width:32, height:32, borderRadius:"50%", flexShrink:0,
          background:"#1e3a2f", border:"1px solid #2d5a3d",
          display:"flex", alignItems:"center", justifyContent:"center",
          fontSize:13, marginLeft:10, marginTop:2, color:C.accent }}>U</div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display:"flex", padding:"0 16px", marginBottom:16, alignItems:"center" }}>
      <div style={{ width:32, height:32, borderRadius:"50%",
        background:`linear-gradient(135deg,${C.accentDim},${C.accent})`,
        display:"flex", alignItems:"center", justifyContent:"center",
        fontSize:14, marginRight:10 }}>S</div>
      <div style={{ background:C.bot, border:`1px solid ${C.border}`,
        borderRadius:"18px 18px 18px 4px", padding:"12px 18px",
        display:"flex", gap:5, alignItems:"center" }}>
        {[0,1,2].map(i => (
          <div key={i} style={{ width:7, height:7, borderRadius:"50%",
            background:C.accentDim,
            animation:`bounce 1.2s ease-in-out ${i*0.2}s infinite` }} />
        ))}
      </div>
    </div>
  );
}

function SidebarItem({ session, active, onSelect, onDelete }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ position:"relative", padding:"11px 14px", cursor:"pointer",
        borderRadius:8, marginBottom:4,
        background:active?"#252525":hovered?"#1e1e1e":"transparent",
        border:active?`1px solid ${C.border}`:"1px solid transparent",
        transition:"all 0.15s" }}
      onClick={() => onSelect(session)}
    >
      <div style={{ color:active?C.text:C.textDim, fontSize:13,
        fontWeight:active?500:400, paddingRight:hovered?24:0,
        whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
        {session.title || "New conversation"}
      </div>
      <div style={{ color:C.textMuted, fontSize:11, marginTop:3 }}>
        {timeAgo(session.last_activity_at)}
        {session.message_count > 0 && ` · ${session.message_count} messages`}
      </div>
      {hovered && (
        <button
          onClick={e => { e.stopPropagation(); onDelete(session.session_id); }}
          title="Delete chat"
          style={{ position:"absolute", right:8, top:"50%",
            transform:"translateY(-50%)", background:"transparent",
            border:"none", cursor:"pointer", fontSize:14, padding:"2px 6px",
            borderRadius:4, color:C.textMuted }}
          onMouseEnter={e => e.currentTarget.style.color=C.flagText}
          onMouseLeave={e => e.currentTarget.style.color=C.textMuted}
        >🗑</button>
      )}
    </div>
  );
}

// ── Login Page ────────────────────────────────────────────────────────────────
function LoginPage({ onLogin }) {
  const [customers, setCustomers] = useState([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(true);
  const [logging, setLogging] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGetCustomers().then(d => { setCustomers(d.customers||[]); setLoading(false); })
      .catch(() => { setError("Cannot connect to server."); setLoading(false); });
  }, []);

  const handleLogin = async () => {
    if (!selected) return;
    setLogging(true); setError("");
    try { const u = await apiLogin(selected); onLogin(u); }
    catch { setError("Login failed."); setLogging(false); }
  };

  return (
    <div style={{ minHeight:"100vh", background:C.bg,
      display:"flex", alignItems:"center", justifyContent:"center",
      fontFamily:"'Playfair Display',Georgia,serif" }}>
      <div style={{ width:440, background:C.sidebar,
        border:`1px solid ${C.border}`, borderRadius:16, padding:"48px 40px", textAlign:"center" }}>
        <div style={{ fontSize:36, letterSpacing:6, color:C.accent, fontWeight:700, marginBottom:8 }}>
          SUNLYTICS
        </div>
        <div style={{ fontSize:12, letterSpacing:3, color:C.textDim,
          textTransform:"uppercase", marginBottom:40, fontFamily:"system-ui,sans-serif" }}>
          Fashion Intelligence System
        </div>
        <div style={{ color:C.textDim, fontSize:13, marginBottom:20, fontFamily:"system-ui,sans-serif" }}>
          Select your customer profile to continue
        </div>
        {loading ? <div style={{color:C.textMuted,fontSize:13}}>Loading...</div> : (
          <>
            <select value={selected} onChange={e=>setSelected(e.target.value)}
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
            <button onClick={handleLogin} disabled={!selected||logging}
              style={{ width:"100%", background:selected?C.accent:C.textMuted,
                border:"none", borderRadius:8, color:"#0f0f0f", padding:"13px",
                fontSize:14, fontWeight:700, cursor:selected?"pointer":"not-allowed",
                fontFamily:"system-ui,sans-serif", letterSpacing:1,
                textTransform:"uppercase", transition:"background 0.2s" }}>
              {logging?"Signing in...":"Enter"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ── Model Select Page ─────────────────────────────────────────────────────────
const MODEL_OPTIONS = [
  {
    id: "m3",
    title: "Member 3 · Conversational RAG",
    desc: "Text-based RAG with session memory, hallucination checking and contradiction detection.",
    tag: "M3",
  },
  {
    id: "m2",
    title: "Member 2 · Multimodal RAG",
    desc: "Multimodal retrieval combining text and visual features for richer recommendations.",
    tag: "M2",
  },
  {
    id: "m1",
    title: "Member 1 · Graph RAG",
    desc: "Knowledge-graph-based retrieval leveraging product relationship networks.",
    tag: "M1",
  },
];

function ModelSelectPage({ onSelect }) {
  const [hovered, setHovered] = useState(null);
  return (
    <div style={{ minHeight:"100vh", background:C.bg,
      display:"flex", alignItems:"center", justifyContent:"center",
      fontFamily:"system-ui,-apple-system,sans-serif" }}>
      <div style={{ width:500, maxWidth:"92vw" }}>
        <div style={{ textAlign:"center", marginBottom:36 }}>
          <div style={{ fontSize:30, letterSpacing:5, color:C.accent, fontWeight:700,
            fontFamily:"'Playfair Display',Georgia,serif", marginBottom:8 }}>SUNLYTICS</div>
          <div style={{ fontSize:13, color:C.textDim }}>Select a recommendation model to begin</div>
        </div>
        {MODEL_OPTIONS.map(opt => (
          <div key={opt.id}
            onClick={() => onSelect(opt.id)}
            onMouseEnter={() => setHovered(opt.id)}
            onMouseLeave={() => setHovered(null)}
            style={{
              background: hovered === opt.id ? "#222" : C.card,
              border: `1px solid ${hovered === opt.id ? C.accent : C.border}`,
              borderRadius:12, padding:"18px 20px", marginBottom:12,
              cursor:"pointer", transition:"all 0.15s", display:"flex",
              alignItems:"center", gap:16,
            }}>
            <div style={{ width:44, height:44, borderRadius:10, flexShrink:0,
              background:`linear-gradient(135deg,${C.accentDim},${C.accent})`,
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:13, fontWeight:700, color:"#0f0f0f", fontFamily:"monospace" }}>
              {opt.tag}
            </div>
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{ color:C.text, fontWeight:600, fontSize:14, marginBottom:4 }}>
                {opt.title}
              </div>
              <div style={{ color:C.textDim, fontSize:12, lineHeight:1.5 }}>
                {opt.desc}
              </div>
            </div>
            <div style={{ color:hovered === opt.id ? C.accent : C.textMuted, fontSize:18, flexShrink:0 }}>
              →
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const MODEL_META = {
  m1: { label:"M1 · Graph RAG",          tag:"M1", color:"#7ec87e", bg:"#1e3a1e" },
  m2: { label:"M2 · Multimodal RAG",     tag:"M2", color:"#6e9bcf", bg:"#1e2a3a" },
  m3: { label:"M3 · Conversational RAG", tag:"M3", color:"#c9a96e", bg:"#2a200e" },
};

function ModelBadge({ model, size = "normal" }) {
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

// ── Chat Page ─────────────────────────────────────────────────────────────────
function ChatPage({ user, onLogout, selectedModel, onResetModel }) {
  const [sessions, setSessions]     = useState([]);
  const [activeSession, setActive]  = useState(null);
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [sending, setSending]       = useState(false);
  const [sidebarOpen, setSidebar]   = useState(true);
  const [forceNewSession, setForceNew] = useState(false);
  const [awaitingConsent, setAwaitingConsent] = useState(false);
  const messagesEndRef              = useRef(null);
  const inputRef                    = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior:"smooth" });
  useEffect(() => { scrollToBottom(); }, [messages, sending]);

  // Load sessions on mount
  useEffect(() => { loadSessions(); }, []);

  async function loadSessions() {
    try {
      const data = await apiGetSessions(user.user_id);
      setSessions(data.sessions || []);
    } catch(e) { console.error("loadSessions failed", e); }
  }

  function mapHistoryMessages(data, sessionId) {
    return (data.messages || []).map(m => ({
      id: m.turn_id || Math.random(),
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
      label: m.label,
      // Restore the product cards and their "why this for you" reasons so a
      // reopened chat looks the same as when it was live.
      items: m.items_recommended || [],
      // Catalogue values that changed after this message quoted them.
      corrections: m.corrections || [],
      recommendation_id: m.recommendation_id || null,
      turn_id: m.turn_id || null,
      session_id: sessionId,
      isConsentQuestion: m.role === "assistant"
        && (m.content || "").includes(CONSENT_TRIGGER),
    }));
  }

  async function selectSession(session) {
    setAwaitingConsent(false);
    setActive(session.session_id);
    try {
      const data = await apiGetHistory(session.session_id, user.user_id);
      setMessages(mapHistoryMessages(data, session.session_id));
    } catch(e) { console.error("selectSession failed", e); }
  }

  /**
   * Re-reads the transcript from the server after a catalogue revision.
   *
   * A revision annotates messages sent EARLIER in this conversation, and only
   * the server knows which assistant turns quoted the old value. Re-reading is
   * cheaper to reason about than trying to locate those messages client-side,
   * and it happens only on the rare turn where a value actually changed.
   */
  async function refreshTranscript(sessionId) {
    if (!sessionId) return;
    try {
      const data = await apiGetHistory(sessionId, user.user_id);
      setMessages(mapHistoryMessages(data, sessionId));
    } catch(e) { console.error("transcript refresh failed", e); }
  }

  async function newChat() {
    // Clear Redis active session pointer
    try { await apiNewSession(user.user_id); } catch(e) {}
    // Return to model selection — the new session will use a fresh model choice
    onResetModel();
  }

  async function handleDeleteSession(sessionId) {
    if (!window.confirm("Delete this chat? All data will be removed.")) return;
    try {
      await apiDeleteSession(sessionId, user.user_id);
      if (sessionId === activeSession) { setActive(null); setMessages([]); }
      await loadSessions();
    } catch(e) { console.error("delete failed", e); }
  }

  async function handleFeedback(msg, rating) {
    // Optimistically update UI immediately
    setMessages(prev => prev.map(m =>
      m.id === msg.id ? { ...m, feedbackGiven: rating } : m
    ));
    // Send to backend for RL signal collection
    await apiSubmitFeedback({
      sessionId:        msg.session_id || activeSession,
      userId:           user.user_id,
      recommendationId: msg.recommendation_id,
      turnId:           msg.turn_id,
      rating,
      articleIds:       (msg.items || []).map(i => i.article_id).filter(Boolean),
    });
  }

  async function handleConsentYes() {
    setAwaitingConsent(false);
    setMessages(prev => [...prev, {
      id: Date.now(), role:"user", content:"Yes",
      timestamp: new Date().toISOString(),
    }]);
    setSending(true);
    try {
      const res = await apiSendMessage({
        userId: user.user_id, customerId: user.customer_id,
        message: "yes", sessionId: activeSession, selectedModel,
      });
      const botMsg = {
        id: Date.now()+1, role:"assistant",
        content: res.response_text,
        timestamp: new Date().toISOString(),
        label: res.label, confidence: res.confidence,
        items: res.items_recommended || [],
        hallucination_flag: res.hallucination_flag,
        contradiction_found: res.contradiction_found,
        recommendation_id: res.recommendation_id || null,
        turn_id: res.turn_id || null,
        session_id: res.session_id || null,
        feedbackGiven: null,
        isConsentQuestion: (res.response_text||"").includes(CONSENT_TRIGGER),
      };
      setMessages(prev => [...prev, botMsg]);
      if (botMsg.isConsentQuestion) setAwaitingConsent(true);
      if ((res.revisions || []).length > 0) {
        await refreshTranscript(res.session_id || activeSession);
      }
    } catch(e) {
      setMessages(prev => [...prev, {
        id:Date.now()+1, role:"assistant",
        content:"Sorry, something went wrong. Please try again.",
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  async function handleConsentNo() {
    setAwaitingConsent(false);
    setMessages(prev => [...prev, {
      id: Date.now(), role:"user", content:"No",
      timestamp: new Date().toISOString(),
    }]);
    // Call backend silently to clear the consent flag — don't show bot response
    try {
      await apiSendMessage({
        userId: user.user_id, customerId: user.customer_id,
        message: "no", sessionId: activeSession, selectedModel,
      });
    } catch(e) {}
    inputRef.current?.focus();
  }

  async function send() {
    const text = input.trim();
    if (!text || sending || awaitingConsent) return;

    const userMsg = { id:Date.now(), role:"user", content:text,
                      timestamp:new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setSending(true);

    try {
      const res = await apiSendMessage({
        userId: user.user_id,
        customerId: user.customer_id,
        message: text,
        sessionId: activeSession,
        forceNew: forceNewSession,
        selectedModel,
      });
      setForceNew(false);  // reset after first message of new chat

      // Set active session FIRST before any state updates
      const newSessionId = res.session_id;
      if (!activeSession) setActive(newSessionId);

      const botMsg = {
        id: Date.now()+1, role:"assistant",
        content: res.response_text,
        timestamp: new Date().toISOString(),
        label: res.label,
        confidence: res.confidence,
        items: res.items_recommended || [],
        hallucination_flag: res.hallucination_flag,
        contradiction_found: res.contradiction_found,
        recommendation_id: res.recommendation_id || null,
        turn_id: res.turn_id || null,
        session_id: res.session_id || null,
        feedbackGiven: null,
        isConsentQuestion: (res.response_text||"").includes(CONSENT_TRIGGER),
      };
      setMessages(prev => [...prev, botMsg]);
      if (botMsg.isConsentQuestion) setAwaitingConsent(true);

      // A catalogue value changed mid-conversation: pull the transcript back
      // from the server so the correction note appears under the earlier
      // message that quoted the old value.
      if ((res.revisions || []).length > 0) {
        await refreshTranscript(newSessionId || activeSession);
      }

      // Reload sessions — retry up to 3 times to handle MongoDB write delay
      const reloadWithRetry = async (retries = 3, delay = 600) => {
        for (let i = 0; i < retries; i++) {
          await new Promise(r => setTimeout(r, delay));
          const data = await apiGetSessions(user.user_id);
          const list = data.sessions || [];
          // Check if our new session is in the list
          if (list.some(s => s.session_id === newSessionId) || i === retries - 1) {
            setSessions(list);
            break;
          }
        }
      };
      reloadWithRetry();
    } catch(e) {
      setMessages(prev => [...prev, {
        id:Date.now()+1, role:"assistant",
        content:"Sorry, something went wrong. Please try again.",
        timestamp:new Date().toISOString(),
      }]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  const handleKey = e => {
    if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div style={{ height:"100vh", display:"flex", background:C.bg,
      fontFamily:"system-ui,-apple-system,sans-serif", overflow:"hidden" }}>

      {/* Sidebar */}
      <div style={{ width:sidebarOpen?260:0, flexShrink:0, background:C.sidebar,
        borderRight:`1px solid ${C.border}`, display:"flex", flexDirection:"column",
        overflow:"hidden", transition:"width 0.25s ease" }}>
        <div style={{ padding:"18px 14px 12px", borderBottom:`1px solid ${C.border}` }}>
          <div style={{ fontSize:18, fontWeight:700, color:C.accent, letterSpacing:3,
            fontFamily:"'Playfair Display',Georgia,serif", marginBottom:12 }}>SUNLYTICS</div>
          <button onClick={newChat}
            style={{ width:"100%", background:"transparent", border:`1px solid ${C.border}`,
              borderRadius:8, color:C.textDim, padding:"8px 12px", fontSize:13,
              cursor:"pointer", display:"flex", alignItems:"center", gap:8, transition:"all 0.15s" }}
            onMouseEnter={e=>{e.currentTarget.style.borderColor=C.accent;e.currentTarget.style.color=C.accent;}}
            onMouseLeave={e=>{e.currentTarget.style.borderColor=C.border;e.currentTarget.style.color=C.textDim;}}>
            <span style={{fontSize:16}}>+</span> New Chat
          </button>
        </div>

        <div style={{ flex:1, overflowY:"auto", padding:"10px" }}>
          {(() => {
            const filtered = sessions.filter(s =>
              (s.selected_model || "m3") === selectedModel
            );
            if (filtered.length === 0)
              return <div style={{color:C.textMuted,fontSize:12,padding:"16px 4px"}}>No previous chats yet.</div>;
            return filtered.map(s => (
              <SidebarItem key={s.session_id} session={s}
                active={s.session_id === activeSession}
                onSelect={selectSession}
                onDelete={handleDeleteSession} />
            ));
          })()}
        </div>

        <div style={{ padding:"8px 14px", borderTop:`1px solid ${C.border}` }}>
          <ModelBadge model={selectedModel} size="small" />
        </div>

        <div style={{ padding:"10px 14px", borderTop:`1px solid ${C.border}` }}>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <div style={{ width:32, height:32, borderRadius:"50%", background:C.user,
              border:"1px solid #2d5a3d", display:"flex", alignItems:"center",
              justifyContent:"center", color:C.accent, fontSize:13, fontWeight:700, flexShrink:0 }}>
              {user.age?user.age.toString()[0]:"U"}
            </div>
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{ color:C.text, fontSize:12, fontWeight:500,
                whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
                {user.customer_id?.slice(0,18)}...
              </div>
              <div style={{ color:C.textDim, fontSize:11 }}>
                {user.purchase_summary?.budget_tier || ""}
              </div>
            </div>
            <button onClick={onLogout} title="Sign out"
              style={{ background:"transparent", border:"none", color:C.textMuted,
                cursor:"pointer", fontSize:18, padding:"2px 6px", borderRadius:6 }}
              onMouseEnter={e=>e.currentTarget.style.color=C.flagText}
              onMouseLeave={e=>e.currentTarget.style.color=C.textMuted}>↩</button>
          </div>
        </div>
      </div>

      {/* Main chat area */}
      <div style={{ flex:1, display:"flex", flexDirection:"column", minWidth:0 }}>
        {/* Header */}
        <div style={{ padding:"0 20px", height:56, borderBottom:`1px solid ${C.border}`,
          display:"flex", alignItems:"center", gap:12, background:C.bg, flexShrink:0 }}>
          <button onClick={()=>setSidebar(v=>!v)}
            style={{ background:"transparent", border:"none", color:C.textDim,
              cursor:"pointer", fontSize:18, padding:"4px 8px", borderRadius:6 }}>☰</button>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <ModelBadge model={selectedModel} size="normal" />
            <span style={{ color:C.textMuted, fontSize:12 }}>
              {activeSession ? `Session · ${activeSession.slice(-8)}` : "New conversation"}
            </span>
          </div>
        </div>

        {/* Messages */}
        <div style={{ flex:1, overflowY:"auto", padding:"24px 0 8px" }}>
          {messages.length === 0 && !sending ? (
            <div style={{ height:"100%", display:"flex", flexDirection:"column",
              alignItems:"center", justifyContent:"center",
              color:C.textMuted, padding:40 }}>
              <div style={{ fontSize:42, letterSpacing:6, color:C.accentDim,
                fontFamily:"'Playfair Display',Georgia,serif", marginBottom:12 }}>SUNLYTICS</div>
              <div style={{ fontSize:14, maxWidth:340, textAlign:"center", lineHeight:1.7 }}>
                Your personalised fashion assistant.<br/>Tell me what you are looking for today.
              </div>
              <div style={{ marginTop:28, display:"flex", gap:10, flexWrap:"wrap", justifyContent:"center" }}>
                {["I want a black dress under £50","Show me casual summer tops","Need 3 shirts in different colours"].map(s => (
                  <button key={s} onClick={()=>{setInput(s);inputRef.current?.focus();}}
                    style={{ background:C.card, border:`1px solid ${C.border}`,
                      borderRadius:20, color:C.textDim, padding:"8px 16px",
                      fontSize:12, cursor:"pointer", transition:"all 0.15s" }}
                    onMouseEnter={e=>{e.currentTarget.style.borderColor=C.accent;e.currentTarget.style.color=C.accent;}}
                    onMouseLeave={e=>{e.currentTarget.style.borderColor=C.border;e.currentTarget.style.color=C.textDim;}}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map(msg => (
              <Message key={msg.id} msg={msg} onFeedback={handleFeedback}
                awaitingConsent={awaitingConsent}
                onConsentYes={handleConsentYes}
                onConsentNo={handleConsentNo} />
            ))}
              {sending && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input bar */}
        <div style={{ padding:"12px 20px 20px", borderTop:`1px solid ${C.border}`,
          background:C.bg, flexShrink:0 }}>
          <div style={{ display:"flex", gap:10, alignItems:"flex-end",
            background:C.card, border:`1px solid ${C.border}`,
            borderRadius:14, padding:"10px 14px" }}>
            <textarea ref={inputRef} value={input}
              onChange={e=>setInput(e.target.value)} onKeyDown={handleKey}
              placeholder={awaitingConsent ? "Please select Yes or No above…" : "Message Sunlytics..."}
              disabled={awaitingConsent}
              rows={1}
              style={{ flex:1, background:"transparent", border:"none",
                color: awaitingConsent ? C.textMuted : C.text,
                fontSize:14, resize:"none", outline:"none",
                lineHeight:1.6, maxHeight:120, overflowY:"auto",
                fontFamily:"system-ui,sans-serif",
                cursor: awaitingConsent ? "not-allowed" : "text" }}
              onInput={e=>{
                e.target.style.height="auto";
                e.target.style.height=Math.min(e.target.scrollHeight,120)+"px";
              }} />
            <button onClick={send} disabled={!input.trim()||sending||awaitingConsent}
              style={{ background:input.trim()&&!sending&&!awaitingConsent?C.accent:C.textMuted,
                border:"none", borderRadius:9, width:36, height:36,
                display:"flex", alignItems:"center", justifyContent:"center",
                cursor:input.trim()&&!sending&&!awaitingConsent?"pointer":"not-allowed",
                fontSize:16, flexShrink:0, transition:"background 0.2s" }}>
              {sending?"…":"↑"}
            </button>
          </div>
          <div style={{ textAlign:"center", fontSize:10, color:C.textMuted, marginTop:8 }}>
            Sunlytics CRS · Fashion Recommendation Research System
          </div>
        </div>
      </div>

      <style>{`
        @keyframes bounce {
          0%,100%{transform:translateY(0);opacity:0.4;}
          50%{transform:translateY(-5px);opacity:1;}
        }
        *{box-sizing:border-box;}
        ::-webkit-scrollbar{width:4px;}
        ::-webkit-scrollbar-track{background:transparent;}
        ::-webkit-scrollbar-thumb{background:#333;border-radius:4px;}
        select option{background:#1c1c1c;color:#f0ebe3;}
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');
      `}</style>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("sunlytics_user")); }
    catch { return null; }
  });
  // selectedModel is kept in state only (not localStorage) so every fresh login
  // requires a new model selection — one model per session, per login.
  const [selectedModel, setSelectedModel] = useState(null);

  const handleLogin       = u => { localStorage.setItem("sunlytics_user", JSON.stringify(u)); setUser(u); };
  const handleLogout      = () => { localStorage.removeItem("sunlytics_user"); setUser(null); setSelectedModel(null); };
  const handleModelSelect = model => setSelectedModel(model);
  const handleResetModel  = () => setSelectedModel(null);

  if (!user)          return <LoginPage onLogin={handleLogin} />;
  if (!selectedModel) return <ModelSelectPage onSelect={handleModelSelect} />;
  return (
    <ChatPage
      user={user}
      onLogout={handleLogout}
      selectedModel={selectedModel}
      onResetModel={handleResetModel}
    />
  );
}
//isura