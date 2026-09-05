import { useState } from "react";
import ProductImage from "../atoms/ProductImage";
import Avatar from "../atoms/Avatar";
import WhyList from "./WhyList";
import { M2_IMAGE_BASE } from "../../services/http";
import { C } from "../../styles/theme";

/** One recommended article: photo, name, colour/type/price, and its reasons. */
export default function ProductCard({ item, model }) {
  const [preview, setPreview] = useState(false);
  const imgSrc = `${M2_IMAGE_BASE}/api/images/${item.article_id}`;

  return (
    <div style={{ background:"#1a1a1a", border:`1px solid ${C.border}`,
      borderRadius:10, padding:"10px 14px", marginTop:8,
      display:"flex", alignItems:"flex-start", gap:14 }}>

      {item.article_id
        ? <ProductImage src={imgSrc} alt={item.name || "product"} onClick={() => setPreview(true)} />
        : <Avatar size={80} radius={10} fontSize={26}>👗</Avatar>}

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
        <WhyList reasons={item.why || []} model={model} />
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
