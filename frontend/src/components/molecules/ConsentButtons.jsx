import Button from "../atoms/Button";
import { C } from "../../styles/theme";

/** Yes/No pair replacing the composer when the assistant asks to re-run a search. */
export default function ConsentButtons({ onYes, onNo }) {
  const base = { borderRadius:8, padding:"6px 18px", fontSize:13,
    cursor:"pointer", fontWeight:600, transition:"all 0.15s" };
  return (
    <div style={{ display:"flex", gap:8, marginTop:10 }}>
      <Button onClick={onYes}
        style={{ ...base, background:"#1e3a2f", border:"1px solid #2d5a3d", color:"#7ec87e" }}
        hoverStyle={{ background:"#253f35" }}>
        Yes
      </Button>
      <Button onClick={onNo}
        style={{ ...base, background:"#1c1c1c", border:`1px solid ${C.border}`, color:C.textDim }}
        hoverStyle={{ border:"1px solid #555" }}>
        No
      </Button>
    </div>
  );
}
