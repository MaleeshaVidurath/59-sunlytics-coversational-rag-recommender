import Wordmark from "../atoms/Wordmark";
import ModelOptionCard from "../molecules/ModelOptionCard";
import { C } from "../../styles/theme";
import { MODEL_OPTIONS } from "../../utils/constants";

/** The list of selectable recommendation models, with its heading. */
export default function ModelGrid({ onSelect }) {
  return (
    <div style={{ width:500, maxWidth:"92vw" }}>
      <div style={{ textAlign:"center", marginBottom:36 }}>
        <Wordmark size={30} letterSpacing={5} color={C.accent} weight={700} marginBottom={8} />
        <div style={{ fontSize:13, color:C.textDim }}>Select a recommendation model to begin</div>
      </div>
      {MODEL_OPTIONS.map(opt => (
        <ModelOptionCard key={opt.id} option={opt} onSelect={onSelect} />
      ))}
    </div>
  );
}
