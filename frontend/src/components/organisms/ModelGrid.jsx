import Wordmark from "../atoms/Wordmark";
import ModelOptionCard from "../molecules/ModelOptionCard";
import { MODEL_OPTIONS } from "../../utils/constants";
import styles from "./ModelGrid.module.css";

/** The list of selectable recommendation models, with its heading. */
export default function ModelGrid({ onSelect }) {
  return (
    <div className={styles.grid}>
      <div className={styles.heading}>
        <Wordmark size={30} letterSpacing={5} marginBottom={8} />
        <div className={styles.subtitle}>Select a recommendation model to begin</div>
      </div>
      {MODEL_OPTIONS.map(opt => (
        <ModelOptionCard key={opt.id} option={opt} onSelect={onSelect} />
      ))}
    </div>
  );
}
