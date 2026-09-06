import Button from "../atoms/Button";
import styles from "./ConsentButtons.module.css";

/** Yes/No pair replacing the composer when the assistant asks to re-run a search. */
export default function ConsentButtons({ onYes, onNo }) {
  return (
    <div className={styles.row}>
      <Button variant="success" onClick={onYes}>Yes</Button>
      <Button variant="neutral" onClick={onNo}>No</Button>
    </div>
  );
}
