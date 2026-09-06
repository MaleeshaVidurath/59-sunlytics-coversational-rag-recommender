import { useDispatch, useSelector } from "react-redux";
import CenteredTemplate from "../components/templates/CenteredTemplate";
import CredentialsForm from "../components/organisms/CredentialsForm";
import Wordmark from "../components/atoms/Wordmark";
import Button from "../components/atoms/Button";
import { register, selectAuthError, authErrorCleared } from "../store/slices/authSlice";
import styles from "./AuthCard.module.css";

/**
 * New accounts start cold: no linked H&M persona, so no purchase history.
 * Recommendations are ranked semantically until preferences build up through
 * conversation. Said plainly on the form so the difference is not a surprise.
 */
export default function RegisterPage({ onShowLogin }) {
  const dispatch = useDispatch();
  const error = useSelector(selectAuthError);

  return (
    <CenteredTemplate font="serif">
      <div className={styles.card}>
        <Wordmark size={36} letterSpacing={6} marginBottom={8} />
        <div className={`${styles.subtitle} ${styles.subtitleTight}`}>Create an account</div>

        <div className={styles.notice}>
          A new account starts with no shopping history, so early recommendations
          are based on what you ask for rather than what you have bought before.
        </div>

        <CredentialsForm
          mode="register"
          error={error}
          onDismissError={() => dispatch(authErrorCleared())}
          onSubmit={({ username, password }) => dispatch(register({ username, password }))}
        />

        <div className={styles.switch}>
          Already have an account?{" "}
          <Button variant="link" onClick={onShowLogin}>Sign in</Button>
        </div>
      </div>
    </CenteredTemplate>
  );
}
